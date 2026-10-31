r"""Post-hoc temporal analysis of forward linear innovations Ridge models.

The analysis reuses the saved blind feature-matrix artifacts. It does not fit
models, refit transformations, or alter any result files.

Example:
  python -m waveform_analysis.scripts.analyze_linear_innovations ^
    --results "UC=C:\results\UC" "FBK=C:\results\FBK" ^
    --cache-root "UC=C:\cache\UC_cache" "FBK=C:\cache\FBK_cache" ^
    --output-root "C:\results\linear_innovation_analysis" ^
    --tshap-root "C:\results\xai"

Cache roots are optional only when the artifact-store paths recorded in the
result configurations are already valid on this machine. Each cache root must
contain preprocessing/ and artifact_store/.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.feature_cache import (
    _CACHE_VERSION,
    prepare_frozen_features,
    prepare_frozen_transform,
)
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import batch_root, discover_runs, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, model_target
from waveform_analysis.engine.train import (
    load_fitted_model,
    release_training_memory,
    saved_model_complete,
)
from waveform_analysis.models import get_model
from waveform_analysis.reporting.stats import metric_values, paired_central_metrics


ANALYSIS_VERSION = 1
INNOVATION_MODELS = ("direct_linear_innovations", "shared_linear_innovations")
FEATURE_CACHE_IMPLEMENTATION_VERSION = _CACHE_VERSION


def _parse_board_paths(values, option_name):
    parsed = {}
    for value in values or ():
        board, separator, raw_path = str(value).partition("=")
        board = board.strip().upper()
        raw_path = raw_path.strip()
        if not separator or board not in {"UC", "FBK"} or not raw_path:
            raise ValueError(
                f"{option_name} entries must have the form UC=PATH or FBK=PATH; got {value!r}"
            )
        if board in parsed:
            raise ValueError(f"{option_name} contains more than one path for {board}")
        parsed[board] = Path(raw_path).expanduser().resolve()
    return parsed


def _apply_cache_root(config, cache_root):
    if cache_root is None:
        return
    root = Path(cache_root).expanduser().resolve()
    preprocessing = root / "preprocessing"
    artifact_root = root / "artifact_store"
    missing = [str(path) for path in (preprocessing, artifact_root) if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Cache bundle at {root} is incomplete; missing: {', '.join(missing)}"
        )
    config["preprocessing"]["cache_dir"] = str(preprocessing)
    config["preprocessing"]["artifact_root"] = str(artifact_root)


def _adapt_feature_matrix(features, formulation, n_samples, model_name):
    """Validate the canonical feature layout for the registered formulation."""
    values = np.asarray(features)
    if values.ndim != 2:
        raise ValueError(f"{model_name} feature cache must be a 2D matrix")
    expected = 2 * int(n_samples) if formulation == "direct" else int(n_samples)
    if values.shape[1] != expected:
        raise ValueError(
            f"{model_name} feature cache has {values.shape[1]} columns; "
            f"expected {expected} for {formulation} formulation. "
            "This usually indicates a stale or incorrectly shared feature cache; "
            "rebuild the formulation-specific feature matrix and retrain the model."
        )
    return values, "paired_detector_innovations" if formulation == "direct" else "shared_detector_difference"


def _load_cached_feature_matrix(spec, frozen_transform, blind, config, logger=None):
    """Load or materialize blind features using the already-fitted frozen transform.

    This writes/reuses only the blind feature matrix. The transform itself is
    resolved separately and is never refitted by this helper.
    """
    if frozen_transform is None or frozen_transform.transform is None:
        raise ValueError(f"{spec.name} has no resolved frozen feature transformation")
    feature_set = prepare_frozen_features(
        spec,
        frozen_transform,
        config,
        blind,
        role="blind",
        cache_root=config["preprocessing"]["cache_dir"],
        logger=logger,
    )
    return feature_set.features, feature_set.identity, frozen_transform.identity

def _load_saved_predictions(run_store, blind, mode):
    if not run_store.predictions_path.is_file():
        raise FileNotFoundError(
            f"Saved blind predictions are missing: {run_store.predictions_path}"
        )
    payload = run_store.load_predictions()
    required = {"event_id", "prediction_ps", "corrected_ps", "led_residual_ps"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Saved blind predictions are missing arrays: {sorted(missing)}")
    saved_ids = np.asarray(payload["event_id"], dtype=np.int64).reshape(-1)
    if np.unique(saved_ids).size != saved_ids.size:
        raise ValueError("Saved blind prediction event IDs are not unique")
    dataset_ids = np.asarray(blind.event_index, dtype=np.int64).reshape(-1)
    if np.unique(dataset_ids).size != dataset_ids.size:
        raise ValueError("Prepared blind dataset event IDs are not unique")
    lookup = {int(event_id): row for row, event_id in enumerate(saved_ids)}
    missing_ids = [int(event_id) for event_id in dataset_ids if int(event_id) not in lookup]
    if missing_ids:
        raise ValueError(
            f"Saved blind predictions do not cover {len(missing_ids)} prepared events; "
            f"first missing IDs: {missing_ids[:5]}"
        )
    order = np.asarray([lookup[int(event_id)] for event_id in dataset_ids], dtype=np.int64)
    prediction = np.asarray(payload["prediction_ps"], dtype=np.float64)[order]
    corrected = np.asarray(payload["corrected_ps"], dtype=np.float64)[order]
    led = np.asarray(payload["led_residual_ps"], dtype=np.float64)[order]
    target = np.asarray(model_target(blind, mode), dtype=np.float64)
    expected = (int(blind.n_events),)
    if any(value.shape != expected for value in (prediction, corrected, led, target)):
        raise ValueError("Blind prediction/target arrays have incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (prediction, corrected, led, target)):
        raise ValueError("Blind prediction/target arrays contain non-finite values")
    if not np.allclose(led, target, rtol=1e-10, atol=1e-10):
        raise ValueError("Saved LED residuals differ from the prepared blind targets")
    if not np.allclose(corrected, target - prediction, rtol=1e-10, atol=1e-10):
        raise ValueError("Saved corrected residuals are inconsistent with target - prediction")
    return dataset_ids, prediction, target, corrected


def _build_temporal_groups(sample_indices, time_ps, full_time_ps, window_samples, stride_samples):
    """Build complete-coverage groups without crossing masked gaps in the source grid."""
    sample_indices = np.asarray(sample_indices, dtype=np.int64).reshape(-1)
    time_ps = np.asarray(time_ps, dtype=np.float64).reshape(-1)
    full_time_ps = np.asarray(full_time_ps, dtype=np.float64).reshape(-1)
    if sample_indices.size != time_ps.size or not sample_indices.size:
        raise ValueError("The retained sample index/time arrays are empty or inconsistent")
    if window_samples < 1 or stride_samples < 1 or stride_samples > window_samples:
        raise ValueError("Require 1 <= stride_samples <= window_samples to avoid temporal gaps")
    if np.any(np.diff(sample_indices) <= 0):
        raise ValueError("Retained sample indices must be strictly increasing")

    boundaries = np.flatnonzero(np.diff(sample_indices) != 1) + 1
    runs = np.split(np.arange(sample_indices.size, dtype=np.int64), boundaries)
    groups = []
    for run_positions in runs:
        for local_start in range(0, len(run_positions), stride_samples):
            positions = run_positions[local_start:local_start + window_samples]
            if not positions.size:
                continue
            first_position = int(positions[0])
            last_position = int(positions[-1])
            first_original = int(sample_indices[first_position])
            last_original = int(sample_indices[last_position])
            if last_original + 1 < full_time_ps.size:
                dt = float(full_time_ps[last_original + 1] - full_time_ps[last_original])
            elif last_original > 0:
                dt = float(full_time_ps[last_original] - full_time_ps[last_original - 1])
            else:
                dt = 1.0
            start_ps = float(time_ps[first_position])
            end_ps = float(time_ps[last_position] + dt)
            groups.append({
                "index": len(groups),
                "positions": np.asarray(positions, dtype=np.int64),
                "start_sample": first_original,
                "stop_sample_exclusive": last_original + 1,
                "width_samples": int(len(positions)),
                "start_ps": start_ps,
                "end_ps": end_ps,
                "center_ps": 0.5 * (start_ps + end_ps),
                "width_ps": end_ps - start_ps,
            })
    covered = np.unique(np.concatenate([group["positions"] for group in groups]))
    if not np.array_equal(covered, np.arange(sample_indices.size, dtype=np.int64)):
        raise RuntimeError("Temporal groups do not cover every retained sample")
    return groups


def _contribution_batch(features, coefficients, formulation, n_samples):
    """Return pair-level contributions and, for direct models, detector contributions."""
    values = np.asarray(features, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if formulation == "direct":
        if values.ndim != 2 or values.shape[1] != 2 * n_samples or weights.size != 2 * n_samples:
            raise ValueError("Direct innovations feature/weight dimensions are inconsistent")
        by_detector = values.reshape(len(values), 2, n_samples) * weights.reshape(1, 2, n_samples)
        return np.sum(by_detector, axis=1), by_detector
    if formulation == "shared":
        if values.ndim != 2 or values.shape[1] != n_samples or weights.size != n_samples:
            raise ValueError("Shared innovations feature/weight dimensions are inconsistent")
        return values * weights[None, :], None
    raise ValueError(f"Unsupported innovations formulation: {formulation}")


def _compute_profiles(features, rows, coefficients, intercept, formulation,
                      n_samples, groups, batch_size):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if not rows.size:
        raise ValueError("At least one blind event is required for innovation analysis")
    n_groups = len(groups)
    group_contributions = np.empty((len(rows), n_groups), dtype=np.float32)
    sum_abs = np.zeros(n_samples, dtype=np.float64)
    sum_sq = np.zeros(n_samples, dtype=np.float64)
    sum_signed = np.zeros(n_samples, dtype=np.float64)
    detector_abs = np.zeros((2, n_samples), dtype=np.float64) if formulation == "direct" else None
    detector_sq = np.zeros((2, n_samples), dtype=np.float64) if formulation == "direct" else None
    detector_signed = np.zeros((2, n_samples), dtype=np.float64) if formulation == "direct" else None
    raw_prediction = np.empty(len(rows), dtype=np.float64)

    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        matrix = np.asarray(features[rows[start:stop]], dtype=np.float64)
        pair_contribution, detector_contribution = _contribution_batch(
            matrix, coefficients, formulation, n_samples
        )
        raw_prediction[start:stop] = np.sum(pair_contribution, axis=1) + float(intercept)
        sum_abs += np.sum(np.abs(pair_contribution), axis=0)
        sum_sq += np.sum(pair_contribution * pair_contribution, axis=0)
        sum_signed += np.sum(pair_contribution, axis=0)
        if detector_contribution is not None:
            detector_abs += np.sum(np.abs(detector_contribution), axis=0)
            detector_sq += np.sum(detector_contribution * detector_contribution, axis=0)
            detector_signed += np.sum(detector_contribution, axis=0)
        for group in groups:
            group_contributions[start:stop, group["index"]] = np.sum(
                pair_contribution[:, group["positions"]], axis=1
            ).astype(np.float32)

    count = float(len(rows))
    profile = {
        "mean_abs_pair": sum_abs / count,
        "rms_pair": np.sqrt(sum_sq / count),
        "mean_signed_pair": sum_signed / count,
    }
    if detector_abs is not None:
        profile.update({
            "mean_abs_detector": detector_abs / count,
            "rms_detector": np.sqrt(detector_sq / count),
            "mean_signed_detector": detector_signed / count,
        })
    return profile, group_contributions, raw_prediction


def _profile_only(features, rows, coefficients, formulation, n_samples, batch_size):
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    if not rows.size:
        raise ValueError("Cannot calculate an innovation profile for zero events")
    sum_abs = np.zeros(n_samples, dtype=np.float64)
    sum_sq = np.zeros(n_samples, dtype=np.float64)
    sum_signed = np.zeros(n_samples, dtype=np.float64)
    for start in range(0, len(rows), batch_size):
        selected = rows[start:start + batch_size]
        matrix = np.asarray(features[selected], dtype=np.float64)
        pair, _ = _contribution_batch(matrix, coefficients, formulation, n_samples)
        sum_abs += np.sum(np.abs(pair), axis=0)
        sum_sq += np.sum(pair * pair, axis=0)
        sum_signed += np.sum(pair, axis=0)
    count = float(len(rows))
    return {
        "mean_abs_pair": sum_abs / count,
        "rms_pair": np.sqrt(sum_sq / count),
        "mean_signed_pair": sum_signed / count,
    }


def _safe_correlation(first, second, method="pearson"):
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    if a.shape != b.shape or a.size < 2 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        return float("nan")
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    if method == "spearman":
        result = spearmanr(a, b).statistic
        return float(result) if np.isfinite(result) else float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _normalized_profile(values, time_ps):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    time_ns = np.asarray(time_ps, dtype=np.float64).reshape(-1) / 1000.0
    if values.shape != time_ns.shape or values.size < 2:
        raise ValueError("Profile and time grid must have matching lengths >= 2")
    if np.any(np.diff(time_ns) <= 0):
        raise ValueError("Profile time grid must be strictly increasing")
    area = float(np.trapezoid(np.maximum(values, 0.0), time_ns))
    if not np.isfinite(area) or area <= 0:
        return np.zeros_like(values)
    return np.maximum(values, 0.0) / area


def _metric_or_nan(residuals, fit_config, seed):
    try:
        values = metric_values(residuals, fit_config, seed=int(seed))
        return float(values["ctr_ps"]), float(values["rmse_ps"]), ""
    except Exception as exc:
        residuals = np.asarray(residuals, dtype=np.float64)
        finite = residuals[np.isfinite(residuals)]
        rmse = float(np.sqrt(np.mean(finite * finite))) if finite.size else float("nan")
        return float("nan"), rmse, f"{type(exc).__name__}: {exc}"


def _safe_slug(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return text or "item"


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_contribution_profile(destination, time_ps, profile, model_name, board, mode, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    axes[0].plot(time_ns, profile["mean_abs_pair"], label="Pair", linewidth=1.2)
    if "mean_abs_detector" in profile:
        axes[0].plot(time_ns, profile["mean_abs_detector"][0], label="Detector 1", alpha=0.8)
        axes[0].plot(time_ns, profile["mean_abs_detector"][1], label="Detector 2", alpha=0.8)
    axes[0].set_ylabel("Mean absolute contribution (ps)")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].plot(time_ns, profile["mean_signed_pair"], color="tab:orange", linewidth=1.0)
    axes[1].axhline(0.0, color="0.4", linewidth=0.7)
    axes[1].set_ylabel("Mean signed contribution (ps)")
    axes[1].set_xlabel("Waveform time (ns)")
    fig.suptitle(f"{board} | {mode} | {window} | {model_name}")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_grouped_deletion(destination, groups, deletion, model_name, board, mode, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = np.asarray([group["center_ps"] for group in groups], dtype=np.float64) / 1000.0
    widths = np.asarray([group["width_ps"] for group in groups], dtype=np.float64) / 1000.0
    widths = np.maximum(widths, np.finfo(float).eps)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    axes[0].bar(centers, deletion["delta_ctr_ps"], width=widths, align="center", alpha=0.85)
    axes[0].axhline(0.0, color="0.3", linewidth=0.8)
    axes[0].set_ylabel("CTR change (ps)")
    axes[1].bar(centers, deletion["delta_rmse_ps"], width=widths, align="center", alpha=0.85)
    axes[1].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1].set_ylabel("RMSE change (ps)")
    axes[1].set_xlabel("Innovation window time (ns)")
    fig.suptitle(f"Grouped innovation deletion | {board} | {mode} | {window} | {model_name}")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _write_run_artifacts(destination, identity, fingerprint, board, config, spec,
                         event_ids, prediction, target, time_ps, sample_indices,
                         profile, groups, deletion, baseline, summary_extra,
                         overwrite):
    destination = Path(destination)
    summary_path = destination / "summary.json"
    if destination.exists() and not overwrite:
        if summary_path.is_file():
            try:
                previous = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                previous = {}
            if previous.get("identity_sha256") == fingerprint:
                return False
        raise FileExistsError(
            f"Analysis output already exists with a different identity: {destination}. "
            "Use --overwrite to replace it."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".linear-innovation-analysis-", dir=destination.parent))
    try:
        profile_arrays = {
            "time_ps": np.asarray(time_ps, dtype=np.float64),
            "sample_indices": np.asarray(sample_indices, dtype=np.int64),
            **{key: np.asarray(value) for key, value in profile.items()},
        }
        np.savez_compressed(temporary / "contribution_profiles.npz", **profile_arrays)
        np.savez_compressed(
            temporary / "evaluation.npz",
            event_id=np.asarray(event_ids, dtype=np.int64),
            prediction_ps=np.asarray(prediction, dtype=np.float64),
            target_ps=np.asarray(target, dtype=np.float64),
        )
        group_rows = []
        for group, values in zip(groups, zip(
            deletion["mean_abs_group_contribution_ps"],
            deletion["rms_group_contribution_ps"],
            deletion["mean_signed_group_contribution_ps"],
            deletion["deleted_ctr_ps"], deletion["delta_ctr_ps"],
            deletion["deleted_rmse_ps"], deletion["delta_rmse_ps"],
            deletion["ctr_error"],
        )):
            mean_abs, rms, signed, deleted_ctr, delta_ctr, deleted_rmse, delta_rmse, error = values
            group_rows.append({
                "group_index": group["index"],
                "start_sample": group["start_sample"],
                "stop_sample_exclusive": group["stop_sample_exclusive"],
                "width_samples": group["width_samples"],
                "start_ps": group["start_ps"],
                "end_ps": group["end_ps"],
                "center_ps": group["center_ps"],
                "mean_abs_group_contribution_ps": mean_abs,
                "rms_group_contribution_ps": rms,
                "mean_signed_group_contribution_ps": signed,
                "baseline_ctr_ps": baseline["ctr_ps"],
                "deleted_ctr_ps": deleted_ctr,
                "delta_ctr_ps": delta_ctr,
                "baseline_rmse_ps": baseline["rmse_ps"],
                "deleted_rmse_ps": deleted_rmse,
                "delta_rmse_ps": delta_rmse,
                "ctr_error": error,
            })
        _write_csv(
            temporary / "contribution_profile.csv",
            [
                {
                    "sample_position": i,
                    "original_sample_index": int(sample_indices[i]),
                    "time_ps": float(time_ps[i]),
                    "mean_abs_pair_contribution_ps": float(profile["mean_abs_pair"][i]),
                    "rms_pair_contribution_ps": float(profile["rms_pair"][i]),
                    "mean_signed_pair_contribution_ps": float(profile["mean_signed_pair"][i]),
                    **({
                        "mean_abs_detector_1_contribution_ps": float(profile["mean_abs_detector"][0, i]),
                        "mean_abs_detector_2_contribution_ps": float(profile["mean_abs_detector"][1, i]),
                        "rms_detector_1_contribution_ps": float(profile["rms_detector"][0, i]),
                        "rms_detector_2_contribution_ps": float(profile["rms_detector"][1, i]),
                        "mean_signed_detector_1_contribution_ps": float(profile["mean_signed_detector"][0, i]),
                        "mean_signed_detector_2_contribution_ps": float(profile["mean_signed_detector"][1, i]),
                    } if "mean_abs_detector" in profile else {}),
                }
                for i in range(len(time_ps))
            ],
            [
                "sample_position", "original_sample_index", "time_ps",
                "mean_abs_pair_contribution_ps", "rms_pair_contribution_ps",
                "mean_signed_pair_contribution_ps",
                "mean_abs_detector_1_contribution_ps", "mean_abs_detector_2_contribution_ps",
                "rms_detector_1_contribution_ps", "rms_detector_2_contribution_ps",
                "mean_signed_detector_1_contribution_ps", "mean_signed_detector_2_contribution_ps",
            ],
        )
        _write_csv(
            temporary / "grouped_deletion.csv",
            group_rows,
            list(group_rows[0]) if group_rows else [],
        )
        deletion_arrays = {
            "group_index": np.asarray([g["index"] for g in groups], dtype=np.int64),
            "group_start_sample": np.asarray([g["start_sample"] for g in groups], dtype=np.int64),
            "group_stop_sample_exclusive": np.asarray([g["stop_sample_exclusive"] for g in groups], dtype=np.int64),
            "group_start_ps": np.asarray([g["start_ps"] for g in groups], dtype=np.float64),
            "group_end_ps": np.asarray([g["end_ps"] for g in groups], dtype=np.float64),
            "group_center_ps": np.asarray([g["center_ps"] for g in groups], dtype=np.float64),
            **{key: np.asarray(value) for key, value in deletion.items()
               if key not in {"ctr_error"}},
        }
        np.savez_compressed(temporary / "grouped_deletion.npz", **deletion_arrays)
        _plot_contribution_profile(
            temporary / "contribution_profile.png", time_ps, profile,
            spec.name, board, config["mode"], config["window_name"],
        )
        _plot_grouped_deletion(
            temporary / "grouped_deletion.png", groups, deletion,
            spec.name, board, config["mode"], config["window_name"],
        )
        summary = {
            "analysis_version": ANALYSIS_VERSION,
            "identity": identity,
            "identity_sha256": fingerprint,
            "board": board,
            "model": spec.name,
            "mode": config["mode"],
            "window_name": config["window_name"],
            "n_events": int(len(event_ids)),
            "n_input_samples": int(len(sample_indices)),
            "n_temporal_groups": int(len(groups)),
            "baseline_metrics": baseline,
            "output_files": [
                "contribution_profiles.npz", "contribution_profile.csv",
                "contribution_profile.png", "grouped_deletion.npz",
                "grouped_deletion.csv", "grouped_deletion.png", "evaluation.npz",
            ],
            "interpretation": {
                "contribution": "Per-sample additive contribution of the standardized forward linear innovation to the fitted Ridge correction; pair contribution sums detector contributions in the direct formulation.",
                "deletion": ("Fixed-model deletion: subtract the group's additive contribution before output clipping. "
                         "For direct models this sets both detector innovations to zero at those positions; "
                         "for shared models it removes the detector-difference innovation feature. No retraining is performed."),
                "conditioning": "Each innovation is conditional on preceding retained samples in the forward Cholesky ordering; it is not conditional on future samples.",
                "cohort": "Blind evaluation cohort; this post-hoc analysis is descriptive and is not used for model selection.",
                "masked_samples": "Temporal groups are built separately within contiguous runs of retained original sample indices and never cross masked gaps.",
                "limitations": "Fixed-model deletion measures dependence of the fitted predictor, not irrecoverability of information after retraining on the remaining features.",
            },
            **summary_extra,
        }
        atomic_json(temporary / "summary.json", summary)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
        return True
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_run_record(destination):
    with np.load(destination / "contribution_profiles.npz", allow_pickle=False) as data:
        profile = {key: np.asarray(data[key]) for key in data.files}
    with np.load(destination / "grouped_deletion.npz", allow_pickle=False) as data:
        deletion = {key: np.asarray(data[key]) for key in data.files}
    with np.load(destination / "evaluation.npz", allow_pickle=False) as data:
        evaluation = {key: np.asarray(data[key]) for key in data.files}
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    return {
        "destination": destination,
        "summary": summary,
        "time_ps": profile["time_ps"],
        "sample_indices": profile["sample_indices"],
        "profile": {key: value for key, value in profile.items()
                    if key not in {"time_ps", "sample_indices"}},
        "deletion": deletion,
        "event_ids": evaluation["event_id"],
        "prediction": evaluation["prediction_ps"],
        "target": evaluation["target_ps"],
    }


def _run_output_directory(output_root, board, run_directory):
    batch = batch_root(run_directory)
    relative = Path(run_directory).resolve().relative_to(batch)
    return Path(output_root) / board / batch.name / relative / "linear_innovation_analysis"


def _analyze_run(run, board, output_root, *, cache_root=None,
                 source_artifact_root=None, max_events=0,
                 batch_size=256, window_samples=20, stride_samples=20,
                 seed=42, overwrite=False, tshap_profiles=(), logger=None):
    directory = Path(run["directory"]).resolve()
    manifest = run["manifest"]
    config = resolve_run_config(directory)
    configured_source = config["preprocessing"].get("artifact_root")
    source_root = (
        Path(source_artifact_root).expanduser().resolve()
        if source_artifact_root is not None
        else Path(configured_source).expanduser().resolve()
        if configured_source and Path(configured_source).expanduser().is_dir()
        else None
    )
    if source_artifact_root is not None and not source_root.is_dir():
        raise FileNotFoundError(f"Source artifact store does not exist: {source_root}")

    spec = get_model(manifest["model"])
    if spec.name not in INNOVATION_MODELS or spec.feature_transform is None:
        raise ValueError(f"Expected a linear innovations model, got {spec.name}")

    # Saved model and blind predictions belong to the original run's artifact
    # store, not the new cache being prepared for this post-hoc analysis.
    run_store = RunStore(directory, artifact_root=source_root)
    local_model_directory = directory / "model"
    if run_store.model_reference_path.is_file() and source_root is not None:
        model_directory = run_store.model_dir
    elif saved_model_complete(spec, local_model_directory):
        model_directory = local_model_directory
    else:
        raise FileNotFoundError(
            f"Cannot resolve the saved model for {directory}. "
            "Provide --source-artifact-root pointing to the original run artifact_store, "
            "or ensure the run contains a local model/ directory."
        )
    if not saved_model_complete(spec, model_directory):
        raise FileNotFoundError(f"Saved innovations model is incomplete: {model_directory}")

    model_artifact_manifest = model_directory / ".artifact.json"
    expected_transform_identity = None
    if model_artifact_manifest.is_file():
        try:
            artifact_metadata = json.loads(model_artifact_manifest.read_text(encoding="utf-8"))
            expected_transform_identity = (
                artifact_metadata.get("recipe", {}).get("parameters", {})
                .get("feature_transform_identity")
            )
        except (OSError, ValueError):
            expected_transform_identity = None

    fitted = load_fitted_model(
        spec, model_directory, run["best"].get("parameters", {}), config
    )
    if expected_transform_identity is None:
        expected_transform_identity = fitted.metadata.get("feature_transform_identity")
    if not expected_transform_identity:
        raise ValueError(
            f"Cannot verify the original frozen-transform identity for {spec.name}. "
            "The saved model artifact manifest/metadata must be available; refusing "
            "to analyse features generated by an unverified transform."
        )

    _apply_cache_root(config, cache_root)
    lookup = PreparedCacheLookup()
    control = lookup.find("control", config, manifest)
    development = lookup.find("development", config, manifest)
    blind = lookup.find("blind", config, manifest)
    # Loads the cached transform. If the cache is absent it can deterministically
    # reconstruct it from the same control cohort; identity must still match the
    # transform used by the saved final model.
    frozen_transform = prepare_frozen_transform(
        spec,
        config["model"]["space"],
        config,
        control_dataset=control,
        development_dataset=development,
        cache_root=config["preprocessing"]["cache_dir"],
        logger=logger,
    )
    if frozen_transform is None:
        raise ValueError(f"{spec.name} frozen innovations transform could not be resolved")
    if frozen_transform.identity != str(expected_transform_identity):
        raise ValueError(
            f"Frozen-transform identity mismatch for {spec.name}: "
            f"saved model={expected_transform_identity}, analysis cache={frozen_transform.identity}. "
            "The analysis cache was built from different data/configuration or with a "
            "different transform implementation. Do not proceed with this comparison."
        )
    fitted.feature_transform = frozen_transform.transform
    family = mode_family(config["mode"])
    full_time_ps = np.asarray(getattr(blind, f"{family}_time_ps"), dtype=np.float64)
    sample_mask = np.asarray(fitted.sample_mask, dtype=bool).reshape(-1)
    if sample_mask.shape != full_time_ps.shape:
        raise ValueError(
            f"Saved model sample mask {sample_mask.shape} differs from waveform grid {full_time_ps.shape}"
        )
    if not np.array_equal(sample_mask, np.asarray(
        getattr(blind, "sample_mask", np.ones_like(sample_mask)) if config["mode"] != "energy_to_energy"
        else np.ones_like(sample_mask), dtype=bool
    )):
        # model_sample_mask() is the authoritative cross-role mask; import lazily
        # to keep this check identical to training.
        from waveform_analysis.data.view import model_sample_mask
        if not np.array_equal(sample_mask, model_sample_mask(blind, config["mode"])):
            raise ValueError("Saved model sample mask differs from the prepared blind mask")
    time_ps = full_time_ps[sample_mask]
    sample_indices = np.flatnonzero(sample_mask)
    n_samples = len(time_ps)
    if n_samples < 2:
        raise ValueError("Innovations model has fewer than two retained waveform samples")

    features, feature_product_id, transform_identity = _load_cached_feature_matrix(
        spec, frozen_transform, blind, config, logger=logger
    )
    formulation = "direct" if spec.name.startswith("direct_") else "shared"
    coefficients = np.asarray(
        fitted.artifact.regressor.coef_, dtype=np.float64
    ).reshape(-1)
    features, feature_matrix_adapter = _adapt_feature_matrix(
        features, formulation, n_samples, spec.name
    )
    if coefficients.size != features.shape[1]:
        raise ValueError(
            f"{spec.name} saved Ridge has {coefficients.size} coefficients, "
            f"but its canonical {formulation} input has {features.shape[1]} features. "
            "The saved model was trained with an incompatible feature layout; "
            "retrain it before temporal analysis."
        )
    contribution_formulation = formulation

    event_ids, saved_prediction, target, corrected = _load_saved_predictions(
        run_store, blind, config["mode"]
    )
    rows = np.arange(blind.n_events, dtype=np.int64)
    if max_events and max_events < len(rows):
        selection_seed = semantic_seed(
            int(seed), board, config["mode"], config["window_name"], "innovation_analysis_events"
        )
        rows = np.sort(np.random.default_rng(selection_seed).choice(
            rows, size=int(max_events), replace=False
        ))
    selected_event_ids = event_ids[rows]
    intercept = float(np.asarray(fitted.artifact.regressor.intercept_).reshape(()))
    if coefficients.size != features.shape[1]:
        raise ValueError(
            f"Saved Ridge coefficient count ({coefficients.size}) differs from "
            f"adapted feature count ({features.shape[1]})"
        )
    groups = _build_temporal_groups(
        sample_indices, time_ps, full_time_ps, window_samples, stride_samples
    )
    profile, group_contributions, raw_prediction = _compute_profiles(
        features, rows, coefficients, intercept, contribution_formulation, n_samples, groups, batch_size
    )
    limit = fitted.output_max_abs_ps
    replay_prediction = (
        raw_prediction if limit is None
        else np.clip(raw_prediction, -float(limit), float(limit))
    )
    replay_error = float(np.max(np.abs(replay_prediction - saved_prediction[rows])))
    if not np.allclose(replay_prediction, saved_prediction[rows], rtol=2e-5, atol=2e-3):
        raise ValueError(
            f"Innovation contributions do not reproduce saved predictions "
            f"(max absolute difference {replay_error:.6g} ps). "
            "The cached feature matrix and final model may not belong to the same run."
        )
    corrected_selected = target[rows] - replay_prediction
    metric_seed = semantic_seed(int(config["seed"]), spec.name, "innovation_deletion")
    baseline_ctr, baseline_rmse, baseline_error = _metric_or_nan(
        corrected_selected, config["fit"], metric_seed
    )
    central = paired_central_metrics(
        corrected_selected, target[rows], config["fit"],
        seed=semantic_seed(int(config["seed"]), spec.name, "blind_central"),
    )
    deletion = {
        "mean_abs_group_contribution_ps": np.mean(np.abs(group_contributions), axis=0),
        "rms_group_contribution_ps": np.sqrt(np.mean(group_contributions.astype(np.float64) ** 2, axis=0)),
        "mean_signed_group_contribution_ps": np.mean(group_contributions, axis=0),
        "deleted_ctr_ps": np.full(len(groups), np.nan, dtype=np.float64),
        "delta_ctr_ps": np.full(len(groups), np.nan, dtype=np.float64),
        "deleted_rmse_ps": np.full(len(groups), np.nan, dtype=np.float64),
        "delta_rmse_ps": np.full(len(groups), np.nan, dtype=np.float64),
        "ctr_error": [],
    }
    for group in groups:
        index = group["index"]
        raw_deleted = raw_prediction - group_contributions[:, index].astype(np.float64)
        deleted_prediction = (
            raw_deleted if limit is None
            else np.clip(raw_deleted, -float(limit), float(limit))
        )
        deleted_corrected = target[rows] - deleted_prediction
        ctr, rmse, error = _metric_or_nan(deleted_corrected, config["fit"], metric_seed)
        deletion["deleted_ctr_ps"][index] = ctr
        deletion["delta_ctr_ps"][index] = ctr - baseline_ctr
        deletion["deleted_rmse_ps"][index] = rmse
        deletion["delta_rmse_ps"][index] = rmse - baseline_rmse
        deletion["ctr_error"].append(error)

    direct_symmetry_ratio = None
    if formulation == "direct":
        weights = coefficients.reshape(2, n_samples)
        denominator = float(np.linalg.norm(weights[0]) + np.linalg.norm(weights[1]))
        direct_symmetry_ratio = (
            float(np.linalg.norm(weights[0] + weights[1]) / denominator)
            if denominator > 0 else float("nan")
        )

    identity = {
        "final_fit_fingerprint": manifest.get("stage_fingerprints", {}).get("final_fit"),
        "blind_fingerprint": manifest.get("stage_fingerprints", {}).get("blind"),
        "blind_dataset_identity": blind.manifest.get("analysis_protocol_identity"),
        "transform_identity": transform_identity,
        "feature_matrix_product_id": feature_product_id,
        "model": spec.name,
        "mode": config["mode"],
        "window_name": config["window_name"],
        "board": board,
        "max_events": int(max_events),
        "batch_size": int(batch_size),
        "window_samples": int(window_samples),
        "stride_samples": int(stride_samples),
        "seed": int(seed),
        "analysis_version": ANALYSIS_VERSION,
    }
    fingerprint = canonical_hash(identity)
    destination = _run_output_directory(output_root, board, directory)
    summary_extra = {
        "formulation": formulation,
        "feature_matrix_product_id": feature_product_id,
        "feature_matrix_adapter": feature_matrix_adapter,
        "contribution_layout": contribution_formulation,
        "transform_identity": transform_identity,
        "prediction_replay_max_abs_error_ps": replay_error,
        "n_output_clipped_events": int(np.count_nonzero(
            np.abs(raw_prediction) > float(limit)
        )) if limit is not None else 0,
        "direct_antisymmetry_weight_ratio": direct_symmetry_ratio,
        "sample_grouping": {
            "window_samples": int(window_samples),
            "stride_samples": int(stride_samples),
            "complete_coverage": True,
            "masked_gaps_crossed": False,
            "partial_final_groups": True,
        },
        "central_blind_metrics": central,
        "deletion_metric_seed": int(metric_seed),
        "baseline_deletion_metric_error": baseline_error,
    }
    written = _write_run_artifacts(
        destination, identity, fingerprint, board, config, spec,
        selected_event_ids, replay_prediction, target[rows], time_ps, sample_indices,
        profile, groups, deletion,
        {"ctr_ps": baseline_ctr, "rmse_ps": baseline_rmse},
        summary_extra, overwrite,
    )
    if logger:
        logger.info(
            "Innovation analysis | %s | %s | %s | events=%d | samples=%d | groups=%d | cache=%s | feature_adapter=%s",
            spec.name, config["mode"], config["window_name"], len(rows),
            n_samples, len(groups), "written" if written else "hit",
            feature_matrix_adapter,
        )
    record = _load_run_record(destination)

    tshap_rows = _compare_tshap_profiles(
        record, tshap_profiles, features, event_ids, coefficients,
        contribution_formulation, n_samples, full_time_ps, sample_mask, batch_size,
        output_root,
    )
    record["tshap_comparisons"] = tshap_rows
    del fitted, blind, features, group_contributions, raw_prediction
    release_training_memory()
    return record


def _read_tshap_profiles(root):
    profiles = []
    if root is None:
        return profiles
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"TSHAP output root does not exist: {root}")
    for summary_path in sorted(root.rglob("summary.json")):
        parent = summary_path.parent
        attribution_path = parent / "attributions.npz"
        csv_path = parent / "global_importance.csv"
        if not attribution_path.is_file() or not csv_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            identity = summary.get("identity", {}) or {}
            model = str(summary.get("model") or identity.get("model") or parent.parent.name)
            mode = str(summary.get("mode") or identity.get("mode") or "")
            window = str(summary.get("window_name") or "")
            board = str(summary.get("board") or "")
            if board.upper() not in {"UC", "FBK"}:
                board = next(
                    (part.upper() for part in reversed(parent.parts) if part.upper() in {"UC", "FBK"}),
                    "",
                )
            with np.load(attribution_path, allow_pickle=False) as saved:
                if "event_id" not in saved or "time_ps" not in saved:
                    continue
                event_ids = np.asarray(saved["event_id"], dtype=np.int64).reshape(-1)
                time_from_npz = np.asarray(saved["time_ps"], dtype=np.float64).reshape(-1)
            with csv_path.open("r", newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            times = np.asarray([float(row["time_ps"]) for row in rows], dtype=np.float64)
            values = np.asarray([
                float(row["mean_absolute_tshap_importance_ps_per_sample"]) for row in rows
            ], dtype=np.float64)
            if (
                board not in {"UC", "FBK"} or not mode or not window
                or times.shape != values.shape or times.shape != time_from_npz.shape
                or not np.allclose(times, time_from_npz, rtol=0, atol=1e-6)
                or not np.all(np.isfinite(values)) or np.any(values < 0)
            ):
                continue
            profiles.append({
                "path": summary_path,
                "model": model,
                "mode": mode,
                "window": window,
                "board": board,
                "event_ids": event_ids,
                "time_ps": times,
                "importance": values,
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return profiles


def _compare_tshap_profiles(record, tshap_profiles, features, dataset_event_ids,
                            coefficients, formulation, n_samples, full_time_ps,
                            sample_mask, batch_size, output_root):
    comparisons = []
    if not tshap_profiles:
        return comparisons
    board = record["summary"]["board"]
    mode = record["summary"]["mode"]
    window = record["summary"]["window_name"]
    model = record["summary"]["model"]
    event_lookup = {
        int(event_id): row for row, event_id in enumerate(np.asarray(dataset_event_ids, dtype=np.int64))
    }
    time_ps = np.asarray(record["time_ps"], dtype=np.float64)
    for tshap in tshap_profiles:
        if (tshap["board"], tshap["mode"], tshap["window"]) != (board, mode, window):
            continue
        if tshap["time_ps"].shape != full_time_ps.shape or not np.allclose(
            tshap["time_ps"], full_time_ps, rtol=0, atol=1e-6
        ):
            continue
        missing = [int(value) for value in tshap["event_ids"] if int(value) not in event_lookup]
        if missing or not len(tshap["event_ids"]):
            continue
        matched_rows = np.asarray(
            [event_lookup[int(value)] for value in tshap["event_ids"]], dtype=np.int64
        )
        innovation = _profile_only(
            features, matched_rows, coefficients, formulation, n_samples, batch_size
        )["mean_abs_pair"]
        tshap_importance = tshap["importance"][sample_mask]
        if tshap_importance.shape != time_ps.shape:
            continue
        innovation_normalized = _normalized_profile(innovation, time_ps)
        tshap_normalized = _normalized_profile(tshap_importance, time_ps)
        row = {
            "board": board,
            "mode": mode,
            "window": window,
            "innovations_model": model,
            "tshap_model": tshap["model"],
            "tshap_source": str(tshap["path"]),
            "n_matched_events": int(len(matched_rows)),
            "pearson_profile_correlation": _safe_correlation(
                innovation_normalized, tshap_normalized, "pearson"
            ),
            "spearman_profile_correlation": _safe_correlation(
                innovation_normalized, tshap_normalized, "spearman"
            ),
        }
        comparisons.append(row)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        destination = (
            Path(output_root) / "comparisons" / "tshap" / board
            / mode_family(mode) / window
            / f"{_safe_slug(model)}_vs_{_safe_slug(tshap['model'])}_"
              f"{_safe_slug(tshap['path'].parent.parent.name)}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(time_ps / 1000.0, innovation_normalized,
                label=f"{model}: innovation contribution")
        ax.plot(time_ps / 1000.0, tshap_normalized,
                label=f"{tshap['model']}: TSHAP")
        ax.set_xlabel("Waveform time (ns)")
        ax.set_ylabel("Area-normalized importance (1/ns)")
        ax.set_title(
            f"{board} | {mode} | {window} | r={row['pearson_profile_correlation']:.3f}"
        )
        ax.legend(loc="best", fontsize=8)
        fig.savefig(destination, dpi=180)
        plt.close(fig)
    return comparisons


def _write_tshap_comparisons(output_root, rows):
    if not rows:
        return
    _write_csv(
        Path(output_root) / "comparisons" / "tshap_comparisons.csv",
        rows,
        [
            "board", "mode", "window", "innovations_model", "tshap_model",
            "tshap_source", "n_matched_events", "pearson_profile_correlation",
            "spearman_profile_correlation",
        ],
    )


def _compare_direct_shared(records, output_root):
    groups = {}
    for record in records:
        summary = record["summary"]
        key = (summary["board"], summary["mode"], summary["window_name"])
        groups.setdefault(key, {})[summary["model"]] = record
    rows = []
    for (board, mode, window), pair in sorted(groups.items()):
        direct = pair.get("direct_linear_innovations")
        shared = pair.get("shared_linear_innovations")
        if direct is None or shared is None:
            continue
        if direct["time_ps"].shape != shared["time_ps"].shape or not np.allclose(
            direct["time_ps"], shared["time_ps"], rtol=0, atol=1e-6
        ):
            continue
        profile_corr = _safe_correlation(
            direct["profile"]["mean_abs_pair"], shared["profile"]["mean_abs_pair"]
        )
        delta_rmse_corr = _safe_correlation(
            direct["deletion"]["delta_rmse_ps"], shared["deletion"]["delta_rmse_ps"]
        )
        delta_ctr_corr = _safe_correlation(
            direct["deletion"]["delta_ctr_ps"], shared["deletion"]["delta_ctr_ps"]
        )
        shared_lookup = {
            int(event_id): index for index, event_id in enumerate(shared["event_ids"])
        }
        direct_rows = [
            index for index, event_id in enumerate(direct["event_ids"])
            if int(event_id) in shared_lookup
        ]
        shared_rows = [
            shared_lookup[int(direct["event_ids"][index])] for index in direct_rows
        ]
        prediction_corr = _safe_correlation(
            direct["prediction"][direct_rows], shared["prediction"][shared_rows]
        )
        prediction_rmse_difference = (
            float(np.sqrt(np.mean(
                (direct["prediction"][direct_rows] - shared["prediction"][shared_rows]) ** 2
            )))
            if direct_rows else float("nan")
        )
        direct_summary = direct["summary"]
        rows.append({
            "board": board, "mode": mode, "window": window,
            "direct_shared_prediction_correlation": prediction_corr,
            "direct_shared_prediction_rmse_difference_ps": prediction_rmse_difference,
            "mean_absolute_contribution_profile_correlation": profile_corr,
            "grouped_delta_rmse_profile_correlation": delta_rmse_corr,
            "grouped_delta_ctr_profile_correlation": delta_ctr_corr,
            "direct_antisymmetry_weight_ratio": direct_summary.get(
                "direct_antisymmetry_weight_ratio"
            ),
            "n_common_events": len(direct_rows),
        })
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        destination = (
            Path(output_root) / "comparisons" / "direct_vs_shared"
            / board / mode_family(mode) / window / "direct_vs_shared.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        time_ns = direct["time_ps"] / 1000.0
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
        axes[0].plot(
            time_ns, _normalized_profile(direct["profile"]["mean_abs_pair"], direct["time_ps"]),
            label="Direct",
        )
        axes[0].plot(
            time_ns, _normalized_profile(shared["profile"]["mean_abs_pair"], shared["time_ps"]),
            label="Shared",
        )
        axes[0].set_ylabel("Normalized contribution (1/ns)")
        axes[0].legend(loc="best")
        group_centers = direct["deletion"]["group_center_ps"] / 1000.0
        group_widths = (
            direct["deletion"]["group_end_ps"] - direct["deletion"]["group_start_ps"]
        ) / 1000.0
        axes[1].bar(
            group_centers - group_widths * 0.18,
            direct["deletion"]["delta_rmse_ps"],
            width=group_widths * 0.36, label="Direct",
        )
        axes[1].bar(
            group_centers + group_widths * 0.18,
            shared["deletion"]["delta_rmse_ps"],
            width=group_widths * 0.36, label="Shared",
        )
        axes[1].axhline(0.0, color="0.3", linewidth=0.7)
        axes[1].set_ylabel("Deletion ΔRMSE (ps)")
        axes[1].set_xlabel("Waveform time (ns)")
        axes[1].legend(loc="best")
        fig.suptitle(
            f"{board} | {mode} | {window} | prediction r={prediction_corr:.4f}"
        )
        fig.savefig(destination, dpi=180)
        plt.close(fig)
    if rows:
        _write_csv(
            Path(output_root) / "comparisons" / "direct_vs_shared.csv",
            rows, list(rows[0]),
        )
    return rows


def _compare_boards(records, output_root):
    groups = {}
    for record in records:
        summary = record["summary"]
        key = (summary["model"], summary["mode"], summary["window_name"])
        groups.setdefault(key, {})[summary["board"]] = record
    rows = []
    profile_rows = []
    for (model, mode, window), pair in sorted(groups.items()):
        uc, fbk = pair.get("UC"), pair.get("FBK")
        if uc is None or fbk is None:
            continue
        for record in (uc, fbk):
            normalized = _normalized_profile(
                record["profile"]["mean_abs_pair"], record["time_ps"]
            )
            for index, value in enumerate(normalized):
                profile_rows.append({
                    "board": record["summary"]["board"], "model": model,
                    "mode": mode, "window": window,
                    "sample_index": int(record["sample_indices"][index]),
                    "time_ps": float(record["time_ps"][index]),
                    "normalized_mean_absolute_contribution_per_ns": float(value),
                })
        rows.append({
            "model": model, "mode": mode, "window": window,
            "uc_events": int(len(uc["event_ids"])),
            "fbk_events": int(len(fbk["event_ids"])),
            "uc_samples": int(len(uc["time_ps"])),
            "fbk_samples": int(len(fbk["time_ps"])),
            "comparison": "physical-time overlay; no pointwise correlation because board grids may differ",
        })
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        destination = (
            Path(output_root) / "comparisons" / "UC_vs_FBK"
            / mode_family(mode) / window / f"{_safe_slug(model)}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.plot(
            uc["time_ps"] / 1000.0,
            _normalized_profile(uc["profile"]["mean_abs_pair"], uc["time_ps"]),
            label="UC",
        )
        ax.plot(
            fbk["time_ps"] / 1000.0,
            _normalized_profile(fbk["profile"]["mean_abs_pair"], fbk["time_ps"]),
            label="FBK",
        )
        ax.set_xlabel("Waveform time (ns)")
        ax.set_ylabel("Area-normalized contribution (1/ns)")
        ax.set_title(f"{model} | {mode} | {window}")
        ax.legend(loc="best")
        fig.savefig(destination, dpi=180)
        plt.close(fig)
    if rows:
        _write_csv(
            Path(output_root) / "comparisons" / "UC_vs_FBK.csv", rows, list(rows[0])
        )
        _write_csv(
            Path(output_root) / "comparisons" / "UC_vs_FBK_profiles.csv",
            profile_rows, list(profile_rows[0]),
        )
    return rows


def analyze_linear_innovations(
    results, *, output_root, tshap_root=None, cache_roots=None,
    source_artifact_roots=None, windows=None, modes=None, max_events=0, batch_size=256,
    window_samples=20, stride_samples=20, seed=42, overwrite=False,
    logger=None,
):
    """Analyze completed direct/shared innovations runs across UC and FBK."""
    logger = logger or logging.getLogger(__name__)
    results = {str(board).upper(): Path(path).expanduser().resolve()
               for board, path in results.items()}
    if not results or set(results) - {"UC", "FBK"}:
        raise ValueError("Provide one or both result roots labelled UC and/or FBK")
    if max_events not in (0,) and max_events < 5:
        raise ValueError("max_events must be 0 (all events) or >= 5 for CTR estimation")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if window_samples < 1 or stride_samples < 1 or stride_samples > window_samples:
        raise ValueError("Require 1 <= stride_samples <= window_samples")
    output_root = Path(output_root).expanduser().resolve()
    for root in results.values():
        if output_root == root or output_root.is_relative_to(root):
            raise ValueError("Analysis output-root must be outside the supplied results roots")
    output_root.mkdir(parents=True, exist_ok=True)
    cache_roots = cache_roots or {}
    source_artifact_roots = source_artifact_roots or {}
    selected_windows = None if windows is None else {str(x).lower() for x in windows}
    selected_modes = None if modes is None else {str(x).lower() for x in modes}
    tshap_profiles = _read_tshap_profiles(tshap_root)
    records = []
    all_tshap_comparisons = []

    for board, root in sorted(results.items()):
        if not root.is_dir():
            raise FileNotFoundError(f"{board} results directory does not exist: {root}")
        discovered = discover_runs(root)
        selected = []
        for run in discovered:
            model = str(run["manifest"].get("model", ""))
            if model not in INNOVATION_MODELS:
                continue
            config = resolve_run_config(run["directory"])
            family = mode_family(config["mode"])
            window = str(config.get("window_name", "")).lower()
            if selected_windows is not None and window not in selected_windows:
                continue
            if selected_modes is not None and family not in selected_modes:
                continue
            selected.append(run)
        if logger:
            logger.info(
                "Innovation analysis discovery | %s | complete_runs=%d | selected=%d",
                board, len(discovered), len(selected),
            )
        for run in selected:
            record = _analyze_run(
                run, board, output_root,
                cache_root=cache_roots.get(board),
                source_artifact_root=source_artifact_roots.get(board),
                max_events=max_events, batch_size=batch_size,
                window_samples=window_samples, stride_samples=stride_samples,
                seed=seed, overwrite=overwrite,
                tshap_profiles=tshap_profiles, logger=logger,
            )
            records.append(record)
            all_tshap_comparisons.extend(record.get("tshap_comparisons", []))

    if not records:
        raise ValueError(
            "No completed direct_linear_innovations/shared_linear_innovations runs "
            "matched the supplied UC/FBK result roots and filters"
        )
    _compare_direct_shared(records, output_root)
    _compare_boards(records, output_root)
    _write_tshap_comparisons(output_root, all_tshap_comparisons)

    summary_rows = []
    for record in records:
        summary = record["summary"]
        baseline = summary.get("baseline_metrics", {})
        summary_rows.append({
            "board": summary["board"], "mode": summary["mode"],
            "window": summary["window_name"], "model": summary["model"],
            "n_events": summary["n_events"], "n_samples": summary["n_input_samples"],
            "n_groups": summary["n_temporal_groups"],
            "baseline_ctr_ps": baseline.get("ctr_ps"),
            "baseline_rmse_ps": baseline.get("rmse_ps"),
            "direct_antisymmetry_weight_ratio": summary.get("direct_antisymmetry_weight_ratio"),
            "prediction_replay_max_abs_error_ps": summary.get("prediction_replay_max_abs_error_ps"),
            "tshap_comparisons": len(record.get("tshap_comparisons", [])),
            "output_directory": str(record["destination"]),
        })
    _write_csv(
        output_root / "summary.csv", summary_rows, list(summary_rows[0])
    )
    atomic_json(output_root / "analysis_summary.json", {
        "analysis_version": ANALYSIS_VERSION,
        "results_roots": {board: str(path) for board, path in results.items()},
        "tshap_root": str(Path(tshap_root).expanduser().resolve()) if tshap_root else None,
        "output_root": str(output_root),
        "n_runs": len(records),
        "n_tshap_comparisons": len(all_tshap_comparisons),
        "settings": {
            "max_events": int(max_events),
            "batch_size": int(batch_size),
            "window_samples": int(window_samples),
            "stride_samples": int(stride_samples),
            "seed": int(seed),
        },
        "cohort_note": (
            "Blind data are used only for post-hoc explanation and are never used "
            "for model fitting or hyperparameter selection."
        ),
    })
    if logger:
        logger.info(
            "Innovation analysis complete | runs=%d | TSHAP comparisons=%d | output=%s",
            len(records), len(all_tshap_comparisons), output_root,
        )
    return output_root


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", required=True, metavar="BOARD=PATH")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tshap-root", type=Path)
    parser.add_argument("--cache-root", nargs="*", default=[], metavar="BOARD=PATH",
                        help="Prepared analysis cache bundle per board (preprocessing/ and artifact_store/)")
    parser.add_argument("--source-artifact-root", nargs="*", default=[], metavar="BOARD=PATH",
                        help="Original artifact store containing saved model/prediction products, if external")
    parser.add_argument("--windows", nargs="+", choices=("onishi", "wide"))
    parser.add_argument("--modes", nargs="+", choices=("energy", "timing"))
    parser.add_argument("--max-events", type=int, default=0,
                        help="Events used for contribution/deletion metrics; 0 means all blind events")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-samples", type=int, default=20)
    parser.add_argument("--stride-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    results = _parse_board_paths(args.results, "--results")
    cache_roots = _parse_board_paths(args.cache_root, "--cache-root")
    source_artifact_roots = _parse_board_paths(
        args.source_artifact_root, "--source-artifact-root"
    )
    return analyze_linear_innovations(
        results,
        output_root=args.output_root,
        tshap_root=args.tshap_root,
        cache_roots=cache_roots,
        source_artifact_roots=source_artifact_roots,
        windows=args.windows,
        modes=args.modes,
        max_events=args.max_events,
        batch_size=args.batch_size,
        window_samples=args.window_samples,
        stride_samples=args.stride_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
