"""Local TSHAP case studies selected by signed timing improvement.

This command is separate from global TSHAP. It predicts the blind population
to select cases, but computes TSHAP only for the selected events.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
from importlib.metadata import version

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import model_target, waveform_view
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine.tshap_explainer import (
    _check_prepared_grids, _choose_indices, _explanation_destination,
    _new_explainer, _predict_bounded, _publish_directory, _selection_seed,
)
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model


LOCAL_CASES_VERSION = 1
LOCAL_REQUIRED_ARTIFACTS = ("summary.json", "selected_events.csv", "local_attributions.npz")
CASE_GROUPS = (
    ("good_positive", "Good positive correction"),
    ("good_negative", "Good negative correction"),
    ("bad_positive", "Bad positive correction"),
    ("bad_negative", "Bad negative correction"),
    ("typical", "Typical correction (median predicted output)"),
)


def _select_local_event_groups(target_ps, prediction_ps, *, events_per_group=2,
                               extreme_margin=0.05):
    """Select distinct rows by correction sign and signed improvement.

    Improvement is |target| - |target - prediction|. For good/bad groups, the
    outermost extreme_margin fraction of eligible candidates is skipped so
    selected cases are strong but not the absolute best/worst. Typical events
    are unused events nearest the median predicted correction.
    """
    if isinstance(events_per_group, bool) or not isinstance(events_per_group, int) or events_per_group < 1:
        raise ValueError("events_per_group must be an integer >= 1")
    if not np.isfinite(extreme_margin) or not 0.0 <= extreme_margin < 1.0:
        raise ValueError("extreme_margin must be in [0, 1)")
    target = np.asarray(target_ps, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction_ps, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Target and prediction arrays must have the same nonzero length")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(prediction)):
        raise ValueError("Target and prediction arrays must be finite")

    improvement = np.abs(target) - np.abs(target - prediction)
    median_prediction = float(np.median(prediction))
    specifications = (
        ("good_positive", "Good positive correction",
         (prediction > 0) & (improvement > 0), "high"),
        ("good_negative", "Good negative correction",
         (prediction < 0) & (improvement > 0), "high"),
        ("bad_positive", "Bad positive correction",
         (prediction > 0) & (improvement < 0), "low"),
        ("bad_negative", "Bad negative correction",
         (prediction < 0) & (improvement < 0), "low"),
    )
    groups = {}
    used = set()
    for key, title, mask, direction in specifications:
        candidates = np.flatnonzero(mask)
        if len(candidates) < events_per_group:
            raise ValueError(
                f"Cannot select {events_per_group} events for '{title}': "
                f"only {len(candidates)} events have the required correction "
                "sign and signed-improvement sign"
            )
        if direction == "high":
            order = candidates[np.lexsort((candidates, -improvement[candidates]))]
        else:
            order = candidates[np.lexsort((candidates, improvement[candidates]))]
        n_excluded = min(
            int(np.ceil(extreme_margin * len(order))),
            max(0, len(order) - events_per_group),
        )
        chosen = order[n_excluded:n_excluded + events_per_group]
        if len(chosen) != events_per_group:
            raise ValueError(f"Unable to select enough non-extreme events for '{title}'")
        groups[key] = {"title": title, "indices": chosen.astype(np.int64)}
        used.update(map(int, chosen))

    available = np.asarray([i for i in range(target.size) if i not in used], dtype=np.int64)
    if len(available) < events_per_group:
        raise ValueError("Not enough unused events remain for the typical-correction group")
    distance = np.abs(prediction[available] - median_prediction)
    order = available[np.lexsort((available, distance))]
    groups["typical"] = {
        "title": "Typical correction (median predicted output)",
        "indices": order[:events_per_group].astype(np.int64),
    }
    all_selected = np.concatenate([groups[key]["indices"] for key, _ in CASE_GROUPS])
    if len(np.unique(all_selected)) != len(all_selected):
        raise AssertionError("Local TSHAP case selection produced duplicate events")
    return groups, improvement, median_prediction


def _plot_local_event(destination, *, time_ps, waveform, attribution, group_title,
                      event_id, target_ps, prediction_ps, residual_ps, improvement_ps):
    """Plot a local explanation with waveform and attribution on separate axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    waveform = np.asarray(waveform, dtype=np.float64)
    attribution = np.asarray(attribution, dtype=np.float64)
    if waveform.shape != (2, len(time_ns)) or attribution.shape != waveform.shape:
        raise ValueError("Local waveform/attribution dimensions do not match the time grid")

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 7.5), sharex="col", constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 1.25]},
    )
    for channel in range(2):
        wave_ax, attr_ax = axes[0, channel], axes[1, channel]
        wave_ax.plot(time_ns, waveform[channel], linewidth=1.2, color="tab:blue")
        wave_ax.set_ylim(-0.03, 1.03)
        wave_ax.set_ylabel("Normalized amplitude (0–1)")
        wave_ax.set_title(f"Detector {channel + 1}")
        wave_ax.grid(True, alpha=0.22, linewidth=0.6)

        attr_ax.plot(time_ns, attribution[channel], linewidth=1.15, color="tab:orange")
        attr_ax.axhline(0.0, color="0.35", linestyle=":", linewidth=0.8)
        attr_ax.set_ylabel("TSHAP attribution (ps/sample)")
        attr_ax.set_xlabel("Waveform time (ns)")
        attr_ax.grid(True, alpha=0.22, linewidth=0.6)

    fig.suptitle(
        f"{group_title} | event {int(event_id)}\n"
        f"Predicted correction: {prediction_ps:+.2f} ps  |  "
        f"Signed improvement: {improvement_ps:+.2f} ps  |  "
        f"Target: {target_ps:+.2f} ps  |  Final residual: {residual_ps:+.2f} ps",
        fontsize=12,
    )
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _local_cases_complete(destination, fingerprint, expected_events):
    if not all((destination / name).is_file() for name in LOCAL_REQUIRED_ARTIFACTS):
        return False
    try:
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        if summary.get("identity_sha256") != fingerprint:
            return False
        if summary.get("n_explained_events") != expected_events:
            return False
        plots = summary.get("plot_files", [])
        return len(plots) == expected_events and all((destination / name).is_file() for name in plots)
    except (ValueError, OSError, TypeError):
        return False


def explain_tshap_cases(
    results, *, models=None, cache_root=None, output_root=None,
    events_per_group=2, extreme_margin=0.05, n_baselines=8,
    window_samples=20, stride_samples=10, predict_batch_size=64,
    seed=42, overwrite=False, logger=None,
):
    """Explain ten local cases by default, without running global TSHAP."""
    for name, value in (
        ("events_per_group", events_per_group),
        ("n_baselines", n_baselines),
        ("window_samples", window_samples),
        ("stride_samples", stride_samples),
        ("predict_batch_size", predict_batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not np.isfinite(extreme_margin) or not 0.0 <= extreme_margin < 1.0:
        raise ValueError("extreme_margin must be in [0, 1)")

    results_path = Path(results).expanduser().resolve()
    if output_root is None:
        output_root = (Path.cwd() / "xai").resolve()
        if output_root.is_relative_to(results_path):
            output_root = results_path.parent / "xai"
    else:
        output_root = Path(output_root).expanduser().resolve()
        if output_root == results_path or output_root.is_relative_to(results_path):
            raise ValueError("TSHAP output-root must be outside the supplied results tree")

    runs = discover_runs(results_path)
    if not runs:
        raise ValueError(f"No complete saved-model runs found in {results}")
    available_models = {run["manifest"]["model"] for run in runs}
    if models is not None and set(models) - available_models:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available_models)}")

    lookup = PreparedCacheLookup()
    outputs, failures = [], []
    tshap_version = version("tshap")
    for run in runs:
        manifest = run["manifest"]
        if models is not None and manifest["model"] not in models:
            continue
        run_directory = Path(run["directory"])
        try:
            config = resolve_run_config(run_directory)
            spec = get_model(manifest["model"])
            model_dir = run_directory / "model"
            if not saved_model_complete(spec, model_dir):
                raise FileNotFoundError(f"Saved model or frozen feature transform missing: {model_dir}")

            settings = {
                "events_per_group": events_per_group,
                "extreme_margin": float(extreme_margin),
                "n_baselines": n_baselines,
                "window_samples": window_samples,
                "stride_samples": stride_samples,
                "predict_batch_size": predict_batch_size,
                "seed": int(seed),
            }
            identity = {
                "method": "tshap_local_cases",
                "method_version": LOCAL_CASES_VERSION,
                "upstream_version": tshap_version,
                "final_fit": manifest["stage_fingerprints"]["final_fit"],
                "control": manifest["dataset_populations"]["control"]["analysis_protocol_identity"],
                "blind": manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
                "model": spec.name,
                "mode": config["mode"],
                "settings": settings,
                "selection_metric": "abs(target_ps) - abs(target_ps - predicted_correction_ps)",
                "selection_population": "full blind population",
            }
            fingerprint = canonical_hash(identity)
            destination = _explanation_destination(output_root, run_directory).with_name("tshap-cases")
            expected_events = len(CASE_GROUPS) * events_per_group
            if _local_cases_complete(destination, fingerprint, expected_events) and not overwrite:
                if logger:
                    logger.info("TSHAP cases | cache hit | %s", destination)
                outputs.append(destination)
                continue

            cache = _cache_root(config, run_directory, cache_root)
            control = lookup.find(cache, "control", config, manifest)
            blind = lookup.find(cache, "blind", config, manifest)
            time_ps = _check_prepared_grids(control, blind, config["mode"])
            if window_samples > len(time_ps):
                raise ValueError("TSHAP window longer than input waveform")
            targets = model_target(blind, config["mode"])
            if not np.all(np.isfinite(targets)):
                raise ValueError("Blind target corrections contain non-finite values")

            baseline_rows = _choose_indices(
                control.n_events, n_baselines, _selection_seed(seed, "control_baselines")
            )
            baselines = waveform_view(control, config["mode"], baseline_rows).materialize()
            all_rows = np.arange(blind.n_events, dtype=np.int64)
            if logger:
                logger.info(
                    "TSHAP cases | %s | predicting %d blind events to select %d local cases",
                    spec.name, blind.n_events, expected_events,
                )

            with diagnostic_random_state():
                fitted = load_fitted_model(spec, model_dir, run["best"]["parameters"], config)
                try:
                    predictions = np.asarray(
                        predict_indices(
                            spec, fitted, blind, config["mode"], all_rows,
                            chunk_size=predict_batch_size,
                        ),
                        dtype=np.float64,
                    )
                    if predictions.shape != (blind.n_events,) or not np.all(np.isfinite(predictions)):
                        raise ValueError("Saved model produced invalid blind predictions")
                    groups, improvement, median_prediction = _select_local_event_groups(
                        targets, predictions,
                        events_per_group=events_per_group,
                        extreme_margin=extreme_margin,
                    )
                    selected_rows = np.concatenate([groups[key]["indices"] for key, _ in CASE_GROUPS])
                    group_keys = np.concatenate([
                        np.repeat(key, events_per_group) for key, _ in CASE_GROUPS
                    ])
                    group_titles = np.concatenate([
                        np.repeat(groups[key]["title"], events_per_group) for key, _ in CASE_GROUPS
                    ])
                    selected_targets = np.asarray(targets[selected_rows], dtype=np.float64)
                    selected_predictions = predictions[selected_rows]
                    selected_improvements = improvement[selected_rows]
                    selected_residuals = selected_targets - selected_predictions
                    selected_event_ids = np.asarray(blind.event_index[selected_rows], dtype=np.int64)
                    waves = waveform_view(blind, config["mode"], selected_rows).materialize()
                    if waves.shape[1:] != baselines.shape[1:] or waves.shape[1] != 2:
                        raise ValueError("Local TSHAP inputs must have matching [2, samples] dimensions")
                    if not np.all(np.isfinite(waves)) or not np.all(np.isfinite(baselines)):
                        raise ValueError("Non-finite control or blind waveform values")

                    predict = lambda x: _predict_bounded(spec, fitted, x, predict_batch_size)
                    replay = predict(waves)
                    if not np.allclose(selected_predictions, replay, atol=1e-4, rtol=1e-6):
                        raise ValueError("TSHAP adapter does not reproduce saved-model predictions")

                    explainer = _new_explainer(
                        window_samples, stride_samples, False, predict_batch_size
                    )
                    attribution_maps = np.empty_like(waves, dtype=np.float64)
                    for local_i, pair in enumerate(waves):
                        window_map, _ = explainer.explain(pair[None], baselines, predict)
                        attribution_maps[local_i] = window_map[0]
                        if logger:
                            logger.info(
                                "TSHAP cases | %s | local explanation %d/%d",
                                spec.name, local_i + 1, len(waves),
                            )
                finally:
                    del fitted
                    release_training_memory()

            if not np.all(np.isfinite(attribution_maps)):
                raise ValueError("Non-finite local TSHAP attribution")
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".tshap-cases-", dir=parent))
            try:
                events_dir = temporary / "events"
                events_dir.mkdir(parents=True, exist_ok=True)
                plot_files = []
                with (temporary / "selected_events.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "group_key", "group_name", "rank_in_group", "dataset_row", "event_id",
                        "target_correction_ps", "predicted_correction_ps", "final_residual_ps",
                        "signed_improvement_ps", "median_predicted_correction_ps",
                    ])
                    local_i = 0
                    for key, title in CASE_GROUPS:
                        for rank in range(1, events_per_group + 1):
                            row = int(selected_rows[local_i])
                            event_id = int(selected_event_ids[local_i])
                            slug = key.replace("_", "-")
                            plot_name = f"events/{local_i + 1:02d}_{slug}_event_{event_id}.png"
                            _plot_local_event(
                                temporary / plot_name,
                                time_ps=time_ps,
                                waveform=waves[local_i],
                                attribution=attribution_maps[local_i],
                                group_title=title,
                                event_id=event_id,
                                target_ps=float(selected_targets[local_i]),
                                prediction_ps=float(selected_predictions[local_i]),
                                residual_ps=float(selected_residuals[local_i]),
                                improvement_ps=float(selected_improvements[local_i]),
                            )
                            plot_files.append(plot_name)
                            writer.writerow([
                                key, title, rank, row, event_id,
                                float(selected_targets[local_i]), float(selected_predictions[local_i]),
                                float(selected_residuals[local_i]), float(selected_improvements[local_i]),
                                median_prediction,
                            ])
                            local_i += 1

                np.savez_compressed(
                    temporary / "local_attributions.npz",
                    dataset_row=selected_rows,
                    event_id=selected_event_ids,
                    group_key=group_keys,
                    group_name=group_titles,
                    time_ps=np.asarray(time_ps, dtype=np.float64),
                    waveform_normalized=waves,
                    window_attribution_ps_per_sample=attribution_maps,
                    target_correction_ps=selected_targets,
                    predicted_correction_ps=selected_predictions,
                    final_residual_ps=selected_residuals,
                    signed_improvement_ps=selected_improvements,
                    control_baseline_event_id=np.asarray(control.event_index[baseline_rows], dtype=np.int64),
                )
                atomic_json(temporary / "summary.json", {
                    **identity,
                    "identity_sha256": fingerprint,
                    "n_explained_events": len(selected_rows),
                    "events_per_group": events_per_group,
                    "n_control_baselines": len(baseline_rows),
                    "control_baseline_event_ids": np.asarray(
                        control.event_index[baseline_rows], dtype=np.int64
                    ).tolist(),
                    "median_predicted_correction_ps": median_prediction,
                    "prediction_equivalence_max_error_ps": float(
                        np.max(np.abs(selected_predictions - replay))
                    ),
                    "groups": {
                        key: {
                            "title": groups[key]["title"],
                            "event_ids": np.asarray(
                                blind.event_index[groups[key]["indices"]], dtype=np.int64
                            ).tolist(),
                        }
                        for key, _ in CASE_GROUPS
                    },
                    "plot_files": plot_files,
                    "attribution_units": "ps/sample (upstream window scores redistributed over temporal samples)",
                    "reference": "https://github.com/mlgig/tshap",
                })
                _publish_directory(temporary, destination, overwrite=True)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            if logger:
                logger.info("TSHAP cases | saved %d local explanations | %s", expected_events, destination)
            outputs.append(destination)
        except Exception as exc:
            failures.append(f"{run_directory}: {exc}")
            if logger:
                logger.exception("TSHAP local cases failed | %s", run_directory)

    if failures:
        raise RuntimeError("Some local TSHAP case studies failed; successful results retained:\n"
                           + "\n".join(failures))
    return outputs
