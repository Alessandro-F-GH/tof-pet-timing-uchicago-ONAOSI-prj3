"""Post-training TSHAP for frozen TOF-PET waveform regressors.

Uses the upstream tshap library for window interpolation and ROI extraction.
Only its perturbation scorer is overridden to avoid the upstream all-at-once
allocation and repeated array stacking on long waveforms.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from importlib.metadata import version

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import batch_root, discover_runs, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, waveform_view
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine.xai import _predict_array
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.models import get_model


TSHAP_METHOD_VERSION = 1
REQUIRED_ARTIFACTS = (
    "summary.json", "attributions.npz", "global_importance.csv", "global_importance.png",
)


def _validate_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")


def _choose_indices(n, cap, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, cap), replace=False))


def _check_prepared_grids(control, blind, mode):
    family = mode_family(mode)
    ctime = np.asarray(getattr(control, f"{family}_time_ps"))
    btime = np.asarray(getattr(blind, f"{family}_time_ps"))
    if not np.array_equal(ctime, btime):
        raise ValueError("Control and blind waveform grids differ")
    ct = getattr(control, f"{family}_transform")
    bt = getattr(blind, f"{family}_transform")
    if ct is None or bt is None:
        raise ValueError("Missing frozen waveform normalization")
    if any(not np.array_equal(getattr(ct, name), getattr(bt, name))
           for name in ("minimum", "maximum")):
        raise ValueError("Control and blind waveform normalization differs")
    return np.asarray(ctime, dtype=np.float64)


def _predict_bounded(spec, fitted, values, batch_size):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError("TSHAP expects [events, 2 detectors, samples]")
    chunk = batch_size
    if getattr(spec, "bounded_transform_batches", False) and fitted.feature_transform is not None:
        chunk = min(chunk, int(fitted.feature_transform.artifact.metadata["batch_size"]))
    output = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), chunk):
        stop = min(start + chunk, len(values))
        prediction = _predict_array(spec, fitted, values[start:stop])
        if prediction.shape != (stop - start,) or not np.all(np.isfinite(prediction)):
            raise ValueError("TSHAP model returned invalid scalar predictions")
        output[start:stop] = prediction
    return output


def _new_explainer(window_samples, stride_samples, roi, predict_batch_size):
    try:
        from tshap.tshap import TSHAPExplainer
    except ImportError as exc:
        raise ImportError("TSHAP requires the upstream dependency: pip install tshap==0.0.1") from exc

    class BoundedTSHAP(TSHAPExplainer):
        """Keep upstream TSHAP aggregation/ROI, but bound perturbation allocation."""

        def _calculate_window_attrib_scores(self, fnc, input_sample, baselines, window_positions):
            sample = np.asarray(input_sample)
            backgrounds = np.asarray(baselines)
            n_background = len(backgrounds)
            if not n_background:
                raise ValueError("TSHAP requires at least one control baseline")
            # Upstream evaluates the original and each reference just once per call.
            endpoints = np.asarray(fnc(np.concatenate((sample[None], backgrounds))), dtype=np.float64)
            if endpoints.shape != (n_background + 1,):
                raise ValueError("Unexpected model output for TSHAP endpoints")
            scores = np.empty(len(window_positions), dtype=np.float64)
            # Do not construct every overlapping perturbation in one huge matrix.
            windows_per_batch = max(1, predict_batch_size // (2 * n_background))
            for first in range(0, len(window_positions), windows_per_batch):
                positions = window_positions[first:first + windows_per_batch]
                candidates = np.empty((len(positions) * 2 * n_background, *sample.shape),
                                      dtype=np.float32)
                for i, position in enumerate(positions):
                    channel, start = int(position[0]), int(position[1])
                    width = int(position[2]) if len(position) == 3 else self.window_length
                    # Match upstream half-open masks, including truncated end windows.
                    if not (0 <= channel < sample.shape[0]):
                        raise ValueError("Invalid TSHAP channel")
                    stop = min(start + width, sample.shape[1])
                    if start < 0:
                        raise ValueError("Negative TSHAP window start")
                    block = candidates[i * 2 * n_background:(i + 1) * 2 * n_background]
                    block[0::2] = sample
                    block[1::2] = backgrounds
                    if stop > start:
                        block[0::2, channel, start:stop] = backgrounds[:, channel, start:stop]
                        block[1::2, channel, start:stop] = sample[channel, start:stop]
                predicted = np.asarray(fnc(candidates), dtype=np.float64)
                if predicted.shape != (len(candidates),):
                    raise ValueError("TSHAP predictor returned incorrect batch shape")
                for i in range(len(positions)):
                    block = predicted[i * 2 * n_background:(i + 1) * 2 * n_background]
                    # Identical two-player window/complement formula as upstream.
                    scores[first + i] = np.mean(
                        ((endpoints[0] - block[0::2])
                         + (block[1::2] - endpoints[1:])) / 2
                    )
            return scores

    return BoundedTSHAP(window_length=window_samples, stride=stride_samples,
                        interpolation=True, roi=roi)


def _plot_global(destination, time_ps, mean_absolute, mean_signed, mean_reference):
    """Plot attribution scores and normalized control waveforms on separate axes.

    Model outputs are timing corrections in ps, so the TSHAP scores inherit ps.
    The prepared waveform inputs are min-max normalized and clipped to [0, 1],
    so their amplitude is dimensionless; keep it on a separate row rather than
    a twin axis to prevent the waveform from visually obscuring attribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ps = np.asarray(time_ps, dtype=np.float64)
    time_ns = time_ps / 1000.0
    mean_absolute = np.asarray(mean_absolute, dtype=np.float64)
    mean_signed = np.asarray(mean_signed, dtype=np.float64)
    mean_reference = np.asarray(mean_reference, dtype=np.float64)
    expected_shape = (2, time_ps.size)
    if any(a.shape != expected_shape for a in (mean_absolute, mean_signed, mean_reference)):
        raise ValueError("Global TSHAP summaries must have shape [2 detectors, samples]")

    # One quantity per row: no twin y-axes and no attribution/waveform overlap.
    fig, axes = plt.subplots(
        3, 2, figsize=(13, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.15, 1.15, 1.0]},
    )
    row_labels = (
        ("Mean absolute TSHAP attribution (ps/sample)", mean_absolute, "tab:blue"),
        ("Mean signed TSHAP attribution (ps/sample)", mean_signed, "tab:orange"),
        ("Mean control baseline amplitude (0–1)", mean_reference, "0.35"),
    )
    for channel in range(2):
        axes[0, channel].set_title(f"Detector {channel + 1}")
        for row, (ylabel, values, color) in enumerate(row_labels):
            ax = axes[row, channel]
            ax.plot(time_ns, values[channel], color=color, linewidth=1.15)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.margins(x=0.005)
            if row < 2:
                ax.axhline(0.0, color="0.45", linewidth=0.75, linestyle=":")
            else:
                ax.set_ylim(-0.03, 1.03)
        axes[2, channel].set_xlabel("Waveform time (ns)")

    fig.savefig(destination / "global_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _publish_directory(temporary, destination, overwrite):
    """Publish atomically on the same filesystem; restore an old result on failure."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Incomplete TSHAP result exists: {destination}; use --overwrite")
    old = None
    if destination.exists():
        old = destination.parent / (".old-" + next(tempfile._get_candidate_names()))
        os.replace(destination, old)
    try:
        os.replace(temporary, destination)
    except BaseException:
        if old is not None:
            os.replace(old, destination)
        raise
    else:
        if old is not None:
            shutil.rmtree(old)


def _complete(destination, fingerprint):
    if not all((destination / name).is_file() for name in REQUIRED_ARTIFACTS):
        return False
    try:
        return json.loads((destination / "summary.json").read_text(encoding="utf-8")).get("identity_sha256") == fingerprint
    except (ValueError, OSError):
        return False


def _refresh_presentation(destination):
    """Regenerate plots/tables from saved attributions without rerunning TSHAP."""
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("plot_version", 0) >= 2:
        return
    with np.load(destination / "attributions.npz") as saved:
        time_ps = np.asarray(saved["time_ps"], dtype=np.float64)
        window_maps = np.asarray(saved["window_attribution_ps"], dtype=np.float64)
        reference = np.asarray(saved["reference_mean_normalized"], dtype=np.float64)
        roi_maps = np.asarray(saved["roi_attribution_ps"], dtype=np.float64) if "roi_attribution_ps" in saved else None
    mean_absolute = np.mean(np.abs(window_maps), axis=0)
    mean_signed = np.mean(window_maps, axis=0)
    with (destination / "global_importance.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["channel", "time_ps", "window_mean_abs_ps_per_sample",
                         "window_mean_signed_ps_per_sample", "roi_mean_abs_ps_per_sample",
                         "roi_mean_signed_ps_per_sample"])
        for channel in range(2):
            for j, t in enumerate(time_ps):
                writer.writerow([channel + 1, float(t), float(mean_absolute[channel, j]),
                                 float(mean_signed[channel, j]),
                                 float(np.mean(np.abs(roi_maps[:, channel, j])) ) if roi_maps is not None else "",
                                 float(np.mean(roi_maps[:, channel, j])) if roi_maps is not None else ""])
    _plot_global(destination, time_ps, mean_absolute, mean_signed, reference)
    summary["plot_version"] = 2
    summary["attribution_units"] = "ps/sample (upstream window score redistributed across temporal samples)"
    atomic_json(summary_path, summary)


def _explanation_destination(output_root, run_directory):
    """Return one stable TSHAP directory per run; fingerprints belong in metadata only."""
    batch = batch_root(run_directory)
    relative_run = Path(run_directory).resolve().relative_to(batch)
    return Path(output_root).resolve() / batch.name / relative_run / "tshap"


def _copy_cached_explanation(source, destination, *, remove_source=False, logger=None):
    """Migrate a complete legacy cache into the stable directory without inference."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tshap-migrate-", dir=destination.parent))
    shutil.rmtree(temporary)
    try:
        shutil.copytree(source, temporary)
        _refresh_presentation(temporary)
        _publish_directory(temporary, destination, overwrite=True)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if remove_source:
        shutil.rmtree(source)
        # Remove now-empty legacy containers only; never remove the run itself.
        for parent in (source.parent, source.parent.parent):
            try:
                parent.rmdir()
            except OSError:
                break
    if logger:
        logger.info("TSHAP | migrated cached explanation | %s -> %s", source, destination)


def explain_tshap(results, *, models=None, cache_root=None, output_root=None, max_events=100, n_baselines=8,
                  window_samples=20, stride_samples=10, predict_batch_size=64,
                  roi=False, overwrite=False, logger=None):
    """Explain saved regressors using reproducible blind/control subsets.

    Identical requests reuse completed explanation artifacts without inference.
    Changes in model, cohorts or settings produce separate, fingerprinted results.
    The explicit overwrite flag recomputes/replaces only the matching identity.
    """
    for name, value in (("max_events", max_events), ("n_baselines", n_baselines),
                        ("window_samples", window_samples), ("stride_samples", stride_samples),
                        ("predict_batch_size", predict_batch_size)):
        _validate_positive_int(name, value)
    tshap_version = version("tshap")
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
    available = {run["manifest"]["model"] for run in runs}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    lookup = PreparedCacheLookup()
    outputs, failures = [], []
    for run in runs:
        manifest = run["manifest"]
        if models is not None and manifest["model"] not in models:
            continue
        directory = Path(run["directory"])
        try:
            config = resolve_run_config(directory)
            spec = get_model(manifest["model"])
            model_dir = directory / "model"
            if not saved_model_complete(spec, model_dir):
                raise FileNotFoundError(f"Saved model or frozen feature transform missing: {model_dir}")
            seed = semantic_seed(config["seed"], spec.name, "tshap")
            settings = dict(max_events=max_events, n_baselines=n_baselines,
                            window_samples=window_samples, stride_samples=stride_samples,
                            predict_batch_size=predict_batch_size, roi=bool(roi))
            identity = {
                "method": "tshap_window_roi" if roi else "tshap_window",
                "method_version": TSHAP_METHOD_VERSION, "upstream_version": tshap_version,
                "final_fit": manifest["stage_fingerprints"]["final_fit"],
                "control": manifest["dataset_populations"]["control"]["analysis_protocol_identity"],
                "blind": manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
                "model": spec.name, "mode": config["mode"], "seed": seed, "settings": settings,
            }
            fingerprint = canonical_hash(identity)
            destination = _explanation_destination(output_root, directory)
            legacy_results = directory / "explanations" / "tshap" / fingerprint[:16]
            legacy_xai = destination / fingerprint[:16]

            # A matching flat cache is the canonical hit: no model inference.
            if _complete(destination, fingerprint) and not overwrite:
                _refresh_presentation(destination)
                if logger:
                    logger.info("TSHAP | cache hit | %s", destination)
                outputs.append(destination)
                continue

            # Migrate old hash-folder layouts rather than creating another copy.
            if not overwrite and _complete(legacy_xai, fingerprint):
                _copy_cached_explanation(legacy_xai, destination, logger=logger)
                outputs.append(destination)
                continue
            if not overwrite and _complete(legacy_results, fingerprint):
                _copy_cached_explanation(
                    legacy_results, destination, remove_source=True, logger=logger
                )
                outputs.append(destination)
                continue

            # No matching cache: compute in a temporary directory and replace the
            # stable tshap directory only after all artifacts have been written.
            cache = _cache_root(config, directory, cache_root)
            control = lookup.find(cache, "control", config, manifest)
            blind = lookup.find(cache, "blind", config, manifest)
            time_ps = _check_prepared_grids(control, blind, config["mode"])
            if window_samples > time_ps.size:
                raise ValueError("TSHAP window longer than input waveform")
            baseline_ids = _choose_indices(control.n_events, n_baselines,
                                           semantic_seed(seed, "control_baselines"))
            event_ids = _choose_indices(blind.n_events, max_events, semantic_seed(seed, "blind_events"))
            baselines = waveform_view(control, config["mode"], baseline_ids).materialize()
            waves = waveform_view(blind, config["mode"], event_ids).materialize()
            if baselines.shape[1:] != waves.shape[1:] or waves.shape[1] != 2:
                raise ValueError("TSHAP inputs must have matching [2, samples] dimensions")
            if not np.all(np.isfinite(waves)) or not np.all(np.isfinite(baselines)):
                raise ValueError("Non-finite control or blind waveform values")
            if logger:
                logger.info("TSHAP | %s | events=%d/%d | baselines=%d/%d | samples=%d | ROI=%s",
                            spec.name, len(event_ids), blind.n_events, len(baseline_ids),
                            control.n_events, time_ps.size, roi)
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".tshap-", dir=parent))
            try:
                with diagnostic_random_state():
                    fitted = load_fitted_model(spec, model_dir, run["best"]["parameters"], config)
                    try:
                        predict = lambda x: _predict_bounded(spec, fitted, x, predict_batch_size)
                        # Preprocessing/masks, fitted feature map and clipping must be unchanged.
                        original = predict_indices(spec, fitted, blind, config["mode"], event_ids,
                                                   chunk_size=predict_batch_size)
                        replay = predict(waves)
                        if not np.allclose(original, replay, atol=1e-4, rtol=1e-6):
                            raise ValueError("TSHAP model adapter does not reproduce saved inference")
                        explainer = _new_explainer(window_samples, stride_samples, roi,
                                                   predict_batch_size)
                        window_maps = np.empty_like(waves, dtype=np.float64)
                        roi_maps = np.empty_like(waves, dtype=np.float64) if roi else None
                        for i, pair in enumerate(waves):
                            win, regions = explainer.explain(pair[None], baselines, predict)
                            window_maps[i] = win[0]
                            if roi:
                                roi_maps[i] = regions[0]
                            if logger and (i == 0 or (i + 1) % 10 == 0 or i + 1 == len(waves)):
                                logger.info("TSHAP | %s | %d/%d events", spec.name, i + 1, len(waves))
                    finally:
                        del fitted
                        release_training_memory()
                if not np.all(np.isfinite(window_maps)) or (roi and not np.all(np.isfinite(roi_maps))):
                    raise ValueError("Non-finite TSHAP attribution")
                win_abs = np.mean(np.abs(window_maps), axis=0)
                win_signed = np.mean(window_maps, axis=0)
                payload = dict(event_id=np.asarray(blind.event_index[event_ids], np.int64),
                               control_event_id=np.asarray(control.event_index[baseline_ids], np.int64),
                               time_ps=time_ps, window_attribution_ps=window_maps,
                               reference_mean_normalized=np.mean(baselines, axis=0),
                               original_prediction_ps=original)
                if roi:
                    payload["roi_attribution_ps"] = roi_maps
                np.savez_compressed(temporary / "attributions.npz", **payload)
                with (temporary / "global_importance.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["channel", "time_ps", "window_mean_abs_ps_per_sample", "window_mean_signed_ps_per_sample",
                                     "roi_mean_abs_ps", "roi_mean_signed_ps"])
                    for ch in range(2):
                        for j, t in enumerate(time_ps):
                            writer.writerow([ch + 1, float(t), float(win_abs[ch, j]),
                                             float(win_signed[ch, j]),
                                             float(np.mean(np.abs(roi_maps[:, ch, j]))) if roi else "",
                                             float(np.mean(roi_maps[:, ch, j])) if roi else ""])
                _plot_global(temporary, time_ps, win_abs, win_signed, payload["reference_mean_normalized"])
                atomic_json(temporary / "summary.json", {
                    **identity, "identity_sha256": fingerprint, "plot_version": 2,
                    "n_explained_events": len(event_ids),
                    "n_control_baselines": len(baseline_ids), "waveform_samples": len(time_ps),
                    "formulation": spec.estimator_formulation,
                    "attribution_units": "ps/sample (upstream window score redistributed across temporal samples)",
                    "interpretation": "Two-player window/complement SHAP with reference substitution; "
                                      "overlapping-window maps are not exact sample-level Shapley values.",
                    "cohort": "blind", "input_units": "frozen normalized waveform",
                    "prediction_equivalence_max_error_ps": float(np.max(np.abs(original - replay))),
                    "reference": "https://github.com/mlgig/tshap",
                })
                # Replace stale settings/incomplete artifacts in-place. The cache
                # check above prevents recalculation when the fingerprint matches.
                _publish_directory(temporary, destination, overwrite=True)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            outputs.append(destination)
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
            if logger:
                logger.exception("TSHAP failed | %s", directory)
    if failures:
        raise RuntimeError("Some TSHAP explanations failed; successful results retained:\n"
                           + "\n".join(failures))
    return outputs
