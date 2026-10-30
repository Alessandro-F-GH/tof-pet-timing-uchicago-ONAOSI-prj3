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
from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.named_cache import imputer_alias, get_or_build_named, cache_root_from_config
from waveform_analysis.data.results import batch_root, discover_runs, read_run, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, model_target, waveform_view
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, release_training_memory, saved_model_complete,
)
from waveform_analysis.signal.sample_mask import apply_sample_mask
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.models import get_model
from waveform_analysis.signal.baseline import baseline_level_mV


TSHAP_METHOD_VERSION = 3
GLOBAL_PLOT_VERSION = 13
LOCAL_PREDICTION_REPLAY_ATOL_PS = 0.1
LOCAL_PREDICTION_REPLAY_RTOL = 1e-3
REQUIRED_ARTIFACTS = (
    "summary.json", "attributions.npz", "global_importance.csv", "global_importance.png",
)


def _validate_positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1")


def _matches_tshap_window(config, ml_window):
    """Select an ML input window by its resolved run configuration name."""
    if ml_window not in {"onishi", "wide", "all"}:
        raise ValueError("ml_window must be 'onishi', 'wide', or 'all'")
    return ml_window == "all" or str(config.get("window_name", "")).strip().lower() == ml_window




def _resolve_model_run_directory(model_dir):
    """Resolve a run directory from its result folder or its local model/ child.

    The run directory owns the scientific metadata and model reference. A raw
    shared ArtifactStore product is not sufficient because it has no unique
    control/blind cohort or experiment configuration.
    """
    path = Path(model_dir).expanduser().resolve()
    for candidate in (path, path.parent):
        if (candidate / "metadata" / "manifest.json").is_file():
            return candidate
    raise ValueError(
        f"Cannot identify a saved-model run from {path}. Pass the run directory "
        "containing metadata/manifest.json, or its local model/ subdirectory; "
        "a raw ArtifactStore model product is not a complete run."
    )


def _tshap_run_inputs(results=None, model_dir=None):
    """Load either all runs under a result root or one explicitly selected run."""
    if (results is None) == (model_dir is None):
        raise ValueError("Specify exactly one of results or model_dir")
    if model_dir is not None:
        directory = _resolve_model_run_directory(model_dir)
        return [read_run(directory)], directory
    root = Path(results).expanduser().resolve()
    runs = discover_runs(root)
    return runs, root


def _apply_tshap_cache(config, *, cache_root=None, artifact_root=None):
    """Override run-local cache paths with a portable shared cache bundle."""
    if cache_root is not None and artifact_root is not None:
        raise ValueError("Specify cache_root or artifact_root, not both")
    if cache_root is not None:
        root = Path(cache_root).expanduser().resolve()
        store = root / "artifact_store"
        if not store.is_dir():
            raise FileNotFoundError(
                f"Shared cache artifact store not found: {store}. "
                "Expected the prepare-cache bundle layout."
            )
        config["preprocessing"]["cache_dir"] = str(root / "preprocessing")
        config["preprocessing"]["artifact_root"] = str(store)
    elif artifact_root is not None:
        config["preprocessing"]["artifact_root"] = str(
            Path(artifact_root).expanduser().resolve()
        )
    return config


def _effective_tshap_window(ml_window, *, model_dir):
    """Keep the historical Wide default for result-root scans.

    With one explicitly selected model run, its own window is authoritative
    unless the caller explicitly requests a filter.
    """
    if ml_window is None:
        return "all" if model_dir is not None else "wide"
    _matches_tshap_window({}, ml_window)
    return ml_window


def _choose_indices(n, cap, seed):
    return np.sort(np.random.default_rng(seed).choice(n, size=min(n, cap), replace=False))


def _selection_seed(seed, role):
    """Stable sampling seed independent of model identity."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("TSHAP seed must be an integer")
    if role not in {"blind_events", "control_baselines"}:
        raise ValueError(f"Unknown TSHAP sampling role: {role}")
    return semantic_seed(seed, "tshap", role)


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


def _predict_array(spec, fitted, pair):
    # Match train._model_input exactly: feature transforms such as MultiRocket
    # may rely on contiguous [event, detector, sample] arrays after masking.
    values = np.ascontiguousarray(
        apply_sample_mask(np.asarray(pair, dtype=np.float32), fitted.sample_mask),
        dtype=np.float32,
    )
    if fitted.feature_transform is not None:
        # Never look up a cached feature row here: each occluded waveform is a
        # new signal and must pass through the fitted transform independently.
        values = fitted.feature_transform.apply(values)
    prediction = np.asarray(
        spec.predict(fitted.artifact, values), dtype=np.float64
    ).reshape(-1)
    if fitted.output_max_abs_ps is not None:
        prediction = np.clip(
            prediction, -fitted.output_max_abs_ps, fitted.output_max_abs_ps
        )
    return prediction


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


class _HermitePPCAImputer:
    """Position-specific Hermite baseline with a calibration-fitted PPCA correction."""

    def __init__(self, models, tuning_waves, *, slope_points=5, batch_size=64):
        from waveform_analysis.scripts.train_ppca_interpolation_blend import (
            hermite_interpolation_batch, smooth_blend_envelope,
        )
        self.models = models
        self.tuning_waves = np.asarray(tuning_waves, dtype=np.float64)
        self.slope_points = int(slope_points)
        self.batch_size = int(batch_size)
        self._prepared = {}
        self._envelopes = {}
        self._alphas = {}
        self._hermite = hermite_interpolation_batch
        self._envelope = smooth_blend_envelope

    def _prepare(self, detector, start, width):
        key = (int(detector), int(start), int(width))
        if key in self._alphas:
            return key
        model = self.models[detector]
        stop = start + width
        prepared = model.prepare_mask(start, width)
        envelope = self._envelope(start, width, len(model.mean))
        alpha = 0.0
        if start > 0 and stop < len(model.mean) and np.any(envelope):
            numerator = 0.0
            denominator = 0.0
            for offset in range(0, len(self.tuning_waves), self.batch_size):
                batch = self.tuning_waves[offset:offset + self.batch_size, detector, :]
                truth = batch[:, start:stop]
                hermite = self._hermite(batch, start, width, self.slope_points)
                ppca = model.conditional_mean_batch(
                    batch, start, width, prepared=prepared
                )
                delta = envelope[None, :] * (ppca - hermite)
                numerator += float(np.sum(delta * (truth - hermite)))
                denominator += float(np.sum(delta * delta))
            if denominator > 1e-18:
                alpha = float(np.clip(numerator / denominator, 0.0, 1.0))
        self._prepared[key] = prepared
        self._envelopes[key] = envelope
        self._alphas[key] = alpha
        return key

    def impute_segment_batch(self, waveforms, detector, start, width):
        """Return imputed samples only, for [events, detectors, time] waveforms."""
        from waveform_analysis.scripts.train_ppca_interpolation_blend import (
            linear_interpolation_batch,
        )
        batch = np.asarray(waveforms, dtype=np.float64)
        if batch.ndim != 3 or batch.shape[1] != 2:
            raise ValueError("Imputation expects [events, 2 detectors, samples]")
        key = self._prepare(detector, start, width)
        if self._alphas[key] == 0.0 and (
            start == 0 or start + width >= batch.shape[-1]
        ):
            return linear_interpolation_batch(batch[:, detector, :], start, width)
        model = self.models[detector]
        signal = batch[:, detector, :]
        hermite = self._hermite(signal, start, width, self.slope_points)
        ppca = model.conditional_mean_batch(
            signal, start, width, prepared=self._prepared[key]
        )
        envelope = self._envelopes[key]
        return hermite + self._alphas[key] * envelope[None, :] * (ppca - hermite)

    @property
    def weight_rows(self):
        return [
            {"detector": detector + 1, "start_sample": start, "width_samples": width,
             "alpha_ppca": float(alpha)}
            for (detector, start, width), alpha in sorted(self._alphas.items())
        ]

    @property
    def mean_alpha_by_detector(self):
        return [
            float(np.mean([a for (d, _, _), a in self._alphas.items() if d == detector]))
            if any(d == detector for d, _, _ in self._alphas) else 0.0
            for detector in range(2)
        ]


def _fit_hermite_ppca_imputer(control, mode, baseline_ids, *, components,
                              fit_events, tuning_events, slope_points,
                              batch_size, seed):
    """Fit PPCA on control events with disjoint fitting and tuning cohorts."""
    from waveform_analysis.models.ppca_imputer import ProbabilisticPCA

    all_ids = np.arange(control.n_events, dtype=np.int64)
    available = np.setdiff1d(all_ids, np.asarray(baseline_ids, dtype=np.int64), assume_unique=False)
    rng = np.random.default_rng(int(seed) + 1)
    available = rng.permutation(available)
    n_tune = min(int(tuning_events), max(2, len(available) // 5))
    n_fit = min(int(fit_events), len(available) - n_tune)
    if n_fit <= components or n_tune < 2:
        raise ValueError(
            "Not enough control events to fit the imputation-based TSHAP method "
            f"(available={len(available)}, fit={n_fit}, tuning={n_tune}, components={components}). "
            "Reduce --imputer-components/--imputer-fit-events or use more control events."
        )
    fit_ids = available[:n_fit]
    tune_ids = available[n_fit:n_fit + n_tune]
    fit_waves = waveform_view(control, mode, fit_ids).materialize()
    tuning_waves = waveform_view(control, mode, tune_ids).materialize()
    models = []
    explained = []
    for detector in range(2):
        model = ProbabilisticPCA(components).fit(
            np.asarray(fit_waves[:, detector, :], dtype=np.float64)
        )
        models.append(model)
        explained.append(float(model.explained_variance_ratio))
    del fit_waves
    imputer = _HermitePPCAImputer(
        models, tuning_waves, slope_points=slope_points, batch_size=batch_size
    )
    del tuning_waves
    metadata = {
        "training_indices": fit_ids,
        "tuning_indices": tune_ids,
        "training_event_ids": np.asarray(control.event_index[fit_ids], dtype=np.int64),
        "tuning_event_ids": np.asarray(control.event_index[tune_ids], dtype=np.int64),
        "n_training_events": int(len(fit_ids)),
        "n_tuning_events": int(len(tune_ids)),
        "components": int(components),
        "slope_points": int(slope_points),
        "explained_variance_ratio_by_detector": explained,
        "training_and_tuning_exclude_reference_baselines": True,
    }
    return imputer, metadata



def _cached_hermite_ppca_imputer(
    control, mode, config, *, window_samples, stride_samples,
    components, fit_events, tuning_events, slope_points, batch_size, seed,
):
    """Fit or resolve one immutable conditional Hermite-PPCA imputer product."""
    from waveform_analysis.models.ppca_imputer import ProbabilisticPCA

    family = mode_family(mode)
    times = np.asarray(getattr(control, f"{family}_time_ps"), np.float64)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("Invalid control time grid for imputer cache")
    board = str(config.get("_board") or _find_board_label(config.get("output_dir", "")))
    if board == "Unknown board":
        board = str(control.manifest.get("board") or "unknown_board")
    settings = dict(
        cache_schema=2, board=board, channel=family, mode=mode,
        window_ns=config["window_ns"],
        control_identity=control.manifest.get("analysis_protocol_identity"),
        control_population=control.manifest.get("event_population_identity"),
        samples=int(times.size), time_grid_hash=canonical_hash(times.tolist()),
        window_samples=int(window_samples), stride_samples=int(stride_samples),
        components=int(components), fit_events=int(fit_events),
        tuning_events=int(tuning_events), slope_points=int(slope_points),
        batch_size=int(batch_size), seed=int(seed),
    )
    recipe = ProductRecipe(
        "tshap-imputers",
        settings,
        parents=(str(control.directory.name),),
        implementation="conditional-hermite-ppca-v1",
    )
    store = ArtifactStore(config["preprocessing"]["artifact_root"])
    prior = store.lookup(recipe)

    def build(directory, effective_recipe=None):
        imputer, fit_meta = _fit_hermite_ppca_imputer(
            control, mode, np.empty(0, np.int64), components=components,
            fit_events=fit_events, tuning_events=tuning_events,
            slope_points=slope_points, batch_size=batch_size, seed=seed,
        )
        starts = np.arange(0, len(times), int(stride_samples), dtype=np.int64)
        widths = np.minimum(int(window_samples), len(times) - starts).astype(np.int64)
        for start, width in zip(starts, widths):
            for detector in range(2):
                imputer._prepare(detector, int(start), int(width))
        alphas = np.asarray([
            [imputer._alphas[(detector, int(start), int(width))]
             for start, width in zip(starts, widths)]
            for detector in range(2)
        ], dtype=np.float64)
        metadata = dict(
            n_training_events=int(fit_meta["n_training_events"]),
            n_tuning_events=int(fit_meta["n_tuning_events"]),
            explained_variance_ratio_by_detector=fit_meta["explained_variance_ratio_by_detector"],
            training_and_tuning_exclude_reference_baselines=True,
        )
        np.savez_compressed(
            directory / "imputer.npz",
            metadata_json=np.asarray(json.dumps(metadata)),
            starts=starts, widths=widths, alphas=alphas,
            **{f"mean_{d}": model.mean for d, model in enumerate(imputer.models)},
            **{f"loadings_{d}": model.W for d, model in enumerate(imputer.models)},
            **{f"noise_{d}": np.asarray(model.sigma2) for d, model in enumerate(imputer.models)},
            **{f"variance_{d}": np.asarray(model.explained_variance_ratio)
               for d, model in enumerate(imputer.models)},
        )

    alias = imputer_alias(
        cache_root_from_config(config), config, window_samples=window_samples,
        stride_samples=stride_samples, components=components, fit_events=fit_events,
        tuning_events=tuning_events, slope_points=slope_points, seed=seed,
    )
    reference = get_or_build_named(
        store, recipe, alias, build,
        metadata={
            "control_identity": settings["control_identity"],
            "control_population": settings["control_population"],
            "fit_events": int(fit_events),
            "tuning_events": int(tuning_events),
        },
    )
    cache_file = reference.payload_path("imputer.npz")
    with np.load(cache_file, allow_pickle=False) as saved:
        models = []
        for detector in range(2):
            model = ProbabilisticPCA(components)
            model.mean = np.asarray(saved[f"mean_{detector}"], np.float64)
            model.W = np.asarray(saved[f"loadings_{detector}"], np.float64)
            model.sigma2 = float(saved[f"noise_{detector}"])
            model.explained_variance_ratio = float(saved[f"variance_{detector}"])
            if model.mean.shape != (len(times),) or model.W.shape != (len(times), components):
                raise ValueError(f"Imputer artifact shape mismatch: {cache_file}")
            models.append(model)
        imputer = _HermitePPCAImputer(
            models, np.empty((0, 2, len(times))), slope_points=slope_points,
            batch_size=batch_size,
        )
        starts = np.asarray(saved["starts"], np.int64)
        widths = np.asarray(saved["widths"], np.int64)
        alphas = np.asarray(saved["alphas"], np.float64)
        if alphas.shape != (2, len(starts)) or not np.all(np.isfinite(alphas)):
            raise ValueError(f"Imputer artifact weights invalid: {cache_file}")
        for i, (start, width) in enumerate(zip(starts, widths)):
            for detector in range(2):
                key = (detector, int(start), int(width))
                imputer._alphas[key] = float(alphas[detector, i])
                imputer._prepared[key] = models[detector].prepare_mask(int(start), int(width))
                imputer._envelopes[key] = imputer._envelope(int(start), int(width), len(times))
        metadata = json.loads(str(saved["metadata_json"].item()))
    return imputer, {
        **metadata,
        "cache_file": str(cache_file),
        "cache_hit": prior.status == "hit",
        "cache_identity": reference.product_id,
    }


def _new_explainer(window_samples, stride_samples, roi, predict_batch_size, imputer=None):
    try:
        from tshap.tshap import TSHAPExplainer
    except ImportError as exc:
        raise ImportError("TSHAP requires the upstream dependency: pip install tshap==0.0.1") from exc

    if imputer is None:
        raise ValueError("The conditional TSHAP adapter requires a fitted imputer")

    class BoundedTSHAP(TSHAPExplainer):
        """Upstream temporal aggregation with conditional single-event occlusion."""

        def _calculate_window_attrib_scores(self, fnc, input_sample, baselines, window_positions):
            sample = np.asarray(input_sample, np.float32)
            if sample.ndim != 2 or sample.shape[0] != 2:
                raise ValueError("Expected one two-detector waveform")
            original = np.asarray(fnc(sample[None]), np.float64)
            if original.shape != (1,):
                raise ValueError("TSHAP predictor must return one scalar per event")
            scores = np.empty(len(window_positions), np.float64)
            for first in range(0, len(window_positions), predict_batch_size):
                part = window_positions[first:first + predict_batch_size]
                candidates = np.repeat(sample[None], len(part), axis=0)
                for i, position in enumerate(part):
                    channel, start = int(position[0]), int(position[1])
                    width = int(position[2]) if len(position) == 3 else self.window_length
                    stop = min(start + width, sample.shape[-1])
                    if not 0 <= channel < 2 or start < 0 or stop <= start:
                        raise ValueError("Invalid temporal occlusion window")
                    imputed = imputer.impute_segment_batch(
                        sample[None], channel, start, stop-start
                    )[0]
                    candidates[i, channel, start:stop] = imputed
                predicted = np.asarray(fnc(candidates), np.float64)
                if predicted.shape != (len(part),):
                    raise ValueError("TSHAP returned an invalid prediction batch")
                scores[first:first+len(part)] = original[0] - predicted
            return scores

    return BoundedTSHAP(window_length=window_samples, stride=stride_samples,
                        interpolation=True, roi=roi)


def _format_global_model_name(model_name):
    """Return publication-friendly model labels while preserving common acronyms."""
    known = {
        "direct_linear_ridge": "Direct Linear Ridge",
        "shared_linear_ridge": "Shared Linear Ridge",
        "direct_minirocket": "Direct MiniRocket",
        "shared_minirocket": "Shared MiniRocket",
        "direct_rdst": "Direct RDST",
        "shared_rdst": "Shared RDST",
        "direct_hydra": "Direct Hydra",
        "shared_hydra": "Shared Hydra",
        "direct_multirocket": "Direct MultiRocket",
        "shared_multirocket": "Shared MultiRocket",
        "direct_multirocket_rdst": "Direct MultiRocket + RDST",
        "shared_multirocket_rdst": "Shared MultiRocket + RDST",
        "direct_mlp": "Direct MLP",
        "antisymmetric_mlp": "Antisymmetric MLP",
        "independent_cnn1d": "Independent CNN",
        "shared_cnn1d": "Shared CNN",
        "onishi_cnn": "Onishi CNN",
        "locally_connected_mlp": "Locally Connected MLP",
    }
    return known.get(str(model_name).lower(), str(model_name).replace("_", " ").title())


def _find_board_label(*paths):
    """Find a known detector-board label from result or XAI paths."""
    for path in paths:
        if path is None:
            continue
        for part in reversed(Path(path).expanduser().parts):
            if part.upper() in {"UC", "FBK"}:
                return part.upper()
    return "Unknown board"


def _global_plot_context(destination, summary):
    """Recover plot labels from saved metadata and the stable XAI path."""
    identity = summary.get("identity", {})
    model_name = str(identity.get("model") or summary.get("model_name") or destination.parent.name)
    mode = str(identity.get("mode") or summary.get("mode") or "")
    window_name = str(summary.get("window_name") or destination.parent.parent.name)
    formulation = str(summary.get("formulation") or (
        "shared" if model_name.lower().startswith("shared_") else "direct"
    )).lower()
    board_name = str(summary.get("board") or _find_board_label(destination))
    settings = summary.get("settings", {})
    window_samples = int(
        summary.get("tshap_window_samples", settings.get("window_samples", 1))
    )
    if window_samples < 1:
        window_samples = 1
    return {
        "model_name": model_name,
        "formulation": formulation,
        "mode": mode,
        "window_name": window_name,
        "board_name": board_name,
        "n_events": int(summary.get("n_explained_events", 0)),
        "window_samples": window_samples,
    }


def _write_global_importance_csv(destination, time_ps, mean_absolute, formulation, roi_maps=None):
    """Write only unnormalized absolute-importance summaries to the global CSV."""
    time_ps = np.asarray(time_ps, dtype=np.float64)
    mean_absolute = np.asarray(mean_absolute, dtype=np.float64)
    if mean_absolute.shape != (2, time_ps.size):
        raise ValueError("Global absolute importance must have shape [2 detectors, samples]")
    roi_abs = None
    if roi_maps is not None:
        roi_maps = np.asarray(roi_maps, dtype=np.float64)
        if roi_maps.ndim != 3 or roi_maps.shape[1:] != mean_absolute.shape:
            raise ValueError("Global ROI attributions must have shape [events, 2 detectors, samples]")
        roi_abs = np.mean(np.abs(roi_maps), axis=0)

    shared = str(formulation).lower() == "shared"
    if shared:
        curves = [(
            "shared",
            np.mean(mean_absolute, axis=0),
            np.mean(roi_abs, axis=0) if roi_abs is not None else None,
        )]
    else:
        curves = [
            (f"detector_{channel + 1}", mean_absolute[channel],
             roi_abs[channel] if roi_abs is not None else None)
            for channel in range(2)
        ]

    with (Path(destination) / "global_importance.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "detector_or_shared",
            "time_ps",
            "mean_absolute_tshap_importance_ps_per_sample",
            "roi_mean_absolute_tshap_importance_ps",
        ])
        for label, values, roi_values in curves:
            for index, time_value in enumerate(time_ps):
                writer.writerow([
                    label,
                    float(time_value),
                    float(values[index]),
                    float(roi_values[index]) if roi_values is not None else "",
                ])


def _add_saliency_line(ax, time_ns, waveform_mv, importance, *, norm, cmap, linestyle):
    """Draw a physical waveform whose segment colours encode absolute importance."""
    from matplotlib.collections import LineCollection

    points = np.column_stack((time_ns, waveform_mv))
    segments = np.stack((points[:-1], points[1:]), axis=1)
    segment_importance = 0.5 * (importance[:-1] + importance[1:])
    line = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=3.5,
        linestyles=linestyle,
        zorder=3,
        capstyle="round",
        joinstyle="round",
    )
    line.set_array(segment_importance)
    ax.add_collection(line)
    return line


def _plot_global(
    destination, time_ps, mean_absolute, mean_waveform_mv, *,
    model_name, formulation, mode, window_name, board_name, n_events,
    window_samples,
):
    """Plot physical waveforms with common-scale TSHAP saliency and aligned histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm
    from matplotlib.cm import ScalarMappable

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }):
        time_ps = np.asarray(time_ps, dtype=np.float64)
        time_ns = time_ps / 1000.0
        mean_absolute = np.asarray(mean_absolute, dtype=np.float64)
        mean_waveform_mv = np.asarray(mean_waveform_mv, dtype=np.float64)
        expected_shape = (2, time_ps.size)
        if time_ps.ndim != 1 or time_ps.size < 2 or not np.all(np.isfinite(time_ps)):
            raise ValueError("Global TSHAP time grid must contain at least two finite samples")
        time_steps = np.diff(time_ps)
        if np.any(time_steps <= 0):
            raise ValueError("Global TSHAP time grid must be strictly increasing")
        dt_ps = float(np.median(time_steps))
        if not np.allclose(time_steps, dt_ps, rtol=1e-5, atol=1e-3):
            raise ValueError("Global TSHAP histogram requires a uniformly sampled time grid")
        if isinstance(window_samples, bool) or int(window_samples) < 1:
            raise ValueError("window_samples must be an integer >= 1")
        window_samples = int(window_samples)
        if mean_absolute.shape != expected_shape or mean_waveform_mv.shape != expected_shape:
            raise ValueError("Global TSHAP arrays must have shape [2 detectors, samples]")
        if not np.all(np.isfinite(mean_absolute)) or np.any(mean_absolute < 0):
            raise ValueError("Global absolute importance must be finite and non-negative")
        if not np.all(np.isfinite(mean_waveform_mv)):
            raise ValueError("Global physical waveforms must be finite")

        shared = str(formulation).lower() == "shared"
        saliency = np.mean(mean_absolute, axis=0) if shared else mean_absolute
        # Direct models use the maximum across BOTH detectors, preserving a common
        # quantitative colour scale while PowerNorm(gamma > 1) suppresses weak
        # attribution colours and improves separation of low and moderate importance.
        vmax = float(np.max(saliency) if shared else np.max(mean_absolute))
        if not np.isfinite(vmax) or vmax <= 0.0:
            vmax = 1.0
        norm = PowerNorm(gamma=1.6, vmin=0.0, vmax=vmax, clip=True)
        cmap = LinearSegmentedColormap.from_list(
            "neutral_to_saliency",
            [
                (0.00, "#b8b8b8"),
                (0.06, "#4c78a8"),
                (0.52, "#1f5f99"),
                (1.00, "#f2d94e"),
            ],
            N=256,
        )

        window_width_ns = window_samples * dt_ps / 1000.0
        bin_starts = list(range(0, time_ps.size, window_samples))
        bin_centers_ns = np.asarray([
            np.mean(time_ns[start:min(start + window_samples, time_ps.size)])
            for start in bin_starts
        ], dtype=np.float64)
        detector_window_importance = np.asarray([
            [
                np.sum(mean_absolute[channel, start:min(start + window_samples, time_ps.size)])
                for start in bin_starts
            ]
            for channel in range(2)
        ], dtype=np.float64)
        shared_window_importance = np.asarray([
            np.sum(saliency[start:min(start + window_samples, time_ps.size)])
            for start in bin_starts
        ], dtype=np.float64)

        if shared:
            # Stack both waveforms and the pooled importance histogram in one
            # column so every time axis has exactly the same horizontal extent.
            fig = plt.figure(figsize=(12.2, 7.8), constrained_layout=True)
            gs = fig.add_gridspec(
                3, 2,
                height_ratios=[1.0, 1.0, 0.62],
                width_ratios=[1.0, 0.035],
                hspace=0.025,
                wspace=0.08,
            )
            ax_signal1 = fig.add_subplot(gs[0, 0])
            ax_signal2 = fig.add_subplot(gs[1, 0], sharex=ax_signal1)
            ax_shared_hist = fig.add_subplot(gs[2, 0], sharex=ax_signal1)
            colorbar_ax = fig.add_subplot(gs[0:2, 1])
            hist_axes = [ax_shared_hist]
        else:
            fig = plt.figure(figsize=(13.2, 6.3), constrained_layout=True)
            gs = fig.add_gridspec(
                2, 2,
                height_ratios=[2.35, 1.0],
                hspace=0.025,
                wspace=0.12,
            )
            ax_signal1 = fig.add_subplot(gs[0, 0])
            ax_signal2 = fig.add_subplot(gs[0, 1])
            ax_detector1_hist = fig.add_subplot(gs[1, 0], sharex=ax_signal1)
            ax_detector2_hist = fig.add_subplot(gs[1, 1], sharex=ax_signal2)
            hist_axes = [ax_detector1_hist, ax_detector2_hist]
            colorbar_ax = None
        signal_axes = [ax_signal1, ax_signal2]

        title = f"Global TSHAP Saliency — {_format_global_model_name(model_name)}"
        fig.suptitle(title, fontsize=12.5, y=1.015)

        for ax in [*signal_axes, *hist_axes]:
            ax.axvspan(-1.5, 2.0, color="0.75", alpha=0.25, linewidth=0, zorder=0)

        low = float(np.min(mean_waveform_mv))
        high = float(np.max(mean_waveform_mv))
        span = high - low
        y_margin = 0.06 * span if span > 0 else max(abs(high) * 0.06, 1.0)
        for channel, ax in enumerate(signal_axes):
            importance = saliency if shared else saliency[channel]
            ax.plot(
                time_ns, mean_waveform_mv[channel],
                color="#d8dadd", linewidth=3.8, linestyle="solid", zorder=2,
            )
            _add_saliency_line(
                ax, time_ns, mean_waveform_mv[channel], importance,
                norm=norm, cmap=cmap, linestyle="solid",
            )
            timing_channel = "timing" in str(mode).lower()
            label_x = 0.982 if timing_channel else 0.018
            label_ha = "right" if timing_channel else "left"
            ax.text(
                label_x, 0.965, f"Detector {channel + 1}",
                transform=ax.transAxes, ha=label_ha, va="top",
                fontsize=9.0, zorder=5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.6},
            )
            ax.set_ylabel("Amplitude (mV)")
            ax.set_ylim(low - y_margin, high + y_margin)
            ax.set_xlim(float(time_ns[0]), float(time_ns[-1]))
            ax.set_axisbelow(True)
            ax.grid(True, alpha=0.18, linewidth=0.5)
            ax.tick_params(axis="x", labelbottom=False, length=3, width=0.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        histogram_color = "#4c78a8"
        if shared:
            ax_shared_hist.bar(
                bin_centers_ns, shared_window_importance, width=window_width_ns,
                align="center", color=histogram_color, edgecolor="0.25",
                linewidth=0.35, zorder=2,
            )
            ax_shared_hist.set_ylabel("Importance (ps)")
            ax_shared_hist.set_title("Shared importance", loc="left", fontsize=9.5, pad=2)
            ax_shared_hist.grid(True, axis="y", alpha=0.18, linewidth=0.5)
        else:
            hist_specs = (
                (ax_detector1_hist, detector_window_importance[0]),
                (ax_detector2_hist, detector_window_importance[1]),
            )
            # One absolute y-scale keeps detector histograms directly comparable.
            histogram_ymax = float(np.max(detector_window_importance))
            if not np.isfinite(histogram_ymax) or histogram_ymax <= 0.0:
                histogram_ymax = 1.0
            histogram_ymax *= 1.05
            for ax, values in hist_specs:
                ax.bar(
                    bin_centers_ns, values, width=window_width_ns,
                    align="center", color=histogram_color, edgecolor="0.25",
                    linewidth=0.35, zorder=2,
                )
                ax.set_ylim(0.0, histogram_ymax)
                ax.set_ylabel("Importance (ps)")
                ax.grid(True, axis="y", alpha=0.18, linewidth=0.5)

        for ax in hist_axes:
            ax.set_xlabel("Waveform time (ns)")
            ax.tick_params(axis="both", length=3, width=0.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

        if shared:
            colorbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                cax=colorbar_ax,
            )
        else:
            colorbar = fig.colorbar(
                ScalarMappable(norm=norm, cmap=cmap),
                ax=signal_axes, pad=0.015, fraction=0.025, location="right",
            )
        from matplotlib.ticker import MaxNLocator

        tick_locator = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        colorbar_ticks = tick_locator.tick_values(0.0, vmax)
        colorbar_ticks = colorbar_ticks[
            (colorbar_ticks >= 0.0) & (colorbar_ticks <= vmax + 1e-12)
        ]
        if colorbar_ticks.size == 0 or not np.isclose(colorbar_ticks[0], 0.0):
            colorbar_ticks = np.insert(colorbar_ticks, 0, 0.0)
        colorbar.set_ticks(colorbar_ticks)
        colorbar.set_label("Absolute TSHAP importance (ps/sample)", fontsize=9)
        colorbar.ax.tick_params(labelsize=8, width=0.6, length=3)

        destination = Path(destination)
        fig.savefig(destination / "global_importance.png", dpi=240, bbox_inches="tight")
        fig.savefig(
            destination / "global_importance.pdf",
            bbox_inches="tight",
            metadata={"Title": title, "Creator": "Matplotlib"},
        )
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


LOCAL_GROUP_LABELS = {
    "good_positive": "Good positive correction",
    "good_negative": "Good negative correction",
    "bad_positive": "Bad positive correction",
    "bad_negative": "Bad negative correction",
    "typical": "Typical correction (median predicted output)",
}


def _select_local_events(predictions, targets, events_per_group=2, extreme_margin=0.05):
    """Select two non-overlapping examples per correction/improvement category.

    Good examples target upper improvement quantiles; bad examples target lower
    improvement quantiles. The outer extreme_margin fraction is excluded, so
    examples are near the good/bad tails without selecting absolute extremes.
    Typical examples are closest to the median predicted correction.
    """
    _validate_positive_int("events_per_group", events_per_group)
    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if predictions.ndim != 1 or targets.shape != predictions.shape or not predictions.size:
        raise ValueError("Local TSHAP selection requires matching nonempty prediction/target vectors")
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(targets)):
        raise ValueError("Local TSHAP selection requires finite predictions and targets")
    if not np.isfinite(extreme_margin) or not 0 < extreme_margin < 0.25:
        raise ValueError("extreme_margin must be greater than 0 and less than 0.25")

    improvement = np.abs(targets) - np.abs(targets - predictions)
    masks = {
        "good_positive": (predictions > 0) & (improvement > 0),
        "good_negative": (predictions < 0) & (improvement > 0),
        "bad_positive": (predictions > 0) & (improvement < 0),
        "bad_negative": (predictions < 0) & (improvement < 0),
    }
    selected = []
    used = set()
    for group in ("good_positive", "good_negative", "bad_positive", "bad_negative"):
        candidates = np.flatnonzero(masks[group])
        if candidates.size < events_per_group:
            raise ValueError(
                f"Cannot select {events_per_group} events for {group}: "
                f"only {candidates.size} eligible events"
            )
        values = improvement[candidates]
        order = candidates[np.argsort(values, kind="stable")]
        if group.startswith("good_"):
            quantiles = np.linspace(1.0 - 2.0 * extreme_margin, 1.0 - extreme_margin,
                                    events_per_group)
        else:
            quantiles = np.linspace(extreme_margin, 2.0 * extreme_margin, events_per_group)
        used_ranks = set()
        for quantile in quantiles:
            desired_rank = int(round(float(quantile) * (len(order) - 1)))
            available_ranks = [rank for rank in range(len(order)) if rank not in used_ranks]
            rank = min(available_ranks, key=lambda candidate: (abs(candidate - desired_rank), candidate))
            used_ranks.add(rank)
            index = int(order[rank])
            selected.append({"group": group, "index": index})
            used.add(index)

    median_prediction = float(np.median(predictions))
    typical_candidates = np.asarray([i for i in range(len(predictions)) if i not in used], dtype=np.int64)
    if typical_candidates.size < events_per_group:
        raise ValueError(
            f"Cannot select {events_per_group} non-overlapping typical events: "
            f"only {typical_candidates.size} events remain"
        )
    distance = np.abs(predictions[typical_candidates] - median_prediction)
    typical_order = typical_candidates[np.argsort(distance, kind="stable")]
    for index in typical_order[:events_per_group]:
        selected.append({"group": "typical", "index": int(index)})

    if len({item["index"] for item in selected}) != len(selected):
        raise AssertionError("Local TSHAP selection produced duplicate events")
    return selected, improvement, median_prediction


LOCAL_PLOT_VERSION = 9


def _detector_difference(values):
    """Return the raw detector 1 minus detector 2 waveform."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2:
        raise ValueError("Expected an array with shape [2 detectors, samples]")
    return values[0] - values[1]


def _center_detector_difference(values, dataset_mean_difference):
    """Subtract the per-sample mean detector difference estimated across events."""
    difference = _detector_difference(values)
    dataset_mean = np.asarray(dataset_mean_difference, dtype=np.float64)
    if dataset_mean.shape != difference.shape:
        raise ValueError("Dataset mean difference must match the waveform sample grid")
    if not np.all(np.isfinite(dataset_mean)):
        raise ValueError("Dataset mean difference must be finite")
    return difference - dataset_mean


def _dataset_mean_difference_mv(
    dataset, mode, waveform_transform, seed, max_events=1000
):
    """Estimate mean detector difference per sample from a reproducible event sample."""
    _validate_positive_int("max_events", max_events)
    if dataset.n_events < 1:
        raise ValueError("Cannot estimate dataset mean from an empty population")
    indices = _choose_indices(
        dataset.n_events, max_events, _selection_seed(int(seed), "blind_events")
    )
    normalized = waveform_view(dataset, mode, indices).materialize()
    waveform_mv = np.asarray(waveform_transform.inverse(normalized), dtype=np.float64)
    if waveform_mv.ndim != 3 or waveform_mv.shape[1] != 2:
        raise ValueError("Dataset waveforms must have shape [events, 2 detectors, samples]")
    differences = waveform_mv[:, 0, :] - waveform_mv[:, 1, :]
    if not np.all(np.isfinite(differences)):
        raise ValueError("Dataset sample used for mean difference contains non-finite values")
    return np.mean(differences, axis=0), indices


def _native_led_baselines(
    artifact_root, dataset, mode, dataset_indices, baseline_window_ns
):
    """Read trigger-relative baselines from the exact referenced native product."""
    from waveform_analysis.data.artifact_store import ArtifactStore

    dataset_indices = np.asarray(dataset_indices, dtype=np.int64).reshape(-1)
    if dataset_indices.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if np.any(dataset_indices < 0) or np.any(dataset_indices >= dataset.n_events):
        raise ValueError("LED baseline dataset indices are out of range")
    native_ref = dataset.manifest.get("native_artifact_ref")
    if not native_ref or not native_ref.get("product_id"):
        raise ValueError("Prepared dataset has no native waveform artifact reference")
    reference = ArtifactStore(artifact_root).resolve(
        str(native_ref["product_type"]), str(native_ref["product_id"])
    )
    native_manifest = json.loads((reference.path / "manifest.json").read_text(encoding="utf-8"))
    family = mode_family(mode)
    if (
        native_manifest.get("source") != dataset.manifest.get("analysis_source")
        or native_manifest.get("mode") != mode
        or native_manifest.get("family") != family
    ):
        raise ValueError("Native waveform artifact does not match prepared dataset provenance")
    source_event_ids = np.asarray(dataset.event_index, dtype=np.int64)[dataset_indices]
    native_event_ids = np.asarray(
        np.load(reference.payload_path("event_index.npy"), mmap_mode="r"), dtype=np.int64
    )
    row_lookup = {int(event_id): row for row, event_id in enumerate(native_event_ids)}
    missing = [int(event_id) for event_id in source_event_ids if int(event_id) not in row_lookup]
    if missing:
        raise ValueError(f"Native waveform artifact is missing prepared events: {missing[:5]}")
    rows = np.asarray([row_lookup[int(event_id)] for event_id in source_event_ids], dtype=np.int64)
    waves = np.load(reference.payload_path(f"{family}_windows_mV.npy"), mmap_mode="r")
    intervals = np.load(reference.payload_path(f"{family}_sample_interval_s.npy"), mmap_mode="r")
    from waveform_analysis.signal.baseline import baseline_level_mV
    baseline = np.empty((len(rows), 2), dtype=np.float64)
    for output_row, native_row in enumerate(rows):
        for detector in range(2):
            baseline[output_row, detector] = baseline_level_mV(
                waves[native_row, detector],
                float(intervals[native_row, detector]),
                float(native_manifest["materialized_window_ns"]["before"]),
                baseline_window_ns,
            )
    if not np.all(np.isfinite(baseline)):
        raise ValueError("Native LED baselines contain non-finite values")
    return baseline

def _led_reference_levels(baseline_mv, threshold_mv):
    """Convert the calibrated baseline-relative LED threshold to absolute mV."""
    baseline = np.asarray(baseline_mv, dtype=np.float64).reshape(-1)
    if baseline.shape != (2,) or not np.all(np.isfinite(baseline)):
        raise ValueError("Expected two finite detector baseline levels")
    if not np.isfinite(threshold_mv):
        raise ValueError("LED threshold must be finite")
    return baseline + float(threshold_mv)


def _led_line_extent(time_ns):
    """Limit the LED reference line to the waveform start and time zero."""
    time_ns = np.asarray(time_ns, dtype=np.float64).reshape(-1)
    if time_ns.size == 0 or not np.all(np.isfinite(time_ns)):
        raise ValueError("LED line time grid must contain finite samples")
    start = float(time_ns[0])
    end = float(time_ns[-1])
    return (start, 0.0) if start <= 0.0 <= end else (start, end)


def _plot_local_event(path, *, time_ps, waveform_mv, dataset_mean_difference_mv,
                      led_baseline_mv, attribution, group, event_id, target_ps,
                      prediction_ps, improvement_ps, led_threshold_mv):
    """Plot detector waveforms in mV, their mean-centered difference, and TSHAP."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    waveform_mv = np.asarray(waveform_mv, dtype=np.float64)
    attribution = np.asarray(attribution, dtype=np.float64)
    if waveform_mv.shape != (2, time_ns.size) or attribution.shape != waveform_mv.shape:
        raise ValueError("Local TSHAP plot arrays must have shape [2 detectors, samples]")
    if not np.all(np.isfinite(waveform_mv)) or not np.isfinite(led_threshold_mv):
        raise ValueError("Waveforms and LED threshold must be finite")

    waveform_difference = _center_detector_difference(
        waveform_mv, dataset_mean_difference_mv
    )
    led_levels_mv = _led_reference_levels(led_baseline_mv, led_threshold_mv)
    led_line_start_ns, led_line_end_ns = _led_line_extent(time_ns)

    fig, axes = plt.subplots(
        3, 1, figsize=(12.5, 8.5), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [1.0, 0.85, 1.1]},
    )
    label = LOCAL_GROUP_LABELS[group]
    fig.suptitle(
        f"{label} | event {int(event_id)}\n"
        f"Predicted correction: {prediction_ps:+.2f} ps | "
        f"Signed improvement: {improvement_ps:+.2f} ps | "
        f"Target correction: {target_ps:+.2f} ps | "
        f"Final residual: {target_ps - prediction_ps:+.2f} ps",
        fontsize=11,
    )

    ax = axes[0]
    ax.plot(time_ns, waveform_mv[0], linewidth=1.15, label="Detector 1")
    ax.plot(time_ns, waveform_mv[1], linewidth=1.15, label="Detector 2")
    ax.hlines(
        led_levels_mv[0], xmin=led_line_start_ns, xmax=led_line_end_ns,
        linestyle="--", linewidth=0.9,
        label=f"Detector 1 LED threshold ({led_levels_mv[0]:.1f} mV)",
    )
    ax.hlines(
        led_levels_mv[1], xmin=led_line_start_ns, xmax=led_line_end_ns,
        linestyle=":", linewidth=0.9,
        label=f"Detector 2 LED threshold ({led_levels_mv[1]:.1f} mV)",
    )
    ax.set_title("Detector waveforms")
    ax.set_ylabel("Amplitude (mV)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="best", fontsize=9, ncols=3)

    ax = axes[1]
    ax.plot(time_ns, waveform_difference, linewidth=1.15,
            label="Detector difference − dataset mean")
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=0.8)
    ax.set_title("Detector difference relative to dataset mean")
    ax.set_ylabel("Amplitude difference (mV)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="best", fontsize=9)

    ax = axes[2]
    ax.plot(time_ns, attribution[0], linewidth=1.1, label="Detector 1")
    ax.plot(time_ns, attribution[1], linewidth=1.1, label="Detector 2")
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=0.8)
    ax.set_title("Local TSHAP attributions")
    ax.set_ylabel("Attribution (ps/sample)")
    ax.set_xlabel("Waveform time (ns)")
    ax.grid(True, alpha=0.22, linewidth=0.6)
    ax.legend(loc="best", fontsize=9, ncols=2)

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _refresh_local_presentation(
    destination, *, waveform_transform, led_threshold_mv, baseline_window_ns,
    dataset, mode, seed, artifact_root
):
    """Refresh plots from cached attributions and a cached dataset-mean waveform."""
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with np.load(destination / "local_attributions.npz") as saved:
        arrays = {key: np.asarray(saved[key]) for key in saved.files}
    if (
        summary.get("plot_version", 0) >= LOCAL_PLOT_VERSION
        and "dataset_mean_difference_mv" in arrays
        and "led_baseline_mv" in arrays
    ):
        return

    time_ps = np.asarray(arrays["time_ps"], dtype=np.float64)
    waves = np.asarray(arrays["waveform_normalized"], dtype=np.float64)
    attributions = np.asarray(arrays["attribution_ps_per_sample"], dtype=np.float64)
    event_ids = np.asarray(arrays["event_id"], dtype=np.int64)
    groups = np.asarray(arrays["group"]).astype(str)
    targets = np.asarray(arrays["target_correction_ps"], dtype=np.float64)
    predictions = np.asarray(arrays["predicted_correction_ps"], dtype=np.float64)
    improvements = np.asarray(arrays["signed_improvement_ps"], dtype=np.float64)

    waveform_mv = np.asarray(
        arrays["waveform_mv"] if "waveform_mv" in arrays
        else waveform_transform.inverse(waves),
        dtype=np.float64,
    )
    selected_led_times_ps = np.asarray(arrays["led_time_ps"], dtype=np.float64)
    if selected_led_times_ps.shape != (len(event_ids), 2):
        raise ValueError("Cached local LED crossing times must have shape [events, 2]")
    threshold = float(np.asarray(
        arrays["led_threshold_mv"] if "led_threshold_mv" in arrays
        else led_threshold_mv
    ).reshape(()))
    if "dataset_index" not in arrays:
        raise ValueError("Cached local explanations are missing dataset indices")
    led_baselines_mv = _native_led_baselines(
        artifact_root, dataset, mode, arrays["dataset_index"], baseline_window_ns
    )

    if "dataset_mean_difference_mv" in arrays:
        dataset_mean = np.asarray(arrays["dataset_mean_difference_mv"], dtype=np.float64)
        mean_indices = np.asarray(
            arrays.get("dataset_mean_sample_indices", np.empty(0, dtype=np.int64)),
            dtype=np.int64,
        )
    else:
        dataset_mean, mean_indices = _dataset_mean_difference_mv(
            dataset, mode, waveform_transform, seed, max_events=1000
        )
    if dataset_mean.shape != time_ps.shape or not np.all(np.isfinite(dataset_mean)):
        raise ValueError("Cached dataset mean difference does not match the waveform grid")

    arrays["waveform_mv"] = waveform_mv
    arrays["led_time_ps"] = selected_led_times_ps
    arrays["led_baseline_mv"] = led_baselines_mv
    arrays["led_threshold_mv"] = np.asarray(threshold, dtype=np.float64)
    arrays["dataset_mean_difference_mv"] = dataset_mean
    arrays["dataset_mean_sample_indices"] = mean_indices

    plot_names = []
    for i in range(len(event_ids)):
        plot_name = f"{groups[i]}__event_{int(event_ids[i])}.png"
        _plot_local_event(
            destination / plot_name,
            time_ps=time_ps,
            waveform_mv=waveform_mv[i],
            dataset_mean_difference_mv=dataset_mean,
            led_baseline_mv=led_baselines_mv[i],
            attribution=attributions[i],
            group=groups[i],
            event_id=int(event_ids[i]),
            target_ps=float(targets[i]),
            prediction_ps=float(predictions[i]),
            improvement_ps=float(improvements[i]),
            led_threshold_mv=threshold,
        )
        plot_names.append(plot_name)

    refreshed = destination / "local_attributions.refresh.npz"
    np.savez_compressed(refreshed, **arrays)
    os.replace(refreshed, destination / "local_attributions.npz")
    summary["plots"] = plot_names
    summary["plot_version"] = LOCAL_PLOT_VERSION
    summary["signal_units"] = "mV (inverse of frozen prepared waveform normalization)"
    summary["led_threshold_mV"] = threshold
    summary["led_baseline_source"] = "original blind-native waveform using the timing pipeline baseline window"
    summary["led_reference_line_extent"] = "waveform start to time zero"
    summary["difference_centering"] = "per-sample mean across up to 1000 seeded blind events"
    summary["difference_centering_sample_size"] = int(mean_indices.size)
    atomic_json(summary_path, summary)


def _local_explanation_destination(output_root, run_directory):
    """Return one stable local-case-study folder per run."""
    batch = batch_root(run_directory)
    relative_run = Path(run_directory).resolve().relative_to(batch)
    return Path(output_root).resolve() / batch.name / relative_run / "tshap_local"


def _complete_local(destination, fingerprint, expected_events):
    required = ("summary.json", "events.csv", "local_attributions.npz")
    if not all((destination / name).is_file() for name in required):
        return False
    try:
        summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        if summary.get("identity_sha256") != fingerprint:
            return False
        if summary.get("n_explained_events") != expected_events:
            return False
        plots = summary.get("plots", [])
        return len(plots) == expected_events and all((destination / name).is_file() for name in plots)
    except (ValueError, OSError, TypeError):
        return False


def _refresh_presentation(destination):
    """Regenerate global CSV and plot from cached attributions, without SHAP inference."""
    destination = Path(destination)
    summary_path = destination / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current_plot_files = (
        "global_importance.png", "global_importance.pdf", "global_importance.csv"
    )
    if (
        summary.get("plot_version", 0) >= GLOBAL_PLOT_VERSION
        and all((destination / name).is_file() for name in current_plot_files)
    ):
        return

    with np.load(destination / "attributions.npz") as saved:
        time_ps = np.asarray(saved["time_ps"], dtype=np.float64)
        window_maps = np.asarray(saved["window_attribution_ps"], dtype=np.float64)
        if "mean_waveform_mv" not in saved:
            raise ValueError(
                "Cannot refresh global TSHAP saliency plot: cached attributions.npz "
                "does not contain mean_waveform_mv. TSHAP values were not recalculated."
            )
        mean_waveform_mv = np.asarray(saved["mean_waveform_mv"], dtype=np.float64)
        roi_maps = (
            np.asarray(saved["roi_attribution_ps"], dtype=np.float64)
            if "roi_attribution_ps" in saved else None
        )

    if window_maps.ndim != 3 or window_maps.shape[1:] != (2, time_ps.size):
        raise ValueError("Cached TSHAP attribution array must have shape [events, 2 detectors, samples]")
    mean_absolute = np.mean(np.abs(window_maps), axis=0)
    context = _global_plot_context(destination, summary)
    _write_global_importance_csv(
        destination, time_ps, mean_absolute, context["formulation"], roi_maps
    )
    _plot_global(
        destination, time_ps, mean_absolute, mean_waveform_mv,
        model_name=context["model_name"],
        formulation=context["formulation"],
        mode=context["mode"],
        window_name=context["window_name"],
        board_name=context["board_name"],
        n_events=context["n_events"] or len(window_maps),
        window_samples=context["window_samples"],
    )

    summary["plot_version"] = GLOBAL_PLOT_VERSION
    summary["board"] = context["board_name"]
    summary["window_name"] = context["window_name"]
    summary["attribution_units"] = (
        "ps/sample (upstream window score redistributed across temporal samples)"
    )
    summary["input_units"] = "mV (inverse of frozen prepared waveform normalization)"
    summary["global_importance_aggregation"] = {
        "direct": "mean(abs(attribution)) across explained blind events separately for each detector and sample",
        "shared": "mean(abs(attribution)) across explained blind events and both detectors at each sample",
        "normalization": "importance values are unnormalized; waveform colours use a shared PowerNorm(gamma=1.6) scale and the colourbar retains original ps/sample units",
    }
    summary["tshap_window_samples"] = context["window_samples"]
    summary["global_plot"] = {
        "style": "separate physical detector waveforms with matched solid line styles and aligned importance histograms",
        "layout": (
            "three vertically stacked panels (detector 1 waveform, detector 2 waveform, shared importance histogram), with a common time axis"
            if context["formulation"] == "shared"
            else "two-column grid with each detector waveform directly above its importance histogram"
        ),
        "onishi_interval_ns": [-1.5, 2.0],
        "importance_colour_scale": "shared PowerNorm(gamma=1.6) across both detectors; colourbar ticks in original ps/sample units",
        "histogram_colour": "#4c78a8",
        "histogram_y_scale": "direct models use a common upper limit based on the maximum bin importance across both detectors",
        "detector_label_placement": "inside upper-right corner for timing channels and upper-left corner otherwise",
        "output_formats": ["PNG (240 dpi)", "PDF (vector)"],
        "importance_histogram": {
            "bin_width_samples": context["window_samples"],
            "bar_width_ns": float(context["window_samples"] * np.median(np.diff(time_ps)) / 1000.0),
            "bar_height": "sum of mean absolute sample attributions within each non-overlapping window-length bin",
            "unit": "ps",
        },
        "detector_identity": (
            "shared absolute-importance curve and pooled histogram for both signals"
            if context["formulation"] == "shared"
            else "detector-specific absolute-importance curves and histograms"
        ),
        "legend": "omitted; detector labels are placed inside waveform panels according to channel type",
    }
    atomic_json(summary_path, summary)




def _explanation_destination(output_root, run_directory):
    """Return one stable TSHAP directory per run; fingerprints belong in metadata only."""
    batch = batch_root(run_directory)
    relative_run = Path(run_directory).resolve().relative_to(batch)
    return Path(output_root).resolve() / batch.name / relative_run / "tshap"


def explain_tshap(results=None, *, model_dir=None, models=None, ml_window=None, cache_root=None, artifact_root=None, output_root=None, max_events=100, n_baselines=8,
                  window_samples=20, stride_samples=20, predict_batch_size=64, seed=42,
                  roi=False, overwrite=False, imputation="hermite_ppca",
                  imputer_components=32, imputer_fit_events=2000,
                  imputer_tuning_events=300, imputer_slope_points=5,
                  logger=None):
    """Explain saved regressors using reproducible blind/control subsets.

    Identical requests reuse completed explanation artifacts without inference.
    Changes in model, cohorts or settings produce separate, fingerprinted results.
    The explicit overwrite flag recomputes/replaces only the matching identity.
    """
    for name, value in (("max_events", max_events), ("n_baselines", n_baselines),
                        ("window_samples", window_samples), ("stride_samples", stride_samples),
                        ("predict_batch_size", predict_batch_size)):
        _validate_positive_int(name, value)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("TSHAP seed must be an integer")
    if imputation != "hermite_ppca":
        raise ValueError("Only conditional Hermite-PPCA imputation is supported")
    if imputation == "hermite_ppca":
        for name, value in (("imputer_components", imputer_components),
                            ("imputer_fit_events", imputer_fit_events),
                            ("imputer_tuning_events", imputer_tuning_events),
                            ("imputer_slope_points", imputer_slope_points)):
            _validate_positive_int(name, value)
        if imputer_slope_points < 2:
            raise ValueError("imputer_slope_points must be >= 2")
    ml_window = _effective_tshap_window(ml_window, model_dir=model_dir)
    tshap_version = version("tshap")
    runs, source_path = _tshap_run_inputs(results, model_dir)
    if not runs:
        raise ValueError(f"No complete saved-model runs found in {results or model_dir}")
    if output_root is None:
        output_root = (Path.cwd() / "xai").resolve()
        if output_root.is_relative_to(source_path):
            output_root = source_path.parent / "xai"
    else:
        output_root = Path(output_root).expanduser().resolve()
        if output_root == source_path or output_root.is_relative_to(source_path):
            raise ValueError("TSHAP output-root must be outside the supplied results/model directory")
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
            config["_tshap_run_manifest"] = manifest
            _apply_tshap_cache(config, cache_root=cache_root, artifact_root=artifact_root)
            if not _matches_tshap_window(config, ml_window):
                if model_dir is not None:
                    raise ValueError(
                        f"Selected model uses ML window {config.get('window_name')!r}, "
                        f"which does not match --ml-window {ml_window!r}"
                    )
                if logger:
                    logger.info(
                        "TSHAP skipped | unselected window=%s | requested=%s | model=%s",
                        config.get("window_name"), ml_window, manifest["model"],
                    )
                continue
            spec = get_model(manifest["model"])
            run_store = RunStore(
                directory, artifact_root=config["preprocessing"]["artifact_root"]
            )
            # An explicitly supplied local model directory is authoritative.
            # Otherwise resolve the model product from the run's artifact reference.
            local_model = Path(directory) / "model"
            fitted_model_dir = (
                local_model if model_dir is not None and saved_model_complete(spec, local_model)
                else run_store.model_dir
            )
            if not saved_model_complete(spec, fitted_model_dir):
                raise FileNotFoundError(f"Saved model or frozen feature transform missing: {fitted_model_dir}")
            sampling_seed = int(seed)
            settings = dict(max_events=max_events, n_baselines=n_baselines,
                            window_samples=window_samples, stride_samples=stride_samples,
                            predict_batch_size=predict_batch_size, seed=sampling_seed,
                            roi=bool(roi), imputation=imputation)
            if imputation == "hermite_ppca":
                settings.update(imputer_components=imputer_components,
                                imputer_fit_events=imputer_fit_events,
                                imputer_tuning_events=imputer_tuning_events,
                                imputer_slope_points=imputer_slope_points)
            identity = {
                "method": "conditional_hermite_ppca_occlusion",
                "method_version": TSHAP_METHOD_VERSION, "upstream_version": tshap_version,
                "final_fit": manifest["stage_fingerprints"]["final_fit"],
                "control": manifest["dataset_populations"]["control"]["analysis_protocol_identity"],
                "blind": manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
                "model": spec.name, "mode": config["mode"], "seed": sampling_seed, "settings": settings,
            }
            fingerprint = canonical_hash(identity)
            destination = _explanation_destination(output_root, directory)
            # A matching flat cache is the canonical hit: no model inference.
            if _complete(destination, fingerprint) and not overwrite:
                _refresh_presentation(destination)
                if logger:
                    logger.info("TSHAP | cache hit | %s", destination)
                outputs.append(destination)
                continue

            # No matching cache: compute in a temporary directory and replace the
            # stable tshap directory only after all artifacts have been written.
            control = lookup.find("control", config, manifest)
            blind = lookup.find("blind", config, manifest)
            time_ps = _check_prepared_grids(control, blind, config["mode"])
            if window_samples > time_ps.size:
                raise ValueError("TSHAP window longer than input waveform")
            event_ids = _choose_indices(
                blind.n_events, max_events, _selection_seed(sampling_seed, "blind_events")
            )
            waves = waveform_view(blind, config["mode"], event_ids).materialize()
            if waves.shape[1] != 2 or not np.all(np.isfinite(waves)):
                raise ValueError("Invalid blind waveforms")
            imputer = None
            imputer_metadata = None
            if imputation == "hermite_ppca":
                imputer, imputer_metadata = _cached_hermite_ppca_imputer(
                    control, config["mode"], config,
                    window_samples=window_samples, stride_samples=stride_samples,
                    components=imputer_components, fit_events=imputer_fit_events,
                    tuning_events=imputer_tuning_events,
                    slope_points=imputer_slope_points, batch_size=predict_batch_size,
                    seed=sampling_seed,
                )
                if logger:
                    logger.info(
                        "TSHAP imputation | Hermite-PPCA | train=%d | tune=%d | components=%d",
                        imputer_metadata["n_training_events"],
                        imputer_metadata["n_tuning_events"], imputer_components,
                    )
            if logger:
                logger.info("TSHAP conditional occlusion | %s | events=%d/%d | samples=%d | ROI=%s",
                            spec.name, len(event_ids), blind.n_events, time_ps.size, roi)
            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".tshap-", dir=parent))
            try:
                with diagnostic_random_state():
                    fitted = load_fitted_model(spec, fitted_model_dir, run["best"]["parameters"], config)
                    try:
                        predict = lambda x: _predict_bounded(spec, fitted, x, predict_batch_size)
                        # Preprocessing/masks, fitted feature map and clipping must be unchanged.
                        # Deliberately omit frozen_features: TSHAP inference always starts
                        # from original waveforms and reapplies the fitted transform.
                        original = predict_indices(spec, fitted, blind, config["mode"], event_ids,
                                                   chunk_size=predict_batch_size)
                        replay = predict(waves)
                        if not np.allclose(original, replay, atol=1e-4, rtol=1e-6):
                            raise ValueError("TSHAP model adapter does not reproduce saved inference")
                        explainer = _new_explainer(window_samples, stride_samples, roi,
                                                   predict_batch_size, imputer=imputer)
                        window_maps = np.empty_like(waves, dtype=np.float64)
                        roi_maps = np.empty_like(waves, dtype=np.float64) if roi else None
                        for i, pair in enumerate(waves):
                            win, regions = explainer.explain(pair[None], pair[None], predict)
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
                waveform_transform = getattr(
                    blind, f"{mode_family(config['mode'])}_transform"
                )
                if waveform_transform is None:
                    raise ValueError("Missing frozen waveform normalization for mV plotting")
                waveform_mv = np.asarray(waveform_transform.inverse(waves), dtype=np.float64)
                if waveform_mv.shape != waves.shape or not np.all(np.isfinite(waveform_mv)):
                    raise ValueError("Inverse-transformed blind waveforms must be finite and match TSHAP inputs")
                mean_waveform_mv = np.mean(waveform_mv, axis=0)
                payload = dict(
                    event_id=np.asarray(blind.event_index[event_ids], np.int64),
                    control_event_id=np.empty(0, np.int64),
                    time_ps=time_ps,
                    window_attribution_ps=window_maps,
                    mean_waveform_mv=mean_waveform_mv,
                    original_prediction_ps=original,
                )
                if roi:
                    payload["roi_attribution_ps"] = roi_maps
                np.savez_compressed(temporary / "attributions.npz", **payload)
                board_name = _find_board_label(directory, output_root)
                window_name = str(config.get("window_name", directory.parent.name))
                _write_global_importance_csv(
                    temporary, time_ps, win_abs, spec.estimator_formulation, roi_maps
                )
                _plot_global(
                    temporary, time_ps, win_abs, payload["mean_waveform_mv"],
                    model_name=spec.name,
                    formulation=spec.estimator_formulation,
                    mode=config["mode"],
                    window_name=window_name,
                    board_name=board_name,
                    n_events=len(event_ids),
                    window_samples=window_samples,
                )
                atomic_json(temporary / "summary.json", {
                    **identity, "identity_sha256": fingerprint, "plot_version": GLOBAL_PLOT_VERSION,
                    "board": board_name, "window_name": window_name,
                    "imputation": imputation,
                    **({"imputer": {
                        "method": "position-specific Hermite baseline plus smooth-envelope PPCA correction",
                        "components": int(imputer_components),
                        "slope_points": int(imputer_slope_points),
                        "training_events": int(imputer_metadata["n_training_events"]),
                        "tuning_events": int(imputer_metadata["n_tuning_events"]),
                        "explained_variance_ratio_by_detector": imputer_metadata["explained_variance_ratio_by_detector"],
                        "cache_file": imputer_metadata["cache_file"],
                        "cache_identity": imputer_metadata["cache_identity"],
                        "cache_hit": imputer_metadata["cache_hit"],
                        "blend_weights_by_detector_and_window": imputer.weight_rows,
                    }} if imputer is not None else {}),
                    "tshap_window_samples": int(window_samples),
                    "tshap_stride_samples": int(stride_samples),
                    "n_explained_events": len(event_ids),
                    "n_control_baselines": 0, "waveform_samples": len(time_ps),
                    "formulation": spec.estimator_formulation,
                    "attribution_units": "ps/sample (upstream window score redistributed across temporal samples)",
                    "interpretation": "Conditional window-occlusion prediction differences in ps; "
                                      "these are not exact Shapley values.",
                    "cohort": "blind", "input_units": "mV (inverse of frozen prepared waveform normalization)",
                    "global_importance_aggregation": {
                        "direct": "mean(abs(attribution)) across explained blind events separately for each detector and sample",
                        "shared": "mean(abs(attribution)) across explained blind events and both detectors at each sample",
                        "normalization": "importance values are unnormalized; waveform colours use a shared PowerNorm(gamma=1.6) scale and the colourbar retains original ps/sample units",
                    },
                    "global_plot": {
                        "style": "separate physical detector waveforms with matched solid line styles and aligned importance histograms",
                        "layout": (
                            "three vertically stacked panels (detector 1 waveform, detector 2 waveform, shared importance histogram), with a common time axis"
                            if spec.estimator_formulation == "shared"
                            else "two-column grid with each detector waveform directly above its importance histogram"
                        ),
                        "onishi_interval_ns": [-1.5, 2.0],
                        "importance_colour_scale": "shared PowerNorm(gamma=1.6) across both detectors; colourbar ticks in original ps/sample units",
                        "histogram_colour": "#4c78a8",
        "histogram_y_scale": "direct models use a common upper limit based on the maximum bin importance across both detectors",
        "detector_label_placement": "inside the upper-left corner of each waveform panel",
                        "output_formats": ["PNG (240 dpi)", "PDF (vector)"],
                        "importance_histogram": {
                            "bin_width_samples": int(window_samples),
                            "bar_width_ns": float(window_samples * np.median(np.diff(time_ps)) / 1000.0),
                            "bar_height": "sum of mean absolute sample attributions within each non-overlapping window-length bin",
                            "unit": "ps",
                        },
                        "detector_identity": (
                            "shared absolute-importance curve and pooled histogram for both signals"
                            if spec.estimator_formulation == "shared"
                            else "detector-specific absolute-importance curves and histograms"
                        ),
                        "legend": "omitted; detector labels are placed inside waveform panels",
                    },
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


def explain_tshap_local(
    results=None, *, model_dir=None, models=None, ml_window=None, cache_root=None, artifact_root=None, output_root=None, events_per_group=2,
    extreme_margin=0.05, n_baselines=8, window_samples=20, stride_samples=10,
    predict_batch_size=64, seed=42, overwrite=False, logger=None,
    imputer_components=32, imputer_fit_events=2000,
    imputer_tuning_events=300, imputer_slope_points=5,
):
    """Explain a small, reproducible set of good/bad and typical blind events.

    Categories use the sign of the predicted correction and signed improvement:
    abs(target) - abs(target - prediction). Good/bad cases are selected near
    improvement quantiles while excluding the outer extreme_margin fraction.
    """
    for name, value in (
        ("events_per_group", events_per_group),
        ("n_baselines", n_baselines),
        ("window_samples", window_samples),
        ("stride_samples", stride_samples),
        ("predict_batch_size", predict_batch_size),
    ):
        _validate_positive_int(name, value)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("TSHAP seed must be an integer")
    if not np.isfinite(extreme_margin) or not 0 < extreme_margin < 0.25:
        raise ValueError("extreme_margin must be greater than 0 and less than 0.25")

    ml_window = _effective_tshap_window(ml_window, model_dir=model_dir)
    tshap_version = version("tshap")
    runs, source_path = _tshap_run_inputs(results, model_dir)
    if not runs:
        raise ValueError(f"No complete saved-model runs found in {results or model_dir}")
    if output_root is None:
        output_root = (Path.cwd() / "xai").resolve()
        if output_root.is_relative_to(source_path):
            output_root = source_path.parent / "xai"
    else:
        output_root = Path(output_root).expanduser().resolve()
        if output_root == source_path or output_root.is_relative_to(source_path):
            raise ValueError("TSHAP output-root must be outside the supplied results/model directory")
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
            config["_tshap_run_manifest"] = manifest
            _apply_tshap_cache(config, cache_root=cache_root, artifact_root=artifact_root)
            if not _matches_tshap_window(config, ml_window):
                if model_dir is not None:
                    raise ValueError(
                        f"Selected model uses ML window {config.get('window_name')!r}, "
                        f"which does not match --ml-window {ml_window!r}"
                    )
                if logger:
                    logger.info(
                        "TSHAP local skipped | unselected window=%s | requested=%s | model=%s",
                        config.get("window_name"), ml_window, manifest["model"],
                    )
                continue
            spec = get_model(manifest["model"])
            run_store = RunStore(
                directory, artifact_root=config["preprocessing"]["artifact_root"]
            )
            # An explicitly supplied local model directory is authoritative.
            # Otherwise resolve the model product from the run's artifact reference.
            local_model = Path(directory) / "model"
            fitted_model_dir = (
                local_model if model_dir is not None and saved_model_complete(spec, local_model)
                else run_store.model_dir
            )
            if not saved_model_complete(spec, fitted_model_dir):
                raise FileNotFoundError(
                    f"Saved model or frozen feature transform missing: {fitted_model_dir}"
                )
            settings = dict(
                events_per_group=events_per_group,
                extreme_margin=float(extreme_margin),
                n_baselines=n_baselines,
                window_samples=window_samples,
                stride_samples=stride_samples,
                predict_batch_size=predict_batch_size,
                seed=int(seed),
                imputer_components=imputer_components,
                imputer_fit_events=imputer_fit_events,
                imputer_tuning_events=imputer_tuning_events,
                imputer_slope_points=imputer_slope_points,
            )
            identity = {
                "method": "conditional_hermite_ppca_local_case_study",
                "method_version": 2,
                "upstream_version": tshap_version,
                "final_fit": manifest["stage_fingerprints"]["final_fit"],
                "control": manifest["dataset_populations"]["control"]["analysis_protocol_identity"],
                "blind": manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
                "model": spec.name,
                "mode": config["mode"],
                "seed": int(seed),
                "settings": settings,
            }
            fingerprint = canonical_hash(identity)
            destination = _local_explanation_destination(output_root, directory)
            if _complete_local(destination, fingerprint, len(LOCAL_GROUP_LABELS) * events_per_group) and not overwrite:
                blind = lookup.find("blind", config, manifest)
                family = mode_family(config["mode"])
                waveform_transform = getattr(blind, f"{family}_transform")
                if waveform_transform is None:
                    raise ValueError("Missing frozen waveform normalization for mV plotting")
                led_threshold_mv = float(blind.manifest["fixed_led_threshold_mV"])
                led_times = getattr(blind, f"{family}_led_time_ps")
                if led_times is None:
                    raise ValueError("Missing frozen LED crossing times for mV plotting")
                with np.load(destination / "local_attributions.npz") as saved:
                    cached_indices = np.asarray(saved["dataset_index"], dtype=np.int64)
                selected_led_times_ps = np.asarray(led_times[cached_indices], dtype=np.float64)
                _refresh_local_presentation(
                    destination,
                    waveform_transform=waveform_transform,
                    led_threshold_mv=led_threshold_mv,
                    baseline_window_ns=config["preprocessing"]["selection"]["baseline_window_ns"],
                    dataset=blind,
                    mode=config["mode"],
                    seed=int(seed),
                    artifact_root=Path(config["preprocessing"]["artifact_root"]),
                )
                if logger:
                    logger.info("TSHAP local | cache hit | %s", destination)
                outputs.append(destination)
                continue

            control = lookup.find("control", config, manifest)
            blind = lookup.find("blind", config, manifest)
            time_ps = _check_prepared_grids(control, blind, config["mode"])
            if window_samples > time_ps.size:
                raise ValueError("TSHAP window longer than input waveform")
            targets = np.asarray(model_target(blind, config["mode"]), dtype=np.float64)
            if targets.shape != (blind.n_events,) or not np.all(np.isfinite(targets)):
                raise ValueError("Blind targets must be finite and match the event population")

            imputer, imputer_metadata = _cached_hermite_ppca_imputer(
                control, config["mode"], config,
                window_samples=window_samples, stride_samples=stride_samples,
                components=imputer_components, fit_events=imputer_fit_events,
                tuning_events=imputer_tuning_events, slope_points=imputer_slope_points,
                batch_size=predict_batch_size, seed=int(seed),
            )

            parent = destination.parent
            parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix=".tshap-local-", dir=parent))
            try:
                with diagnostic_random_state():
                    fitted = load_fitted_model(spec, fitted_model_dir, run["best"]["parameters"], config)
                    try:
                        # Predictions over the full blind population are needed only
                        # to select cases; the expensive TSHAP stage explains 10 by default.
                        all_indices = np.arange(blind.n_events, dtype=np.int64)
                        # Do not use precomputed feature matrices for case selection.
                        # Keep the same waveform -> fitted-transform -> model path as occlusion.
                        predictions = predict_indices(
                            spec, fitted, blind, config["mode"], all_indices,
                            chunk_size=predict_batch_size,
                        )
                        selected, improvement, median_prediction = _select_local_events(
                            predictions, targets, events_per_group=events_per_group,
                            extreme_margin=extreme_margin,
                        )
                        dataset_indices = np.asarray(
                            [item["index"] for item in selected], dtype=np.int64
                        )
                        groups = [item["group"] for item in selected]
                        event_ids = np.asarray(blind.event_index[dataset_indices], dtype=np.int64)
                        waves = waveform_view(
                            blind, config["mode"], dataset_indices
                        ).materialize()
                        family = mode_family(config["mode"])
                        waveform_transform = getattr(blind, f"{family}_transform")
                        if waveform_transform is None:
                            raise ValueError("Missing frozen waveform normalization for mV plotting")
                        raw_waves_mv = np.asarray(waveform_transform.inverse(waves), dtype=np.float64)
                        dataset_mean_difference_mv, dataset_mean_sample_indices = (
                            _dataset_mean_difference_mv(
                                blind, config["mode"], waveform_transform, int(seed),
                                max_events=1000,
                            )
                        )
                        led_threshold_mv = float(blind.manifest["fixed_led_threshold_mV"])
                        led_times = getattr(blind, f"{family}_led_time_ps")
                        if led_times is None:
                            raise ValueError("Missing frozen LED crossing times for mV plotting")
                        selected_led_times_ps = np.asarray(led_times[dataset_indices], dtype=np.float64)
                        baseline_window_ns = config["preprocessing"]["selection"]["baseline_window_ns"]
                        led_baselines_mv = _native_led_baselines(
                            config["preprocessing"]["artifact_root"], blind, config["mode"], dataset_indices, baseline_window_ns
                        )
                        if not np.isfinite(led_threshold_mv) or not np.all(np.isfinite(selected_led_times_ps)):
                            raise ValueError("LED threshold and crossing times must be finite")
                        selected_predictions = predictions[dataset_indices]
                        selected_targets = targets[dataset_indices]
                        selected_improvements = improvement[dataset_indices]

                        predict = lambda x: _predict_bounded(
                            spec, fitted, x, predict_batch_size
                        )
                        replay = predict(waves)
                        prediction_difference = replay - selected_predictions
                        if not np.allclose(
                            selected_predictions, replay,
                            atol=LOCAL_PREDICTION_REPLAY_ATOL_PS,
                            rtol=LOCAL_PREDICTION_REPLAY_RTOL
                        ):
                            mismatch = {
                                "model": spec.name,
                                "run": str(directory),
                                "max_absolute_difference_ps": float(
                                    np.max(np.abs(prediction_difference))
                                ),
                                "atol_ps": LOCAL_PREDICTION_REPLAY_ATOL_PS,
                                "rtol": LOCAL_PREDICTION_REPLAY_RTOL,
                                "events": [
                                    {
                                        "event_id": int(event_ids[i]),
                                        "saved_model_prediction_ps": float(selected_predictions[i]),
                                        "tshap_replay_prediction_ps": float(replay[i]),
                                        "difference_ps": float(prediction_difference[i]),
                                    }
                                    for i in range(len(event_ids))
                                ],
                            }
                            print(
                                "TSHAP local prediction mismatch before failure:\\n"
                                + json.dumps(mismatch, indent=2),
                                flush=True,
                            )
                            raise ValueError(
                                "TSHAP local adapter does not reproduce saved-model predictions "
                                f"(maximum absolute difference: "
                                f"{mismatch['max_absolute_difference_ps']:.9g} ps)"
                            )

                        explainer = _new_explainer(
                            window_samples, stride_samples, roi=False,
                            predict_batch_size=predict_batch_size, imputer=imputer,
                        )
                        local_maps = np.empty_like(waves, dtype=np.float64)
                        for i, pair in enumerate(waves):
                            window_map, _ = explainer.explain(
                                pair[None, ...], pair[None, ...], predict
                            )
                            local_maps[i] = window_map[0]
                            if logger:
                                logger.info(
                                    "TSHAP local | %s | %d/%d | %s | event=%s",
                                    spec.name, i + 1, len(waves), groups[i], event_ids[i],
                                )
                    finally:
                        del fitted
                        release_training_memory()

                if not np.all(np.isfinite(local_maps)):
                    raise ValueError("Non-finite local TSHAP attribution values")

                plot_names = []
                with (temporary / "events.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    writer = csv.writer(stream)
                    writer.writerow([
                        "group", "event_id", "dataset_index", "target_correction_ps",
                        "predicted_correction_ps", "residual_ps", "initial_abs_error_ps",
                        "final_abs_error_ps", "signed_improvement_ps",
                        "median_predicted_correction_ps",
                    ])
                    for i, (group, event_id, dataset_index) in enumerate(
                        zip(groups, event_ids, dataset_indices)
                    ):
                        target_ps = float(selected_targets[i])
                        prediction_ps = float(selected_predictions[i])
                        residual_ps = target_ps - prediction_ps
                        improvement_ps = float(selected_improvements[i])
                        plot_name = f"{group}__event_{int(event_id)}.png"
                        _plot_local_event(
                            temporary / plot_name,
                            time_ps=time_ps,
                            waveform_mv=raw_waves_mv[i],
                            dataset_mean_difference_mv=dataset_mean_difference_mv,
                            led_baseline_mv=led_baselines_mv[i],
                            attribution=local_maps[i],
                            group=group,
                            event_id=int(event_id),
                            target_ps=target_ps,
                            prediction_ps=prediction_ps,
                            improvement_ps=improvement_ps,
                            led_threshold_mv=led_threshold_mv,
                        )
                        plot_names.append(plot_name)
                        writer.writerow([
                            group, int(event_id), int(dataset_index), target_ps,
                            prediction_ps, residual_ps, abs(target_ps), abs(residual_ps),
                            improvement_ps, median_prediction,
                        ])

                np.savez_compressed(
                    temporary / "local_attributions.npz",
                    event_id=event_ids,
                    dataset_index=dataset_indices,
                    group=np.asarray(groups, dtype="U"),
                    time_ps=time_ps,
                    waveform_normalized=waves,
                    waveform_mv=raw_waves_mv,
                    led_baseline_mv=led_baselines_mv,
                    dataset_mean_difference_mv=dataset_mean_difference_mv,
                    dataset_mean_sample_indices=dataset_mean_sample_indices,
                    led_threshold_mv=np.asarray(led_threshold_mv, dtype=np.float64),
                    led_time_ps=selected_led_times_ps,
                    attribution_ps_per_sample=local_maps,
                    target_correction_ps=selected_targets,
                    predicted_correction_ps=selected_predictions,
                    residual_ps=selected_targets - selected_predictions,
                    signed_improvement_ps=selected_improvements,
                    control_event_id=np.empty(0, np.int64),
                    median_predicted_correction_ps=np.asarray(median_prediction, dtype=np.float64),
                )
                group_counts = {
                    group: sum(item == group for item in groups)
                    for group in LOCAL_GROUP_LABELS
                }
                atomic_json(temporary / "summary.json", {
                    **identity,
                    "identity_sha256": fingerprint,
                    "plot_version": LOCAL_PLOT_VERSION,
                    "n_explained_events": len(event_ids),
                    "n_events_per_group": events_per_group,
                    "n_control_baselines": 0,
                    "imputer_cache_identity": imputer_metadata["cache_identity"],
                    "imputer_cache_file": imputer_metadata["cache_file"],
                    "waveform_samples": int(time_ps.size),
                    "median_predicted_correction_ps": median_prediction,
                    "group_counts": group_counts,
                    "selection": {
                        "correction_sign": "sign of model-predicted correction",
                        "signed_improvement_definition": "abs(target) - abs(target - prediction)",
                        "good_quantiles": [
                            float(1.0 - 2.0 * extreme_margin),
                            float(1.0 - extreme_margin),
                        ],
                        "bad_quantiles": [
                            float(extreme_margin), float(2.0 * extreme_margin),
                        ],
                        "extreme_margin": float(extreme_margin),
                        "typical": "closest predicted corrections to the blind-population median, excluding other selected events",
                    },
                    "plots": plot_names,
                    "attribution_units": "ps/sample (upstream TSHAP window scores redistributed over samples)",
                    "input_units": "dimensionless min-max-normalized waveform amplitude",
                    "signal_units": "mV (inverse of frozen prepared waveform normalization)",
                    "led_threshold_mV": led_threshold_mv,
                    "led_baseline_source": "original blind-native waveform using the timing pipeline baseline window",
                    "led_reference_line_extent": "waveform start to time zero",
                    "difference_centering": "per-sample mean across up to 1000 seeded blind events",
                    "difference_centering_sample_size": int(dataset_mean_sample_indices.size),
                    "prediction_equivalence_max_error_ps": float(
                        np.max(np.abs(selected_predictions - replay))
                    ),
                    "reference": "https://github.com/mlgig/tshap",
                })
                _publish_directory(temporary, destination, overwrite=True)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            outputs.append(destination)
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
            if logger:
                logger.exception("TSHAP local failed | %s", directory)

    if failures:
        raise RuntimeError(
            "Some local TSHAP case-study explanations failed; successful results retained:\n"
            + "\n".join(failures)
        )
    return outputs
