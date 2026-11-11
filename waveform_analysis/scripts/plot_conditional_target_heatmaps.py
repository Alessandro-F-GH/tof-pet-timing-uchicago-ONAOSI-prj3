"""Conditional heatmaps of SAVED model predictions versus paired waveform representations.

Compare original, per-event baseline-subtracted, and per-detector peak-normalized
waveforms on the same development-training events. Each representation has its
OWN train-only per-sample mean and scale; paired difference is z2-z1 (not
standardized again). Optional legacy first-difference views are supported for
the original input. No model inference, transformations, or refitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waveform_analysis.engine.sample_normalization import (
    fit_first_difference_stats, fit_time_sample_stats)
from waveform_analysis.scripts.evaluate_conditional_knn import _load_inputs, _logger

CHANNEL_NAMES = ("detector_1", "detector_2", "detector_2_minus_1")
PERCENTILE_BANDS = ((2., 12.), (45., 55.), (88., 98.))
REPRESENTATIONS = ("original", "baseline_subtracted", "amplitude_normalized")


def effective_representations(mode, requested):
    """Timing waveforms saturate: skip amplitude normalization in timing mode."""
    selected = tuple(dict.fromkeys(requested))
    if mode == "timing_to_timing":
        selected = tuple(r for r in selected if r != "amplitude_normalized")
    if not selected:
        raise ValueError("No applicable heatmap representations for this mode")
    return selected


def mask_excluded_spans(mask, time_ps):
    """Contiguous excluded spans in ns on the full LED-relative input grid."""
    kept = np.asarray(mask, dtype=bool)
    times = np.asarray(time_ps, dtype=np.float64)
    if kept.ndim != 1 or times.ndim != 1 or kept.size != times.size:
        raise ValueError("Sample mask and time grid must have equal 1D shape")
    edges = sample_centers_to_edges(times / 1000.)
    excluded = ~kept
    changes = np.diff(np.r_[False, excluded, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(float(edges[start]), float(edges[stop]))
            for start, stop in zip(starts, stops)]


def onishi_window_bounds(model_dir, *, start_ns=None, end_ns=None,
                         disabled=False):
    """Read Onishi context window from the SAME batch/mode.

    No guessed default bounds: a missing context disables the overlay unless
    the user supplies both limits explicitly.
    """
    if disabled:
        return None
    if (start_ns is None) != (end_ns is None):
        raise ValueError("Specify both --onishi-start-ns and --onishi-end-ns")
    if start_ns is not None:
        left, right = float(start_ns), float(end_ns)
    else:
        context_path = Path(model_dir).expanduser().resolve().parent.parent / "onishi" / "context.json"
        if not context_path.is_file():
            return None
        context = json.loads(context_path.read_text(encoding="utf-8-sig"))
        window = context.get("window_ns") or {}
        left, right = float(window["start"]), float(window["end"])
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        raise ValueError("Invalid Onishi window limits")
    return (left, right)


def baseline_indices(time_ps, start_ns=-2., end_ns=-1.):
    """Indices in the prepared LED-relative grid, covering [start,end] ns."""
    time_ps = np.asarray(time_ps, dtype=np.float64)
    start_ps, end_ps = 1000. * float(start_ns), 1000. * float(end_ns)
    if (time_ps.ndim != 1 or time_ps.size < 2 or
            not np.all(np.isfinite(time_ps)) or
            not np.all(np.diff(time_ps) > 0) or
            not np.isfinite(start_ps) or not np.isfinite(end_ps) or
            start_ps >= end_ps):
        raise ValueError("Invalid time grid or baseline interval")
    if time_ps[0] > start_ps + 1e-6 or time_ps[-1] < end_ps - 1e-6:
        raise ValueError(
            f"Prepared LED-relative window [{time_ps[0]/1000:.3f},"
            f"{time_ps[-1]/1000:.3f}] ns does not cover baseline interval "
            f"[{start_ns:g}, {end_ns:g}] ns")
    indices = np.flatnonzero((time_ps >= start_ps - 1e-6) &
                            (time_ps <= end_ps + 1e-6))
    if len(indices) < 2:
        raise ValueError("Baseline interval contains fewer than two prepared samples")
    return indices


def sample_centers_to_edges(centers):
    """Construct plotting-cell edges from monotonic physical time coordinates.

    Midpoints preserve real time gaps when --retained-only selects a
    non-uniform subset. A single sample uses a symmetric nominal half-width
    solely for rendering.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if (centers.ndim != 1 or centers.size < 2 or
            not np.all(np.isfinite(centers)) or
            not np.all(np.diff(centers) > 0)):
        raise ValueError("Need at least two strictly increasing finite time centers")
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2.
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2.
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2.
    return edges


class WaveformRepresentation:
    """Lazy transformed waveform view; no full-data copy or ML preprocessing.

    Prepared ML samples are affinely mapped to [0,1] per detector (and clipped).
    Baseline subtraction before z-scoring is equivalent in physical mV up to
    that pre-existing clipping. Peak normalization is invariant to this affine
    scaling because the event baseline is subtracted first.
    """

    def __init__(self, waves, representation, baseline_positions, *, peak_floor=1e-8):
        if representation not in REPRESENTATIONS or representation == "original":
            raise ValueError("WaveformRepresentation requires a transformed representation")
        self.waves = waves
        self.shape = waves.shape
        self.representation = representation
        self.baseline_positions = np.asarray(baseline_positions, dtype=np.int64)
        self.peak_floor = float(peak_floor)
        if (self.baseline_positions.size < 2 or not np.isfinite(self.peak_floor)
                or self.peak_floor <= 0):
            raise ValueError("Invalid baseline indices or peak floor")

    def __getitem__(self, indices):
        x = np.asarray(self.waves[indices], dtype=np.float64)
        one_event = x.ndim == 2
        if one_event:
            x = x[None]
        baseline = np.mean(x[:, :, self.baseline_positions], axis=2, keepdims=True)
        values = x - baseline
        if self.representation == "amplitude_normalized":
            peak = np.max(values, axis=2, keepdims=True)
            invalid = (~np.isfinite(peak)) | (peak <= self.peak_floor)
            if np.any(invalid):
                raise ValueError(
                    f"Cannot peak-normalize {int(np.count_nonzero(invalid))} "
                    "event-detector waveforms with nonpositive/near-zero peak. "
                    "Event selection is not changed silently.")
            values = values / peak
        values = values.astype(np.float32)
        return values[0] if one_event else values



class EventSubsetView:
    """Lazy reference waveform subset indexed in matched-event order.

    The original prepared waveform array is never copied as a whole; only
    requested rows/chunks are materialized for normalization/heatmaps.
    """

    def __init__(self, waves, reference_rows):
        self.waves = waves
        self.reference_rows = np.asarray(reference_rows, dtype=np.int64)
        self.shape = (len(self.reference_rows),) + waves.shape[1:]
        # The grouped physical-waveform path validates waves.ndim, as it
        # does for a normal NumPy array. Keep the lazy view array-compatible.
        self.ndim = len(self.shape)

    def __getitem__(self, indices):
        return self.waves[self.reference_rows[indices]]


def match_event_populations(reference_ids, comparison_ids, reference_targets,
                            comparison_targets, *, target_tolerance_ps=1e-3):
    """Return reference-order intersection rows, aligned comparison rows and IDs.

    Match event IDs, never array positions. Reject ambiguous IDs and target
    mismatches rather than silently mixing incompatible event calibrations.
    """
    a = np.asarray(reference_ids, dtype=np.int64)
    b = np.asarray(comparison_ids, dtype=np.int64)
    if (a.ndim != 1 or b.ndim != 1 or
            np.unique(a).size != a.size or np.unique(b).size != b.size):
        raise ValueError("Ambiguous or duplicate development event IDs")
    if a.size < 2 or b.size < 2:
        raise ValueError("Both development populations must contain >=2 events")
    common, a_rows, b_rows = np.intersect1d(
        a, b, assume_unique=True, return_indices=True)
    if common.size < 2:
        raise ValueError("Fewer than two common development events")
    order = np.argsort(a_rows, kind="stable")
    reference_rows = a_rows[order]
    comparison_rows = b_rows[order]
    event_ids = a[reference_rows]

    targets_a = np.asarray(reference_targets, dtype=np.float64)
    targets_b = np.asarray(comparison_targets, dtype=np.float64)
    if targets_a.shape != a.shape or targets_b.shape != b.shape:
        raise ValueError("Prepared target array does not match event IDs")
    left = targets_a[reference_rows]
    right = targets_b[comparison_rows]
    if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
        raise ValueError("Nonfinite calibrated targets in common events")
    delta = np.abs(left - right)
    if np.max(delta) > target_tolerance_ps:
        raise ValueError(
            "Calibrated targets disagree on matched event IDs: "
            f"max absolute difference={np.max(delta):.6g} ps "
            f"(tolerance={target_tolerance_ps:g} ps)")
    info = {
        "reference_events": int(a.size),
        "comparison_events": int(b.size),
        "common_events": int(common.size),
        "reference_only_events": int(a.size - common.size),
        "comparison_only_events": int(b.size - common.size),
        "reference_overlap_fraction": float(common.size / a.size),
        "comparison_overlap_fraction": float(common.size / b.size),
        "max_target_difference_ps": float(np.max(delta)),
        "event_order": "reference prepared development order",
    }
    return reference_rows, comparison_rows, event_ids, info


def load_prediction_difference(args, reference_predictions, reference_provenance,
                               reference_waves, log):
    """Compare saved corrections on common events; visualize reference window.

    Same-window runs retain waveform/grid identity validation. Different-window
    runs can have different sample lengths and slightly different populations;
    their comparison is allowed only when mode, control calibration and paired
    targets agree. Never crop/resize wide waveforms to the comparison window.
    """
    from argparse import Namespace
    from waveform_analysis.data.results import read_run, resolve_run_config
    from waveform_analysis.data.prepared_cache import PreparedCacheLookup
    from waveform_analysis.data.view import mode_family

    second_args = Namespace(**vars(args))
    second_args.model_dir = args.comparison_model_dir
    other_waves, other_predictions, _, second = _load_inputs(second_args, log)
    if reference_provenance["mode"] != second["mode"]:
        raise ValueError("Different modes: cannot compare prediction outputs")

    def dataset_for(model_dir):
        path = model_dir.expanduser().resolve()
        run = read_run(path)
        config = resolve_run_config(path)
        if args.cache_root is not None:
            root = args.cache_root.expanduser().resolve()
            config["preprocessing"]["cache_dir"] = str(root / "preprocessing")
            config["preprocessing"]["artifact_root"] = str(root / "artifact_store")
            config["preprocessing"]["named_cache_root"] = str(root)
        elif args.artifact_root is not None:
            config["preprocessing"]["artifact_root"] = str(
                args.artifact_root.expanduser().resolve())
            config["preprocessing"].pop("named_cache_root", None)
        return (PreparedCacheLookup().find("development", config, run["manifest"]),
                config)

    ref_dataset, ref_config = dataset_for(args.model_dir)
    other_dataset, other_config = dataset_for(args.comparison_model_dir)
    if ref_config["mode"] != other_config["mode"]:
        raise ValueError("Different modes: cannot compare prediction outputs")
    if ref_config["development"].get("channels") != other_config["development"].get("channels"):
        raise ValueError("Different detector channel definitions")
    ref_manifest, cmp_manifest = ref_dataset.manifest, other_dataset.manifest
    for key in ("mode", "control_mode_fingerprint", "target_definition"):
        if ref_manifest.get(key) != cmp_manifest.get(key):
            raise ValueError(f"Incompatible prepared metadata: {key}")
    for key in ("true_tof_ps", "fixed_led_threshold_mV", "led_control_mean_ps"):
        a = ref_manifest.get(key)
        b = cmp_manifest.get(key)
        if a is None or b is None or not np.isclose(
                float(a), float(b), rtol=0, atol=1e-3):
            raise ValueError(f"Incompatible calibration metadata: {key}")

    family = mode_family(ref_config["mode"])
    ref_targets = getattr(ref_dataset, f"{family}_target_ps")
    other_targets = getattr(other_dataset, f"{family}_target_ps")
    if ref_targets is None or other_targets is None:
        raise ValueError("Prepared calibrated targets are required for comparison")
    ref_ids = np.asarray(ref_dataset.event_index, dtype=np.int64)
    other_ids = np.asarray(other_dataset.event_index, dtype=np.int64)
    if len(ref_ids) != len(reference_predictions) or len(other_ids) != len(other_predictions):
        raise ValueError("Saved predictions and prepared event populations disagree")
    ref_rows, other_rows, common_ids, info = match_event_populations(
        ref_ids, other_ids, ref_targets, other_targets)
    same_window = (ref_config["window_name"] == other_config["window_name"] and
                   ref_config["window_ns"] == other_config["window_ns"])
    if same_window:
        if reference_waves.shape[1:] != other_waves.shape[1:]:
            raise ValueError("Same-window prepared waveform shapes differ")
        # Preserve strict waveform identity checks when comparing runs with the
        # same window. For different windows the cropped input is expected to
        # differ, so this check is not meaningful.
        ref_time = getattr(ref_dataset, f"{family}_time_ps")
        other_time = getattr(other_dataset, f"{family}_time_ps")
        if not np.array_equal(np.asarray(ref_time), np.asarray(other_time)):
            raise ValueError("Same-window waveform time grids differ")
        sample_rows = np.unique(np.linspace(0, len(ref_rows) - 1, 64, dtype=np.int64))
        sample_times = np.unique(np.linspace(
            0, reference_waves.shape[-1] - 1, 128, dtype=np.int64))
        for row in sample_rows:
            left = np.asarray(reference_waves[int(ref_rows[row])], dtype=np.float32)
            right = np.asarray(other_waves[int(other_rows[row])], dtype=np.float32)
            if not np.allclose(left[:, sample_times], right[:, sample_times],
                               rtol=1e-5, atol=1e-6):
                raise ValueError("Prepared waveforms differ for matched same-window events")
    info["reference_window"] = ref_config["window_name"]
    info["comparison_window"] = other_config["window_name"]
    info["same_window"] = bool(same_window)
    info["waveform_display"] = "reference window only"
    info["comparison_sign"] = "reference saved prediction minus comparison saved prediction"
    log.info(
        "DIFFERENCE | %s/%s minus %s/%s | common=%d | reference-only=%d | "
        "comparison-only=%d | target max delta=%.6g ps | reference waveform retained",
        ref_config["window_name"], reference_provenance["model"],
        other_config["window_name"], second["model"],
        info["common_events"], info["reference_only_events"],
        info["comparison_only_events"], info["max_target_difference_ps"])
    return (reference_predictions[ref_rows] - other_predictions[other_rows],
            second, other_predictions[other_rows], ref_rows, common_ids, info)


def prediction_comparison_metrics(reference, comparison, *, reference_name="reference",
                                  comparison_name="comparison"):
    """Paired saved-output agreement; never claim these are errors to physical truth."""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(comparison, dtype=np.float64)
    if a.ndim != 1 or b.shape != a.shape or a.size < 2:
        raise ValueError("Need matching 1D prediction arrays with >=2 events")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite model predictions")
    delta = a - b
    abs_delta = np.abs(delta)
    std_a = float(np.std(a))
    std_b = float(np.std(b))
    corr = (float(np.corrcoef(a, b)[0, 1])
            if std_a > 0 and std_b > 0 else None)
    thresholds = np.percentile(a, [0, 2, 10, 25, 50, 75, 90, 98, 100])
    bins = []
    for i in range(len(thresholds)-1):
        lower, upper = float(thresholds[i]), float(thresholds[i+1])
        subset = (a >= lower) & ((a <= upper) if i == len(thresholds)-2
                                 else (a < upper))
        if not np.any(subset):
            continue
        error = delta[subset]
        bins.append({
            "reference_quantiles": [int(v) for v in (0, 2, 10, 25, 50, 75, 90, 98, 100)[i:i+2]],
            "n": int(subset.sum()), "bias_ps": float(np.mean(error)),
            "rmse_ps": float(np.sqrt(np.mean(error**2))),
            "mae_ps": float(np.mean(np.abs(error))),
        })
    return {
        "n_events": int(len(a)),
        "reference": reference_name, "comparison": comparison_name,
        "difference_definition": "reference prediction minus comparison prediction",
        "reference_mean_ps": float(np.mean(a)),
        "reference_std_ps": std_a,
        "comparison_mean_ps": float(np.mean(b)),
        "comparison_std_ps": std_b,
        "pearson_correlation": corr,
        "bias_ps": float(np.mean(delta)),
        "difference_std_ps": float(np.std(delta)),
        "rmse_difference_ps": float(np.sqrt(np.mean(delta ** 2))),
        "mae_difference_ps": float(np.mean(abs_delta)),
        "median_abs_difference_ps": float(np.median(abs_delta)),
        "p90_abs_difference_ps": float(np.percentile(abs_delta, 90)),
        "p95_abs_difference_ps": float(np.percentile(abs_delta, 95)),
        "p99_abs_difference_ps": float(np.percentile(abs_delta, 99)),
        "max_abs_difference_ps": float(abs_delta.max()),
        "by_reference_prediction_quantile": bins,
    }


def log_prediction_comparison(log, label, result):
    log.info("MODEL AGREEMENT | %s | N=%d | Pearson r=%s | "
             "ref std=%.4f ps comparison std=%.4f ps",
             label, result["n_events"], result["pearson_correlation"],
             result["reference_std_ps"], result["comparison_std_ps"])
    log.info("MODEL DIFFERENCE | %s | reference-comparison | bias=%.4f ps "
             "std=%.4f ps RMSE=%.4f ps MAE=%.4f ps | "
             "median|delta|=%.4f ps p90=%.4f p95=%.4f p99=%.4f max=%.4f ps",
             label, result["bias_ps"], result["difference_std_ps"],
             result["rmse_difference_ps"], result["mae_difference_ps"],
             result["median_abs_difference_ps"], result["p90_abs_difference_ps"],
             result["p95_abs_difference_ps"], result["p99_abs_difference_ps"],
             result["max_abs_difference_ps"])
    for band in result["by_reference_prediction_quantile"]:
        lo, hi = band["reference_quantiles"]
        log.info("MODEL DIFFERENCE BY ROCKET QUANTILE | %s | P%d-P%d n=%d "
                 "bias=%.4f ps RMSE=%.4f ps MAE=%.4f ps",
                 label, lo, hi, band["n"], band["bias_ps"],
                 band["rmse_ps"], band["mae_ps"])


def grouped_physical_waveforms(waves, predictions, training, positions, minimum, maximum,
                               *, bands=PERCENTILE_BANDS, chunk_size=128):
    """Mean original ML input waveform in mV by SAVED-output percentile band.

    Inverse the fixed prepared-input [0,1] affine transformation, not a
    representation-specific z-score. Prepared clipping remains in effect.
    Grouping and means use exactly the heatmap training subset.
    Returns group means [group, D1/D2/(D2-D1), plotted_time].
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    training = np.asarray(training, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64)
    minimum = np.asarray(minimum, dtype=np.float64).reshape(2, 1)
    maximum = np.asarray(maximum, dtype=np.float64).reshape(2, 1)
    if (waves.ndim != 3 or waves.shape[1] != 2
            or predictions.shape != (waves.shape[0],)
            or training.ndim != 1 or training.size < 2
            or positions.ndim != 1 or not positions.size
            or chunk_size < 1 or not np.all(np.isfinite(minimum))
            or not np.all(np.isfinite(maximum))
            or not np.all(maximum > minimum)):
        raise ValueError("Invalid waveform input or physical affine transform")
    train_predictions = predictions[training]
    group_means, group_info = [], []
    for lower_pct, upper_pct in bands:
        lower, upper = np.percentile(train_predictions, [lower_pct, upper_pct])
        selected = training[
            (train_predictions >= lower) & (train_predictions < upper)]
        if len(selected) == 0:
            raise ValueError(
                f"No events in prediction band P{lower_pct:g}-P{upper_pct:g}")
        total = np.zeros((2, len(positions)), dtype=np.float64)
        for start in range(0, len(selected), chunk_size):
            ids = selected[start:start + chunk_size]
            batch = np.asarray(waves[ids], dtype=np.float64)[:, :, positions]
            restored_mv = batch * (maximum - minimum)[None, :, :] + minimum[None, :, :]
            if not np.all(np.isfinite(restored_mv)):
                raise ValueError("Nonfinite physical waveform samples in selected group")
            total += restored_mv.sum(axis=0)
        mean = total / len(selected)
        group_means.append(np.stack([mean[0], mean[1], mean[1] - mean[0]]))
        group_info.append({
            "percentile_lower": float(lower_pct),
            "percentile_upper": float(upper_pct),
            "prediction_lower_ps": float(lower),
            "prediction_upper_ps": float(upper),
            "mean_prediction_ps": float(np.mean(predictions[selected])),
            "n_events": int(len(selected)),
        })
    return np.stack(group_means), group_info


def grouped_amplitude_normalized_waveforms(waves, predictions, training, positions,
                                           baseline_positions, group_info, *,
                                           chunk_size=128, peak_floor=1e-8):
    """Group mean (x - event baseline) / event peak, separately per detector.

    Use the EXACT view supplying the amplitude-normalized heatmap, before
    train-only timewise z-scoring. Output is dimensionless, not physical mV.
    Each group has D1, D2 and their difference D2-D1.
    """
    training = np.asarray(training, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    view = WaveformRepresentation(waves, "amplitude_normalized",
                                  baseline_positions, peak_floor=peak_floor)
    group_means = []
    for info in group_info:
        lo, hi = info["prediction_lower_ps"], info["prediction_upper_ps"]
        selected = training[(predictions[training] >= lo) &
                            (predictions[training] < hi)]
        if selected.size != info["n_events"]:
            raise ValueError("Inconsistent saved-prediction group selection")
        total = np.zeros((2, len(positions)), dtype=np.float64)
        for start in range(0, len(selected), chunk_size):
            batch = np.asarray(view[selected[start:start + chunk_size]],
                               dtype=np.float64)[:, :, positions]
            if not np.all(np.isfinite(batch)):
                raise ValueError("Nonfinite amplitude-normalized waveform")
            total += batch.sum(axis=0)
        avg = total / selected.size
        group_means.append(np.stack([avg[0], avg[1], avg[1] - avg[0]]))
    return np.stack(group_means)


def accumulate_heatmaps(waves, labels, rows, positions, mean, scale,
                        derivative_mean, derivative_scale, *,
                        bins=80, z_limit=3., chunk_size=128, signal_only=False):
    """Return target sums/counts [kind, channel, z-bin, time].

    Signal-only mode uses just one kind; the existing two-kind derivative
    behavior remains available for legacy plots and tests.
    Values outside the z range are omitted, not clipped. Difference is z2-z1.
    """
    if bins < 2 or z_limit <= 0 or chunk_size < 1:
        raise ValueError("Invalid binning")
    positions = np.asarray(positions, dtype=np.int64)
    if positions.ndim != 1 or not len(positions):
        raise ValueError("At least one time position required")
    ntime = len(positions)
    flat_size = bins * ntime
    kinds = 1 if signal_only else 2
    sums = np.zeros((kinds, 3, bins, ntime), dtype=np.float64)
    counts = np.zeros((kinds, 3, bins, ntime), dtype=np.int64)
    t_offsets = np.arange(ntime, dtype=np.int64)[None, :]
    edges = np.linspace(-z_limit, z_limit, bins + 1)
    m = np.asarray(mean[:, positions], dtype=np.float64)
    s = np.asarray(scale[:, positions], dtype=np.float64)
    if not signal_only:
        dm = np.asarray(derivative_mean[:, positions], dtype=np.float64)
        ds = np.asarray(derivative_scale[:, positions], dtype=np.float64)
    for left in range(0, len(rows), chunk_size):
        ids = rows[left:left + chunk_size]
        x = np.asarray(waves[ids], dtype=np.float64)
        signal = (x[:, :, positions] - m[None]) / s[None]
        sources = (signal,)
        if not signal_only:
            # First differences in raw prepared input units before z-scoring.
            difference = np.zeros((len(ids), 2, ntime), dtype=np.float64)
            valid_derivative = positions > 0
            if valid_derivative.any():
                p = positions[valid_derivative]
                difference[:, :, valid_derivative] = x[:, :, p] - x[:, :, p - 1]
            derivative = (difference - dm[None]) / ds[None]
            sources = (signal, derivative)
        for kind, source in enumerate(sources):
            for channel in range(3):
                values = (source[:, channel, :] if channel < 2 else
                          source[:, 1, :] - source[:, 0, :])
                which = np.floor((values + z_limit) * bins / (2 * z_limit)).astype(np.int64)
                # Include exact upper edge in the last bin.
                which = np.where(values == z_limit, bins - 1, which)
                valid = (np.isfinite(values) & (values >= -z_limit) &
                         (values <= z_limit) & (which >= 0) & (which < bins))
                flat_indices = which * ntime + t_offsets
                selected = flat_indices[valid]
                count = np.bincount(selected, minlength=flat_size)
                weighted = np.bincount(
                    selected,
                    weights=np.broadcast_to(np.asarray(labels[ids], dtype=np.float64)[:, None],
                                            valid.shape)[valid],
                    minlength=flat_size)
                counts[kind, channel] += count.reshape(bins, ntime)
                sums[kind, channel] += weighted.reshape(bins, ntime)
    return edges, sums, counts


def run(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    log = _logger(root)
    waves, predictions, retained, provenance = _load_inputs(args, log)
    comparison_provenance = None
    reference_predictions = predictions
    comparison_predictions = None
    matching = None
    if args.comparison_model_dir is not None:
        (predictions, comparison_provenance, comparison_predictions,
         reference_rows, common_event_ids, matching) = load_prediction_difference(
            args, reference_predictions, provenance, waves, log)
        # All subsequent statistics and plots use exactly the matched events,
        # but only reference (typically wide) waveform time coordinates.
        reference_predictions = reference_predictions[reference_rows]
        waves = EventSubsetView(waves, reference_rows)
        np.save(root / "matched_event_ids.npy", common_event_ids)
    target_kind = getattr(args, "comparison_target", "difference")
    if target_kind not in ("difference", "reference", "comparison"):
        raise ValueError("Invalid comparison target")
    if target_kind != "difference" and comparison_predictions is None:
        raise ValueError("--comparison-target reference/comparison requires --comparison-model-dir")
    if comparison_predictions is not None:
        if target_kind == "reference":
            predictions = reference_predictions
        elif target_kind == "comparison":
            predictions = comparison_predictions
    is_difference = comparison_predictions is not None and target_kind == "difference"
    label_description = (
        "Mean prediction difference (reference - comparison, ps)" if is_difference else
        ("Mean saved comparison-model prediction (ps)" if target_kind == "comparison"
         else "Mean saved reference-model prediction (ps)"))
    n = len(predictions)
    if args.tuning_queries < 1 or args.holdout_queries < 1:
        raise ValueError("Tuning/holdout must be nonempty")
    if args.tuning_queries + args.holdout_queries >= n:
        raise ValueError("Insufficient training population")
    if args.time_stride < 1 or args.z_bins < 2 or args.z_limit <= 0:
        raise ValueError("Invalid grid settings")
    if args.derivative_end_sample is not None and args.derivative_end_sample <= 0:
        raise ValueError("--derivative-end-sample must be positive")
    rng = np.random.default_rng(args.seed)
    excluded = rng.choice(n, args.tuning_queries + args.holdout_queries, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[excluded] = False
    training = np.flatnonzero(mask)
    comparison_metrics = None
    if comparison_predictions is not None:
        comparison_metrics = {}
        subsets = {"common_development": np.arange(n),
                   "training": training,
                   "tuning": excluded[:args.tuning_queries],
                   "holdout": excluded[args.tuning_queries:]}
        for split_name, indices in subsets.items():
            result = prediction_comparison_metrics(
                reference_predictions[indices], comparison_predictions[indices],
                reference_name=provenance["model"],
                comparison_name=comparison_provenance["model"])
            comparison_metrics[split_name] = result
            log_prediction_comparison(log, split_name, result)
        (root / "prediction_comparison_metrics.json").write_text(
            json.dumps(comparison_metrics, indent=2), encoding="utf-8")
    log.info("TRAIN ONLY | n=%d | tuning=%d holdout=%d seed=%d",
             len(training), args.tuning_queries, args.holdout_queries, args.seed)
    # Prepared ML samples are already clipped and affinely scaled by detector,
    # but they are NOT baseline-subtracted event by event.
    family = "energy" if provenance["mode"] == "energy_to_energy" else "timing"
    time_file = Path(provenance["dataset"]) / f"{family}_time_ps.npy"
    time_ps = np.asarray(np.load(time_file, mmap_mode="r"), dtype=np.float64)
    if time_ps.shape != (waves.shape[-1],):
        raise ValueError("Prepared time grid does not match waveform samples")
    choices = effective_representations(provenance["mode"], args.representations)
    if "amplitude_normalized" in args.representations and "amplitude_normalized" not in choices:
        log.info("TIMING MODE | amplitude_normalized skipped: saturated timing pulses")
    cropped_spans = (mask_excluded_spans(retained, time_ps)
                     if provenance["mode"] == "timing_to_timing" else [])
    onishi_bounds = onishi_window_bounds(
        args.model_dir,
        start_ns=args.onishi_start_ns, end_ns=args.onishi_end_ns,
        disabled=args.disable_onishi_highlight)
    if onishi_bounds is None and not args.disable_onishi_highlight:
        log.warning("ONISHI | sibling onishi/context.json not found; "
                    "use --onishi-start-ns/--onishi-end-ns to highlight the window")
    elif onishi_bounds is not None:
        log.info("ONISHI | highlight signal region [%.3f, %.3f] ns",
                 *onishi_bounds)
    if cropped_spans:
        log.info("TIMING CROP | %d excluded time spans shaded", len(cropped_spans))
    baseline_positions = None
    if any(name != "original" for name in choices):
        baseline_positions = baseline_indices(
            time_ps, args.baseline_start_ns, args.baseline_end_ns)
        log.info("BASELINE | prepared LED-relative [%g,%g] ns | %d samples "
                 "(indices %d..%d), before per-time z-score",
                 args.baseline_start_ns, args.baseline_end_ns,
                 len(baseline_positions), baseline_positions[0],
                 baseline_positions[-1])

    positions = np.arange(0, waves.shape[-1], args.time_stride, dtype=np.int64)
    if args.retained_only:
        positions = positions[retained[positions]]
    if not len(positions):
        raise ValueError("No plotted sample positions remain")

    # Identical events, coordinates, binning, and original saved labels for
    # all representations; fit independent train-only means/stds per view.
    results = {}
    for representation in choices:
        view = (waves if representation == "original" else WaveformRepresentation(
            waves, representation, baseline_positions, peak_floor=args.peak_floor))
        log.info("REPRESENTATION | %s | fitting train-only per-sample statistics",
                 representation)
        means, scales, signal_meta = fit_time_sample_stats(
            view, training, chunk_size=args.chunk_size,
            scale_floor_ratio=args.normalization_scale_floor_ratio)
        derivative = (representation == "original" and args.include_original_derivative)
        if derivative:
            dmeans, dscales, derivative_meta = fit_first_difference_stats(
                view, training, chunk_size=args.chunk_size,
                scale_floor_ratio=args.normalization_scale_floor_ratio)
        else:
            dmeans, dscales, derivative_meta = None, None, None
        edges, sums, counts = accumulate_heatmaps(
            view, predictions, training, positions, means, scales, dmeans, dscales,
            bins=args.z_bins, z_limit=args.z_limit,
            chunk_size=args.chunk_size, signal_only=not derivative)
        mean_target = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=mean_target, where=counts >= args.min_count)
        results[representation] = {
            "edges": edges, "sums": sums, "counts": counts,
            "mean_target": mean_target, "means": means, "scales": scales,
            "normalization": signal_meta, "derivative_normalization": derivative_meta,
            "derivative_normalization_mean": dmeans,
            "derivative_normalization_scale": dscales,
            "feature_names": (["waveform", "first_difference"] if derivative
                              else ["waveform"]),
        }
        log.info("REPRESENTATION | %s | heatmaps accumulated", representation)

    # Share one ps color scale across all views; z axes remain representation-
    # specific because every transformed population gets its OWN z-score.
    baseline = float(np.mean(predictions[training]))
    deviations = [np.abs(result["mean_target"][np.isfinite(result["mean_target"])]
                         - baseline) for result in results.values()]
    all_deviations = np.concatenate([a for a in deviations if a.size]) if any(
        a.size for a in deviations) else np.array([], dtype=float)
    radius = float(np.percentile(all_deviations, 98)) if all_deviations.size else 1.
    radius = max(radius, 1.)
    target_norm = TwoSlopeNorm(vmin=baseline - radius, vcenter=baseline,
                               vmax=baseline + radius)
    # Time coordinates are LED-relative from the prepared waveform grid.
    # Use actual cell edges so retained-only positions remain geometrically
    # correct even when their spacing is not uniform.
    time_ns = time_ps[positions] / 1000.0
    time_edges_ns = sample_centers_to_edges(time_ns)

    # Recover physical mV at the prepared ML time samples. The fixed input
    # transform contains per-detector amplitude bounds; it is NOT the z-score
    # fitted above. Read only the existing cached transform and waveforms.
    transform_file = Path(provenance["dataset"]) / f"{family}_transform.npz"
    if not transform_file.is_file():
        raise FileNotFoundError(f"Prepared physical transform not found: {transform_file}")
    with np.load(transform_file, allow_pickle=False) as transform:
        physical_min = np.asarray(transform["minimum"], dtype=np.float64)
        physical_max = np.asarray(transform["maximum"], dtype=np.float64)
    grouped_mv, prediction_groups = grouped_physical_waveforms(
        waves, predictions, training, positions, physical_min, physical_max,
        chunk_size=args.chunk_size)
    for group in prediction_groups:
        log.info("WAVEFORM GROUP | P%g-P%g | n=%d | mean prediction=%.3f ps",
                 group["percentile_lower"], group["percentile_upper"],
                 group["n_events"], group["mean_prediction_ps"])
    np.savez_compressed(
        root / "grouped_physical_waveforms.npz",
        time_ns=time_ns, positions=positions,
        group_mean_waveform_mv=grouped_mv,
        channel_names=np.array(CHANNEL_NAMES),
        percentile_bands=np.array(PERCENTILE_BANDS, dtype=np.float64),
        group_mean_prediction_ps=np.array(
            [g["mean_prediction_ps"] for g in prediction_groups]),
        group_counts=np.array([g["n_events"] for g in prediction_groups]),
        group_prediction_bounds_ps=np.array(
            [[g["prediction_lower_ps"], g["prediction_upper_ps"]]
             for g in prediction_groups]),
        physical_transform_minimum_mv=physical_min,
        physical_transform_maximum_mv=physical_max)
    grouped_normalized = None
    if "amplitude_normalized" in choices:
        grouped_normalized = grouped_amplitude_normalized_waveforms(
            waves, predictions, training, positions,
            baseline_positions, prediction_groups,
            chunk_size=args.chunk_size, peak_floor=args.peak_floor)
        np.savez_compressed(
            root / "grouped_amplitude_normalized_waveforms.npz",
            time_ns=time_ns, positions=positions,
            group_mean_waveform=grouped_normalized,
            channel_names=np.array(CHANNEL_NAMES),
            percentile_bands=np.array(PERCENTILE_BANDS, dtype=np.float64),
            group_mean_prediction_ps=np.array(
                [g["mean_prediction_ps"] for g in prediction_groups]),
            group_counts=np.array([g["n_events"] for g in prediction_groups]),
            definition=np.array(
                "(prepared waveform - per-event detector baseline) / "
                "per-event detector peak; no timewise z-scoring"))
    result_files = []
    for representation, result in results.items():
        representation_root = root / representation
        representation_root.mkdir(parents=True, exist_ok=True)
        for kind, feature_name in enumerate(result["feature_names"]):
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                # Vertically aligned panels: the corresponding detector
                # (or D2-D1) is shown above its heatmap. The amplitude-
                # normalized view uses normalized signals (dimensionless);
                # the other views retain the original physical mV curves.
                # A dedicated colorbar column preserves the shared time axis.
                fig = plt.figure(figsize=(13, 7.3), constrained_layout=True)
                gs = fig.add_gridspec(
                    2, 2, width_ratios=[1, 0.035], height_ratios=[1, 1.7])
                ax_signal = fig.add_subplot(gs[0, 0])
                ax = fig.add_subplot(gs[1, 0], sharex=ax_signal)
                cax = fig.add_subplot(gs[1, 1])
                for group_idx, info in enumerate(prediction_groups):
                    color = ("#454545" if group_idx == 1 else
                             plt.get_cmap("coolwarm")(
                                 target_norm(info["mean_prediction_ps"])))
                    label = (
                        f'P{info["percentile_lower"]:g}–'
                        f'{info["percentile_upper"]:g} | '
                        f'{info["mean_prediction_ps"]:+.1f} ps | '
                        f'n={info["n_events"]}')
                    signal_means = (grouped_normalized if
                                    representation == "amplitude_normalized"
                                    else grouped_mv)
                    ax_signal.plot(
                        time_ns, signal_means[group_idx, channel],
                        linewidth=1.7, color=color, label=label)
                signal_units = ("relative amplitude" if
                                representation == "amplitude_normalized" else "mV")
                ax_signal.set_ylabel(
                    (f"Detector 2 - detector 1 ({signal_units})"
                     if channel == 2 else
                     f"Detector {channel + 1} ({signal_units})"))
                ax_signal.set_title(
                    f"{representation} | {feature_name} | {channel_name} | " +
                    ("peak-normalized waveforms by saved prediction" if
                     representation == "amplitude_normalized" else
                     "original physical waveforms by saved prediction"))
                # Onishi marks the baseline+rising-edge region of interest.
                # Crop spans indicate samples unused by the timing model.
                # Keep the overlays behind the physical waveforms.
                if onishi_bounds is not None:
                    ax_signal.axvspan(
                        *onishi_bounds, facecolor="0.82",
                        alpha=0.45, edgecolor="none", zorder=0.3)
                for crop_left, crop_right in cropped_spans:
                    ax_signal.axvspan(
                        crop_left, crop_right, facecolor="0.91",
                        alpha=0.75, edgecolor="none", zorder=0.5)
                ax_signal.grid(True, which="major", alpha=0.18, linewidth=0.6)
                ax_signal.legend(loc="best", fontsize="small")
                ax_signal.tick_params(labelbottom=False)
                # Support threshold still masks sparse heatmap cells;
                # all counts remain available in each representation NPZ.
                im = ax.pcolormesh(
                    time_edges_ns, result["edges"],
                    np.ma.masked_invalid(result["mean_target"][kind, channel]),
                    shading="flat", cmap="coolwarm", norm=target_norm)
                fig.colorbar(im, cax=cax, label=(
                    label_description))
                # Shade excluded timing samples without removing existing
                # heatmap cells. No crop shading for energy mode.
                for crop_left, crop_right in cropped_spans:
                    ax.axvspan(
                        crop_left, crop_right, facecolor="0.88",
                        alpha=0.38, edgecolor="none", zorder=3)
                if kind == 1:
                    ylabel = ("Difference of standardized first differences"
                              if channel == 2 else "Standardized first difference (z)")
                else:
                    ylabel = ("Difference of detector z-scores (z2 - z1)"
                              if channel == 2 else
                              f"{representation.replace('_', ' ')} waveform (z)")
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Time relative to LED (ns)")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=12))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
                ax.grid(True, which="major", color="black",
                        alpha=0.25, linewidth=0.65)
                ax.set_axisbelow(False)
                ax.set_xlim(time_edges_ns[0], time_edges_ns[-1])
                if kind == 1 and args.derivative_end_sample is not None:
                    endpoint = min(args.derivative_end_sample, waves.shape[-1] - 1)
                    ax.set_xlim(time_edges_ns[0], time_ps[endpoint] / 1000.0)
                filename = f"{feature_name}_{channel_name}_target_heatmap.png"
                figure = representation_root / filename
                fig.savefig(figure, dpi=180)
                plt.close(fig)
                result_files.append(str(figure.relative_to(root)))
                log.info("FIGURE | %s", figure)

        # Keep every representation independently reproducible and debuggable.
        np.savez_compressed(
            representation_root / "conditional_target_heatmaps.npz",
            positions=positions, time_ps=time_ps[positions],
            time_ns=time_ns, time_edges_ns=time_edges_ns,
            z_edges=result["edges"], target_mean_ps=result["mean_target"],
            support_counts=result["counts"], target_sums_ps=result["sums"],
            training_prediction_mean_ps=baseline,
            normalization_mean=result["means"],
            normalization_scale=result["scales"],
            derivative_normalization_mean=(result["derivative_normalization_mean"]
                                           if derivative else np.empty((0,), dtype=np.float32)),
            derivative_normalization_scale=(result["derivative_normalization_scale"]
                                            if derivative else np.empty((0,), dtype=np.float32)),
            prediction_type=np.array("reference_minus_comparison" if is_difference
                                     else ("saved_comparison" if target_kind == "comparison"
                                           else "saved_reference")),
            channel_names=np.array(CHANNEL_NAMES),
            feature_names=np.array(result["feature_names"]),
            representation=np.array(representation))
        (representation_root / "summary.json").write_text(json.dumps({
            "representation": representation,
            "events": int(len(training)),
            "normalization": result["normalization"],
            "derivative_normalization": result["derivative_normalization"],
            "baseline_window_ns": ([args.baseline_start_ns, args.baseline_end_ns]
                                   if representation != "original" else None),
            "peak_definition": ("max of per-event per-detector baseline-subtracted "
                                "prepared waveform" if representation == "amplitude_normalized"
                                else None),
            "peak_floor_prepared_units": (args.peak_floor
                                          if representation == "amplitude_normalized" else None),
            "z_definition": "(transformed waveform - train timewise mean) / "
                            "train timewise safe std, fitted separately for this view",
            "differential_definition": "z(detector2)-z(detector1); not separately standardized",
            "physical_note": ("Prepared ML inputs are already affinely scaled and "
                              "clipped to [0,1]; baseline subtraction precedes the "
                              "new timewise z-score. Peak-normalized view is dimensionless."),
            "min_count": args.min_count,
            "plot_x_axis": "Time relative to LED (ns), prepared grid",
            "plot_layout": "matching detector mean waveform above heatmap; shared ns axis",
            "upper_panel_units": ("relative amplitude" if
                                  representation == "amplitude_normalized" else "mV"),
            "upper_panel_definition": (
                "per-event baseline-subtracted and peak-normalized waveform; "
                "group average before timewise z-scoring"
                if representation == "amplitude_normalized" else
                "original physical waveform, inverse prepared transform"),
            "waveform_groups_file": (
                "grouped_amplitude_normalized_waveforms.npz" if
                representation == "amplitude_normalized" else
                "grouped_physical_waveforms.npz"),
            "support_counts": "retained in NPZ, not plotted",
            "timing_cropped_spans_ns": cropped_spans,
            "onishi_highlight_ns": onishi_bounds,
        }, indent=2), encoding="utf-8")

    (root / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "comparison_provenance": comparison_provenance,
        "matched_population": matching,
        "matched_event_ids_file": ("matched_event_ids.npy" if matching else None),
        "comparison_metrics_file": ("prediction_comparison_metrics.json"
                                     if comparison_metrics is not None else None),
        "prediction_sign": "reference minus comparison" if is_difference else None,
        "comparison_target": target_kind if comparison_predictions is not None else "reference",
        "events": int(len(training)), "seed": args.seed,
        "target": ("SAVED reference minus comparison prediction, not calibrated ground truth"
                   if is_difference else
                   ("SAVED comparison-model prediction visualized on reference waveforms"
                    if target_kind == "comparison" else
                    "SAVED reference-model prediction, not calibrated ground truth")),
        "representations": list(choices),
        "baseline_window_ns": [args.baseline_start_ns, args.baseline_end_ns],
        "baseline_sample_count": (int(len(baseline_positions))
                                  if baseline_positions is not None else None),
        "baseline_sample_indices": ([int(baseline_positions[0]),
                                     int(baseline_positions[-1])]
                                    if baseline_positions is not None else None),
        "normalization": "independent train-only time-sample z-scoring per representation",
        "differential_definition": "z(detector2)-z(detector1); not separately standardized",
        "z_range": [-args.z_limit, args.z_limit],
        "z_bins": args.z_bins, "time_stride": args.time_stride,
        "min_count": args.min_count, "figures": result_files,
        "plot_x_axis": "Time relative to LED (ns), prepared grid",
        "plot_layout": "grouped mean waveforms above each heatmap, shared ns axis",
        "overlay_definition": (
            "amplitude_normalized: per-event baseline-subtracted/peak-normalized "
            "dimensionless signals; original and baseline_subtracted: original mV"),
        "waveform_groups": prediction_groups,
        "waveform_group_means_file": "grouped_physical_waveforms.npz",
        "waveform_units": "mV (inverse of clipped prepared input transform)",
        "support_counts": "saved in each NPZ, not plotted",
        "timing_cropped_spans_ns": cropped_spans,
        "onishi_highlight_ns": onishi_bounds,
        "target_mean_ps": baseline, "shared_symmetric_color_radius_ps": radius,
        "note": "Descriptive 1D conditional mean per time coordinate, not model attribution.",
    }, indent=2), encoding="utf-8")
    log.info("COMPLETE | %d figures across %d representations | %s",
             len(result_files), len(choices), root)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--comparison-model-dir", type=Path,
                   help="Second saved model run; map reference minus comparison predictions")
    p.add_argument("--comparison-target", choices=("difference", "reference", "comparison"),
                   default="difference",
                   help="With --comparison-model-dir, select difference (default), "
                        "reference saved prediction, or comparison saved prediction; "
                        "always visualize reference-model waveforms on common event IDs")
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tuning-queries", type=int, default=500)
    p.add_argument("--holdout-queries", type=int, default=500)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--normalization-scale-floor-ratio", type=float, default=0.05)
    p.add_argument("--time-stride", type=int, default=8)
    p.add_argument("--z-bins", type=int, default=80)
    p.add_argument("--z-limit", type=float, default=3.)
    p.add_argument("--min-count", type=int, default=25)
    p.add_argument("--representations", nargs="+", choices=REPRESENTATIONS,
                   default=list(REPRESENTATIONS),
                   help="Views to compare; default: original, baseline-subtracted, peak-normalized")
    p.add_argument("--baseline-start-ns", type=float, default=-2.)
    p.add_argument("--baseline-end-ns", type=float, default=-1.)
    p.add_argument("--peak-floor", type=float, default=1e-8,
                   help="Fail if baseline-subtracted peak in prepared [0,1] units is smaller")
    p.add_argument("--include-original-derivative", action="store_true",
                   help="Also emit legacy first-difference plots for original view")
    p.add_argument("--derivative-end-sample", type=int, default=1200,
                   help="Rightmost sample shown in derivative plots; 0-1200 by default. "
                        "The saved numerical heatmap remains full-length.")
    p.add_argument("--retained-only", action="store_true")
    p.add_argument("--onishi-start-ns", type=float,
                   help="Override Onishi window start; otherwise read sibling onishi/context.json")
    p.add_argument("--onishi-end-ns", type=float,
                   help="Override Onishi window end; requires --onishi-start-ns")
    p.add_argument("--disable-onishi-highlight", action="store_true",
                   help="Disable the Onishi reference band in signal panels")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
