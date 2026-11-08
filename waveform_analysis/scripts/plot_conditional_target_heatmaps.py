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
REPRESENTATIONS = ("original", "baseline_subtracted", "amplitude_normalized")


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



def load_prediction_difference(args, reference_predictions, reference_provenance,
                               reference_waves, log):
    """Match two independently loaded models by prepared event ID, not row index.

    The output heatmap always uses reference-model waveform coordinates.
    Refuse a comparison if matched waveform inputs disagree.
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
        run = read_run(model_dir.expanduser().resolve())
        config = resolve_run_config(model_dir.expanduser().resolve())
        if args.cache_root is not None:
            root = args.cache_root.expanduser().resolve()
            config["preprocessing"]["cache_dir"] = str(root / "preprocessing")
            config["preprocessing"]["artifact_root"] = str(root / "artifact_store")
            config["preprocessing"]["named_cache_root"] = str(root)
        elif args.artifact_root is not None:
            config["preprocessing"]["artifact_root"] = str(
                args.artifact_root.expanduser().resolve())
            config["preprocessing"].pop("named_cache_root", None)
        return PreparedCacheLookup().find("development", config, run["manifest"])

    ref_dataset = dataset_for(args.model_dir)
    other_dataset = dataset_for(args.comparison_model_dir)
    ref_ids = np.asarray(ref_dataset.event_index, dtype=np.int64)
    other_ids = np.asarray(other_dataset.event_index, dtype=np.int64)
    if (len(ref_ids) != len(reference_predictions) or
            len(other_ids) != len(other_predictions) or
            np.unique(ref_ids).size != len(ref_ids) or
            np.unique(other_ids).size != len(other_ids)):
        raise ValueError("Ambiguous or inconsistent development event IDs")
    if not np.array_equal(np.sort(ref_ids), np.sort(other_ids)):
        raise ValueError("Development event populations differ")
    order = np.argsort(other_ids, kind="stable")
    aligned_rows = order[np.searchsorted(other_ids[order], ref_ids)]
    if reference_waves.shape != other_waves.shape:
        raise ValueError("Prepared waveform shapes differ")
    # Check waveform identity using paired channels and selected positions,
    # not only event IDs, because model runs may have different preprocessing.
    sample_rows = np.unique(np.linspace(0, len(ref_ids) - 1, 64, dtype=np.int64))
    sample_times = np.unique(np.linspace(
        0, reference_waves.shape[-1] - 1, 128, dtype=np.int64))
    for row in sample_rows:
        left = np.asarray(reference_waves[int(row)], dtype=np.float32)
        right = np.asarray(other_waves[int(aligned_rows[row])], dtype=np.float32)
        if not np.allclose(left[:, sample_times], right[:, sample_times],
                           rtol=1e-5, atol=1e-6):
            raise ValueError("Prepared waveforms differ between runs after event-ID alignment")
    log.info("DIFFERENCE | reference %s minus %s | aligned events=%d | "
             "waveform checks passed",
             reference_provenance["model"], second["model"], len(ref_ids))
    return reference_predictions - other_predictions[aligned_rows], second, other_predictions[aligned_rows]


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
    if args.comparison_model_dir is not None:
        predictions, comparison_provenance, comparison_predictions = load_prediction_difference(
            args, predictions, provenance, waves, log)
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
        subsets = {"all_development": np.arange(n),
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
    choices = tuple(dict.fromkeys(args.representations))
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
    result_files = []
    for representation, result in results.items():
        representation_root = root / representation
        representation_root.mkdir(parents=True, exist_ok=True)
        for kind, feature_name in enumerate(result["feature_names"]):
            for channel, channel_name in enumerate(CHANNEL_NAMES):
                fig, ax = plt.subplots(
                    figsize=(13, 5.0), constrained_layout=True)
                # Single-panel plot. Empirical-support counts are still saved
                # in the numerical NPZ; min_count still masks sparse cells.
                im = ax.pcolormesh(
                    time_edges_ns, result["edges"],
                    np.ma.masked_invalid(result["mean_target"][kind, channel]),
                    shading="flat", cmap="coolwarm", norm=target_norm)
                fig.colorbar(im, ax=ax, label=(
                    "Mean prediction difference (reference - comparison, ps)"
                    if comparison_provenance else "Mean saved teacher prediction (ps)"))
                ax.set_title(
                    f"{representation} | {feature_name} | {channel_name} | "
                    f"mean prediction (min count {args.min_count})")
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
            prediction_type=np.array("reference_minus_comparison" if comparison_provenance
                                     else "saved_teacher"),
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
            "plot_layout": "single-panel heatmap with major gridlines",
            "support_counts": "retained in NPZ, not plotted",
        }, indent=2), encoding="utf-8")

    (root / "summary.json").write_text(json.dumps({
        "provenance": provenance,
        "comparison_provenance": comparison_provenance,
        "comparison_metrics_file": ("prediction_comparison_metrics.json"
                                     if comparison_metrics is not None else None),
        "prediction_sign": "reference minus comparison" if comparison_provenance else None,
        "events": int(len(training)), "seed": args.seed,
        "target": ("SAVED reference minus comparison prediction, not calibrated ground truth"
                   if comparison_provenance else "SAVED full teacher prediction, not calibrated ground truth"),
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
        "plot_layout": "single-panel heatmaps, major gridlines",
        "support_counts": "saved in each NPZ, not plotted",
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
    return p


if __name__ == "__main__":
    run(parser().parse_args())
