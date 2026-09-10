from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from .common import atomic_json, canonical_hash, channel_limits, dataset_cache_dir
from .dataset import DATASET_FORMAT_VERSION, load_prepared_dataset
from .diagnostics import plot_missing_led_example, plot_ml_window_exceeds_example
from .splits import semantic_seed, split_training_validation
from utils_fit import CTR_METRIC_NAME

from .stats import ctr_estimate, format_residual_summary, residual_summary
from .timing import cfd_grid, family_arrays, interpolate_relative, led_grid, pair_delta
from .view import source_family, target_family


def _families(config):
    mode = config["mode"]
    return {source_family(mode)}, {target_family(mode)}


def dataset_fingerprint(preprocessed, config):
    mode = config["mode"]
    return canonical_hash(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "preprocessed": preprocessed.manifest["fingerprint"],
            "true_tof_ps": config["data"]["true_tof_ps"],
            "validation": {
                "seed": config["validation"]["seed"],
                "validation_fraction": config["validation"]["validation_fraction"],
            },
            "standard_methods": config["standard_methods"],
            "fit": config.get("fit"),
            "ml_input": config["ml_input"],
            "mode": mode,
            "cfd": config["cfd"],
            "normalization_limits": config["preprocessing"][source_family(mode)]["vertical_scale_limit_mV"],
        }
    )


def _best_column(
    grid,
    candidates,
    true_tof,
    fit,
    *,
    minimum_efficiency=0.0,
    coincidence_window_ps=None,
    logger=None,
    label="standard method",
):
    total = int(np.asarray(grid).shape[0])
    best = None
    best_observed_efficiency = 0.0
    if total <= 0:
        raise RuntimeError(f"No development events available for {label} selection")

    # Candidate selection uses the same robust CTR estimator as final reporting,
    # but skips bootstrap because uncertainty is not part of threshold ranking.
    for i, candidate in enumerate(candidates):
        residual = pair_delta(np.asarray(grid[:, :, i], dtype=np.float64)) - float(true_tof)
        valid = np.isfinite(residual)
        if coincidence_window_ps is not None:
            valid &= np.abs(residual) <= float(coincidence_window_ps)
        coverage = int(np.count_nonzero(valid))
        efficiency = coverage / total
        best_observed_efficiency = max(best_observed_efficiency, efficiency)
        if coverage == 0:
            continue
        if efficiency < float(minimum_efficiency):
            if logger is not None:
                logger.info(
                    "%s candidate %.6g rejected | coincidence efficiency %.2f%% < %.2f%%",
                    label,
                    float(candidate),
                    100.0 * efficiency,
                    100.0 * float(minimum_efficiency),
                )
            continue
        try:
            score = float(ctr_estimate(residual[valid], fit, bootstrap=False).ctr_ps)
        except ValueError as exc:
            if logger is not None:
                logger.warning(
                    "%s candidate %.6g robust CTR unavailable | coincidence efficiency %.2f%% (%d/%d) | reason=%s | %s",
                    label,
                    float(candidate),
                    100.0 * efficiency,
                    coverage,
                    total,
                    exc,
                    format_residual_summary(residual_summary(residual[valid])),
                )
            continue
        key = (score, float(candidate))
        if best is None or key < best[0]:
            best = (key, float(candidate), score, coverage, efficiency)

    if best is None:
        raise RuntimeError(
            f"No {label} candidate satisfies the minimum coincidence efficiency "
            f"{100.0 * float(minimum_efficiency):.1f}% and provides a valid robust CTR; "
            f"best observed efficiency={100.0 * best_observed_efficiency:.2f}%"
        )
    return best[1], best[2], best[3], best[4]


def _input_time_grid(interval_s, config):
    """Common continuous time grid relative to the interpolated LED crossing."""
    window = config["ml_input"]["window_ns"]
    dt_ps = float(interval_s) * 1.0e12
    factor = int(config["ml_input"].get("subsampling", 1))
    step_ps = dt_ps * factor
    start_ps = 1000.0 * float(window["start"])
    end_ps = 1000.0 * float(window["end"])
    first = int(np.ceil(start_ps / step_ps - 1e-12))
    last = int(np.floor(end_ps / step_ps + 1e-12))
    time_ps = np.arange(first, last + 1, dtype=np.float64) * step_ps
    if time_ps.size < 2:
        raise ValueError("ML input window contains fewer than two interpolated samples")
    if not np.any(np.isclose(time_ps, 0.0, rtol=0.0, atol=max(1e-9, abs(step_ps) * 1e-12))):
        raise ValueError("Interpolated ML grid must contain t=0 at the selected LED crossing")
    return time_ps


def _family_time_grid_and_invalid(data, family, led_time_ps, config):
    waves, starts, intervals, _rising_start, _rising_stop = family_arrays(data, family)
    ref = float(np.asarray(intervals)[0, 0])
    if not np.allclose(np.asarray(intervals), ref, rtol=1e-9, atol=0):
        raise ValueError(f"{family} sampling interval must be common")
    time_ps = _input_time_grid(ref, config)
    led = np.asarray(led_time_ps, dtype=np.float64)
    if led.shape != (data.n_events, 2):
        raise ValueError(f"{family} LED timing array has unexpected shape {led.shape}")
    target_start_s = led * 1.0e-12 + float(time_ps[0]) * 1.0e-12
    target_end_s = led * 1.0e-12 + float(time_ps[-1]) * 1.0e-12
    source_start_s = np.asarray(starts, dtype=np.float64)
    source_end_s = source_start_s + (waves.shape[2] - 1) * np.asarray(intervals, dtype=np.float64)
    tolerance = max(abs(ref) * 1.0e-9, 1.0e-18)
    invalid = np.any(
        ~np.isfinite(led)
        | (target_start_s < source_start_s - tolerance)
        | (target_end_s > source_end_s + tolerance),
        axis=1,
    )
    return time_ps, invalid


def _materialize_family(data, family, led_time_ps, kept_rows, config, threshold_mV):
    waves, starts, intervals, _rising_start, _rising_stop = family_arrays(data, family)
    ref = float(np.asarray(intervals)[0, 0])
    time_ps = _input_time_grid(ref, config)
    output = np.empty((kept_rows.size, 2, time_ps.size), dtype=np.float32)
    zero = int(np.flatnonzero(np.isclose(time_ps, 0.0, rtol=0.0, atol=max(1e-9, abs(ref) * 1e3)))[0])
    for out_event, event in enumerate(kept_rows):
        for detector in range(2):
            values = interpolate_relative(
                waves[event, detector],
                starts[event, detector],
                intervals[event, detector],
                led_time_ps[event, detector],
                time_ps,
            )
            values[zero] = float(threshold_mV)
            output[out_event, detector] = values.astype(np.float32, copy=False)
    return output, time_ps


def _dominant_fraction(values: np.ndarray) -> float:
    x = np.asarray(values).reshape(-1)
    if x.size == 0:
        return 0.0
    _unique, counts = np.unique(x, return_counts=True)
    return float(np.max(counts) / x.size)


def _learn_dead_time_mask(raw: np.ndarray, development_indices: np.ndarray, time_ps: np.ndarray, threshold: float = 0.99):
    """Learn time coordinates carrying no useful variation on development only."""
    x = np.asarray(raw, dtype=np.float32)
    dev = np.asarray(development_indices, dtype=np.int64)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected [event, detector=2, time], got {x.shape}")
    if dev.size == 0:
        raise ValueError("Cannot learn dead-region mask without development events")
    fractions = np.empty((2, x.shape[2]), dtype=np.float64)
    for detector in range(2):
        for sample in range(x.shape[2]):
            fractions[detector, sample] = _dominant_fraction(x[dev, detector, sample])
    dead = np.all(fractions >= float(threshold), axis=0)
    zero = np.isclose(np.asarray(time_ps, dtype=np.float64), 0.0, rtol=0.0, atol=1e-9)
    dead |= zero
    keep = ~dead
    if np.count_nonzero(keep) < 2:
        raise ValueError("Dead-region removal leaves fewer than two ML input coordinates")
    return keep, fractions


def _normalize_family(raw, family, config):
    limits = channel_limits(config["preprocessing"][family]["vertical_scale_limit_mV"])
    minimum = limits[:, 0, None].astype(np.float32)
    maximum = limits[:, 1, None].astype(np.float32)
    normalized = ((np.asarray(raw, dtype=np.float32) - minimum[None, :, :]) / (maximum - minimum)[None, :, :]).astype(np.float32)
    return normalized, minimum, maximum

def _remap_split(indices, keep):
    mapping = np.full(keep.size, -1, dtype=np.int64)
    mapping[np.flatnonzero(keep)] = np.arange(np.count_nonzero(keep), dtype=np.int64)
    kept = np.asarray(indices, dtype=np.int64)
    kept = kept[keep[kept]]
    return mapping[kept]


def _ensure_diagnostics(preprocessed, config, manifest):
    examples = manifest.get("diagnostic_examples") or {}
    directory = Path(config["experiment"]["output_dir"]).resolve() / "diagnostic_plots"
    missing = examples.get("missing_led")
    if missing:
        plot_missing_led_example(
            preprocessed,
            str(missing["family"]),
            int(missing["event_row"]),
            float(missing["threshold_mV"]),
            directory,
        )
    window = examples.get("ml_window_exceeds_materialized")
    if window:
        plot_ml_window_exceeds_example(
            preprocessed,
            str(window["family"]),
            int(window["event_row"]),
            float(window["threshold_mV"]),
            config["ml_input"]["window_ns"],
            directory,
        )


def prepare_ml_dataset(preprocessed, config, *, rebuild, logger):
    base = dataset_cache_dir(config, "prepared_dir", preprocessed.manifest["source"])
    if base.is_dir() and not rebuild:
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != dataset_fingerprint(preprocessed, config):
            raise ValueError(f"Prepared dataset cache is stale: {base}")
        _ensure_diagnostics(preprocessed, config, manifest)
        return load_prepared_dataset(base)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    development = np.asarray(preprocessed.development, dtype=np.int64)
    test = np.asarray(preprocessed.test, dtype=np.int64)
    split = split_training_validation(
        development,
        validation_fraction=float(config["validation"]["validation_fraction"]),
        seed=semantic_seed(int(config["validation"]["seed"]), Path(preprocessed.manifest["source"]).name),
    )
    training, validation = split.training, split.validation
    true_tof = float(config["data"]["true_tof_ps"])
    sources, targets = _families(config)
    families = sorted(sources | targets)
    thresholds = np.asarray(config["standard_methods"]["led_thresholds_mV"], dtype=float)
    fractions = np.asarray(config["standard_methods"]["cfd_fractions"], dtype=float)
    led_min_eff = float(config["standard_methods"].get("led_minimum_crossing_efficiency", 0.95))
    coincidence_window_ps = 1000.0 * float(config["standard_methods"].get("led_coincidence_window_ns", 2.0))
    if not 0.0 < led_min_eff <= 1.0:
        raise ValueError("standard_methods.led_minimum_crossing_efficiency must be in (0, 1]")
    if coincidence_window_ps <= 0:
        raise ValueError("standard_methods.led_coincidence_window_ns must be positive")

    led_choice = {}
    led_score = {}
    led_development_coverage = {}
    led_development_efficiency = {}
    cfd_choice = {}
    cfd_score = {}
    led_times = {}
    cfd_times = {}
    led_coverage = {}
    led_missing_crossing = {}
    led_noncoincidence = {}
    diagnostic_examples = {}

    for family in families:
        dev_led = led_grid(preprocessed, family, development, thresholds)
        (
            led_choice[family],
            led_score[family],
            led_development_coverage[family],
            led_development_efficiency[family],
        ) = _best_column(
            dev_led,
            thresholds,
            true_tof,
            config.get("fit"),
            minimum_efficiency=led_min_eff,
            coincidence_window_ps=coincidence_window_ps,
            logger=logger,
            label=f"{family} LED",
        )
        logger.info(
            "Selected %s LED | threshold %.6g mV | development robust CTR %.3f ps | coincidence efficiency %.2f%% (%d/%d) | window ±%.3f ns | minimum %.1f%%",
            family,
            led_choice[family],
            led_score[family],
            100.0 * led_development_efficiency[family],
            led_development_coverage[family],
            development.size,
            coincidence_window_ps / 1000.0,
            100.0 * led_min_eff,
        )
        led_times[family] = led_grid(
            preprocessed,
            family,
            np.arange(preprocessed.n_events),
            np.asarray([led_choice[family]]),
        )[:, :, 0]
        finite_pair = np.all(np.isfinite(led_times[family]), axis=1)
        residual_pair = pair_delta(led_times[family]) - true_tof
        in_coincidence = finite_pair & np.isfinite(residual_pair) & (np.abs(residual_pair) <= coincidence_window_ps)
        led_missing_crossing[family] = ~finite_pair
        led_noncoincidence[family] = finite_pair & ~in_coincidence
        led_coverage[family] = in_coincidence

        if config["cfd"] and family in targets:
            dev_cfd = cfd_grid(preprocessed, family, development, fractions)
            cfd_choice[family], cfd_score[family], _coverage, _efficiency = _best_column(
                dev_cfd,
                fractions,
                true_tof,
                config.get("fit"),
                logger=logger,
                label=f"{family} CFD",
            )
            cfd_times[family] = cfd_grid(
                preprocessed,
                family,
                np.arange(preprocessed.n_events),
                np.asarray([cfd_choice[family]]),
            )[:, :, 0]

    invalid_led = np.zeros(preprocessed.n_events, dtype=bool)
    all_missing = np.zeros(preprocessed.n_events, dtype=bool)
    all_noncoincidence = np.zeros(preprocessed.n_events, dtype=bool)
    missing_by_family = {}
    noncoincidence_by_family = {}
    for family in families:
        missing = led_missing_crossing[family]
        noncoincidence = led_noncoincidence[family]
        invalid = ~led_coverage[family]
        invalid_led |= invalid
        all_missing |= missing
        all_noncoincidence |= noncoincidence
        missing_by_family[family] = int(np.count_nonzero(missing))
        noncoincidence_by_family[family] = int(np.count_nonzero(noncoincidence))
        if np.any(missing) and "missing_led" not in diagnostic_examples:
            event_row = int(np.flatnonzero(missing)[0])
            diagnostic_examples["missing_led"] = {
                "family": family,
                "event_row": event_row,
                "event_index": int(np.asarray(preprocessed.event_index)[event_row]),
                "threshold_mV": float(led_choice[family]),
            }

    event_index = np.asarray(preprocessed.event_index, dtype=np.int64)
    invalid_led_index = event_index[invalid_led]
    missing_led_index = event_index[all_missing]
    noncoincidence_index = event_index[all_noncoincidence]
    np.save(base / "excluded_led_event_index.npy", invalid_led_index)
    np.save(base / "excluded_missing_led_event_index.npy", missing_led_index)
    np.save(base / "excluded_noncoincidence_event_index.npy", noncoincidence_index)
    if invalid_led_index.size:
        logger.warning(
            "Discarding events outside selected LED coincidence | discarded=%d/%d | missing_crossing=%d | outside_±%.3fns=%d | missing_by_family=%s | noncoincidence_by_family=%s",
            invalid_led_index.size,
            preprocessed.n_events,
            missing_led_index.size,
            coincidence_window_ps / 1000.0,
            noncoincidence_index.size,
            missing_by_family,
            noncoincidence_by_family,
        )

    ml_window_invalid = np.zeros(preprocessed.n_events, dtype=bool)
    for family in sorted(sources):
        _time, bad = _family_time_grid_and_invalid(preprocessed, family, led_times[family], config)
        ml_window_invalid |= bad
        candidates = np.flatnonzero(bad & ~invalid_led)
        if not candidates.size:
            candidates = np.flatnonzero(bad)
        if candidates.size and "ml_window_exceeds_materialized" not in diagnostic_examples:
            event_row = int(candidates[0])
            diagnostic_examples["ml_window_exceeds_materialized"] = {
                "family": family,
                "event_row": event_row,
                "event_index": int(np.asarray(preprocessed.event_index)[event_row]),
                "threshold_mV": float(led_choice[family]),
            }
    ml_window_index = np.asarray(preprocessed.event_index, dtype=np.int64)[ml_window_invalid]
    np.save(base / "excluded_ml_window_event_index.npy", ml_window_index)
    if ml_window_index.size:
        logger.warning(
            "Discarding events whose ML window exceeds materialized waveform | discarded=%d/%d",
            ml_window_index.size,
            preprocessed.n_events,
        )

    invalid = invalid_led | ml_window_invalid
    keep = ~invalid
    kept_rows = np.flatnonzero(keep)
    if not kept_rows.size:
        raise RuntimeError("No events remain after LED-coincidence and ML-window exclusions")
    training_new = _remap_split(training, keep)
    validation_new = _remap_split(validation, keep)
    test_new = _remap_split(test, keep)

    np.save(base / "event_index.npy", np.asarray(preprocessed.event_index, dtype=np.int64)[keep])
    np.save(base / "bias_voltage_V.npy", np.asarray(preprocessed.bias_voltage_V, dtype=np.float64)[keep])
    np.savez_compressed(base / "splits.npz", training=training_new, validation=validation_new, test=test_new)

    transforms = {}
    dead_regions = {}
    development_new = np.concatenate([training_new, validation_new])
    for family in sorted(sources):
        raw, full_time_ps = _materialize_family(
            preprocessed,
            family,
            led_times[family],
            kept_rows,
            config,
            led_choice[family],
        )
        feature_keep, dominant_fraction = _learn_dead_time_mask(
            raw,
            development_new,
            full_time_ps,
            threshold=0.99,
        )
        time_ps = np.asarray(full_time_ps[feature_keep], dtype=np.float64)
        raw = raw[:, :, feature_keep]
        normalized, minimum, maximum = _normalize_family(raw, family, config)
        target = open_memmap(base / f"{family}_windows.npy", mode="w+", dtype=np.float32, shape=normalized.shape)
        target[:] = normalized
        target.flush()
        del target
        np.save(base / f"{family}_time_ps.npy", time_ps)
        np.save(base / f"{family}_feature_keep_mask.npy", feature_keep)
        np.save(base / f"{family}_dead_dominant_fraction.npy", dominant_fraction)
        np.savez_compressed(base / f"{family}_transform.npz", minimum=minimum, maximum=maximum)
        transforms[family] = {
            "type": "min_max",
            "feature_range": [0.0, 1.0],
            "source": f"preprocessing.{family}.vertical_scale_limit_mV",
            "minimum_mV": minimum[:, 0].tolist(),
            "maximum_mV": maximum[:, 0].tolist(),
        }
        dead_regions[family] = {
            "criterion": "same_exact_value_in_each_detector_for_at_least_99_percent_of_development_events",
            "threshold": 0.99,
            "n_before": int(full_time_ps.size),
            "n_removed": int(np.count_nonzero(~feature_keep)),
            "n_after": int(time_ps.size),
            "removed_time_ps": np.asarray(full_time_ps[~feature_keep], dtype=float).tolist(),
            "feature_keep_mask_file": f"{family}_feature_keep_mask.npy",
            "dominant_fraction_file": f"{family}_dead_dominant_fraction.npy",
        }
        logger.info(
            "%s interpolated ML grid | samples=%d -> %d | removed dead=%d | t=0 removed=%s",
            family,
            full_time_ps.size,
            time_ps.size,
            np.count_nonzero(~feature_keep),
            not np.any(np.isclose(time_ps, 0.0, rtol=0.0, atol=1e-9)),
        )

    led_training_mean = {}
    calibration_bias = {}
    for family in families:
        led_pair = pair_delta(led_times[family][keep])
        mean_led = float(np.mean(led_pair[training_new]))
        c_hat = mean_led - true_tof
        led_training_mean[family] = mean_led
        calibration_bias[family] = c_hat
        np.save(base / f"{family}_led_time_ps.npy", led_times[family][keep])
        if family in cfd_times:
            np.save(base / f"{family}_cfd_time_ps.npy", cfd_times[family][keep])

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": dataset_fingerprint(preprocessed, config),
        "source": preprocessed.manifest["source"],
        "preprocessed_dir": str(preprocessed.directory),
        "mode": config["mode"],
        "true_tof_ps": true_tof,
        "n_input_events": preprocessed.n_events,
        "n_events": int(kept_rows.size),
        "excluded_led_invalid": int(invalid_led_index.size),
        "excluded_led_event_index_file": "excluded_led_event_index.npy",
        "excluded_missing_led": int(missing_led_index.size),
        "excluded_missing_led_event_index_file": "excluded_missing_led_event_index.npy",
        "excluded_noncoincidence": int(noncoincidence_index.size),
        "excluded_noncoincidence_event_index_file": "excluded_noncoincidence_event_index.npy",
        "excluded_ml_window_exceeds_preprocessing": int(ml_window_index.size),
        "excluded_ml_window_event_index_file": "excluded_ml_window_event_index.npy",
        "split": {"training": int(training_new.size), "validation": int(validation_new.size), "test": int(test_new.size)},
        "led_threshold_mV": led_choice,
        "led_minimum_crossing_efficiency": led_min_eff,
        "led_coincidence_window_ns": coincidence_window_ps / 1000.0,
        "led_development_ctr_ps": led_score,
        "led_development_coverage": led_development_coverage,
        "led_development_efficiency": led_development_efficiency,
        "led_missing_by_family": missing_by_family,
        "led_noncoincidence_by_family": noncoincidence_by_family,
        "led_training_mean_ps": led_training_mean,
        "calibration_bias_ps": calibration_bias,
        "cfd_fraction": cfd_choice,
        "cfd_development_ctr_ps": cfd_score,
        "ctr_selection_metric": CTR_METRIC_NAME,
        "ctr_coverage_fraction": float(config["fit"].get("coverage_fraction", 0.90)),
        "ctr_core_bin_width_ps": float(config["fit"]["bin_width_ps"]),
        "ml_input": config["ml_input"],
        "normalization": transforms,
        "dead_region_mask": dead_regions,
        "diagnostic_examples": diagnostic_examples,
        "target_definition": "calibrated_led = delta_t_led - true_tof - calibration_bias",
        "corrected_definition": "calibrated_led - paired_model_prediction",
        "time_reference": "continuous_time_relative_to_interpolated_led_crossing",
        "interpolation": "linear",
        "crossing_sample_policy": "t=0 forced to selected LED threshold then removed from ML features",
    }
    atomic_json(base / "manifest.json", manifest)
    _ensure_diagnostics(preprocessed, config, manifest)
    logger.info(
        "ML dataset %s | train=%d validation=%d test=%d | excluded_led=%d (missing=%d, noncoincidence=%d) | excluded_ml_window=%d | subsampling=%d",
        Path(preprocessed.manifest["source"]).name,
        training_new.size,
        validation_new.size,
        test_new.size,
        invalid_led_index.size,
        missing_led_index.size,
        noncoincidence_index.size,
        ml_window_index.size,
        int(config["ml_input"].get("subsampling", 1)),
    )
    return load_prepared_dataset(base)
