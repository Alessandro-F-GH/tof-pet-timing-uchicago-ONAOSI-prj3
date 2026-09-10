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
from .stats import ctr_estimate, format_residual_summary, residual_summary
from .timing import anchor_grid, cfd_grid, led_grid, pair_delta
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


def _input_offsets(interval_s, config):
    window = config["ml_input"]["window_ns"]
    dt_ns = float(interval_s) * 1e9
    first = int(np.ceil(float(window["start"]) / dt_ns - 1e-9))
    last = int(np.floor(float(window["end"]) / dt_ns + 1e-9))
    offsets = np.arange(first, last + 1, dtype=np.int64)
    factor = int(config["ml_input"].get("subsampling", 1))
    offsets = offsets[::factor]
    if offsets.size < 2:
        raise ValueError("ML input window contains fewer than two samples")
    return offsets, offsets.astype(float) * dt_ns * 1000.0


def _family_offsets_and_invalid(data, family, anchor_index, config):
    waves = data.energy_windows_mV if family == "energy" else data.timing_windows_mV
    intervals = data.energy_sample_interval_s if family == "energy" else data.timing_sample_interval_s
    if waves is None or intervals is None:
        raise ValueError(f"{family} waveform unavailable")
    ref = float(np.asarray(intervals)[0, 0])
    if not np.allclose(np.asarray(intervals), ref, rtol=1e-9, atol=0):
        raise ValueError(f"{family} sampling interval must be common")
    offsets, time_ps = _input_offsets(ref, config)
    idx = np.asarray(anchor_index, dtype=np.int64)
    invalid = np.any(
        (idx + int(offsets[0]) < 0) | (idx + int(offsets[-1]) >= waves.shape[2]),
        axis=1,
    )
    return offsets, time_ps, invalid


def _materialize_family(data, family, anchor_index, kept_rows, config):
    waves = data.energy_windows_mV if family == "energy" else data.timing_windows_mV
    intervals = data.energy_sample_interval_s if family == "energy" else data.timing_sample_interval_s
    if waves is None or intervals is None:
        raise ValueError(f"{family} waveform unavailable")
    ref = float(np.asarray(intervals)[0, 0])
    offsets, time_ps = _input_offsets(ref, config)
    output = np.empty((kept_rows.size, 2, offsets.size), dtype=np.float32)
    for out_event, event in enumerate(kept_rows):
        for detector in range(2):
            idx = int(anchor_index[event, detector]) + offsets
            output[out_event, detector] = np.asarray(waves[event, detector, idx], dtype=np.float32)
    limits = channel_limits(config["preprocessing"][family]["vertical_scale_limit_mV"])
    minimum = limits[:, 0, None].astype(np.float32)
    maximum = limits[:, 1, None].astype(np.float32)
    normalized = ((output - minimum[None, :, :]) / (maximum - minimum)[None, :, :]).astype(np.float32)
    return normalized, time_ps, minimum, maximum


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
    anchor_idx = {}
    anchor_times = {}
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
        anchor_idx[family], anchor_times[family] = anchor_grid(preprocessed, family, led_choice[family])

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
        _offsets, _time, bad = _family_offsets_and_invalid(preprocessed, family, anchor_idx[family], config)
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
    for family in sorted(sources):
        normalized, time_ps, minimum, maximum = _materialize_family(preprocessed, family, anchor_idx[family], kept_rows, config)
        target = open_memmap(base / f"{family}_windows.npy", mode="w+", dtype=np.float32, shape=normalized.shape)
        target[:] = normalized
        target.flush()
        del target
        np.save(base / f"{family}_time_ps.npy", time_ps)
        np.savez_compressed(base / f"{family}_transform.npz", minimum=minimum, maximum=maximum)
        transforms[family] = {
            "type": "min_max",
            "feature_range": [0.0, 1.0],
            "source": f"preprocessing.{family}.vertical_scale_limit_mV",
            "minimum_mV": minimum[:, 0].tolist(),
            "maximum_mV": maximum[:, 0].tolist(),
        }

    led_training_mean = {}
    calibration_bias = {}
    for family in families:
        led_pair = pair_delta(led_times[family][keep])
        mean_led = float(np.mean(led_pair[training_new]))
        c_hat = mean_led - true_tof
        target_values = led_pair - true_tof - c_hat
        led_training_mean[family] = mean_led
        calibration_bias[family] = c_hat
        np.save(base / f"{family}_led_time_ps.npy", led_times[family][keep])
        np.save(base / f"{family}_anchor_time_ps.npy", anchor_times[family][keep])
        np.save(base / f"{family}_target_ps.npy", target_values)
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
        "ctr_selection_metric": "gaussian_equivalent_shortest_coverage_interval",
        "ctr_coverage_fraction": float(config["fit"].get("coverage_fraction", 0.90)),
        "ctr_core_bin_width_ps": float(config["fit"]["bin_width_ps"]),
        "ml_input": config["ml_input"],
        "normalization": transforms,
        "diagnostic_examples": diagnostic_examples,
        "target_definition": "delta_t_led - true_tof - calibration_bias",
        "anchor_definition": "native sample nearest in time to interpolated selected LED crossing",
        "corrected_definition": "target - paired_model_prediction",
        "time_reference": "native_grid_anchor_nearest_interpolated_led",
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
