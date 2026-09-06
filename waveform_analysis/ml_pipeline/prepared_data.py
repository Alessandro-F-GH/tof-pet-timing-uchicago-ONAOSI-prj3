from __future__ import annotations

import copy
import hashlib
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
from scipy.signal import butter, sosfiltfilt

from utils.photopeak import fit_photopeak, photopeak_mask
from utils.signal import INVALID_TIME_FS

from .common import atomic_json, canonical_hash, read_json, source_signature
from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset
from .preprocessing_diagnostics import write_preprocessing_diagnostics
from .selection_store import load_or_compute_selection

if TYPE_CHECKING:
    from .data import EnergyCache

PREPARED_SELECTION_VERSION = 6

# Permanent v7 data contain one canonical trigger-referenced window for each
# active waveform family.  The historical timing-aligned duplicate energy matrix
# is intentionally NOT copied from the temporary raw cache.
_COPY_ARRAYS = (
    "event_id",
    "event_index",
    "source_file_id",
    "source_run_index",
    "bias_voltage_V",
    "amplitude_mV",
    "noise_rms_mV",
    "trigger_index",
    "windows_mV",
    "energy_led_time_fs",
    "timing_led_time_fs",
    "energy_cfd_time_fs",
    "timing_cfd_time_fs",
    "energy_window_anchor_time_fs",
    "timing_window_anchor_time_fs",
    "timing_windows_mV",
)


def _active_trigger_families(study_or_config: dict[str, Any]) -> tuple[str, ...]:
    """Return only waveform families actually used by configured experiment modes."""
    modes = [str(value).strip().lower() for value in study_or_config.get("channel_modes", [])]
    if not modes:
        # Direct materialize_selected_dataset callers from older code do not pass
        # channel_modes. Energy is always present because it defines photopeak.
        return ("energy",)
    families: list[str] = []
    if any("energy" in mode for mode in modes):
        families.append("energy")
    if any("timing" in mode for mode in modes):
        families.append("timing")
    return tuple(families or ["energy"])


def _preparation_request_fingerprint(study: dict[str, Any], root_file: Path) -> str:
    """Hash only inputs that change canonical prepared preprocessing.

    Model families, train/validation settings, true TOF and runtime input windows
    do not trigger another ROOT/photopeak pass.  Physical preprocessing settings,
    active channel families, materialized windowing and selection do.
    """
    preprocessing = copy.deepcopy(study["preprocessing"])

    # These no longer define preprocessing semantics. Baseline uses a
    # trigger-relative ns interval; LED thresholds are selected downstream.
    for section in ("common", "energy", "timing"):
        values = preprocessing.get(section)
        if isinstance(values, dict):
            values.pop("baseline_samples", None)
            values.pop("led_threshold_mV", None)
    for key in (
        "prepared_dir",
        "selection_store_dir",
        "cleanup_raw_cache",
        "materialization_chunk_size",
        "parallelization",
        "input_variants",
        "input_variant_by_channel",
        "subsampling_factors",
    ):
        preprocessing.pop(key, None)
    io = preprocessing.get("io")
    if isinstance(io, dict):
        preprocessing["io"] = {"max_events": int(io.get("max_events", 0))}
    return canonical_hash(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "selection_version": PREPARED_SELECTION_VERSION,
            "source": source_signature(root_file),
            "channels": study["data"]["channels"],
            "active_trigger_families": _active_trigger_families(study),
            "materialized_window_ns": preprocessing.get("materialized_window_ns"),
            "preprocessing": preprocessing,
        }
    )


def _selection_request_fingerprint(
    cache: "EnergyCache", config: dict[str, Any], source_root: Path
) -> str:
    """Fingerprint the reusable physical/photopeak cohort without patching selection_store.py."""
    return canonical_hash(
        {
            "selection_schema": PREPARED_SELECTION_VERSION,
            "source": source_signature(source_root),
            # raw-cache fingerprint includes waveform decoding, clipping, trigger
            # extraction and max-events. Therefore cached selection is safe to
            # reuse only when the quantities used by selection are unchanged.
            "raw_cache": cache.manifest.get("fingerprint"),
            "energy_channels": cache.manifest.get("energy_channels_one_based", []),
            "timing_channels": cache.manifest.get("timing_channels_one_based", []),
            "active_trigger_families": list(config.get("active_trigger_families", ("energy",))),
            "selection": copy.deepcopy(config.get("selection", {})),
            "photopeak": copy.deepcopy(config.get("photopeak", {"enabled": False})),
        }
    )


def _hash_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(indices, dtype=np.int64).tobytes()
    ).hexdigest()


def _copy_selected(
    source: np.ndarray, selected: np.ndarray, path: Path, chunk_size: int
) -> None:
    shape = (int(selected.size),) + tuple(int(value) for value in source.shape[1:])
    target = open_memmap(path, mode="w+", dtype=source.dtype, shape=shape)
    for start in range(0, selected.size, chunk_size):
        idx = selected[start : start + chunk_size]
        target[start : start + idx.size] = np.asarray(source[idx])
    target.flush()
    mmap = getattr(target, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(data, ddof=1)) if data.size > 1 else 0.0
    return center, sigma


def _absolute_rmse_limits(
    selection: dict[str, Any], channel_count: int
) -> np.ndarray | None:
    value = selection.get("baseline_rmse_max_mV")
    if value is None:
        return None
    if isinstance(value, (int, float, np.number)):
        limits = np.full(channel_count, float(value), dtype=np.float64)
    elif isinstance(value, (list, tuple)) and len(value) == channel_count:
        limits = np.asarray([float(item) for item in value], dtype=np.float64)
    else:
        raise ValueError(
            "preprocessing.selection.baseline_rmse_max_mV must be one number "
            "or one value per energy channel"
        )
    if np.any(~np.isfinite(limits)) or np.any(limits < 0.0):
        raise ValueError(
            "preprocessing.selection.baseline_rmse_max_mV values must be finite and >= 0"
        )
    return limits


def _trigger_anchor_array(cache: "EnergyCache", family: str) -> np.ndarray | None:
    if family == "energy":
        return getattr(cache, "energy_window_anchor_time_fs", None)
    if family == "timing":
        return getattr(cache, "timing_window_anchor_time_fs", None)
    return None


def _trigger_location_scale_ns(
    values_ns: np.ndarray, cache: "EnergyCache", family: str
) -> tuple[float, float]:
    """Robust trigger center/scale with a one-native-sample resolution floor."""
    data = np.asarray(values_ns, dtype=np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return float("nan"), float("nan")
    center = float(np.median(data))
    mad = float(np.median(np.abs(data - center)))
    sigma = 1.4826 * mad
    key = (
        "timing_native_sample_interval_ps"
        if family == "timing"
        else "native_sample_interval_ps"
    )
    interval_ps = cache.manifest.get(key)
    try:
        resolution_floor_ns = abs(float(interval_ps)) / 1000.0
    except (TypeError, ValueError):
        resolution_floor_ns = 0.0
    if not np.isfinite(resolution_floor_ns) or resolution_floor_ns <= 0.0:
        resolution_floor_ns = 1.0e-3  # 1 ps conservative fallback
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = resolution_floor_ns
    else:
        sigma = max(float(sigma), resolution_floor_ns)
    return center, float(sigma)


def _physical_photopeak_selection(
    cache: "EnergyCache", config: dict[str, Any], logger: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reusable permanent physical selection before any ML split.

    Order:
      1. finite energy amplitudes;
      2. two-channel energy photopeak;
      3. two-channel energy baseline-RMSE cut;
      4. trigger computability and absolute trigger-time outlier cut, restricted
         to channel families active in the experiment.
    """
    amplitudes_all = np.asarray(cache.amplitude_mV, dtype=np.float64)
    baseline_rmse_all = np.asarray(cache.noise_rms_mV, dtype=np.float64)
    trigger_all = np.asarray(cache.trigger_index, dtype=np.int64)
    selection = copy.deepcopy(config.get("selection", {}))

    finite_amplitude = np.all(np.isfinite(amplitudes_all), axis=1)
    valid = finite_amplitude.copy()
    summary: dict[str, Any] = {
        "scope": "physical_photopeak_noise_trigger_before_ml",
        "finite_amplitude_before_photopeak": int(np.count_nonzero(finite_amplitude)),
    }

    # PHOTOPEAK: both fits use the identical initial finite-amplitude cohort so
    # channel iteration order cannot alter the fit population.
    photopeak_cfg = copy.deepcopy(config.get("photopeak", {"enabled": False}))
    photopeak_rows: list[dict[str, Any]] = []
    if bool(photopeak_cfg.get("enabled", False)):
        fit_indices = np.flatnonzero(finite_amplitude)
        if fit_indices.size == 0:
            raise RuntimeError("No finite energy amplitudes are available for photopeak selection")
        photopeak_accept = finite_amplitude.copy()
        for position, channel_number in enumerate(
            cache.manifest.get("energy_channels_one_based", [1, 2])
        ):
            result = fit_photopeak(
                amplitudes_all[fit_indices, position],
                channel=int(channel_number),
                config=photopeak_cfg,
            )
            if not result.success:
                raise RuntimeError(
                    f"Photopeak fit failed for energy channel {channel_number}: {result.message}"
                )
            photopeak_accept &= photopeak_mask(amplitudes_all[:, position], result)
            photopeak_rows.append(result.as_dict())
        valid &= photopeak_accept
        logger.info(
            "Physical photopeak selection | retained=%d",
            int(np.count_nonzero(valid)),
        )
    summary["photopeak"] = photopeak_rows
    summary["events_after_photopeak"] = int(np.count_nonzero(valid))

    # BASELINE RMSE: prefer the requested absolute threshold.  Keep the old
    # robust-z option as a compatibility fallback for existing configurations.
    rmse_population = valid.copy()
    rmse_accept = np.ones(valid.shape, dtype=bool)
    rmse_rows: list[dict[str, Any]] = []
    absolute_limits = _absolute_rmse_limits(selection, baseline_rmse_all.shape[1])
    if absolute_limits is not None:
        for position, channel_number in enumerate(
            cache.manifest.get("energy_channels_one_based", [1, 2])
        ):
            limit = float(absolute_limits[position])
            channel_accept = (
                np.isfinite(baseline_rmse_all[:, position])
                & (baseline_rmse_all[:, position] <= limit)
            )
            rmse_accept &= channel_accept
            center, sigma = _robust_location_scale(
                baseline_rmse_all[rmse_population, position]
            )
            rmse_rows.append(
                {
                    "channel": int(channel_number),
                    "position": int(position),
                    "population_events": int(np.count_nonzero(rmse_population)),
                    "median_rmse_mV": center,
                    "robust_sigma_mV": sigma,
                    "mode": "absolute_threshold",
                    "upper_limit_mV": limit,
                    "rejected_from_photopeak": int(
                        np.count_nonzero(rmse_population & ~channel_accept)
                    ),
                }
            )
        valid &= rmse_accept
        rmse_filter = {
            "enabled": True,
            "mode": "absolute_threshold",
            "configured_max_mV": absolute_limits.tolist(),
            "channels": rmse_rows,
            "events_after_filter": int(np.count_nonzero(valid)),
        }
    else:
        rmse_z_max = selection.get("baseline_rmse_robust_z", 5.0)
        if rmse_z_max is not None:
            rmse_z_max = float(rmse_z_max)
            if not np.isfinite(rmse_z_max) or rmse_z_max <= 0.0:
                raise ValueError(
                    "preprocessing.selection.baseline_rmse_robust_z must be positive or null"
                )
            for position, channel_number in enumerate(
                cache.manifest.get("energy_channels_one_based", [1, 2])
            ):
                center, sigma = _robust_location_scale(
                    baseline_rmse_all[rmse_population, position]
                )
                if not np.isfinite(center):
                    raise RuntimeError(
                        f"No finite baseline RMSE values for energy channel {channel_number}"
                    )
                upper = center + rmse_z_max * sigma if sigma > 0.0 else center
                channel_accept = (
                    np.isfinite(baseline_rmse_all[:, position])
                    & (baseline_rmse_all[:, position] <= upper)
                )
                rmse_accept &= channel_accept
                rmse_rows.append(
                    {
                        "channel": int(channel_number),
                        "position": int(position),
                        "population_events": int(np.count_nonzero(rmse_population)),
                        "median_rmse_mV": center,
                        "robust_sigma_mV": sigma,
                        "mode": "robust_z_fallback",
                        "robust_z_max": rmse_z_max,
                        "upper_limit_mV": float(upper),
                        "rejected_from_photopeak": int(
                            np.count_nonzero(rmse_population & ~channel_accept)
                        ),
                    }
                )
            valid &= rmse_accept
            rmse_filter = {
                "enabled": True,
                "mode": "robust_z_fallback",
                "robust_z_max": rmse_z_max,
                "channels": rmse_rows,
                "events_after_filter": int(np.count_nonzero(valid)),
            }
        else:
            rmse_filter = {
                "enabled": False,
                "mode": "disabled",
                "channels": [],
                "events_after_filter": int(np.count_nonzero(valid)),
            }
    summary["baseline_rmse_filter"] = rmse_filter
    logger.info(
        "Baseline RMSE filter | retained=%d",
        int(np.count_nonzero(valid)),
    )

    # Energy trigger index must exist because energy channels are always used for
    # photopeak and window materialization.
    valid &= np.all(trigger_all >= 0, axis=1)
    trigger_range = selection.get("energy_trigger_index_range")
    if trigger_range is not None:
        if not isinstance(trigger_range, (list, tuple)) or len(trigger_range) != 2:
            raise ValueError("preprocessing.selection.energy_trigger_index_range must be [low, high]")
        low, high = int(trigger_range[0]), int(trigger_range[1])
        valid &= np.all((trigger_all > low) & (trigger_all < high), axis=1)
    summary["events_after_trigger_computability"] = int(np.count_nonzero(valid))

    # ABSOLUTE TRIGGER-TIME REJECTION.  The interval is estimated only from the
    # post-photopeak/post-noise population and only for active experiment family
    # channels.  Values are absolute on the original oscilloscope time axis.
    trigger_z = selection.get("trigger_time_robust_z", 5.0)
    trigger_rows: list[dict[str, Any]] = []
    active_families = tuple(config.get("active_trigger_families", ("energy",)))
    if trigger_z is not None:
        trigger_z = float(trigger_z)
        if not np.isfinite(trigger_z) or trigger_z <= 0.0:
            raise ValueError(
                "preprocessing.selection.trigger_time_robust_z must be positive or null"
            )
        trigger_population = valid.copy()
        trigger_accept = np.ones(valid.shape, dtype=bool)
        for family in active_families:
            anchors = _trigger_anchor_array(cache, family)
            if anchors is None:
                raise RuntimeError(
                    f"{family} is active but its trigger-anchor array is unavailable"
                )
            anchors = np.asarray(anchors, dtype=np.int64)
            channels = cache.manifest.get(
                f"{family}_channels_one_based",
                cache.manifest.get("energy_channels_one_based", [1, 2]),
            )
            for position in range(anchors.shape[1]):
                values_fs = anchors[:, position]
                usable = (
                    trigger_population
                    & (values_fs != int(INVALID_TIME_FS))
                )
                values_ns = values_fs[usable].astype(np.float64) / 1.0e6
                center, sigma = _trigger_location_scale_ns(values_ns, cache, family)
                if not np.isfinite(center):
                    raise RuntimeError(
                        f"No finite absolute search-trigger times for {family} channel {position + 1}"
                    )
                half_width = trigger_z * sigma if sigma > 0.0 else 0.0
                lower = center - half_width
                upper = center + half_width
                all_ns = values_fs.astype(np.float64) / 1.0e6
                valid_anchor = values_fs != int(INVALID_TIME_FS)
                if sigma > 0.0:
                    channel_accept = valid_anchor & (all_ns >= lower) & (all_ns <= upper)
                else:
                    channel_accept = valid_anchor & np.isclose(all_ns, center, rtol=0.0, atol=1e-12)
                trigger_accept &= channel_accept
                channel_number = int(channels[position]) if position < len(channels) else position + 1
                trigger_rows.append(
                    {
                        "family": str(family),
                        "channel": channel_number,
                        "position": int(position),
                        "population_events": int(np.count_nonzero(trigger_population)),
                        "median_ns": center,
                        "robust_sigma_ns": sigma,
                        "robust_z_max": trigger_z,
                        "lower_ns": float(lower),
                        "upper_ns": float(upper),
                        "rejected_from_population": int(
                            np.count_nonzero(trigger_population & ~channel_accept)
                        ),
                    }
                )
        valid &= trigger_accept
        trigger_summary = {
            "enabled": True,
            "reference": "absolute_time_from_original_oscilloscope_acquisition_origin",
            "robust_z_max": trigger_z,
            "active_families": list(active_families),
            "channels": trigger_rows,
            "events_after_filter": int(np.count_nonzero(valid)),
        }
    else:
        trigger_summary = {
            "enabled": False,
            "reference": "absolute_time_from_original_oscilloscope_acquisition_origin",
            "active_families": list(active_families),
            "channels": [],
            "events_after_filter": int(np.count_nonzero(valid)),
        }
    summary["trigger_time_filter"] = trigger_summary

    selected = np.flatnonzero(valid).astype(np.int64)
    summary["selected_events"] = int(selected.size)
    return selected, summary


def _finite_waveform_rows(
    windows: np.ndarray, *, chunk_size: int = 2048
) -> np.ndarray:
    """Check waveform finiteness without loading the whole memmap into RAM."""
    n_events = int(windows.shape[0])
    output = np.empty(n_events, dtype=bool)
    chunk = max(1, int(chunk_size))
    for start in range(0, n_events, chunk):
        stop = min(start + chunk, n_events)
        block = np.asarray(windows[start:stop])
        output[start:stop] = np.all(np.isfinite(block), axis=(1, 2))
    return output


def _family_cache_validity(cache: "EnergyCache", family: str) -> np.ndarray:
    """Preprocessing validity depends only on the prepared signal and trigger anchor.

    LED/CFD timestamps are intentionally INVALID_TIME_FS at this stage because
    their thresholds are selected later by standard_methods. They must therefore
    never enter permanent preprocessing event selection.
    """
    n = int(cache.event_id.size)
    valid = np.ones(n, dtype=bool)

    if family == "energy":
        windows = getattr(cache, "windows_mV", None)
        anchor = getattr(cache, "energy_window_anchor_time_fs", None)
    elif family == "timing":
        windows = getattr(cache, "timing_windows_mV", None)
        anchor = getattr(cache, "timing_window_anchor_time_fs", None)
    else:
        raise ValueError(f"Unknown waveform family: {family}")

    if windows is None or anchor is None:
        return np.zeros(n, dtype=bool)

    valid &= _finite_waveform_rows(windows)
    valid &= np.all(
        np.asarray(anchor, dtype=np.int64) != int(INVALID_TIME_FS),
        axis=1,
    )
    return valid


def _dataset_level_selection(
    cache: "EnergyCache",
    config: dict[str, Any],
    logger: Any,
    *,
    physical_selected: np.ndarray | None = None,
    physical_summary: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    valid = np.zeros(int(cache.event_id.size), dtype=bool)
    if physical_selected is None:
        physical_selected, physical_summary = _physical_photopeak_selection(cache, config, logger)
    valid[np.asarray(physical_selected, dtype=np.int64)] = True

    # Do not use cache.valid here: current data.py includes the historical
    # timing-LED-aligned duplicate energy matrix in that aggregate flag.  Format
    # v7 deliberately stores only the canonical search-trigger window, so we
    # check exactly the families/arrays that v7 actually requires.
    active_families = tuple(config.get("active_trigger_families", ("energy",)))
    required_families = set(active_families)
    required_families.add("energy")  # energy is always needed for photopeak
    family_counts: dict[str, int] = {}
    for family in sorted(required_families):
        family_valid = _family_cache_validity(cache, family)
        valid &= family_valid
        family_counts[family] = int(np.count_nonzero(family_valid))

    selection = copy.deepcopy(config.get("selection", {}))
    minimum = int(
        selection.get(
            "minimum_events",
            selection.get("minimum_events_per_split", 100),
        )
    )
    selected = np.flatnonzero(valid).astype(np.int64)
    summary: dict[str, Any] = {
        "scope": "physical_and_canonical_signal_quality_before_ml_split",
        "physical_selection": physical_summary or {},
        "canonical_family_valid_counts": family_counts,
        "valid_after_waveform_preparation": int(np.count_nonzero(valid)),
        "selected_events": int(selected.size),
        "canonical_window_alignment": "native_search_trigger_sample_then_relative_baseline",
        "timing_aligned_energy_duplicate_required": False,
        "search_time_outlier_rejection": {
            "permanent_absolute_trigger_stage": bool(
                (physical_summary or {}).get("trigger_time_filter", {}).get("enabled", False)
            ),
            "development_relative_timing_stage": "unchanged_in_study_pipeline",
        },
    }
    if selected.size < minimum:
        raise RuntimeError(
            f"Only {selected.size} events remain after dataset preparation; need {minimum}"
        )
    return selected, summary


def _denoise_windows(
    source: np.ndarray,
    destination: Path,
    *,
    relative_time_ps: np.ndarray,
    config: dict[str, Any],
    chunk_size: int,
) -> None:
    values = source
    if values.ndim != 3:
        raise ValueError("Waveform array must have shape [event, detector, sample]")
    times = np.asarray(relative_time_ps, dtype=np.float64)
    if times.size < 2:
        raise ValueError("Need at least two time samples for denoising")
    interval_s = float(np.median(np.diff(times))) * 1e-12
    fs = 1.0 / interval_s
    cutoff_hz = float(config["cutoff_GHz"]) * 1e9
    if not 0.0 < cutoff_hz < 0.5 * fs:
        raise ValueError("Denoising cutoff must be below Nyquist")
    order = int(config.get("order", 4))
    sos = butter(order, cutoff_hz, btype="lowpass", fs=fs, output="sos")
    target = open_memmap(destination, mode="w+", dtype=np.float32, shape=values.shape)
    for start in range(0, values.shape[0], chunk_size):
        stop = min(start + chunk_size, values.shape[0])
        block = np.asarray(values[start:stop], dtype=np.float64)
        zero_count = min(
            int(np.count_nonzero(sos[:, 2] == 0.0)),
            int(np.count_nonzero(sos[:, 5] == 0.0)),
        )
        default_padlen = 3 * (2 * int(sos.shape[0]) + 1 - zero_count)
        padlen = min(default_padlen, max(0, block.shape[-1] - 1))
        filtered = sosfiltfilt(sos, block, axis=-1, padlen=padlen)
        target[start:stop] = np.asarray(filtered, dtype=np.float32)
    target.flush()
    mmap = getattr(target, "_mmap", None)
    if mmap is not None:
        mmap.close()


def _prepared_fingerprint(
    cache: "EnergyCache", selected: np.ndarray, config: dict[str, Any]
) -> str:
    return canonical_hash(
        {
            "format_version": DATASET_FORMAT_VERSION,
            "selection_version": PREPARED_SELECTION_VERSION,
            "raw_cache": cache.manifest["fingerprint"],
            "selected_hash": _hash_indices(selected),
            "selection": config.get("selection", {}),
            "photopeak": config.get("photopeak", {}),
            "active_trigger_families": list(config.get("active_trigger_families", ("energy",))),
            "denoising": config.get("denoising", {}),
            "permanent_window_storage": "single_search_trigger_referenced_window_with_relative_baseline_v2",
        }
    )


def materialize_selected_dataset(
    cache: "EnergyCache",
    *,
    output: Path,
    config: dict[str, Any],
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    selection_store_root = Path(
        config.get("selection_store_dir", output.parent / "selected_events")
    )
    source_root = Path(config["source_root"])
    selection_fp = str(config.get("selection_request_fingerprint", ""))
    if not selection_fp:
        selection_fp = _selection_request_fingerprint(cache, config, source_root)

    physical_selected, physical_summary, physical_store = load_or_compute_selection(
        root=selection_store_root,
        root_file=source_root,
        fingerprint=selection_fp,
        rebuild=bool(config.get("rebuild_selection", False)),
        compute=lambda: _physical_photopeak_selection(cache, config, logger),
        logger=logger,
    )
    selected, selection_summary = _dataset_level_selection(
        cache,
        config,
        logger,
        physical_selected=physical_selected,
        physical_summary=physical_summary,
    )
    selection_summary["physical_selection_store"] = str(physical_store)
    selection_summary["physical_selection_fingerprint"] = selection_fp

    fingerprint = _prepared_fingerprint(cache, selected, config)
    manifest_path = output / "manifest.json"
    if output.is_dir() and not rebuild and manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
            if manifest.get("fingerprint") == fingerprint:
                logger.info("Reusing permanent prepared dataset: %s", output)
                return load_prepared_dataset(output)
        except Exception:
            pass

    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    chunk_size = max(1, int(config.get("materialization_chunk_size", 2048)))

    for name in _COPY_ARRAYS:
        source = getattr(cache, name, None)
        if source is not None:
            _copy_selected(source, selected, temporary / f"{name}.npy", chunk_size)

    np.save(
        temporary / "relative_time_ps.npy",
        np.asarray(cache.relative_time_ps, dtype=np.float64),
    )
    if cache.timing_relative_time_ps is not None and cache.timing_windows_mV is not None:
        np.save(
            temporary / "timing_relative_time_ps.npy",
            np.asarray(cache.timing_relative_time_ps, dtype=np.float64),
        )

    denoise_cfg = copy.deepcopy(config.get("denoising", {}))
    denoise_enabled = bool(denoise_cfg.get("enabled", False))
    if denoise_enabled:
        _denoise_windows(
            np.load(temporary / "windows_mV.npy", mmap_mode="r"),
            temporary / "denoised_windows_mV.npy",
            relative_time_ps=np.asarray(cache.relative_time_ps),
            config=denoise_cfg,
            chunk_size=chunk_size,
        )

    diagnostics_dir = temporary / "diagnostics"
    diagnostics = write_preprocessing_diagnostics(
        cache,
        diagnostics_dir,
        physical_summary=physical_summary,
        active_families=tuple(config.get("active_trigger_families", ("energy",))),
        dpi=int(config.get("diagnostic_dpi", 180)),
        logger=logger,
    )

    raw_cache_manifest = copy.deepcopy(cache.manifest)
    # data.py remains backward-compatible and may describe its temporary
    # compatibility arrays using the old target-aligned terminology.  The
    # canonical windows produced by this signal.py are search-trigger anchored.
    raw_cache_manifest["canonical_window_alignment_source"] = (
        "native_search_trigger_sample"
    )
    raw_cache_manifest["ml_window_alignment_source"] = (
        "native_search_trigger_sample"
    )
    if cache.timing_windows_mV is not None:
        raw_cache_manifest["timing_window_alignment_source"] = (
            "native_search_trigger_sample"
        )

    waveform_manifest = raw_cache_manifest.get("preprocessing", {}).get("waveform", {})
    vertical_limits = None
    if isinstance(waveform_manifest, dict):
        for key in ("vertical_scale_limit_mV", "vertical_scale_limit", "vertical scale limit"):
            if key in waveform_manifest:
                vertical_limits = waveform_manifest.get(key)
                break

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "request_fingerprint": str(config.get("request_fingerprint", "")),
        "name": str(config.get("name", output.name)),
        "role": "prepared_full_file",
        "subset_kind": "dataset_level_selected",
        "source_root": str(config["source_root"]),
        "true_tof_ps": float(config["true_tof_ps"]),
        "event_count": int(selected.size),
        "input_length": int(cache.windows_mV.shape[-1]),
        "selection": selection_summary,
        "raw_cache_manifest": raw_cache_manifest,
        "energy_channels_one_based": cache.manifest.get("energy_channels_one_based", []),
        "timing_channels_one_based": cache.manifest.get("timing_channels_one_based", []),
        "active_trigger_families": list(config.get("active_trigger_families", ("energy",))),
        "timing_channel_waveforms_saved": (
            cache.timing_windows_mV is not None
            and (temporary / "timing_windows_mV.npy").is_file()
        ),
        "timing_aligned_energy_waveforms_saved": False,
        "denoised_waveforms_saved": denoise_enabled,
        "denoised_energy_waveforms_saved": denoise_enabled,
        "denoised_timing_waveforms_saved": False,
        "denoising_scope": "energy_channels_only",
        "denoising": denoise_cfg if denoise_enabled else {"enabled": False},
        "waveform_grid": cache.manifest.get("waveform_grid", "native_samples"),
        "native_sample_interval_ps": cache.manifest.get("native_sample_interval_ps"),
        "timing_native_sample_interval_ps": cache.manifest.get("timing_native_sample_interval_ps"),
        "baseline_handling": str(
            config.get("baseline_handling", "quality_only_no_shift_v1")
        ),
        "baseline_quality_metric": "rmse_on_trigger_relative_materialized_baseline_window",
        "vertical_scale_limit_mV": vertical_limits,
        "led_timestamp_source": "selected_downstream_from_standard_methods_candidates",
        "cfd_timestamp_source": "selected_downstream_from_standard_methods_candidates",
        "ml_window_alignment_source": "native_search_trigger_sample",
        "window_anchor_timestamps_saved": True,
        "single_canonical_energy_waveform_matrix": True,
        "timing_target_energy_alignment": "lazy_from_canonical_anchor",
        "dataset_selection_is_independent_of_ml_split": True,
        "arrays_are_post_selection": True,
        "ml_split_materialized": False,
        "diagnostics_directory": "diagnostics",
        "diagnostic_files": diagnostics,
    }
    atomic_json(temporary / "manifest.json", manifest)

    if output.exists():
        shutil.rmtree(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, output)
    logger.info(
        "Permanent prepared dataset written | %s | events=%d",
        output,
        selected.size,
    )
    return load_prepared_dataset(output)


def _raw_preprocess_config(
    study: dict[str, Any], root_file: Path, cache_dir: Path
) -> dict[str, Any]:
    preprocessing = copy.deepcopy(study["preprocessing"])
    common = copy.deepcopy(preprocessing.get("common", {}))
    energy = copy.deepcopy(common)
    energy.update(copy.deepcopy(preprocessing.get("energy", {})))
    timing = copy.deepcopy(common)
    timing.update(copy.deepcopy(preprocessing.get("timing", {})))

    # Denoising is excluded from ROOT conversion. Permanent denoised energy
    # windows, when requested, are derived from the canonical raw materialized
    # window after event selection.
    energy["denoising"] = {"enabled": False}
    timing["denoising"] = {"enabled": False}

    # Baseline is estimated only after trigger-centered materialization.
    # Fixed sample-count baselines and fixed preprocessing LED thresholds are
    # deliberately removed before constructing the raw-cache configuration.
    for family_cfg in (energy, timing):
        family_cfg.pop("baseline_samples", None)
        family_cfg.pop("led_threshold_mV", None)

    energy["baseline_handling"] = (
        "materialized_trigger_relative_mean_subtracted_v2"
        if bool(energy.get("subtract_baseline", True))
        else "materialized_trigger_relative_quality_only_v2"
    )
    timing["baseline_handling"] = (
        "materialized_trigger_relative_mean_subtracted_v2"
        if bool(timing.get("subtract_baseline", True))
        else "materialized_trigger_relative_quality_only_v2"
    )

    # Participates in data.py's raw-cache fingerprint so old caches cannot be
    # reused with the new trigger -> materialize -> baseline semantics.
    energy["canonical_window_alignment_version"] = (
        "search_trigger_then_materialized_baseline_v2"
    )
    timing["canonical_window_alignment_version"] = (
        "search_trigger_then_materialized_baseline_v2"
    )

    materialized = preprocessing.get("materialized_window_ns") or {
        "before": max(float(window["before_ns"]) for window in study["windows_ns"]),
        "after": max(float(window["after_ns"]) for window in study["windows_ns"]),
    }
    energy["ml_window_ns"] = {
        "before": float(materialized["before"]),
        "after": float(materialized["after"]),
    }
    timing["ml_window_ns"] = {
        "before": float(materialized["before"]),
        "after": float(materialized["after"]),
    }

    active_families = _active_trigger_families(study)
    timing["enabled"] = "timing" in active_families
    energy["timing_channel_led"] = timing

    return {
        "data": {
            "input_root": str(root_file),
            "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),
        },
        "channels": copy.deepcopy(study["data"]["channels"]),
        "waveform": energy,
        "io": copy.deepcopy(
            preprocessing.get(
                "io",
                {
                    "step_size": "128 MB",
                    "max_events": 0,
                    "progress_every": 1000,
                },
            )
        ),
        "parallelization": copy.deepcopy(
            preprocessing.get(
                "parallelization",
                {
                    "preprocessing_backend": "process",
                    "preprocessing_workers": 0,
                    "preprocessing_chunksize": 8,
                },
            )
        ),
        "cache": {"raw_cache_dir": str(cache_dir)},
    }


def prepare_file_dataset(
    study: dict[str, Any],
    root_file: Path,
    *,
    rebuild: bool,
    logger: Any,
) -> PreparedDataset:
    root_id = root_file.stem
    prepared_root = Path(study["preprocessing"]["prepared_dir"])
    output = prepared_root / root_id
    request_fingerprint = _preparation_request_fingerprint(study, root_file)

    if output.is_dir() and not rebuild:
        manifest_path = output / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = read_json(manifest_path)
                if manifest.get("request_fingerprint") == request_fingerprint:
                    logger.info(
                        "Reusing permanent prepared dataset without ROOT reconversion: %s",
                        output,
                    )
                    loaded = load_prepared_dataset(output)
                    updated_manifest = dict(loaded.manifest)
                    updated_manifest["true_tof_ps"] = float(
                        study["data"].get("true_tof_ps", 0.0)
                    )
                    return replace(loaded, manifest=updated_manifest)
            except Exception as exc:
                logger.warning(
                    "Cannot reuse permanent prepared dataset %s: %s", output, exc
                )

    raw_cache_dir = prepared_root / ".raw_cache" / root_id
    raw_cfg = _raw_preprocess_config(study, root_file, raw_cache_dir)
    cache_cfg = {
        "channels": raw_cfg["channels"],
        "waveform": raw_cfg["waveform"],
        "io": raw_cfg["io"],
        "parallelization": raw_cfg["parallelization"],
    }
    from .data import prepare_energy_cache

    cache = prepare_energy_cache(
        root_file,
        raw_cache_dir,
        cache_cfg,
        rebuild=rebuild,
        logger=logger,
    )

    permanent_cfg = {
        "name": root_id,
        "source_root": str(root_file),
        "request_fingerprint": request_fingerprint,
        "baseline_handling": raw_cfg["waveform"].get(
            "baseline_handling", "quality_only_no_shift_v1"
        ),
        "true_tof_ps": float(study["data"].get("true_tof_ps", 0.0)),
        "selection_store_dir": str(
            study["preprocessing"].get(
                "selection_store_dir",
                Path(study["preprocessing"]["prepared_dir"]).parent
                / "selected_events",
            )
        ),
        "rebuild_selection": bool(rebuild),
        "selection": copy.deepcopy(
            study["preprocessing"].get("selection", {})
        ),
        "photopeak": copy.deepcopy(
            study["preprocessing"].get("photopeak", {"enabled": False})
        ),
        "denoising": copy.deepcopy(
            study["preprocessing"].get("denoising", {"enabled": False})
        ),
        "materialization_chunk_size": int(
            study["preprocessing"].get("materialization_chunk_size", 2048)
        ),
        "active_trigger_families": _active_trigger_families(study),
        "diagnostic_dpi": int(study.get("reporting", {}).get("dpi", 180)),
    }
    permanent_cfg["selection_request_fingerprint"] = _selection_request_fingerprint(
        cache, permanent_cfg, root_file
    )

    dataset = materialize_selected_dataset(
        cache,
        output=output,
        config=permanent_cfg,
        rebuild=rebuild,
        logger=logger,
    )

    if bool(study["preprocessing"].get("cleanup_raw_cache", True)):
        # Close source memmaps before deleting their directory on Windows.
        del cache
        shutil.rmtree(raw_cache_dir, ignore_errors=True)

    manifest = dict(dataset.manifest)
    manifest["true_tof_ps"] = float(study["data"].get("true_tof_ps", 0.0))
    return replace(dataset, manifest=manifest)


def plot_prepared_signal_examples(
    dataset: PreparedDataset,
    destination: Path,
    *,
    dpi: int = 180,
) -> None:
    if dataset.event_id.size == 0:
        return
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None]] = [
        (
            "Energy ch. 1",
            dataset.relative_time_ps,
            np.asarray(dataset.windows_mV[0, 0]),
            None
            if dataset.denoised_windows_mV is None
            else np.asarray(dataset.denoised_windows_mV[0, 0]),
        ),
        (
            "Energy ch. 2",
            dataset.relative_time_ps,
            np.asarray(dataset.windows_mV[0, 1]),
            None
            if dataset.denoised_windows_mV is None
            else np.asarray(dataset.denoised_windows_mV[0, 1]),
        ),
    ]
    if (
        dataset.timing_windows_mV is not None
        and dataset.timing_relative_time_ps is not None
    ):
        rows.extend(
            [
                (
                    "Timing ch. 1",
                    dataset.timing_relative_time_ps,
                    np.asarray(dataset.timing_windows_mV[0, 0]),
                    None
                    if dataset.denoised_timing_windows_mV is None
                    else np.asarray(dataset.denoised_timing_windows_mV[0, 0]),
                ),
                (
                    "Timing ch. 2",
                    dataset.timing_relative_time_ps,
                    np.asarray(dataset.timing_windows_mV[0, 1]),
                    None
                    if dataset.denoised_timing_windows_mV is None
                    else np.asarray(dataset.denoised_timing_windows_mV[0, 1]),
                ),
            ]
        )
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(10.5, 2.7 * len(rows)), squeeze=False
    )
    for axis, (title, time_ps, raw, denoised) in zip(axes[:, 0], rows):
        axis.plot(
            np.asarray(time_ps, dtype=np.float64) / 1000.0,
            raw,
            linewidth=1.0,
            label="raw",
        )
        if denoised is not None:
            axis.plot(
                np.asarray(time_ps, dtype=np.float64) / 1000.0,
                denoised,
                linewidth=1.0,
                label="denoised",
            )
            axis.legend(loc="best")
        axis.set_title(title)
        axis.set_xlabel("Time relative to native search-trigger sample [ns]")
        axis.set_ylabel("Voltage [mV]")
        axis.minorticks_on()
        axis.grid(True, which="major", alpha=0.35)
        axis.grid(True, which="minor", alpha=0.15)
    fig.suptitle(
        "Prepared waveform example | "
        f"{Path(dataset.manifest.get('source_root', dataset.directory)).name}"
    )
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def input_variant_dataset_view(
    dataset: PreparedDataset, variant: str
) -> PreparedDataset:
    """Return a zero-copy raw/denoised waveform view of one prepared dataset."""
    key = str(variant).strip().lower()
    if key == "raw":
        manifest = dict(dataset.manifest)
        manifest["ml_input_variant"] = "raw"
        return replace(dataset, manifest=manifest)
    if key != "denoised":
        raise ValueError("ML input variant must be 'raw' or 'denoised'")
    if dataset.denoised_windows_mV is None:
        raise ValueError(
            f"Dataset {dataset.directory} has no materialized denoised waveforms"
        )
    manifest = dict(dataset.manifest)
    manifest["ml_input_variant"] = "denoised"
    return replace(
        dataset,
        manifest=manifest,
        windows_mV=dataset.denoised_windows_mV,
        timing_aligned_energy_windows_mV=(
            dataset.denoised_timing_aligned_energy_windows_mV
            if dataset.denoised_timing_aligned_energy_windows_mV is not None
            else dataset.timing_aligned_energy_windows_mV
        ),
        timing_windows_mV=(
            dataset.denoised_timing_windows_mV
            if dataset.denoised_timing_windows_mV is not None
            else dataset.timing_windows_mV
        ),
    )
