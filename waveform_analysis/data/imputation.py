"""Shared data-loading helpers for waveform imputation experiments."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from waveform_analysis.data.results import (
    discover_runs,
    resolve_run_config,
    read_run_manifest,
)
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root

ROOT = Path(__file__).resolve().parents[2]


def root_relative(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_wide_development(results, channel, cache_root=None, *, return_transform=False,
                          include_control=False, include_sample_mask=False):
    """Load prepared wide-window development waveforms for one channel.

    If ``return_transform`` is true, also return the frozen input transform so
    diagnostics can be expressed in the original configured signal units.
    ``include_sample_mask`` additionally returns the retained-sample mask from
    preprocessing (all samples are valid for energy-channel inputs).
    """
    matches = []
    for run in discover_runs(results):
        directory = Path(run["directory"])
        cfg = resolve_run_config(directory)
        if (
            "wide" in str(cfg.get("window_name", "")).lower()
            and cfg.get("mode") == f"{channel}_to_{channel}"
        ):
            matches.append((directory, cfg))
    if not matches:
        raise ValueError(f"No wide {channel} runs found under {results}")
    directory, cfg = matches[0]
    manifest = read_run_manifest(directory)
    roles = ("development", "control") if include_control else ("development",)
    cache = _cache_root(cfg, directory, cache_root, roles=roles)
    lookup = PreparedCacheLookup()
    dataset = lookup.find(cache, "development", cfg, manifest)
    control_dataset = lookup.find(cache, "control", cfg, manifest) if include_control else None
    waves = dataset.energy_windows if channel == "energy" else dataset.timing_windows
    times = dataset.energy_time_ps if channel == "energy" else dataset.timing_time_ps
    transform = dataset.energy_transform if channel == "energy" else dataset.timing_transform
    control_waves = None
    if control_dataset is not None:
        control_waves = control_dataset.energy_windows if channel == "energy" else control_dataset.timing_windows
        control_times = control_dataset.energy_time_ps if channel == "energy" else control_dataset.timing_time_ps
        control_transform = control_dataset.energy_transform if channel == "energy" else control_dataset.timing_transform
        if control_waves is None or control_times is None or not np.array_equal(times, control_times):
            raise ValueError("Control and development waveform grids differ")
        if transform is None or control_transform is None or any(
            not np.array_equal(getattr(transform, name), getattr(control_transform, name))
            for name in ("minimum", "maximum")
        ):
            raise ValueError("Control and development waveform transforms differ")
    if waves is None or times is None or waves.ndim != 3 or waves.shape[1] != 2:
        raise ValueError("Expected development waveforms [events, 2 detectors, samples]")
    if include_sample_mask and not return_transform:
        raise ValueError("include_sample_mask requires return_transform=True")
    if return_transform:
        if transform is None:
            raise ValueError(f"Missing frozen {channel} input transform")
        extra = (control_waves,) if include_control else ()
        if include_sample_mask:
            if channel == "timing":
                mask = dataset.sample_mask
                if mask is None:
                    raise ValueError("Timing-channel sample mask is unavailable")
                mask = np.asarray(mask, dtype=bool)
                if mask.shape != (len(times),):
                    raise ValueError("Timing-channel sample mask does not match waveform grid")
            else:
                mask = np.ones(len(times), dtype=bool)
            extra += (mask,)
        return (waves, np.asarray(times), directory, dataset.manifest, transform, *extra)
    return waves, np.asarray(times), directory, dataset.manifest
