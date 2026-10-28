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


def load_wide_development(results, channel, cache_root=None):
    """Load prepared wide-window development waveforms for one channel."""
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
    cache = _cache_root(cfg, directory, cache_root, roles=("development",))
    dataset = PreparedCacheLookup().find(cache, "development", cfg, manifest)
    waves = dataset.energy_windows if channel == "energy" else dataset.timing_windows
    times = dataset.energy_time_ps if channel == "energy" else dataset.timing_time_ps
    if waves is None or times is None or waves.ndim != 3 or waves.shape[1] != 2:
        raise ValueError("Expected development waveforms [events, 2 detectors, samples]")
    return waves, np.asarray(times), directory, dataset.manifest
