"""Shared data-loading helpers for waveform imputation experiments."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from waveform_analysis.data.results import (
    resolve_run_config,
    read_run_manifest,
)
from waveform_analysis.data.prepared_cache import PreparedCacheLookup

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
    ``cache_root`` explicitly selects the portable named prepared cache when
    provided; stored run paths are otherwise used as a fallback.
    """
    matches = []
    # Scan run configs directly: discover_runs() opens every manifest and fails
    # when a separate model's result directory is only partially populated.
    # Missing manifests must not block this waveform-only experiment.
    for config_path in sorted(Path(results).expanduser().rglob("metadata/config.json")):
        directory = config_path.parent.parent
        if not (directory / "metadata" / "manifest.json").is_file():
            continue
        cfg = resolve_run_config(directory)
        if (
            "wide" in str(cfg.get("window_name", "")).lower()
            and cfg.get("mode") == f"{channel}_to_{channel}"
        ):
            manifest = read_run_manifest(directory)
            if manifest.get("status") == "complete":
                matches.append((directory, cfg, manifest))
    if not matches:
        raise ValueError(f"No wide {channel} runs found under {results}")
    directory, cfg, manifest = matches[0]
    cfg = dict(cfg)
    cfg["preprocessing"] = dict(cfg["preprocessing"])
    prep = cfg["preprocessing"]
    if cache_root is not None:
        root = Path(cache_root).expanduser().resolve()
        prep["named_cache_root"] = str(root)
        prep["cache_dir"] = str(root / "preprocessing")
        prep["artifact_root"] = str(root / "artifact_store")
    elif "artifact_root" not in prep:
        raise ValueError(
            "This result has no stored artifact_root; supply --cache-root "
            "pointing to the prepared waveform cache"
        )
    # Keep cache_root in the public signature for older callers, but canonical
    # prepared datasets are resolved through immutable manifest references.
    lookup = PreparedCacheLookup()
    dataset = lookup.find("development", cfg, manifest)
    control_dataset = lookup.find("control", cfg, manifest) if include_control else None
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
