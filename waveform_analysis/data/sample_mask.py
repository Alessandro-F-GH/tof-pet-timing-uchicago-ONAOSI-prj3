"""Persist one control-fitted clipping mask for every timing input grid."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.dataset import PreparedDataset
from waveform_analysis.data.storage import _atomic_npz
from waveform_analysis.signal.sample_mask import SAMPLE_CONSTANT_FRACTION, control_clipping_sample_mask

SAMPLE_MASK_VERSION = 1


def attach_control_sample_mask(
    dataset: PreparedDataset, control: PreparedDataset | None, *, cache_root: Path,
    artifact_root: Path | None = None, logger=None,
) -> PreparedDataset:
    """Attach/persist the frozen mask without modifying stored full-grid waveforms.

    Timing masks are learned only from prepared control events on the same window
    and subsampled time axis. Energy uses an all-true mask without fitting.
    """
    mode = dataset.manifest["mode"]
    family = dataset.manifest["family"]
    time = np.asarray(getattr(dataset, f"{family}_time_ps"), dtype=np.float64)
    n_control = 0
    control_fp = None
    if mode == "timing_to_timing":
        if control is None or control.manifest.get("dataset_role") != "control":
            raise ValueError("Timing sample masking requires the control dataset")
        if not np.array_equal(time, np.asarray(control.timing_time_ps)):
            raise ValueError("Control and target timing grids differ; cannot transfer sample mask")
        control_fp = control.manifest["fingerprint"]
        n_control = control.n_events
    identity = canonical_hash({
        "version": SAMPLE_MASK_VERSION, "mode": mode,
        "control_prepared_fingerprint": control_fp,
        "time_ps": time.tolist(), "fraction": SAMPLE_CONSTANT_FRACTION,
    })
    directory = Path(cache_root) / "sample_masks" / identity[:16]
    metadata_path = directory / "manifest.json"
    artifact_path = directory / "mask.npz"
    mask = None
    if artifact_path.is_file() and metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with np.load(artifact_path) as stored:
                candidate = np.asarray(stored["mask"], dtype=bool)
                if (metadata.get("identity") == identity and candidate.shape == time.shape
                        and candidate.any() and np.array_equal(stored["time_ps"], time)):
                    mask = candidate
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass
    if mask is None:
        mask = (control_clipping_sample_mask(control.timing_windows)
                if mode == "timing_to_timing" else np.ones(time.size, dtype=bool))
        _atomic_npz(artifact_path, mask=mask, time_ps=time)
        atomic_json(metadata_path, {
            "version": SAMPLE_MASK_VERSION, "identity": identity,
            "fit_role": "control" if control_fp is not None else "not_applied",
            "control_prepared_fingerprint": control_fp, "control_events": n_control,
            "min_clipped_fraction": SAMPLE_CONSTANT_FRACTION,
            "input_samples": int(mask.size), "retained_samples": int(mask.sum()),
        })
        if logger:
            logger.info("Sample mask | %s | fit=%s | retained=%d/%d",
                        mode, "control" if control_fp else "not applied", mask.sum(), mask.size)
    manifest = dict(dataset.manifest)
    manifest.update({
        "sample_mask_version": SAMPLE_MASK_VERSION, "sample_mask_identity": identity,
        "sample_mask_fit_role": "control" if control_fp is not None else "not_applied",
        "sample_mask_control_events": n_control, "sample_mask_control_fingerprint": control_fp,
        "input_samples_before_mask": int(mask.size), "input_samples_after_mask": int(mask.sum()),
    })
    protocol = canonical_hash({"prepared_fingerprint": manifest["fingerprint"],
                               "event_population_identity": manifest["event_population_identity"],
                               "sample_mask_identity": identity})
    manifest["analysis_protocol_identity"] = protocol
    manifest["analysis_population_identity"] = protocol
    if dataset.manifest != manifest or dataset.sample_mask is None:
        _atomic_npz(dataset.directory / "sample_mask.npz", mask=mask)
        atomic_json(dataset.directory / "manifest.json", manifest)
    return replace(dataset, manifest=manifest, sample_mask=mask)
