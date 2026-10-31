"""Persist one control-fitted clipping mask for every timing input grid."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.core.artifact_identity import ProductRecipe
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.dataset import PreparedDataset
from waveform_analysis.signal.sample_mask import SAMPLE_CONSTANT_FRACTION, control_clipping_sample_mask

SAMPLE_MASK_VERSION = 1


def attach_control_sample_mask(
    dataset: PreparedDataset, control: PreparedDataset | None, *, cache_root: Path,
    artifact_root: Path | None = None, logger=None,
) -> PreparedDataset:
    """Resolve a control-fitted sample mask without mutating prepared waveforms."""
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
        control_fp = str(control.manifest["fingerprint"])
        n_control = control.n_events
    recipe = ProductRecipe(
        "sample-masks",
        {
            "version": SAMPLE_MASK_VERSION,
            "mode": mode,
            "control_prepared_fingerprint": control_fp,
            "time_ps": time.tolist(),
            "fraction": SAMPLE_CONSTANT_FRACTION,
        },
        parents=(str(control.directory.name),) if control_fp and control is not None else (),
        implementation=f"control-clipping-mask-v{SAMPLE_MASK_VERSION}",
    )
    resolved_artifact_root = (
        Path(artifact_root).expanduser().resolve()
        if artifact_root is not None
        else Path(cache_root).expanduser().resolve() / "artifact_store"
    )
    store = ArtifactStore(resolved_artifact_root)

    def build(directory):
        mask = (
            control_clipping_sample_mask(control.timing_windows)
            if mode == "timing_to_timing"
            else np.ones(time.size, dtype=bool)
        )
        np.savez(directory / "mask.npz", mask=np.asarray(mask, dtype=bool), time_ps=time)

    reference = store.get_or_build(recipe, build, metadata={
        "fit_role": "control" if control_fp is not None else "not_applied",
        "control_events": n_control,
    })
    with np.load(reference.payload_path("mask.npz"), allow_pickle=False) as data:
        mask = np.asarray(data["mask"], dtype=bool)
        stored_time = np.asarray(data["time_ps"], dtype=np.float64)
    if mask.shape != time.shape or not mask.any() or not np.array_equal(stored_time, time):
        raise ValueError(f"Invalid stored sample mask {reference.product_id}")

    manifest = dict(dataset.manifest)
    manifest.update({
        "sample_mask_version": SAMPLE_MASK_VERSION,
        "sample_mask_identity": reference.product_id,
        "sample_mask_fit_role": "control" if control_fp is not None else "not_applied",
        "sample_mask_control_events": n_control,
        "sample_mask_control_fingerprint": control_fp,
        "input_samples_before_mask": int(mask.size),
        "input_samples_after_mask": int(mask.sum()),
    })
    protocol = canonical_hash({
        "prepared_fingerprint": manifest["fingerprint"],
        "event_population_identity": manifest["event_population_identity"],
        "sample_mask_identity": reference.product_id,
    })
    manifest["analysis_protocol_identity"] = protocol
    manifest["analysis_population_identity"] = protocol
    return replace(dataset, manifest=manifest, sample_mask=mask)
