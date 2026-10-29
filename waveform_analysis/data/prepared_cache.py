"""Verified prepared cache loading for TSHAP and waveform imputation."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.artifact_store import ArtifactStore

def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class PreparedCacheLookup:
    """Resolve exact prepared products referenced by a completed run manifest."""

    def __init__(self):
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(self, role: str, config: dict[str, Any], manifest: dict[str, Any]) -> PreparedDataset:
        reference_data = manifest.get("prepared_artifact_refs", {}).get(role)
        if not reference_data:
            raise ValueError(
                f"No prepared artifact reference for role={role!r}; "
                "new runs must publish prepared_artifact_refs in their manifest."
            )
        store = ArtifactStore(config["preprocessing"]["artifact_root"])
        reference = store.resolve(
            str(reference_data["product_type"]), str(reference_data["product_id"])
        )
        if reference.path in self.datasets:
            return self.datasets[reference.path]
        dataset = load_prepared_dataset(reference.path)
        if dataset.manifest.get("dataset_role") != role:
            raise ValueError(f"Prepared artifact role mismatch for {role}")
        mode = str(config["mode"])
        if role == "control":
            control_dataset = dataset if mode == "timing_to_timing" else None
        elif mode == "timing_to_timing":
            control_dataset = self.find("control", config, manifest)
        else:
            control_dataset = None
        from waveform_analysis.data.sample_mask import attach_control_sample_mask
        dataset = attach_control_sample_mask(
            dataset,
            control_dataset,
            cache_root=Path(config["preprocessing"]["cache_dir"]),
            artifact_root=Path(config["preprocessing"]["artifact_root"]),
        )
        expected_protocol = (
            manifest.get("dataset_populations", {}).get(role, {})
            .get("analysis_protocol_identity")
        )
        if (
            expected_protocol is not None
            and dataset.manifest.get("analysis_protocol_identity") != expected_protocol
        ):
            raise ValueError(f"Prepared artifact protocol mismatch for {role}")
        self.datasets[reference.path] = dataset
        return dataset
