"""Verified prepared cache loading for TSHAP and waveform imputation."""
from __future__ import annotations
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.artifact_store import ArtifactStore

def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_root(config: dict[str, Any], run: Path, override: Path | None, *, roles=("control", "blind")) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    configured = Path(config["preprocessing"]["cache_dir"]).expanduser()
    candidates = [configured]
    parts = PureWindowsPath(str(configured)).parts
    anchors = [i for i, part in enumerate(parts) if part.lower() == "waveform_analysis"]
    if anchors:
        suffix = Path(*parts[anchors[-1] + 1:])
        roots = [parent for parent in run.parents if parent.name.lower() == "waveform_analysis"]
        roots.append(Path(__file__).resolve().parents[1])
        candidates.extend(root / suffix for root in roots)
    for candidate in candidates:
        if all((candidate / f"{role}_ml" / "prepared").is_dir() for role in roles):
            return candidate.resolve()
    return configured.resolve()


class PreparedCacheLookup:
    """Index metadata once and reuse memory-mapped datasets across saved models."""

    def __init__(self):
        self.index: dict[tuple[Path, str], list[tuple[Path, dict[str, Any]]]] = {}
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(self, root: Path, role: str, config: dict[str, Any], manifest: dict[str, Any]) -> PreparedDataset:
        """Resolve the run's exact prepared product; never search cache directories."""
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
            control_dataset = self.find(root, "control", config, manifest)
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
