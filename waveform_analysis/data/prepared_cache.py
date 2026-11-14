"""Direct lookup of prepared waveform caches by semantic path."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.named_cache import prepared_dataset_alias, cache_root_from_config
from waveform_analysis.data.cache_store import cache_complete
from waveform_analysis.data.sample_mask import attach_control_sample_mask


class PreparedCacheLookup:
    def __init__(self):
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(self, role: str, config: dict[str, Any], manifest: dict[str, Any]) -> PreparedDataset:
        root = cache_root_from_config(config)
        path = prepared_dataset_alias(root, config, role)
        if not cache_complete(path):
            raise FileNotFoundError(f"Prepared cache missing or incomplete: {path}")
        if path in self.datasets:
            return self.datasets[path]

        dataset = load_prepared_dataset(path)
        if dataset.manifest.get("dataset_role") != role:
            raise ValueError(f"Prepared cache role mismatch for {role}")
        mode = str(config["mode"])
        control = (
            dataset if role == "control"
            else self.find("control", config, manifest)
        ) if mode == "timing_to_timing" else None

        dataset = attach_control_sample_mask(
            dataset, control,
            cache_root=Path(config["preprocessing"]["cache_dir"]),
        )
        self.datasets[path] = dataset
        return dataset
