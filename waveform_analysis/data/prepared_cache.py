"""Verified prepared cache loading for TSHAP and waveform imputation."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.data.artifact_store import ArtifactStore
from waveform_analysis.data.named_cache import prepared_dataset_alias, resolve_alias, alias_exists

def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class PreparedCacheLookup:
    """Resolve prepared datasets by semantic cache name, with legacy-ref fallback.

    The selected cache is authoritative: historical run fingerprints are not
    compared with current prepared-cache protocol fingerprints. Role and
    waveform-grid compatibility are still checked by the consumer.
    """

    def __init__(self):
        self.datasets: dict[Path, PreparedDataset] = {}

    def find(
        self,
        role: str,
        config: dict[str, Any],
        manifest: dict[str, Any],
        *,
        control_artifact: dict[str, Any] | None = None,
    ) -> PreparedDataset:
        # Prefer the stable semantic path. Hash references remain a compatibility
        # fallback for old result manifests and caches without named aliases.
        alias = prepared_dataset_alias(
            Path(config["preprocessing"].get("named_cache_root") or
                 Path(config["preprocessing"]["artifact_root"]).resolve().parent),
            config, role,
        )
        reference_data = manifest.get("prepared_artifact_refs", {}).get(role)
        if alias_exists(alias):
            dataset_path = resolve_alias(alias)
        elif reference_data:
            store = ArtifactStore(config["preprocessing"]["artifact_root"])
            reference = store.resolve(str(reference_data["product_type"]), str(reference_data["product_id"]))
            dataset_path = reference.path
        else:
            raise FileNotFoundError(
                f"No named prepared cache or legacy artifact reference for {role}: {alias}"
            )
        if dataset_path in self.datasets:
            return self.datasets[dataset_path]
        dataset = load_prepared_dataset(dataset_path)
        metadata = dataset.manifest
        if metadata.get("dataset_role") != role:
            raise ValueError(f"Prepared artifact role mismatch for {role}")
        mode = str(config["mode"])
        if metadata.get("mode") not in (None, mode):
            raise ValueError(
                f"Prepared artifact mode mismatch for {role}: "
                f"{metadata.get('mode')} != {mode}"
            )
        if "window_ns" in metadata and metadata["window_ns"] != config.get("window_ns"):
            raise ValueError(f"Prepared artifact window mismatch for {role}")
        if "subsampling" in metadata and int(metadata["subsampling"]) != int(
            config.get("ml_input", {}).get("subsampling", 1)
        ):
            raise ValueError(f"Prepared artifact subsampling mismatch for {role}")
        expected_source = Path(config[role]["root_file"]).expanduser().resolve()
        stored_source = metadata.get("dataset_source")
        if stored_source and Path(stored_source).expanduser().resolve() != expected_source:
            raise ValueError(
                f"Prepared artifact source mismatch for {role}: {stored_source}"
            )
        if "true_tof_ps" in metadata and float(metadata["true_tof_ps"]) != float(
            config[role]["true_tof_ps"]
        ):
            raise ValueError(f"Prepared artifact true_tof_ps mismatch for {role}")
        if control_artifact is not None:
            expected_control_fp = (
                control_artifact.get("mode_fingerprints") or {}
            ).get(mode)
            stored_control_fp = metadata.get("control_mode_fingerprint")
            if expected_control_fp and stored_control_fp != expected_control_fp:
                raise ValueError(
                    f"Prepared artifact control fingerprint mismatch for {role}"
                )
        if role == "control":
            control_dataset = dataset if mode == "timing_to_timing" else None
        elif mode == "timing_to_timing":
            control_dataset = self.find(
                "control", config, manifest, control_artifact=control_artifact
            )
        else:
            control_dataset = None
        from waveform_analysis.data.sample_mask import attach_control_sample_mask
        dataset = attach_control_sample_mask(
            dataset,
            control_dataset,
            cache_root=Path(config["preprocessing"]["cache_dir"]),
            artifact_root=Path(config["preprocessing"]["artifact_root"]),
        )
        # Do not compare analysis_protocol_identity with the historical run
        # manifest. Named cache selection is intentional and user-controlled;
        # code-version/fingerprint changes must not make reusable waveforms
        # inaccessible. Structural role and time-grid checks remain in place.
        self.datasets[dataset_path] = dataset
        return dataset
