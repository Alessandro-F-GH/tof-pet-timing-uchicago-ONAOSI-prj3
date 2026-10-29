"""Verified prepared cache loading for TSHAP and waveform imputation."""
from __future__ import annotations
import json
from pathlib import Path, PureWindowsPath
import re
from typing import Any
from waveform_analysis.data.dataset import PreparedDataset, load_prepared_dataset

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
        key = root, role
        if key not in self.index:
            entries = []
            for path in (root / f"{role}_ml" / "prepared").glob("*/manifest.json"):
                try:
                    entries.append((path.parent, _read(path)))
                except (OSError, ValueError):
                    continue
            self.index[key] = entries
        population = manifest.get("dataset_populations", {}).get(role, {})
        expected_protocol = population.get("analysis_protocol_identity")
        expected_population = population.get("population_identity")
        control_prefix = PureWindowsPath(manifest.get("control_artifact", "")).name
        matches = []
        for path, cached in self.index[key]:
            if not (
                cached.get("dataset_role") == role
                and cached.get("dataset_source") == config[role]["root_file"]
                and cached.get("mode") == config["mode"]
                and cached.get("window_ns") == config["window_ns"]
                and cached.get("subsampling") == config["ml_input"]["subsampling"]
                and cached.get("true_tof_ps") == config[role]["true_tof_ps"]
                and (expected_population is None or cached.get("event_population_identity") == expected_population)
            ):
                continue
            if expected_protocol is not None:
                verified = cached.get("analysis_protocol_identity") == expected_protocol
            else:
                verified = (re.fullmatch(r"[0-9a-f]{16}", control_prefix) is not None
                            and str(cached.get("control_fingerprint", "")).startswith(control_prefix))
            if verified:
                matches.append(path)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one verified {role} cache, found {len(matches)} in "
                f"{root / f'{role}_ml' / 'prepared'}. Retain the original prepared "
                f"{role} arrays or supply their location with --cache-root."
            )
        path = matches[0]
        if path not in self.datasets:
            self.datasets[path] = load_prepared_dataset(path)
            from waveform_analysis.data.sample_mask import attach_control_sample_mask
            mode = str(config["mode"])
            if role == "control":
                control_dataset = (
                    self.datasets[path] if mode == "timing_to_timing" else None
                )
            elif mode == "timing_to_timing":
                control_dataset = self.find(root, "control", config, manifest)
            else:
                control_dataset = None
            self.datasets[path] = attach_control_sample_mask(
                self.datasets[path],
                control_dataset,
                cache_root=Path(config["preprocessing"]["cache_dir"]),
                artifact_root=Path(config["preprocessing"]["artifact_root"]),
            )
        return self.datasets[path]

