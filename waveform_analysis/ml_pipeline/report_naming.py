from __future__ import annotations

import json
from pathlib import Path

from .common import atomic_json


_EXACT_RENAMES = {
    "reporting_config_resolved.json": "reporting.json",
    "tables/study_summary.csv": "tables/summary.csv",
    "tables/paired_model_comparisons.csv": "tables/paired.csv",
    "tables/best_by_formulation.csv": "tables/best.csv",
    "tables/correlations/model_output_correlations.csv": "tables/correlations/pairs.csv",
}

_PLOT_PREFIXES = {
    "plots/ctr": "ctr_comparison__",
    "plots/rmse": "rmse_comparison__",
    "plots/led_improvement": "paired_led_improvement__",
    "plots/tradeoffs/rmse_vs_ctr": "rmse_vs_ctr__",
    "plots/tradeoffs/ctr_vs_time": "ctr_vs_time__",
    "plots/window_comparison/ctr": "ctr_by_window__",
    "plots/window_comparison/rmse": "rmse_by_window__",
    "plots/architecture/best_shared_vs_direct": "best_shared_vs_direct__",
    "plots/correlations": "model_output_correlation__",
}

_CORRELATION_TABLE_PREFIX = "model_output_correlation__"
_CORRELATION_COUNT_SUFFIX = "__n_replicas.csv"


def _rename(path: Path, target: Path) -> Path:
    if path == target or not path.exists():
        return target if target.exists() else path
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Compact report filename collision: {target}")
    path.rename(target)
    return target


def _strip_prefix_files(directory: Path, prefix: str) -> None:
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.startswith(prefix):
            continue
        _rename(path, path.with_name(path.name[len(prefix) :]))


def _compact_correlation_tables(directory: Path) -> None:
    if not directory.is_dir():
        return
    for path in sorted(directory.glob(f"{_CORRELATION_TABLE_PREFIX}*.csv")):
        tail = path.name[len(_CORRELATION_TABLE_PREFIX) :]
        if tail.endswith(_CORRELATION_COUNT_SUFFIX):
            tail = f"{tail[:-len(_CORRELATION_COUNT_SUFFIX)]}__n.csv"
        _rename(path, path.with_name(tail))


def compact_report_filenames(report_root) -> Path:
    """Remove filename text already encoded by the report directory hierarchy.

    The operation is deterministic and intentionally limited to report artifacts.
    Mode, window and any context suffix are retained because they distinguish files
    within the same directory; plot/table type prefixes are removed because the
    parent directory already carries that information.
    """

    root = Path(report_root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Report manifest not found: {manifest_path}")

    for source_name, target_name in _EXACT_RENAMES.items():
        _rename(root / source_name, root / target_name)

    for relative_dir, prefix in _PLOT_PREFIXES.items():
        _strip_prefix_files(root / relative_dir, prefix)

    _compact_correlation_tables(root / "tables" / "correlations")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reporting_config"] = "reporting.json"
    manifest["filename_policy"] = (
        "filenames omit information already encoded by their parent directory; "
        "only within-directory discriminators such as mode, window and required context are retained"
    )
    manifest["plots_and_tables"] = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    atomic_json(manifest_path, manifest)
    return root
