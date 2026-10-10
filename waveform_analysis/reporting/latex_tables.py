"""Board-specific LaTeX population tables from persisted dataset metadata."""

from __future__ import annotations

import json
import re
from itertools import groupby
from pathlib import Path
from typing import Any

import numpy as np

from waveform_analysis.core.io import write_csv

DATASET_TABLE_VERSION = 2
ROLES = {
    "control": "Control",
    "development": "Train (development)",
    "blind": "Test (blind)",
}
MODES = {"energy_to_energy": "Energy", "timing_to_timing": "Timing"}


def escape_latex(value: Any) -> str:
    """Escape plain text, including file names, without interpreting TeX commands."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def export_blind_ctr_table(
    path: Path, rows: list[dict[str, Any]], led_ctr: float, *, mode: str, window: str,
    report_id: str,
) -> Path:
    """Export ranked blind CTRs with stable codes matching the companion plot.

    The containing report section must reference the generated label explicitly.
    Values and bootstrap uncertainties are read from completed run metadata.
    """
    def number(value: float) -> str:
        return f"{value:.2f}" if np.isfinite(value) else "---"

    lines = [r"\begin{table*}[t]", r"\centering", r"\small",
             r"\begin{tabular}{lllr}", r"\toprule",
             r"Code & Model & Formulation & Blind CTR [ps] \\", r"\midrule"]
    for row in rows:
        value = number(row["ctr_ps"])
        uncertainty = number(row["ctr_std_ps"])
        lines.append(
            f"{escape_latex(row['code'])} & {escape_latex(row['display_name'])} & "
            f"{escape_latex(row['formulation'].capitalize())} & "
            rf"${value} \pm {uncertainty}$ \\")
    lines.extend([r"\midrule", rf"LED & Reference & & {number(led_ctr)} \\",
                  r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{Blind CTR for {escape_latex(mode)} waveforms, window {escape_latex(window)}, in ascending CTR order. Codes match the companion bar chart; uncertainties are blind-event bootstrap standard deviations.}}",
                  rf"\label{{tab:blind-ctr-{re.sub(r'[^a-zA-Z0-9-]', '-', report_id)}-{mode}-{re.sub(r'[^a-zA-Z0-9-]', '-', window)}}}",
                  r"\end{table*}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def population_metadata(dataset: Any, source: str) -> dict[str, Any]:
    """Snapshot a prepared population; no training or prediction is performed."""
    manifest = dataset.manifest
    bias = np.asarray(dataset.bias_voltage_V, dtype=float)
    return {
        "source": str(manifest.get("dataset_source", source)),
        "n_selected": int(dataset.n_events),
        "bias_voltage_V": np.unique(bias[np.isfinite(bias)]).tolist(),
        "population_identity": manifest.get("event_population_identity"),
        "n_before_fixed_led": manifest.get("n_before_fixed_led"),
        "n_after_fixed_led": manifest.get("n_after_fixed_led"),
        "n_dropped_window": manifest.get("n_dropped_window"),
    }


def _board(config: dict[str, Any]) -> str | None:
    # All shipped batches explicitly identify their board in results.folder.
    folder = str(config.get("results", {}).get("folder", ""))
    boards = set(re.findall(r"(?:^|[/\\])(?P<board>UC|FBK)(?=$|[/\\])", folder.upper()))
    return next(iter(boards)) if len(boards) == 1 else None


def _cached_population(run: dict[str, Any], role: str) -> dict[str, Any] | None:
    """Read matching prepared caches for results predating population snapshots.

    Ambiguous or unavailable caches yield an unknown count, never a guessed one.
    Reporting does not load acquisition files or execute preprocessing.
    """
    config_path = run["directory"] / "metadata" / "config.json"
    control_path = Path(run["manifest"]["control_artifact"]) / "manifest.json"
    if not config_path.is_file() or not control_path.is_file():
        return None
    config = json.loads(config_path.read_text())
    control = json.loads(control_path.read_text())
    cache = Path(config["preprocessing"]["cache_dir"]) / f"{role}_ml" / "prepared"
    matches = []
    for path in cache.glob("*/manifest.json"):
        manifest = json.loads(path.read_text())
        if (
            manifest.get("dataset_role") == role
            and manifest.get("mode") == config["mode"]
            and manifest.get("window_ns") == config["window_ns"]
            and manifest.get("subsampling") == config["ml_input"]["subsampling"]
            and manifest.get("dataset_source") == config[role]["root_file"]
            and manifest.get("control_fingerprint") == control["fingerprint"]
        ):
            bias_path = path.parent / "bias_voltage_V.npy"
            bias = (
                np.load(bias_path, mmap_mode="r")
                if bias_path.is_file()
                else np.array([])
            )
            matches.append(
                {
                    "source": manifest["dataset_source"],
                    "n_selected": manifest["n_final"],
                    "bias_voltage_V": np.unique(bias[np.isfinite(bias)]).tolist(),
                    "population_identity": manifest["event_population_identity"],
                }
            )
    return matches[0] if len(matches) == 1 else None


def dataset_rows(runs: list[dict[str, Any]], board: str | None) -> list[dict[str, Any]]:
    """Deduplicate populations across models sharing one mode/window/acquisition."""
    populations = {}
    for run in runs:
        manifest = run["manifest"]
        window = manifest["window_ns"]
        saved = manifest.get("dataset_populations", {})
        for role in ROLES:
            summary = saved.get(role) or _cached_population(run, role)
            source = str(
                (summary or {}).get("source", manifest[f"{role}_dataset"]["root_file"])
            )
            row = {
                "board": board or "",
                "role": role,
                "mode": manifest["mode"],
                "window": manifest.get("window_name", ""),
                "start_ns": window["start"],
                "end_ns": window["end"],
                "source": source,
                "bias_voltage_V": ", ".join(
                    f"{v:g}" for v in (summary or {}).get("bias_voltage_V", [])
                ),
                "n_selected": (summary or {}).get("n_selected"),
                "population_identity": (summary or {}).get("population_identity"),
            }
            key = tuple(
                row[field]
                for field in ("role", "mode", "window", "start_ns", "end_ns", "source")
            )
            previous = populations.get(key)
            if (
                previous is not None
                and previous["n_selected"] is not None
                and row["n_selected"] is not None
            ):
                if (
                    previous["n_selected"] != row["n_selected"]
                    or previous["population_identity"] != row["population_identity"]
                ):
                    raise ValueError(f"Inconsistent prepared populations for {key}")
            if previous is None or row["n_selected"] is not None:
                populations[key] = row
    return sorted(
        populations.values(),
        key=lambda row: (
            row["mode"],
            str(row["window"]),
            list(ROLES).index(row["role"]),
            row["source"],
        ),
    )


def export_dataset_tables(
    root: str | Path, runs: list[dict[str, Any]], config: dict[str, Any]
) -> list[Path]:
    """Write selected-event CSV and one independent UC or FBK table fragment.

    Counts include frozen selection, finite LED/coincidence requirements and
    waveform availability. Development counts refer to the full population,
    not fold-specific training subsets or neural internal holdouts.
    """
    directory = Path(root) / "report" / "tables" / "datasets"
    directory.mkdir(parents=True, exist_ok=True)
    board = _board(config)
    rows = dataset_rows(runs, board)
    write_csv(directory / "selected_events.csv", rows)
    if board is None:
        return []  # CSV remains usable for batches without declared board folders.
    # Prevent an obsolete table from remaining after changing the board metadata.
    for other in {"UC", "FBK"} - {board}:
        (directory / f"{other}_selected_events.tex").unlink(missing_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lll l r r}",
        r"\toprule",
        r"Mode & Window [ns] & Population & Acquisition & Bias [V] & Selected events \\",
        r"\midrule",
    ]
    for mode_index, (mode, mode_rows) in enumerate(
        groupby(rows, key=lambda row: row["mode"])
    ):
        if mode_index:
            lines.extend([r"\addlinespace[3pt]", r"\hdashline", r"\addlinespace[3pt]"])
        mode_rows = list(mode_rows)
        windows = groupby(
            mode_rows, key=lambda row: (row["window"], row["start_ns"], row["end_ns"])
        )
        for window_index, ((window, start, end), window_rows) in enumerate(windows):
            if window_index:
                lines.extend(
                    [r"\addlinespace[3pt]", r"\cdashline{2-6}", r"\addlinespace[3pt]"]
                )
            window_rows = list(window_rows)
            bias_values = {row["bias_voltage_V"] or "---" for row in window_rows}
            for row_index, row in enumerate(window_rows):
                mode_cell = (
                    rf"\multirow[t]{{{len(mode_rows)}}}{{*}}{{{escape_latex(MODES[mode])}}}"
                    if window_index == row_index == 0
                    else ""
                )
                window_cell = (
                    rf"\multirow[t]{{{len(window_rows)}}}{{*}}{{{escape_latex(f'{window} ({start:g} to {end:g})')}}}"
                    if row_index == 0
                    else ""
                )
                bias_cell = escape_latex(row["bias_voltage_V"] or "---")
                if len(bias_values) == 1:
                    bias_cell = (
                        rf"\multirow{{{len(window_rows)}}}{{*}}{{{bias_cell}}}"
                        if row_index == 0
                        else ""
                    )
                fields = [
                    mode_cell,
                    window_cell,
                    escape_latex(ROLES[row["role"]]),
                    escape_latex(Path(row["source"]).name),
                    bias_cell,
                    str(row["n_selected"]) if row["n_selected"] is not None else "---",
                ]
                lines.append(" & ".join(fields) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{board} selected waveform populations after frozen selection, leading-edge coincidence requirements and window availability. Train denotes the complete development population; test denotes the independent blind population. Counts are reported once per acquisition, mode and window, irrespective of model. A dash indicates unavailable metadata, not zero events.}}",
        rf"\label{{tab:{board.lower()}-selected-events}}",
        r"\end{table*}",
    ]
    path = directory / f"{board}_selected_events.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [path]
