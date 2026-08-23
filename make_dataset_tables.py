from __future__ import annotations

"""
Generate LaTeX dataset tables for:

1) Oscilloscope / waveform datasets
2) Pico-TDC datasets produced by janus_data_analysis

The Pico-TDC reader is designed for the current output layout:

    <pico-folder>/
        summary.csv
        analysis/
            Run4240/
                csv/
                    energy_selection.csv
                    fit.csv
                    matching_model_metrics.csv
                    matching_total.csv
                    matching_training.csv
                    selection.csv
            Run.../

It does NOT assume the old file name main_peak_timing_fit_results.csv.

Typical use from the repository root:

python make_dataset_tables.py ^
  --waveform-folder "waveform_analysis\\processed_data\\ml_prepared" ^
  --pico-folder "janus_data_analysis\\outputs\\07-10" ^
  --output-dir "presentation\\tables"

Outputs:
    waveform_dataset_table.tex
    pico_tdc_dataset_table.tex
    waveform_dataset_summary.csv
    pico_tdc_dataset_summary.csv
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Configuration / aliases
# ---------------------------------------------------------------------------

VOLTAGE_ALIASES = (
    "bias_voltage_V",
    "bias_voltage",
    "voltage_V",
    "voltage",
    "bias_V",
    "bias",
    "V",
)

THRESHOLD_ALIASES = (
    "threshold_mV",
    "timing_threshold_mV",
    "timing_threshold",
    "T_th_mV",
    "T_th",
    "threshold",
    "thr_mV",
    "thr",
)

RUN_ALIASES = (
    "run",
    "run_id",
    "run_number",
    "run_name",
    "Run",
)

RAW_COUNT_ALIASES = (
    "total_events_initial",
    "total_events",
    "raw_events",
    "events_total",
    "n_raw",
    "n_total",
    "n_events_raw",
    "n_events_total",
    "events_before",
    "before_selection",
    "input_events",
    "n_input",
)

PHOTOPEAK_COUNT_ALIASES = (
    "duration_selected_events",
    "photopeak_events",
    "events_after_photopeak",
    "n_photopeak",
    "n_photopeak_events",
    "selected_events",
    "events_selected",
    "n_selected",
    "after_selection",
    "output_events",
    "n_output",
)

# If an event-level CSV is encountered, these names help identify whether a row
# survived a selection. The script only uses them when an explicit count is absent.
SELECTION_FLAG_ALIASES = (
    "selected",
    "is_selected",
    "photopeak_selected",
    "pass",
    "passed",
    "accepted",
    "keep",
)

RUN_RE = re.compile(r"Run(\d+)", re.IGNORECASE)
VOLTAGE_RE = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*V(?!\w)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def alias_lookup(mapping: dict[str, Any], aliases: Iterable[str]) -> Any | None:
    """Case/punctuation-insensitive key lookup."""
    normalized = {norm(k): v for k, v in mapping.items()}
    for alias in aliases:
        key = norm(alias)
        if key in normalized and normalized[key] not in (None, ""):
            return normalized[key]
    return None


def to_float(value: Any, *, field: str, source: Path) -> float:
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{source}: invalid {field}={value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{source}: non-finite {field}={value!r}")
    return out


def to_int(value: Any, *, field: str, source: Path) -> int:
    out = to_float(value, field=field, source=source)
    rounded = int(round(out))
    if abs(out - rounded) > 1e-6:
        raise ValueError(f"{source}: {field} is not an integer: {value!r}")
    if rounded < 0:
        raise ValueError(f"{source}: {field} must be >= 0")
    return rounded


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "pass", "passed", "selected", "keep", "accepted"}:
        return True
    if s in {"0", "false", "no", "n", "fail", "failed", "rejected", "drop"}:
        return False
    return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], []
        return list(reader.fieldnames), list(reader)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return data


def format_float(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def latex_int(value: int) -> str:
    return f"{value:,}".replace(",", r"\,")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Waveform datasets
# ---------------------------------------------------------------------------

def waveform_source_name(manifest: dict[str, Any], directory: Path) -> str:
    for key in ("root_path", "source_root", "source_path", "input_path"):
        value = manifest.get(key)
        if value:
            return Path(str(value)).name

    raw_manifest = manifest.get("raw_cache_manifest")
    if isinstance(raw_manifest, dict):
        source = raw_manifest.get("source")
        if isinstance(source, dict):
            for key in ("path", "name", "file"):
                if source.get(key):
                    return Path(str(source[key])).name
        elif source:
            return Path(str(source)).name

    return directory.name


def waveform_is_prepared(manifest: dict[str, Any], directory: Path) -> bool:
    if "selected_events" in manifest:
        return True
    if (directory / "event_id.npy").is_file() and (
        "raw_cache_manifest" in manifest
        or "source_root" in manifest
        or "condition" in manifest
    ):
        return True
    return False


def waveform_raw_count(manifest: dict[str, Any], source: Path) -> int:
    value = alias_lookup(manifest, RAW_COUNT_ALIASES)
    if value is not None:
        return to_int(value, field="waveform raw events", source=source)

    raw_manifest = manifest.get("raw_cache_manifest")
    if isinstance(raw_manifest, dict):
        value = alias_lookup(
            raw_manifest,
            (
                "event_count",
                "total_events",
                "raw_events",
                "input_event_count",
            ),
        )
        if value is not None:
            return to_int(value, field="waveform raw events", source=source)

    raise ValueError("raw event count not found")


def waveform_selected_count(
    manifest: dict[str, Any], directory: Path, source: Path
) -> int:
    value = alias_lookup(manifest, PHOTOPEAK_COUNT_ALIASES)
    if value is not None:
        return to_int(value, field="waveform photopeak events", source=source)

    event_ids = directory / "event_id.npy"
    if event_ids.is_file():
        arr = np.load(event_ids, mmap_mode="r")
        return int(arr.shape[0])

    raise ValueError("photopeak-selected count not found")


def waveform_voltage(
    manifest: dict[str, Any], directory: Path, source_name: str, source: Path
) -> float:
    bias_path = directory / "bias_voltage_V.npy"
    if bias_path.is_file():
        arr = np.asarray(np.load(bias_path, mmap_mode="r"), dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return float(np.median(arr))

    condition = manifest.get("condition")
    if isinstance(condition, dict):
        value = alias_lookup(condition, VOLTAGE_ALIASES)
        if value is not None:
            return to_float(value, field="waveform voltage", source=source)

    match = VOLTAGE_RE.search(source_name)
    if match:
        return float(match.group(1))

    for part in reversed(directory.parts):
        match = VOLTAGE_RE.search(part)
        if match:
            return float(match.group(1))

    raise ValueError("bias voltage not found")


def collect_waveform_rows(folder: Path) -> list[dict[str, Any]]:
    manifests = sorted(folder.rglob("manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest.json files found below {folder}")

    found: list[dict[str, Any]] = []

    for manifest_path in manifests:
        directory = manifest_path.parent
        try:
            manifest = read_json(manifest_path)
            if not waveform_is_prepared(manifest, directory):
                continue

            source_name = waveform_source_name(manifest, directory)
            raw = waveform_raw_count(manifest, manifest_path)
            selected = waveform_selected_count(manifest, directory, manifest_path)
            voltage = waveform_voltage(
                manifest, directory, source_name, manifest_path
            )

            found.append(
                {
                    "voltage_V": voltage,
                    "raw_events": raw,
                    "photopeak_events": selected,
                    "retained_percent": 100.0 * selected / raw if raw else math.nan,
                    "source": source_name,
                }
            )
        except Exception as exc:
            print(f"WARNING waveform: skip {manifest_path}: {exc}")

    if not found:
        raise RuntimeError(
            "No usable prepared waveform datasets found. "
            "Use the folder containing the per-file ml_prepared datasets."
        )

    # Deduplicate copied/prepared representations of the same source.
    unique: dict[tuple[float, str], dict[str, Any]] = {}
    for row in found:
        key = (round(float(row["voltage_V"]), 9), str(row["source"]))
        old = unique.get(key)
        if old is None:
            unique[key] = row
        elif (
            old["raw_events"] != row["raw_events"]
            or old["photopeak_events"] != row["photopeak_events"]
        ):
            raise RuntimeError(
                f"Conflicting waveform metadata for {row['source']} "
                f"at {row['voltage_V']:g} V"
            )

    return sorted(unique.values(), key=lambda r: (r["voltage_V"], r["source"]))


# ---------------------------------------------------------------------------
# Pico-TDC / janus_data_analysis
# ---------------------------------------------------------------------------

def run_name_from_path(path: Path) -> str | None:
    for part in reversed(path.parts):
        m = RUN_RE.fullmatch(part)
        if m:
            return f"Run{m.group(1)}"
    return None


def normalize_run(value: Any) -> str | None:
    if value in (None, ""):
        return None
    s = str(value).strip()
    m = RUN_RE.search(s)
    if m:
        return f"Run{m.group(1)}"
    if re.fullmatch(r"\d+", s):
        return f"Run{s}"
    return s


def load_pico_summary(folder: Path) -> dict[str, dict[str, Any]]:
    """
    Read <pico-folder>/summary.csv when present.

    The exact column names are allowed to vary.  The summary is used mainly for
    Run -> voltage / timing-threshold metadata.
    """
    path = folder / "summary.csv"
    if not path.is_file():
        print("WARNING Pico-TDC: summary.csv not found; using per-run CSV metadata only.")
        return {}

    _, rows = read_csv_rows(path)
    out: dict[str, dict[str, Any]] = {}

    for row in rows:
        run = normalize_run(alias_lookup(row, RUN_ALIASES))
        if run is None:
            # Some summary files use an unnamed/index-like first column.
            for value in row.values():
                candidate = normalize_run(value)
                if candidate and RUN_RE.fullmatch(candidate):
                    run = candidate
                    break
        if run is None:
            continue

        voltage = alias_lookup(row, VOLTAGE_ALIASES)
        threshold = alias_lookup(row, THRESHOLD_ALIASES)
        raw = alias_lookup(row, RAW_COUNT_ALIASES)
        photopeak = alias_lookup(row, PHOTOPEAK_COUNT_ALIASES)

        out[run] = {
            "voltage_V": float(voltage) if voltage not in (None, "") else None,
            "threshold_mV": float(threshold) if threshold not in (None, "") else None,
            "raw_events": int(float(raw)) if raw not in (None, "") else None,
            "photopeak_events": (
                int(float(photopeak)) if photopeak not in (None, "") else None
            ),
            "_row": row,
        }

    return out


def extract_named_value_from_csv(path: Path, field_name: str) -> int | None:
    """
    Read one explicit scalar field from a Janus CSV.

    IMPORTANT:
    This function never infers an event count from the number of CSV rows.
    Row counts in files such as selection.csv / energy_selection.csv can refer
    to already-filtered or diagnostic populations and are not the acquisition
    event count.
    """
    fields, rows = read_csv_rows(path)
    if not fields or not rows:
        return None

    target = norm(field_name)

    # Standard wide CSV: field is a column.
    normalized_fields = {norm(f): f for f in fields}
    if target in normalized_fields:
        real_field = normalized_fields[target]
        for row in rows:
            value = row.get(real_field)
            if value not in (None, ""):
                try:
                    return to_int(value, field=field_name, source=path)
                except ValueError:
                    continue

    # Key/value CSV: first column contains metric name.
    if len(fields) >= 2:
        for row in rows:
            values = list(row.values())
            if not values:
                continue
            if norm(values[0]) != target:
                continue
            for value in values[1:]:
                if value not in (None, ""):
                    try:
                        return to_int(value, field=field_name, source=path)
                    except ValueError:
                        continue

    return None


def find_explicit_pico_value(
    run_dir: Path,
    field_name: str,
    *,
    preferred_files: tuple[str, ...] = (),
) -> tuple[int | None, str]:
    """
    Find an explicitly stored Janus scalar in the per-run CSV outputs.
    """
    csv_dir = run_dir / "csv"
    if not csv_dir.is_dir():
        return None, ""

    all_files = sorted(csv_dir.glob("*.csv"))
    preferred = [csv_dir / name for name in preferred_files]
    ordered = [p for p in preferred if p.is_file()]
    ordered += [p for p in all_files if p not in ordered]

    for path in ordered:
        value = extract_named_value_from_csv(path, field_name)
        if value is not None:
            return value, path.name

    return None, ""


def pick_pico_counts(run_dir: Path) -> tuple[int | None, int | None, str, str]:
    """
    Extract Pico-TDC raw and photopeak-selected event counts from the
    actual Janus output structure.

    Current Janus semantics:
      energy_selection.csv
        - one row per acquired event
        - duration_selected == 1 for events passing the ToT/photopeak cut

      selection.csv
        - downstream subset used after additional alignment/matching cuts

    Therefore:
      raw_events       = number of rows in energy_selection.csv
      photopeak_events = sum(duration_selected) in energy_selection.csv

    IMPORTANT:
      selection.csv is intentionally NOT used for the photopeak count because
      it is already downstream of the photopeak selection.
    """
    path = run_dir / "csv" / "energy_selection.csv"
    if not path.is_file():
        return None, None, "", ""

    fields, rows = read_csv_rows(path)
    if not rows:
        return 0, 0, path.name, path.name

    normalized_fields = {norm(f): f for f in fields}
    duration_key = normalized_fields.get(norm("duration_selected"))

    if duration_key is None:
        raise ValueError(
            f"{path}: expected a 'duration_selected' column"
        )

    raw = len(rows)

    selected = 0
    unknown = 0
    for row in rows:
        flag = parse_bool(row.get(duration_key))
        if flag is True:
            selected += 1
        elif flag is None:
            unknown += 1

    if unknown:
        raise ValueError(
            f"{path}: {unknown} rows have an invalid duration_selected value"
        )

    return raw, selected, f"{path.name}:row_count", f"{path.name}:sum(duration_selected)"


def collect_pico_rows(folder: Path) -> list[dict[str, Any]]:
    summary = load_pico_summary(folder)

    analysis_dir = folder / "analysis"
    if not analysis_dir.is_dir():
        raise FileNotFoundError(
            f"Expected Pico-TDC analysis directory not found: {analysis_dir}"
        )

    run_dirs = sorted(
        p for p in analysis_dir.iterdir()
        if p.is_dir() and RUN_RE.fullmatch(p.name)
    )
    if not run_dirs:
        raise FileNotFoundError(f"No RunXXXX directories found below {analysis_dir}")

    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for run_dir in run_dirs:
        run = run_dir.name
        meta = summary.get(run, {})

        voltage = meta.get("voltage_V")
        threshold = meta.get("threshold_mV")
        raw = meta.get("raw_events")
        photopeak = meta.get("photopeak_events")
        raw_source = "summary.csv" if raw is not None else ""
        photopeak_source = "summary.csv" if photopeak is not None else ""

        # Fill metadata from run CSVs if summary.csv does not provide it.
        if voltage is None or threshold is None:
            csv_voltage, csv_threshold = extract_metadata_from_run_csvs(run_dir)
            if voltage is None:
                voltage = csv_voltage
            if threshold is None:
                threshold = csv_threshold

        # Fill counts from per-run CSVs.
        if raw is None or photopeak is None:
            r, p, rsrc, psrc = pick_pico_counts(run_dir)
            if raw is None:
                raw, raw_source = r, rsrc
            if photopeak is None:
                photopeak, photopeak_source = p, psrc

        # Last-resort voltage inference from Run name is intentionally narrow:
        # Run4240 -> 42 V is accepted only if it gives a plausible 42--49 V value.
        # Threshold is NOT inferred from the run number because Run IDs such as
        # 4541/4642/4644 are not safely interpretable as mV thresholds.
        if voltage is None:
            m = RUN_RE.fullmatch(run)
            if m:
                digits = m.group(1)
                if len(digits) >= 2:
                    candidate = float(digits[:2])
                    if 40 <= candidate <= 60:
                        voltage = candidate

        problems = []
        if voltage is None:
            problems.append("voltage")
        if threshold is None:
            problems.append("timing threshold")
        if raw is None:
            problems.append("raw event count")
        if photopeak is None:
            problems.append("photopeak event count")

        if problems:
            missing.append(f"{run}: missing {', '.join(problems)}")
            continue

        rows.append(
            {
                "run": run,
                "voltage_V": float(voltage),
                "threshold_mV": float(threshold),
                "raw_events": int(raw),
                "photopeak_events": int(photopeak),
                "retained_percent": (
                    100.0 * int(photopeak) / int(raw) if int(raw) else math.nan
                ),
                "raw_count_source": raw_source,
                "photopeak_count_source": photopeak_source,
            }
        )

    if missing:
        print("\nPico-TDC runs not included:")
        for item in missing:
            print(f"  - {item}")
        print(
            "\nIf threshold metadata are missing, inspect "
            f"{folder / 'summary.csv'}: the script deliberately does not guess "
            "thresholds from RunXXXX names."
        )

    if not rows:
        raise RuntimeError(
            "No complete Pico-TDC operating points could be reconstructed."
        )

    return sorted(rows, key=lambda r: (r["voltage_V"], r["threshold_mV"], r["run"]))


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def waveform_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by make_dataset_tables.py",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"\textbf{Bias [V]} & \textbf{Raw events} & "
        r"\textbf{Photopeak events} & \textbf{Retained [\%]} \\",
        r"\midrule",
    ]

    for row in rows:
        lines.append(
            f"{format_float(row['voltage_V'])} & "
            f"{latex_int(row['raw_events'])} & "
            f"{latex_int(row['photopeak_events'])} & "
            f"{row['retained_percent']:.1f} "
            r"\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def pico_latex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by make_dataset_tables.py",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"\textbf{Bias [V]} & \textbf{$T_{\mathrm{th}}$ [mV]} & "
        r"\textbf{Raw events} & \textbf{Photopeak events} & "
        r"\textbf{Retained [\%]} \\",
        r"\midrule",
    ]

    previous_voltage = None
    for row in rows:
        voltage = float(row["voltage_V"])
        if previous_voltage is not None and not math.isclose(
            voltage, previous_voltage, abs_tol=1e-12
        ):
            lines.append(r"\addlinespace[0.2em]")

        lines.append(
            f"{format_float(voltage)} & "
            f"{format_float(row['threshold_mV'])} & "
            f"{latex_int(row['raw_events'])} & "
            f"{latex_int(row['photopeak_events'])} & "
            f"{row['retained_percent']:.1f} "
            r"\\"
        )
        previous_voltage = voltage

    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate LaTeX dataset tables for waveform and Pico-TDC data."
        )
    )
    parser.add_argument(
        "--waveform-folder",
        type=Path,
        required=True,
        help="Folder containing per-file waveform prepared datasets.",
    )
    parser.add_argument(
        "--pico-folder",
        type=Path,
        required=True,
        help=(
            "Janus output folder containing summary.csv and analysis/RunXXXX/csv/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("presentation") / "tables",
        help="Output directory (default: presentation/tables).",
    )
    args = parser.parse_args()

    waveform_folder = args.waveform_folder.expanduser().resolve()
    pico_folder = args.pico_folder.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not waveform_folder.is_dir():
        raise FileNotFoundError(f"Waveform folder not found: {waveform_folder}")
    if not pico_folder.is_dir():
        raise FileNotFoundError(f"Pico-TDC folder not found: {pico_folder}")

    waveform_rows = collect_waveform_rows(waveform_folder)
    pico_rows = collect_pico_rows(pico_folder)

    output_dir.mkdir(parents=True, exist_ok=True)

    waveform_tex = output_dir / "waveform_dataset_table.tex"
    pico_tex = output_dir / "pico_tdc_dataset_table.tex"

    waveform_tex.write_text(waveform_latex(waveform_rows), encoding="utf-8")
    pico_tex.write_text(pico_latex(pico_rows), encoding="utf-8")

    write_csv(
        output_dir / "waveform_dataset_summary.csv",
        waveform_rows,
        [
            "voltage_V",
            "raw_events",
            "photopeak_events",
            "retained_percent",
            "source",
        ],
    )
    write_csv(
        output_dir / "pico_tdc_dataset_summary.csv",
        pico_rows,
        [
            "run",
            "voltage_V",
            "threshold_mV",
            "raw_events",
            "photopeak_events",
            "retained_percent",
            "raw_count_source",
            "photopeak_count_source",
        ],
    )

    print()
    print(f"Waveform datasets included: {len(waveform_rows)}")
    print(f"Pico-TDC runs included:      {len(pico_rows)}")
    print()
    print(f"Wrote: {waveform_tex}")
    print(f"Wrote: {pico_tex}")
    print(f"Wrote: {output_dir / 'waveform_dataset_summary.csv'}")
    print(f"Wrote: {output_dir / 'pico_tdc_dataset_summary.csv'}")


if __name__ == "__main__":
    main()
