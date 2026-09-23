from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .common import dataset_cache_dir, read_csv, read_json, voltage_from_name
from .config import discover_root_files
from .plot_style import LABELS


def _read_csv(path: Path) -> list[dict[str, Any]]:
    return read_csv(path)


def _float(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _escape(value: Any) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def _number(value: Any, digits: int = 2) -> str:
    number = _float(value)
    if not np.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _measurement(value: Any, uncertainty: Any) -> str:
    central = _float(value)
    error = _float(uncertainty)
    if not np.isfinite(central):
        return "--"
    if not np.isfinite(error) or error <= 0:
        return _number(central, 2)

    exponent = int(np.floor(np.log10(abs(error))))
    decimals = max(0, -exponent)
    scale = 10.0 ** decimals
    rounded_error = round(error * scale) / scale
    rounded_central = round(central * scale) / scale
    return (
        f"{rounded_central:.{decimals}f} "
        f"$\\pm$ {rounded_error:.{decimals}f}"
    )


def _bool_mark(value: Any) -> str:
    return "yes" if str(value).strip().lower() in {"1", "true", "yes"} else "no"


def _write_table(
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
    rows: list[list[str]],
    alignment: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment = alignment or ("l" + "r" * (len(columns) - 1))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path



def _selection_stage_total(rows: list[dict[str, Any]], criterion: str) -> int:
    matches = [row for row in rows if str(row.get("criterion", "")) == criterion]
    if not matches:
        raise ValueError(f"Selection summary has no {criterion!r} stage")
    total = 0
    for row in matches:
        try:
            total += int(row["remaining"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid remaining count for selection stage {criterion!r}"
            ) from exc
    return total


def selection_dataset_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read dataset counts from existing frozen selection caches only.

    This function never opens ROOT data and never creates or rebuilds preprocessing
    artifacts. Source ROOT paths are used only to identify the corresponding cache
    directories through the repository's standard cache helper.
    """
    roots = discover_root_files(config)
    if not roots:
        raise FileNotFoundError("No source ROOT files match the configured data selection")

    rows: list[dict[str, Any]] = []
    for root in roots:
        cache = dataset_cache_dir(config, "selection_store_dir", root)
        manifest_path = cache / "manifest.json"
        summary_path = cache / "selection_summary.csv"
        if not manifest_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError(
                "Missing existing selection cache for "
                f"{root.name}: expected {manifest_path} and {summary_path}. "
                "Dataset-table export never rebuilds preprocessing."
            )

        manifest = read_json(manifest_path)
        summary = read_csv(summary_path)
        if not summary:
            raise ValueError(f"Empty selection summary: {summary_path}")

        n_raw = int(manifest["n_raw"])
        n_selected = int(manifest["n_selected"])
        n_photopeak = _selection_stage_total(summary, "photopeak")

        criteria = list(
            dict.fromkeys(str(row.get("criterion", "")) for row in summary)
        )
        final_criterion = next(
            (criterion for criterion in reversed(criteria) if criterion),
            None,
        )
        if final_criterion is None:
            raise ValueError(f"Selection summary has no criteria: {summary_path}")
        summary_selected = _selection_stage_total(summary, final_criterion)
        if summary_selected != n_selected:
            raise ValueError(
                f"Selection cache is inconsistent for {root.name}: "
                f"manifest n_selected={n_selected}, "
                f"{final_criterion} summary total={summary_selected}"
            )

        voltage = float(voltage_from_name(root))
        if not np.isfinite(voltage):
            raise ValueError(f"Cannot determine bias voltage from dataset name {root.name!r}")

        rows.append(
            {
                "dataset": root.stem,
                "voltage_V": voltage,
                "collected_events": n_raw,
                "photopeak_events": n_photopeak,
                "selected_events": n_selected,
            }
        )

    rows.sort(key=lambda row: (float(row["voltage_V"]), str(row["dataset"])))
    return rows


def make_dataset_latex_table(
    config: dict[str, Any],
    output_file: str | Path,
    *,
    caption: str,
    label: str,
) -> Path:
    """Export one report-ready dataset table from existing selection caches."""
    body = [
        [
            _number(row["voltage_V"], 1),
            str(int(row["collected_events"])),
            str(int(row["photopeak_events"])),
            str(int(row["selected_events"])),
        ]
        for row in selection_dataset_rows(config)
    ]
    return _write_table(
        Path(output_file).resolve(),
        caption=caption,
        label=label,
        columns=[
            "Bias voltage [V]",
            "Collected events",
            "Photopeak events",
            "Final selected events",
        ],
        rows=body,
        alignment="rrrr",
    )


def _final_ctr_table(run: Path, output: Path) -> Path | None:
    rows = [
        row
        for row in _read_csv(run / "csv" / "results.csv")
        if row.get("stage") == "test"
    ]
    if not rows:
        return None

    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        voltage = int(round(_float(row.get("voltage_V"))))
        method = str(row.get("method", ""))
        grouped.setdefault(voltage, {})[method] = row

    model_methods = sorted(
        method
        for methods in grouped.values()
        for method in methods
        if method not in {"led", "cfd"}
    )
    model_method = model_methods[0] if model_methods else None
    body = []
    for voltage in sorted(grouped):
        methods = grouped[voltage]
        led = methods.get("led")
        model = methods.get(model_method) if model_method is not None else None
        if led is None or model is None:
            continue
        body.append(
            [
                str(voltage),
                _measurement(
                    led.get("ctr_ps"),
                    led.get("ctr_uncertainty_ps"),
                ),
                _measurement(
                    model.get("ctr_ps"),
                    model.get("ctr_uncertainty_ps"),
                ),
            ]
        )
    if not body:
        return None

    return _write_table(
        output / "final_ctr.tex",
        caption="Blind-test coincidence timing resolution.",
        label="tab:final-ctr",
        columns=["Bias [V]", "LED CTR [ps]", "Corrected CTR [ps]"],
        rows=body,
        alignment="rrr",
    )


def _threshold_scan_summary_table(run: Path, output: Path) -> Path | None:
    rows = _read_csv(run / "csv" / "threshold_scan.csv")
    if not rows:
        return None

    rows.sort(key=lambda row: _float(row.get("threshold_mV")))
    body = [
        [
            f"{_float(row.get('threshold_mV')):g}",
            _measurement(
                row.get("led_blind_ctr_ps"),
                row.get("led_blind_ctr_uncertainty_ps"),
            ),
            _measurement(
                row.get("model_blind_ctr_ps"),
                row.get("model_blind_ctr_uncertainty_ps"),
            ),
        ]
        for row in rows
    ]
    voltage = _float(rows[0].get("voltage_V"))
    mode = str(rows[0].get("mode", ""))
    mode_label = {
        "energy_to_energy": "energy-channel",
        "timing_to_timing": "timing-channel",
    }.get(mode, mode or "waveform")
    voltage_label = f"{voltage:g} V" if np.isfinite(voltage) else "the scanned dataset"
    return _write_table(
        output / "threshold_scan_summary.tex",
        caption=(
            "Blind-test coincidence timing resolution as a function of the "
            f"LED threshold for the {mode_label} study at {voltage_label}."
        ),
        label="tab:threshold-scan-summary",
        columns=[
            "LED threshold [mV]",
            "LED CTR [ps]",
            "Corrected CTR [ps]",
        ],
        rows=body,
        alignment="rrr",
    )

def _threshold_table(run: Path, output: Path) -> Path | None:
    rows = _read_csv(run / "analyses" / "led_threshold" / "csv" / "threshold_scan.csv")
    if not rows:
        return None
    selected_rows = _read_csv(
        run / "analyses" / "led_threshold" / "csv" / "selected_thresholds.csv"
    )
    selected = {
        str(row.get("dataset")): _float(row.get("selected_threshold_mV"))
        for row in selected_rows
    }
    rows.sort(key=lambda row: (str(row.get("dataset", "")), _float(row.get("threshold_mV"))))
    body = []
    for row in rows:
        threshold = _float(row.get("threshold_mV"))
        is_selected = np.isfinite(selected.get(str(row.get("dataset")), np.nan)) and np.isclose(
            threshold,
            selected[str(row.get("dataset"))],
            rtol=0.0,
            atol=1e-12,
        )
        body.append(
            [
                _escape(row.get("dataset", "")),
                _number(threshold, 1),
                _number(row.get("led_validation_ctr_ps")),
                _number(row.get("model_validation_ctr_ps")),
                _number(row.get("relative_improvement_pct")),
                str(int(_float(row.get("common_validation_events"), 0))),
                "yes" if is_selected else "",
            ]
        )
    return _write_table(
        output / "led_threshold_scan.tex",
        caption="LED-threshold scan evaluated on the common validation population.",
        label="tab:led-threshold-scan",
        columns=[
            "Dataset",
            "Threshold [mV]",
            "LED CTR [ps]",
            "Model CTR [ps]",
            r"Improvement [\%]",
            r"$N_{\mathrm{common}}$",
            "Selected",
        ],
        rows=body,
        alignment="lrrrrrl",
    )


def _window_table(run: Path, output: Path) -> Path | None:
    rows = _read_csv(run / "analyses" / "window" / "csv" / "window_scan.csv")
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            str(row.get("dataset", "")),
            str(row.get("model", "")),
            _float(row.get("right_limit_ns")),
        )
    )
    body = []
    for row in rows:
        body.append(
            [
                _escape(row.get("dataset", "")),
                _escape(LABELS.get(str(row.get("model", "")), str(row.get("model", "")))),
                _number(row.get("right_limit_ns"), 1),
                _number(row.get("validation_ctr_ps")),
                _measurement(row.get("blind_ctr_ps"), row.get("blind_ctr_uncertainty_ps")),
                _number(row.get("relative_improvement_vs_led_pct")),
                _bool_mark(row.get("reused_final_fit")),
            ]
        )
    return _write_table(
        output / "window_scan.tex",
        caption="Sensitivity of timing performance to the waveform right-window limit.",
        label="tab:window-scan",
        columns=[
            "Dataset",
            "Model",
            "Right [ns]",
            "Val. CTR [ps]",
            "Blind CTR [ps]",
            "Improvement [\%]",
            "Reused",
        ],
        rows=body,
        alignment="llrrrrl",
    )


def make_latex_tables(
    run_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Create optional LaTeX tables from persisted numerical study results."""
    run = Path(run_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    generated = []
    for builder in (
        _final_ctr_table,
        _threshold_scan_summary_table,
        _threshold_table,
        _window_table,
    ):
        path = builder(run, output)
        if path is not None:
            generated.append(path)

    include = output / "tables.tex"
    include.write_text(
        "\n".join(f"\\input{{{path.name}}}" for path in generated) + ("\n" if generated else ""),
        encoding="utf-8",
    )
    generated.append(include)
    return generated
