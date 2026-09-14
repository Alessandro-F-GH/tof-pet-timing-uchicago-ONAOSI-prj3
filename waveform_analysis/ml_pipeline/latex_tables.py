from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .plot_style import LABELS


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


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


def _measurement(value: Any, uncertainty: Any, digits: int = 2) -> str:
    central = _float(value)
    error = _float(uncertainty)
    if not np.isfinite(central):
        return "--"
    if not np.isfinite(error):
        return _number(central, digits)
    return f"{central:.{digits}f} $\\pm$ {error:.{digits}f}"


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


def _final_ctr_table(run: Path, output: Path) -> Path | None:
    rows = [
        row
        for row in _read_csv(run / "csv" / "results.csv")
        if row.get("stage") == "test"
    ]
    if not rows:
        return None
    rows.sort(
        key=lambda row: (
            _float(row.get("voltage_V")),
            str(row.get("dataset", "")),
            str(row.get("method", "")),
        )
    )
    body = []
    for row in rows:
        body.append(
            [
                _escape(row.get("dataset", "")),
                _number(row.get("voltage_V"), 1),
                _escape(LABELS.get(str(row.get("method", "")), str(row.get("method", "")))),
                _measurement(row.get("ctr_ps"), row.get("ctr_uncertainty_ps")),
                str(int(_float(row.get("n"), 0))),
            ]
        )
    return _write_table(
        output / "final_ctr.tex",
        caption="Blind-test coincidence timing resolution for the evaluated methods.",
        label="tab:final-ctr",
        columns=["Dataset", "Bias [V]", "Method", "CTR [ps]", "$N$"],
        rows=body,
        alignment="lrlrr",
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
    for builder in (_final_ctr_table, _threshold_table, _window_table):
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
