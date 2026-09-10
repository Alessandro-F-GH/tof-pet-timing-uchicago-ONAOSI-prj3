#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot robust CTR and event efficiency versus timing threshold at fixed bias voltage.")
    parser.add_argument("--summary", type=Path, required=True, help="Path to summary.csv")
    parser.add_argument("--runs-root", type=Path, required=True, help="Root directory containing RunXXXX folders")
    parser.add_argument("--voltage", type=float, default=46.0)
    parser.add_argument("--mode", type=str, default="TRG_MATCHING")
    parser.add_argument("--output", type=Path, default=Path("ctr_efficiency_vs_time_threshold_46V.pdf"))
    return parser.parse_args()


def find_run_directory(runs_root: Path, run_id: str) -> Path:
    exact = [p for p in runs_root.rglob(run_id) if p.is_dir()]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(f"Multiple directories named {run_id} found:\n" + "\n".join(map(str, exact)))
    partial = [p for p in runs_root.rglob("*") if p.is_dir() and run_id.lower() in p.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise FileNotFoundError(f"Could not find directory for {run_id} under {runs_root}")
    raise RuntimeError(f"Multiple possible directories found for {run_id}:\n" + "\n".join(map(str, partial)))


def find_unique_file(run_dir: Path, filename: str) -> Path:
    files = list(run_dir.rglob(filename))
    if len(files) == 1:
        return files[0]
    if not files:
        raise FileNotFoundError(f"{filename} not found under {run_dir}")
    raise RuntimeError(f"Multiple {filename} files found under {run_dir}:\n" + "\n".join(map(str, files)))


def select_ctr_row(frame: pd.DataFrame, threshold_mV: float, run_id: str) -> pd.Series:
    if frame.empty:
        raise RuntimeError(f"{run_id}: fit.csv is empty")
    candidates = frame.copy()
    if "parameter" in candidates:
        parameter = pd.to_numeric(candidates["parameter"], errors="coerce")
        matches = candidates[np.isclose(parameter, threshold_mV, rtol=0.0, atol=1e-6)]
        if not matches.empty:
            candidates = matches
    if len(candidates) > 1 and "method" in candidates:
        matches = candidates[candidates["method"].astype(str).str.contains("Pico-TDC LED", case=False, regex=False)]
        if not matches.empty:
            candidates = matches
    if len(candidates) != 1:
        raise RuntimeError(f"{run_id}: could not uniquely select CTR row for T_th={threshold_mV:g} mV")
    return candidates.iloc[0]


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    required = {"run_id", "Voltage", "AcquisitionMode", "E_th", "T_th", "CTR_ps", "CTR_error_ps"}
    missing = required - set(summary.columns)
    if missing:
        raise RuntimeError("Missing required columns in summary.csv: " + ", ".join(sorted(missing)))
    if "fit_metric" in summary.columns:
        invalid = summary[summary["fit_metric"].astype(str) != "gaussian_equivalent_shortest_coverage_interval"]
        if not invalid.empty:
            raise RuntimeError("summary.csv contains rows not produced with the robust shortest-coverage CTR")

    numeric = ["Voltage", "E_th", "T_th", "CTR_ps", "CTR_error_ps"]
    if "average_delay_corrected_alignments" in summary.columns:
        numeric.append("average_delay_corrected_alignments")
    for column in numeric:
        summary[column] = pd.to_numeric(summary[column], errors="coerce")

    selected = summary[
        np.isclose(summary["Voltage"], args.voltage, rtol=0.0, atol=1e-6)
        & (summary["AcquisitionMode"].astype(str) == args.mode)
    ].copy()
    if selected.empty:
        raise RuntimeError(f"No runs found for Voltage={args.voltage:g} V, AcquisitionMode={args.mode}")

    # When multiple runs exist at one threshold, retain the run with the largest
    # matched/aligned event population. This replaces the obsolete fitted-area criterion.
    ranking = "average_delay_corrected_alignments" if "average_delay_corrected_alignments" in selected.columns else "CTR_ps"
    ascending_rank = ranking == "CTR_ps"
    selected = (
        selected.dropna(subset=["T_th", ranking])
        .sort_values(["T_th", ranking], ascending=[True, ascending_rank])
        .drop_duplicates(subset=["T_th"], keep="first")
        .sort_values("T_th")
        .reset_index(drop=True)
    )

    rows = []
    for _, row in selected.iterrows():
        run_id = str(row["run_id"])
        threshold = float(row["T_th"])
        run_dir = find_run_directory(args.runs_root, run_id)
        energy_df = pd.read_csv(find_unique_file(run_dir, "energy_selection.csv"))
        if "duration_selected" not in energy_df:
            raise RuntimeError(f"{run_id}: duration_selected missing from energy_selection.csv")
        n_photopeak = int((pd.to_numeric(energy_df["duration_selected"], errors="coerce") == 1).sum())
        if n_photopeak <= 0:
            raise RuntimeError(f"{run_id}: N_photopeak=0")

        ctr_row = select_ctr_row(pd.read_csv(find_unique_file(run_dir, "fit.csv")), threshold, run_id)
        count_field = "n_valid" if "n_valid" in ctr_row.index else "n_fit"
        if count_field not in ctr_row.index:
            raise RuntimeError(f"{run_id}: n_valid missing from fit.csv")
        n_valid = int(float(ctr_row[count_field]))
        efficiency = n_valid / n_photopeak
        if efficiency > 1.0 + 1e-12:
            raise RuntimeError(f"{run_id}: invalid efficiency > 1 ({n_valid}/{n_photopeak})")
        efficiency_error = float(np.sqrt(efficiency * (1.0 - efficiency) / n_photopeak))
        rows.append({
            "run_id": run_id,
            "Voltage": float(row["Voltage"]),
            "E_th": float(row["E_th"]),
            "T_th": threshold,
            "CTR_ps": float(row["CTR_ps"]),
            "CTR_error_ps": float(row["CTR_error_ps"]),
            "n_photopeak": n_photopeak,
            "n_valid": n_valid,
            "efficiency": efficiency,
            "efficiency_error": efficiency_error,
        })

    result = pd.DataFrame(rows).sort_values("T_th").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_output = args.output.parent / f"ctr_efficiency_scan_{args.voltage:g}V.csv"
    result.to_csv(csv_output, index=False)

    threshold = result["T_th"].to_numpy(float)
    ctr = result["CTR_ps"].to_numpy(float)
    ctr_error = result["CTR_error_ps"].to_numpy(float)
    efficiency = 100.0 * result["efficiency"].to_numpy(float)

    fig, ax_ctr = plt.subplots(figsize=(10.5, 6.5))
    ax_ctr.errorbar(threshold, ctr, yerr=ctr_error, fmt="s-", markersize=9, linewidth=2.4, capsize=5, label="Robust CTR")
    ax_ctr.set_xlabel(r"Timing threshold $T_{\mathrm{th}}$ [mV]")
    ax_ctr.set_ylabel("Robust CTR [ps]")
    ax_ctr.grid(True, linestyle="--", alpha=0.5)
    ax_ctr.set_xticks(threshold)
    ax_eff = ax_ctr.twinx()
    ax_eff.plot(threshold, efficiency, "o--", markersize=9, linewidth=2.4, label="Event efficiency")
    ax_eff.set_ylabel(r"Event efficiency $N_{\mathrm{valid}}/N_{\mathrm{photopeak}}$ [\%]")
    ax_eff.set_ylim(max(0.0, float(np.min(efficiency)) - 2.0), 100.5)
    h1, l1 = ax_ctr.get_legend_handles_labels(); h2, l2 = ax_eff.get_legend_handles_labels()
    ax_ctr.legend(h1 + h2, l1 + l2, loc="best")
    fig.tight_layout(); fig.savefig(args.output, bbox_inches="tight"); fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    fig_ctr, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.errorbar(threshold, ctr, yerr=ctr_error, fmt="s-", markersize=9, linewidth=2.4, capsize=5)
    ax.set_xlabel(r"Timing threshold $T_{\mathrm{th}}$ [mV]")
    ax.set_ylabel("Robust CTR [ps]")
    ax.set_xticks(threshold); ax.grid(True, linestyle="--", alpha=0.5); fig_ctr.tight_layout()
    ctr_only_pdf = args.output.parent / f"ctr_vs_time_threshold_{args.voltage:g}V.pdf"
    fig_ctr.savefig(ctr_only_pdf, bbox_inches="tight"); fig_ctr.savefig(ctr_only_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight"); plt.close(fig_ctr)

    print(f"Data: {csv_output}\nPDF: {args.output}\nCTR-only PDF: {ctr_only_pdf}")


if __name__ == "__main__":
    main()
