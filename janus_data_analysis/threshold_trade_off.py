#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CTR and event efficiency versus timing threshold "
            "at fixed bias voltage."
        )
    )

    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.csv",
    )

    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="Root directory containing the RunXXXX folders",
    )

    parser.add_argument(
        "--voltage",
        type=float,
        default=46.0,
        help="Bias voltage to select [V] (default: 46)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="TRG_MATCHING",
        help="Acquisition mode (default: TRG_MATCHING)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ctr_efficiency_vs_time_threshold_46V.pdf"),
        help="Output plot path",
    )

    return parser.parse_args()


# ============================================================
# FIND RUN DIRECTORY
# ============================================================

def find_run_directory(
    runs_root: Path,
    run_id: str,
) -> Path:

    # First search for an exact directory name.
    exact = [
        p
        for p in runs_root.rglob(run_id)
        if p.is_dir()
    ]

    if len(exact) == 1:
        return exact[0]

    if len(exact) > 1:
        raise RuntimeError(
            f"Multiple directories named {run_id} found:\n"
            + "\n".join(str(p) for p in exact)
        )

    # Fallback: directory name contains RunXXXX.
    partial = [
        p
        for p in runs_root.rglob("*")
        if p.is_dir()
        and run_id.lower() in p.name.lower()
    ]

    if len(partial) == 1:
        return partial[0]

    if not partial:
        raise FileNotFoundError(
            f"Could not find directory for {run_id} under:\n"
            f"{runs_root}"
        )

    raise RuntimeError(
        f"Multiple possible directories found for {run_id}:\n"
        + "\n".join(str(p) for p in partial)
    )


# ============================================================
# FIND UNIQUE FILE
# ============================================================

def find_unique_file(
    run_dir: Path,
    filename: str,
) -> Path:

    files = list(run_dir.rglob(filename))

    if len(files) == 1:
        return files[0]

    if not files:
        raise FileNotFoundError(
            f"{filename} not found under:\n{run_dir}"
        )

    raise RuntimeError(
        f"Multiple {filename} files found under {run_dir}:\n"
        + "\n".join(str(p) for p in files)
    )


# ============================================================
# SELECT FIT ROW
# ============================================================

def select_fit_row(
    fit_df: pd.DataFrame,
    threshold_mV: float,
    run_id: str,
) -> pd.Series:

    if fit_df.empty:
        raise RuntimeError(
            f"{run_id}: fit.csv is empty"
        )

    candidates = fit_df.copy()

    # Select requested timing threshold when possible.
    if "parameter" in candidates.columns:
        parameter = pd.to_numeric(
            candidates["parameter"],
            errors="coerce",
        )

        mask = np.isclose(
            parameter,
            threshold_mV,
            rtol=0,
            atol=1e-6,
        )

        matches = candidates[mask]

        if not matches.empty:
            candidates = matches

    # If multiple methods exist, prefer Pico-TDC LED.
    if (
        len(candidates) > 1
        and "method" in candidates.columns
    ):
        method_mask = (
            candidates["method"]
            .astype(str)
            .str.contains(
                "Pico-TDC LED",
                case=False,
                regex=False,
            )
        )

        method_matches = candidates[method_mask]

        if not method_matches.empty:
            candidates = method_matches

    if len(candidates) != 1:
        raise RuntimeError(
            f"{run_id}: could not uniquely select fit row "
            f"for T_th = {threshold_mV:g} mV.\n"
            f"Candidate rows:\n"
            f"{candidates.to_string(index=False)}"
        )

    return candidates.iloc[0]


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    # --------------------------------------------------------
    # Load summary
    # --------------------------------------------------------

    summary = pd.read_csv(args.summary)

    required_columns = {
        "run_id",
        "Voltage",
        "AcquisitionMode",
        "E_th",
        "T_th",
        "gaussian_area_events",
        "CTR_ps",
        "CTR_error_ps",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise RuntimeError(
            "Missing required columns in summary.csv:\n"
            + ", ".join(sorted(missing))
        )

    # Numeric conversion
    for column in [
        "Voltage",
        "E_th",
        "T_th",
        "gaussian_area_events",
        "CTR_ps",
        "CTR_error_ps",
    ]:
        summary[column] = pd.to_numeric(
            summary[column],
            errors="coerce",
        )

    # ========================================================
    # SELECT FIXED VOLTAGE
    # ========================================================

    selected = summary[
        np.isclose(
            summary["Voltage"],
            args.voltage,
            rtol=0,
            atol=1e-6,
        )
        & (
            summary["AcquisitionMode"].astype(str)
            == args.mode
        )
    ].copy()

    if selected.empty:
        raise RuntimeError(
            f"No runs found for:\n"
            f"  Voltage = {args.voltage:g} V\n"
            f"  AcquisitionMode = {args.mode}"
        )

    # ========================================================
    # RESOLVE DUPLICATES
    #
    # For each timing threshold, keep the run with the
    # largest gaussian_area_events.
    #
    # E_th is deliberately NOT constrained.
    # ========================================================

    selected = (
        selected
        .dropna(
            subset=[
                "T_th",
                "gaussian_area_events",
            ]
        )
        .sort_values(
            [
                "T_th",
                "gaussian_area_events",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=["T_th"],
            keep="first",
        )
        .sort_values("T_th")
        .reset_index(drop=True)
    )

    print("\n========================================")
    print(
        f"Selected runs at {args.voltage:g} V"
    )
    print("========================================")

    print(
        selected[
            [
                "run_id",
                "Voltage",
                "E_th",
                "T_th",
                "gaussian_area_events",
                "CTR_ps",
                "CTR_error_ps",
            ]
        ].to_string(index=False)
    )

    # ========================================================
    # RECOVER EFFICIENCY
    # ========================================================

    rows = []

    for _, row in selected.iterrows():

        run_id = str(row["run_id"])
        threshold = float(row["T_th"])

        print(
            "\n----------------------------------------"
        )
        print(
            f"{run_id}: "
            f"V={args.voltage:g} V, "
            f"E_th={row['E_th']:g} mV, "
            f"T_th={threshold:g} mV"
        )

        # ----------------------------------------------------
        # Find run
        # ----------------------------------------------------

        run_dir = find_run_directory(
            args.runs_root,
            run_id,
        )

        print(f"Directory: {run_dir}")

        energy_path = find_unique_file(
            run_dir,
            "energy_selection.csv",
        )

        fit_path = find_unique_file(
            run_dir,
            "fit.csv",
        )

        # ----------------------------------------------------
        # Photopeak events
        # ----------------------------------------------------

        energy_df = pd.read_csv(
            energy_path
        )

        if "duration_selected" not in energy_df.columns:
            raise RuntimeError(
                f"{run_id}: 'duration_selected' "
                "missing from energy_selection.csv"
            )

        duration_selected = pd.to_numeric(
            energy_df["duration_selected"],
            errors="coerce",
        )

        n_photopeak = int(
            (duration_selected == 1).sum()
        )

        if n_photopeak <= 0:
            raise RuntimeError(
                f"{run_id}: N_photopeak = 0"
            )

        # ----------------------------------------------------
        # Final-fit events
        # ----------------------------------------------------

        fit_df = pd.read_csv(
            fit_path
        )

        fit_row = select_fit_row(
            fit_df,
            threshold,
            run_id,
        )

        if "n_fit" not in fit_row.index:
            raise RuntimeError(
                f"{run_id}: n_fit missing from fit.csv"
            )

        n_fit = int(
            float(fit_row["n_fit"])
        )

        # ----------------------------------------------------
        # Efficiency
        # ----------------------------------------------------

        efficiency = (
            n_fit / n_photopeak
        )

        if efficiency > 1.0 + 1e-12:
            raise RuntimeError(
                f"{run_id}: invalid efficiency > 1:\n"
                f"N_fit = {n_fit}\n"
                f"N_photopeak = {n_photopeak}"
            )

        # Binomial statistical uncertainty
        efficiency_error = np.sqrt(
            efficiency
            * (1.0 - efficiency)
            / n_photopeak
        )

        print(
            f"N_photopeak = {n_photopeak}"
        )
        print(
            f"N_fit       = {n_fit}"
        )
        print(
            f"Efficiency  = "
            f"{100 * efficiency:.2f} %"
        )

        rows.append(
            {
                "run_id": run_id,
                "Voltage": float(
                    row["Voltage"]
                ),
                "E_th": float(
                    row["E_th"]
                ),
                "T_th": threshold,
                "gaussian_area_events": float(
                    row["gaussian_area_events"]
                ),
                "CTR_ps": float(
                    row["CTR_ps"]
                ),
                "CTR_error_ps": float(
                    row["CTR_error_ps"]
                ),
                "n_photopeak": n_photopeak,
                "n_fit": n_fit,
                "efficiency": efficiency,
                "efficiency_error": efficiency_error,
            }
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    result = (
        pd.DataFrame(rows)
        .sort_values("T_th")
        .reset_index(drop=True)
    )

    print("\n========================================")
    print("FINAL SCAN")
    print("========================================")

    display = result.copy()

    display["efficiency_percent"] = (
        100 * display["efficiency"]
    )

    print(
        display[
            [
                "run_id",
                "E_th",
                "T_th",
                "n_photopeak",
                "n_fit",
                "efficiency_percent",
                "CTR_ps",
                "CTR_error_ps",
            ]
        ].to_string(
            index=False,
            formatters={
                "efficiency_percent":
                    lambda x: f"{x:.2f}",
            },
        )
    )

    # ========================================================
    # SAVE COMBINED NUMERICAL DATA
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_output = (
        args.output.parent
        / f"ctr_efficiency_scan_{args.voltage:g}V.csv"
    )

    result.to_csv(
        csv_output,
        index=False,
    )

    # ========================================================
    # PLOT STYLE
    # ========================================================

    plt.rcParams.update(
        {
            "font.size": 20,
            "axes.labelsize": 23,
            "xtick.labelsize": 19,
            "ytick.labelsize": 19,
            "legend.fontsize": 17,
            "axes.linewidth": 1.4,
        }
    )

    threshold = (
        result["T_th"]
        .to_numpy(dtype=float)
    )

    ctr = (
        result["CTR_ps"]
        .to_numpy(dtype=float)
    )

    ctr_error = (
        result["CTR_error_ps"]
        .to_numpy(dtype=float)
    )

    efficiency_percent = (
        100
        * result["efficiency"]
        .to_numpy(dtype=float)
    )

    efficiency_error_percent = (
        100
        * result["efficiency_error"]
        .to_numpy(dtype=float)
    )

    fig, ax_ctr = plt.subplots(
        figsize=(10.5, 6.5)
    )

    # ========================================================
    # CTR
    # ========================================================

    ax_ctr.errorbar(
        threshold,
        ctr,
        yerr=ctr_error,
        fmt="s-",
        markersize=9,
        linewidth=2.4,
        elinewidth=2.0,
        capsize=5,
        label="CTR",
    )

    ax_ctr.set_xlabel(
        r"Timing threshold $T_{\mathrm{th}}$ [mV]"
    )

    ax_ctr.set_ylabel(
        "CTR [ps]"
    )

    ax_ctr.grid(
        True,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )

    ax_ctr.set_axisbelow(True)

    # ========================================================
    # EVENT EFFICIENCY
    # ========================================================

    ax_eff = ax_ctr.twinx()

    ax_eff.plot(
        threshold,
        efficiency_percent,
        "o--",
        markersize=9,
        linewidth=2.4,
        label="Event efficiency",
    )

    ax_eff.set_ylabel(
        r"Event efficiency "
        r"$N_{\mathrm{fit}}/N_{\mathrm{photopeak}}$ [\%]"
    )

    # Keep the upper edge physically meaningful.
    lower = np.min(
        efficiency_percent
        - efficiency_error_percent
    )

    ax_eff.set_ylim(
        max(
            0.0,
            lower - 2.0,
        ),
        100.5,
    )


    # ========================================================
    # LEGEND
    # ========================================================

    handles_ctr, labels_ctr = (
        ax_ctr.get_legend_handles_labels()
    )

    handles_eff, labels_eff = (
        ax_eff.get_legend_handles_labels()
    )

    ax_ctr.legend(
        handles_ctr + handles_eff,
        labels_ctr + labels_eff,
        loc="best",
        frameon=True,
    )

    # Exact threshold ticks
    ax_ctr.set_xticks(
        threshold
    )

    fig.tight_layout()

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    fig.savefig(
        args.output,
        bbox_inches="tight",
    )

    png_output = (
        args.output.with_suffix(".png")
    )

    fig.savefig(
        png_output,
        dpi=300,
        bbox_inches="tight",
    )

    print("\n========================================")
    print("OUTPUT")
    print("========================================")
    print(f"Data: {csv_output}")
    print(f"PDF : {args.output}")
    print(f"PNG : {png_output}")

    plt.show()

    # ============================================================
    # SECOND PLOT: CTR ONLY VS TIMING THRESHOLD
    # ============================================================

    fig_ctr, ax = plt.subplots(
        figsize=(9.5, 6.0)
    )

    ax.errorbar(
        threshold,
        ctr,
        yerr=ctr_error,
        fmt="s-",
        markersize=9,
        linewidth=2.4,
        elinewidth=2.0,
        capsize=5,
        label="CTR",
    )

    ax.set_xlabel(
        r"Timing threshold $T_{\mathrm{th}}$ [mV]"
    )

    ax.set_ylabel(
        "CTR [ps]"
    )

    # Show exactly the tested thresholds
    ax.set_xticks(threshold)

    ax.grid(
        True,
        linestyle="--",
        linewidth=1.0,
        alpha=0.5,
    )

    ax.set_axisbelow(True)

    fig_ctr.tight_layout()


    # ============================================================
    # SAVE CTR-ONLY PLOT
    # ============================================================

    ctr_only_pdf = (
        args.output.parent
        / f"ctr_vs_time_threshold_{args.voltage:g}V.pdf"
    )

    ctr_only_png = (
        args.output.parent
        / f"ctr_vs_time_threshold_{args.voltage:g}V.png"
    )

    fig_ctr.savefig(
        ctr_only_pdf,
        bbox_inches="tight",
    )

    fig_ctr.savefig(
        ctr_only_png,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"CTR-only PDF: {ctr_only_pdf}")
    print(f"CTR-only PNG: {ctr_only_png}")

    plt.show()



    # ============================================================
    # SAVE CTR-ONLY PLOT
    # ============================================================

    ctr_only_pdf = (
        args.output.parent
        / f"ctr_vs_time_threshold_{args.voltage:g}V.pdf"
    )

    ctr_only_png = (
        args.output.parent
        / f"ctr_vs_time_threshold_{args.voltage:g}V.png"
    )

    fig_ctr.savefig(
        ctr_only_pdf,
        bbox_inches="tight",
    )

    fig_ctr.savefig(
        ctr_only_png,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"CTR-only PDF: {ctr_only_pdf}")
    print(f"CTR-only PNG: {ctr_only_png}")

    plt.show()


if __name__ == "__main__":
    main()