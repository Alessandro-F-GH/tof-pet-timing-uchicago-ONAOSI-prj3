#!/usr/bin/env python3
"""Compare energy and timing residuals for events having the same event_id.

Read two saved prediction NPZ files; no model/data provenance checks, model
loading, retraining, or prediction. Users choose matching input populations.
The corrected_ps values are residuals AFTER the model correction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from waveform_analysis.reporting.stats import metric_values


def read_residuals(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        if "event_id" not in data or "corrected_ps" not in data:
            raise ValueError(f"{path}: requires event_id and corrected_ps")
        event_id = np.asarray(data["event_id"])
        residual = np.asarray(data["corrected_ps"], dtype=np.float64)
    if event_id.ndim != 1 or residual.ndim != 1 or len(event_id) != len(residual):
        raise ValueError(f"{path}: event_id and corrected_ps must be same-length 1D arrays")
    if event_id.dtype.kind not in "iu":
        raise ValueError(f"{path}: event_id must contain integers")
    if len(event_id) != len(np.unique(event_id)):
        raise ValueError(f"{path}: duplicate event_id would produce ambiguous matches")
    if len(event_id) == 0:
        raise ValueError(f"{path}: empty prediction file")
    return event_id.astype(np.int64, copy=False), residual


def align_residuals(energy_ids, energy_residuals, timing_ids, timing_residuals):
    """Join on exact event ID; sorted ascending to guarantee reproducibility."""
    ids, indices_energy, indices_timing = np.intersect1d(
        energy_ids, timing_ids, assume_unique=True, return_indices=True)
    if len(ids) < 3:
        raise ValueError(f"Only {len(ids)} matched event IDs; need at least three")
    energy = np.asarray(energy_residuals[indices_energy], dtype=np.float64)
    timing = np.asarray(timing_residuals[indices_timing], dtype=np.float64)
    valid = np.isfinite(energy) & np.isfinite(timing)
    if int(np.sum(valid)) < 3:
        raise ValueError("Fewer than three finite matched residual pairs")
    return ids[valid], energy[valid], timing[valid], int((~valid).sum())


def correlations(energy, timing):
    if np.ptp(energy) == 0 or np.ptp(timing) == 0:
        raise ValueError("Correlation undefined: constant residual values")
    pearson = pearsonr(energy, timing)
    spearman = spearmanr(energy, timing)
    return {
        "n_matched_finite_events": int(len(energy)),
        "pearson_r": float(pearson.statistic),
        "pearson_p_value": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p_value": float(spearman.pvalue),
        "energy_mean_ps": float(np.mean(energy)),
        "timing_mean_ps": float(np.mean(timing)),
        "energy_std_ps": float(np.std(energy, ddof=1)),
        "timing_std_ps": float(np.std(timing, ddof=1)),
    }


def difference_statistics(energy, timing):
    """Observed difference and variance-subtraction test on paired events."""
    difference = energy - timing
    energy_std = float(np.std(energy, ddof=1))
    timing_std = float(np.std(timing, ddof=1))
    variance_excess = energy_std**2 - timing_std**2
    return {
        "difference_mean_ps": float(np.mean(difference)),
        "difference_std_ps": float(np.std(difference, ddof=1)),
        "difference_rmse_ps": float(np.sqrt(np.mean(difference**2))),
        "energy_variance_minus_timing_variance_ps2": variance_excess,
        "predicted_extra_std_ps_if_timing_is_common_only":
            float(np.sqrt(variance_excess)) if variance_excess >= 0 else None,
        "note_difference": "The quadrature prediction uses full-sample standard "
            "deviations, not fitted CTR. It assumes timing=C and energy=C+N, "
            "with C independent of N.",
    }


def ctr_quadrature_comparison(energy, timing, *, bin_width_ps, seed):
    """Fitted CTR for energy, timing, their difference; common+extra prediction."""
    config = {"histogram_bin_width_ps": float(bin_width_ps)}
    ctr_energy = float(metric_values(energy, config, seed=seed)["ctr_ps"])
    ctr_timing = float(metric_values(timing, config, seed=seed)["ctr_ps"])
    ctr_difference = float(metric_values(energy-timing, config, seed=seed)["ctr_ps"])
    excess = ctr_energy**2 - ctr_timing**2
    predicted = float(np.sqrt(excess)) if excess >= 0 else None
    return {
        "ctr_energy_ps": ctr_energy,
        "ctr_timing_ps": ctr_timing,
        "ctr_difference_ps": ctr_difference,
        "predicted_difference_ctr_ps_if_timing_is_common_only": predicted,
        "difference_ctr_minus_prediction_ps": (
            float(ctr_difference-predicted) if predicted is not None else None),
        "ctr_bin_width_ps": float(bin_width_ps),
        "note_ctr": "CTR comparison requires approximately Gaussian residual "
            "distributions with consistent fitted-width definitions. No "
            "independent-source decomposition can be established from this alone.",
    }


def save_difference_plot(energy, timing, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(energy - timing, bins=100, density=True, alpha=0.75)
    ax.set_xlabel("Energy residual - timing residual (ps)")
    ax.set_ylabel("Density")
    ax.set_title("Difference of matched corrected residuals")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "residual_difference.png", dpi=180)
    fig.savefig(output / "residual_difference.pdf")
    plt.close(fig)


def save_plot(energy, timing, output):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(energy, timing, s=7, alpha=0.25, rasterized=True)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("Energy corrected residual (ps)")
    ax.set_ylabel("Timing corrected residual (ps)")
    ax.set_title("Matched-event energy vs timing residuals")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "residual_correlation.png", dpi=180)
    fig.savefig(output / "residual_correlation.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy", type=Path, required=True,
                        help="Energy predictions .npz with event_id and corrected_ps")
    parser.add_argument("--timing", type=Path, required=True,
                        help="Timing predictions .npz with event_id and corrected_ps")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-bin-width-ps", type=float, default=10.0,
                        help="Histogram bin width for CTR fits (default: 10 ps)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.fit_bin_width_ps <= 0:
        parser.error("--fit-bin-width-ps must be positive")
    energy_ids, energy_residuals = read_residuals(args.energy)
    timing_ids, timing_residuals = read_residuals(args.timing)
    ids, energy, timing, n_nonfinite = align_residuals(
        energy_ids, energy_residuals, timing_ids, timing_residuals)
    results = correlations(energy, timing)
    results.update(difference_statistics(energy, timing))
    results.update(ctr_quadrature_comparison(
        energy, timing, bin_width_ps=args.fit_bin_width_ps, seed=args.seed))
    results.update(
        n_energy_events=int(len(energy_ids)),
        n_timing_events=int(len(timing_ids)),
        n_common_ids=int(len(np.intersect1d(energy_ids, timing_ids, assume_unique=True))),
        n_nonfinite_pairs=int(n_nonfinite),
        energy_file=str(args.energy),
        timing_file=str(args.timing),
        residual_column="corrected_ps",
        note="Event IDs paired without source provenance verification. "
             "Development predictions can be in-sample, not out-of-fold.",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "matched_residuals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_id", "energy_corrected_ps", "timing_corrected_ps", "energy_minus_timing_ps"])
        writer.writerows(zip(ids.tolist(), energy.tolist(), timing.tolist(), (energy-timing).tolist()))
    (args.output / "correlation_summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_plot(energy, timing, args.output)
    save_difference_plot(energy, timing, args.output)
    print(f"Matched events: {len(ids)} | Pearson r={results['pearson_r']:.5f} "
          f"(p={results['pearson_p_value']:.3g}) | "
          f"Spearman rho={results['spearman_rho']:.5f} "
          f"(p={results['spearman_p_value']:.3g})", flush=True)
    print(f"Difference std={results['difference_std_ps']:.3f} ps | "
          f"quadrature prediction={results['predicted_extra_std_ps_if_timing_is_common_only']} ps",
          flush=True)
    print(f"Fitted CTR: energy={results['ctr_energy_ps']:.2f} ps, "
          f"timing={results['ctr_timing_ps']:.2f} ps, "
          f"difference={results['ctr_difference_ps']:.2f} ps | "
          f"quadrature prediction={results['predicted_difference_ctr_ps_if_timing_is_common_only']}",
          flush=True)
    print(f"Results: {args.output}", flush=True)


if __name__ == "__main__":
    main()
