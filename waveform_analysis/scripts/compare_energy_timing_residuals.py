#!/usr/bin/env python3
"""UC/FBK LED-vs-ML residuals: nonparametric Tukey-IQR outlier analysis.

Input root/UC and root/FBK, each containing energy_predictions.npz and
timing_predictions.npz with event_id, led_residual_ps, corrected_ps.
No Gaussian core or tail fit; outliers are outside Q1-3*IQR or Q3+3*IQR
in at least one channel. Quantiles are estimated per board and per state.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_predictions(path):
    with np.load(path, allow_pickle=False) as d:
        required = ("event_id", "led_residual_ps", "corrected_ps")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"{path}: missing {missing}")
        ids = np.asarray(d["event_id"])
        led = np.asarray(d["led_residual_ps"], float)
        ml = np.asarray(d["corrected_ps"], float)
    if any(a.ndim != 1 for a in (ids, led, ml)):
        raise ValueError(f"{path}: expected 1D arrays")
    if len(ids) < 20 or len(ids) != len(led) or len(ids) != len(ml):
        raise ValueError(f"{path}: inconsistent lengths or fewer than 20 events")
    if ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{path}: event_id must be unique integer values")
    if not np.isfinite(led).all() or not np.isfinite(ml).all():
        raise ValueError(f"{path}: residuals contain nonfinite values")
    return ids.astype(np.int64), led, ml


def align_prediction_pairs(energy, timing):
    ids, ei, ti = np.intersect1d(energy[0], timing[0], assume_unique=True,
                                 return_indices=True)
    if len(ids) < 20:
        raise ValueError(f"Only {len(ids)} energy/timing event IDs matched")
    return ids, np.column_stack((energy[1][ei], timing[1][ti])), np.column_stack(
        (energy[2][ei], timing[2][ti]))


def resolve_board_files(root, board):
    folder = Path(root) / board
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    found = {}
    for name in ("energy", "timing"):
        choices = (folder / f"{name}_predictions.npz", folder / f"{name}.npz")
        matches = [p for p in choices if p.is_file()]
        if len(matches) != 1:
            raise ValueError(f"{folder}: expected exactly one {name} file: "
                             f"{', '.join(p.name for p in choices)}")
        found[name] = matches[0]
    return found


def tukey_fences(points, multiplier=3.):
    if multiplier <= 0:
        raise ValueError("IQR multiplier must be positive")
    q1, q3 = np.quantile(points, (.25, .75), axis=0)
    iqr = q3-q1
    if np.any(iqr <= 0):
        raise ValueError("IQR is zero in a residual channel")
    lower = q1-multiplier*iqr
    upper = q3+multiplier*iqr
    outlier = np.any((points < lower) | (points > upper), axis=1)
    return {"q1_ps": q1, "q3_ps": q3, "iqr_ps": iqr,
            "lower_ps": lower, "upper_ps": upper,
            "multiplier": multiplier,
            "rule": "outlier if energy OR timing is outside its Tukey fences"}, outlier


def empirical_statistics(points):
    x = np.asarray(points, float)
    if len(x) < 3:
        return None
    cov = np.cov(x, rowvar=False)
    sd = np.sqrt(np.diag(cov))
    rho = float(cov[0, 1]/np.prod(sd)) if np.all(sd > 0) else None
    return {"n": len(x), "mean_energy_ps": float(x[:, 0].mean()),
            "mean_timing_ps": float(x[:, 1].mean()),
            "sigma_energy_ps": float(sd[0]),
            "sigma_timing_ps": float(sd[1]),
            "correlation": rho, "cov_energy_timing_ps2": float(cov[0, 1])}


def analyze_distribution(points, multiplier=3.):
    fences, mask = tukey_fences(points, multiplier)
    report = {
        "n_events": len(points),
        "method": "Extreme Tukey IQR fences; nonparametric",
        "outlier_fences": fences,
        "inlier_count": int((~mask).sum()),
        "outlier_count": int(mask.sum()),
        "outlier_fraction": float(mask.mean()),
        "empirical_inliers": empirical_statistics(points[~mask]),
        "empirical_outliers": empirical_statistics(points[mask]),
        "empirical_all": empirical_statistics(points),
    }
    return report, mask


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (tuple, list)):
        return [jsonable(x) for x in obj]
    return obj


def save_json(path, payload):
    Path(path).write_text(json.dumps(jsonable(payload), indent=2) + "\n",
                          encoding="utf-8")


def save_plot(fig, folder, basename):
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{basename}.{extension}",
                    dpi=190 if extension == "png" else None)
    plt.close(fig)


def scatter_limits(point_sets):
    stacked = np.concatenate(point_sets, axis=0)
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    pad = np.maximum(.03*(hi-lo), 1.)
    return lo-pad, hi+pad


def scatter_colored(ax, points, outlier, lo, hi, rng, title):
    """Show ALL IQR outliers, and a reproducible sample of inliers."""
    ordinary = np.flatnonzero(~outlier)
    extreme = np.flatnonzero(outlier)
    if len(ordinary) > 20000:
        ordinary = rng.choice(ordinary, 20000, replace=False)
    ax.scatter(points[ordinary, 0], points[ordinary, 1],
               color=".50", s=6, alpha=.16, rasterized=True,
               label="Within IQR fences")
    if len(extreme):
        ax.scatter(points[extreme, 0], points[extreme, 1],
                   color="tab:orange", s=13, alpha=.73, rasterized=True,
                   label=f"IQR outliers (n={len(extreme)})")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_title(title)
    ax.set_xlabel("Energy residual (ps)")
    ax.set_ylabel("Timing residual (ps)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def distribution_panel(ax, items, axis, title):
    width = (10., 5.)[axis]
    boundaries = []
    for label, points, fences, color in items:
        boundaries.append((fences["lower_ps"][axis], fences["upper_ps"][axis]))
    lo = min(v[0] for v in boundaries)
    hi = max(v[1] for v in boundaries)
    edges = np.arange(np.floor(lo/width)*width,
                      np.ceil(hi/width)*width+1.5*width, width)
    maximum = 1.
    for label, points, fences, color in items:
        counts = np.histogram(points[:, axis], bins=edges)[0]
        maximum = max(maximum, float(counts.max()))
        ax.stairs(counts, edges, label=label, color=color, lw=1.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0., maximum*1.12)
    ax.set_title(title)
    ax.set_xlabel("Residual (ps)")
    ax.set_ylabel(f"Events / {width:g} ps")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)


def draw_per_board(board, payload, output, rng):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, channel in enumerate(("Energy", "Timing")):
        items = [(label, payload["pairs"][state],
                  payload["reports"][state]["outlier_fences"], color)
                 for state, label, color in (
                     ("led", "LED", "tab:orange"), ("ml", "ML", "tab:blue"))]
        distribution_panel(axes[axis], items, axis, f"{board} — {channel}")
    save_plot(fig, output, "led_vs_ml_distributions")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    limits = scatter_limits(list(payload["pairs"].values()))
    for ax, state, label in zip(axes, ("led", "ml"), ("LED", "ML")):
        scatter_colored(ax, payload["pairs"][state], payload["masks"][state],
                        *limits, rng, f"{board} — {label}")
    save_plot(fig, output, "led_vs_ml_iqr_scatter")


def draw_cross_board(data, output, rng):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, state in enumerate(("led", "ml")):
        for col, channel in enumerate(("Energy", "Timing")):
            items = [(board, data[board]["pairs"][state],
                      data[board]["reports"][state]["outlier_fences"], color)
                     for board, color in (("UC", "tab:blue"),
                                          ("FBK", "tab:orange"))]
            distribution_panel(axes[row, col], items, col,
                               f"{state.upper()} — {channel}")
    save_plot(fig, output, "uc_vs_fbk_distributions")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    limits = scatter_limits([data[board]["pairs"][state]
                             for board in ("UC", "FBK")
                             for state in ("led", "ml")])
    for row, state in enumerate(("led", "ml")):
        for col, board in enumerate(("UC", "FBK")):
            scatter_colored(axes[row, col], data[board]["pairs"][state],
                            data[board]["masks"][state], *limits, rng,
                            f"{board} — {state.upper()}")
    save_plot(fig, output, "uc_vs_fbk_iqr_scatter")


def bootstrap_rho(points, n_draws, rng):
    """Conditional bootstrap of a fixed, selected empirical population."""
    x = np.asarray(points)
    if len(x) < 4 or np.any(np.std(x, axis=0) <= 0):
        return None
    rho = np.full(n_draws, np.nan)
    for i in range(n_draws):
        sample = x[rng.integers(0, len(x), len(x))]
        if np.all(np.std(sample, axis=0) > 0):
            rho[i] = np.corrcoef(sample[:, 0], sample[:, 1])[0, 1]
    return rho if np.isfinite(rho).all() else None


def compare_boards(data, samples=300, seed=42):
    rng = np.random.default_rng(seed)
    comparison = {
        "method": "Nonparametric Tukey-IQR; no Gaussian assumptions",
        "bootstrap_draws": samples,
        "note": "Intervals resample fixed outlier/inlier memberships, not IQR fences. "
                "Including zero is not proof of statistical equivalence.",
        "states": {},
    }
    for state in ("led", "ml"):
        result = {"outlier_fraction": {}, "populations": {}}
        uc = data["UC"]["reports"][state]["outlier_fraction"]
        fbk = data["FBK"]["reports"][state]["outlier_fraction"]
        result["outlier_fraction"] = {"UC": uc, "FBK": fbk, "UC_minus_FBK": uc-fbk}
        for region in ("inliers", "outliers", "all"):
            selected = {}
            for board in ("UC", "FBK"):
                points = data[board]["pairs"][state]
                mask = data[board]["masks"][state]
                selected[board] = (points[~mask] if region == "inliers" else
                                   points[mask] if region == "outliers" else points)
            stats = {board: empirical_statistics(selected[board])
                     for board in ("UC", "FBK")}
            section = {"n_UC": len(selected["UC"]), "n_FBK": len(selected["FBK"])}
            for key in ("sigma_energy_ps", "sigma_timing_ps", "correlation",
                        "cov_energy_timing_ps2"):
                a = stats["UC"][key] if stats["UC"] else None
                b = stats["FBK"][key] if stats["FBK"] else None
                section[key] = {"UC": a, "FBK": b,
                                "UC_minus_FBK": a-b if a is not None and b is not None else None}
            a, b = (bootstrap_rho(selected[board], samples, rng)
                    for board in ("UC", "FBK"))
            ci = np.quantile(a-b, [.025, .975]).tolist() if a is not None and b is not None else None
            section["rho_difference_bootstrap_95_ci"] = ci
            section["interval_contains_zero"] = (
                bool(ci[0] <= 0 <= ci[1]) if ci is not None else None)
            result["populations"][region] = section
        comparison["states"][state] = result
    comparison["model_corrections_led_minus_ml"] = {}
    for key in ("correlation", "cov_energy_timing_ps2",
                "sigma_energy_ps", "sigma_timing_ps"):
        uc = data["UC"]["correction_stats"][key]
        fbk = data["FBK"]["correction_stats"][key]
        comparison["model_corrections_led_minus_ml"][key] = {
            "UC": uc, "FBK": fbk,
            "UC_minus_FBK": uc - fbk if uc is not None and fbk is not None else None,
        }
    return comparison


def report_board(board, report, output):
    print("\n" + "="*76)
    print(f"{board}: NONPARAMETRIC TUKEY-IQR OUTLIERS", flush=True)
    print("="*76)
    for state in ("led", "ml"):
        result = report[state]
        bounds = result["outlier_fences"]
        print(f" {state.upper()}: {result['n_events']:,} paired events")
        print(f"   Energy fences: [{bounds['lower_ps'][0]:.2f}, "
              f"{bounds['upper_ps'][0]:.2f}] ps")
        print(f"   Timing fences: [{bounds['lower_ps'][1]:.2f}, "
              f"{bounds['upper_ps'][1]:.2f}] ps")
        print(f"   Outliers: {result['outlier_count']:,} "
              f"({100*result['outlier_fraction']:.4f}%)")
        for key, label in (("empirical_inliers", "Within fences"),
                           ("empirical_outliers", "Outside fences"),
                           ("empirical_all", "All events")):
            stats = result[key]
            if stats is None:
                print(f"   {label}: fewer than three observations")
                continue
            corr = (f"{stats['correlation']:.5f}" if stats["correlation"] is not None
                    else "undefined")
            print(f"   {label}: N={stats['n']}, "
                  f"sigmaE={stats['sigma_energy_ps']:.2f} ps, "
                  f"sigmaT={stats['sigma_timing_ps']:.2f} ps, "
                  f"rho={corr}, Cov={stats['cov_energy_timing_ps2']:.2f} ps²")
    print(f" Output: {output}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("iqr_results"))
    parser.add_argument("--iqr-multiplier", type=float, default=3.,
                        help="Extreme Tukey fence multiplier (default 3)")
    parser.add_argument("--bootstrap-resamples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.iqr_multiplier <= 0:
        parser.error("--iqr-multiplier must be positive")
    if args.bootstrap_resamples < 50:
        parser.error("--bootstrap-resamples must be >= 50")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    data = {}
    print(f"Nonparametric Tukey method, multiplier={args.iqr_multiplier:g}*IQR.")
    print("Outlier if E or T is beyond its corresponding quartile fences.")
    print("Fences estimated separately for UC/FBK and LED/ML; NO Gaussian fit.")
    for board in ("UC", "FBK"):
        files = resolve_board_files(args.predictions_root, board)
        _, led, ml = align_prediction_pairs(
            load_predictions(files["energy"]), load_predictions(files["timing"]))
        reports, masks = {}, {}
        for state, points in (("led", led), ("ml", ml)):
            reports[state], masks[state] = analyze_distribution(
                points, multiplier=args.iqr_multiplier)
        output = args.output / board
        output.mkdir(exist_ok=True)
        corrections = led - ml  # Signed model outputs: LED residual minus ML residual.
        correction_stats = empirical_statistics(corrections)
        save_json(output / "iqr_summary.json", {
            "board": board, "energy_file": str(files["energy"]),
            "timing_file": str(files["timing"]), "states": reports,
            "model_corrections_led_minus_ml": correction_stats})
        data[board] = {"pairs": {"led": led, "ml": ml}, "reports": reports,
                       "masks": masks, "corrections": corrections,
                       "correction_stats": correction_stats}
        draw_per_board(board, data[board], output, rng)
        report_board(board, reports, output)
        print(" MODEL OUTPUT CORRELATION (not residual correlation):")
        print("   delta_E = LED_energy_residual - ML_energy_residual")
        print("   delta_T = LED_timing_residual - ML_timing_residual")
        print(f'   Matched events: {correction_stats["n"]:,}')
        print(f'   Pearson rho(delta_E, delta_T) = '
              f'{correction_stats["correlation"]:.5f}')
        print(f'   Cov(delta_E, delta_T) = '
              f'{correction_stats["cov_energy_timing_ps2"]:.2f} ps²')
        print(f'   sigma(delta_E) = {correction_stats["sigma_energy_ps"]:.2f} ps, '
              f'sigma(delta_T) = {correction_stats["sigma_timing_ps"]:.2f} ps')

    comparison = compare_boards(data, samples=args.bootstrap_resamples,
                                seed=args.seed)
    save_json(args.output / "uc_fbk_iqr_comparison.json", comparison)
    draw_cross_board(data, args.output, rng)
    print("\n" + "="*76)
    print("UC vs FBK | EMPIRICAL COMPARISON")
    print("="*76)
    for state, result in comparison["states"].items():
        freq = result["outlier_fraction"]
        print(f"{state.upper()}: outlier fraction UC={freq['UC']:.4%}, "
              f"FBK={freq['FBK']:.4%}, delta={freq['UC_minus_FBK']:+.4%}")
        for region, stats in result["populations"].items():
            corr = stats["correlation"]
            a, b, delta = corr["UC"], corr["FBK"], corr["UC_minus_FBK"]
            if delta is None:
                print(f"  {region}: correlation unavailable")
                continue
            print(f"  {region}: rho_UC={a:.5f}, rho_FBK={b:.5f}, delta={delta:+.5f}")
            ci = stats["rho_difference_bootstrap_95_ci"]
            if ci is not None:
                print(f"    Conditional 95% bootstrap delta-rho interval: "
                      f"[{ci[0]:+.5f}, {ci[1]:+.5f}], includes zero="
                      f"{stats['interval_contains_zero']}")
    model_corr = comparison["model_corrections_led_minus_ml"]
    print("\\nMODEL-OUTPUT CORRELATION COMPARISON (LED residual - ML residual)")
    for key in ("correlation", "cov_energy_timing_ps2",
                "sigma_energy_ps", "sigma_timing_ps"):
        value = model_corr[key]
        print(f'  {key}: UC={value["UC"]:.5f}, FBK={value["FBK"]:.5f}, '
              f'UC-FBK={value["UC_minus_FBK"]:+.5f}')
    print("  Correlation of model outputs is different from correlation of ML residuals.")
    print("Intervals do not include uncertainty from estimating IQR fences.")
    print("Including zero does not establish statistical equivalence.")
    print(f"Outputs: {args.output.resolve()}")
    print("  UC/iqr_summary.json; FBK/iqr_summary.json")
    print("  UC/led_vs_ml_distributions.png/pdf; FBK/led_vs_ml_distributions.png/pdf")
    print("  UC/led_vs_ml_iqr_scatter.png/pdf; FBK/led_vs_ml_iqr_scatter.png/pdf")
    print("  uc_fbk_iqr_comparison.json")
    print("  uc_vs_fbk_distributions.png/pdf; uc_vs_fbk_iqr_scatter.png/pdf")


if __name__ == "__main__":
    main()
