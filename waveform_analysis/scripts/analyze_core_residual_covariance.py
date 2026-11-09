"""Separate ordinary-event covariance from extreme residual populations.

Pairs blind energy/timing predictions by event ID. The primary outlier rule
matches scan_photopeak_ctr._original_outlier_groups: select the largest
absolute ML residuals separately in energy and timing (default 0.2% per
channel). The same ML-defined event memberships are then used for LED and ML.

No waveform loading, inference, retraining, or cross-board event pairing.
The covariance-derived common-mode quantity is conditional on a unit-loading
additive common-component model; it is NOT automatically an intrinsic detector
limit or a CTR lower bound.
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
from scipy.stats import spearmanr

from waveform_analysis.scripts.plot_residual_correlation_percentiles import _read


GROUPS = ("all", "core", "energy_only", "timing_only", "shared")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _paired_board(energy_run: Path, timing_run: Path, artifact_root: Path | None):
    energy = _read(energy_run, artifact_root)
    timing = _read(timing_run, artifact_root)
    ids, ei, ti = np.intersect1d(
        energy[0], timing[0], assume_unique=True, return_indices=True
    )
    if len(ids) < 1000:
        raise ValueError(f"Only {len(ids)} matched blind events: {len(ids)}")
    led = np.column_stack((energy[1][ei], timing[1][ti]))
    ml = np.column_stack((energy[2][ei], timing[2][ti]))
    if not np.isfinite(led).all() or not np.isfinite(ml).all():
        raise ValueError("Non-finite residuals found after event-ID alignment")
    return ids, led, ml, {"energy": energy[3], "timing": timing[3]}


def _outlier_groups(ml: np.ndarray, tail_fraction: float):
    """Exact top-|residual| selection separately in the two ML channels."""
    if not 0 < tail_fraction < 0.5:
        raise ValueError("Each tail_fraction must be in (0, 0.5)")
    n = len(ml)
    count = max(1, int(np.ceil(n * tail_fraction)))
    tails = []
    thresholds = []
    for column in (0, 1):
        values = np.abs(ml[:, column])
        selected = np.argsort(values, kind="stable")[-count:]
        mask = np.zeros(n, dtype=bool)
        mask[selected] = True
        tails.append(mask)
        thresholds.append(float(values[selected].min()))
    energy_tail, timing_tail = tails
    groups = {
        "all": np.ones(n, dtype=bool),
        "core": ~(energy_tail | timing_tail),
        "energy_only": energy_tail & ~timing_tail,
        "timing_only": timing_tail & ~energy_tail,
        "shared": energy_tail & timing_tail,
    }
    return groups, {
        "tail_fraction_per_channel": tail_fraction,
        "events_selected_per_channel": count,
        "energy_abs_residual_threshold_ps": thresholds[0],
        "timing_abs_residual_threshold_ps": thresholds[1],
        "energy_tail_count": int(energy_tail.sum()),
        "timing_tail_count": int(timing_tail.sum()),
        "union_count": int((energy_tail | timing_tail).sum()),
        "intersection_count": int((energy_tail & timing_tail).sum()),
    }


def _statistics(points: np.ndarray) -> dict:
    x = np.asarray(points, dtype=float)
    n = len(x)
    if n < 3:
        return {
            "n": n, "mean_energy_ps": None, "mean_timing_ps": None,
            "variance_energy_ps2": None, "variance_timing_ps2": None,
            "covariance_ps2": None, "pearson": None, "spearman": None,
            "same_sign_fraction": None, "variance_difference_ps2": None,
            "shared_fraction_timing_proxy": None, "common_sigma_proxy_ps": None,
            "common_fwhm_equivalent_proxy_ps": None,
        }
    var_e = float(np.var(x[:, 0], ddof=1))
    var_t = float(np.var(x[:, 1], ddof=1))
    cov = float(np.cov(x[:, 0], x[:, 1], ddof=1)[0, 1])
    rho = cov / np.sqrt(var_e * var_t) if var_e > 0 and var_t > 0 else None
    common_sigma = float(np.sqrt(cov)) if cov > 0 else None
    return {
        "n": n,
        "mean_energy_ps": float(np.mean(x[:, 0])),
        "mean_timing_ps": float(np.mean(x[:, 1])),
        "variance_energy_ps2": var_e,
        "variance_timing_ps2": var_t,
        "covariance_ps2": cov,
        "pearson": float(rho) if rho is not None else None,
        "spearman": float(spearmanr(x[:, 0], x[:, 1]).statistic),
        "same_sign_fraction": float(np.mean(np.sign(x[:, 0]) == np.sign(x[:, 1]))),
        "variance_difference_ps2": float(np.var(x[:, 0] - x[:, 1], ddof=1)),
        "shared_fraction_timing_proxy": float(cov / var_t) if var_t > 0 else None,
        "common_sigma_proxy_ps": common_sigma,
        "common_fwhm_equivalent_proxy_ps": (
            float(2.354820045 * common_sigma) if common_sigma is not None else None
        ),
    }


def _bootstrap_paired(led: np.ndarray, ml: np.ndarray, draws: int, rng):
    """Paired event bootstrap; memberships are held fixed."""
    n = len(ml)
    if n < 4:
        return {"n_bootstrap": 0, "status": "insufficient events"}
    cov_led = np.empty(draws)
    cov_ml = np.empty(draws)
    var_t_led = np.empty(draws)
    var_t_ml = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, n, size=n)
        a = led[idx]
        z = ml[idx]
        cov_led[b] = np.cov(a[:, 0], a[:, 1], ddof=1)[0, 1]
        cov_ml[b] = np.cov(z[:, 0], z[:, 1], ddof=1)[0, 1]
        var_t_led[b] = np.var(a[:, 1], ddof=1)
        var_t_ml[b] = np.var(z[:, 1], ddof=1)

    def interval(values):
        return [float(v) for v in np.quantile(values, [0.025, 0.975])]

    positive_ml = np.maximum(cov_ml, 0.0)
    fwhm_ml = 2.354820045 * np.sqrt(positive_ml)
    return {
        "n_bootstrap": draws,
        "method": "paired event-level percentile bootstrap; fixed ML-defined membership",
        "covariance_led_ci95_ps2": interval(cov_led),
        "covariance_ml_ci95_ps2": interval(cov_ml),
        "covariance_led_minus_ml_ci95_ps2": interval(cov_led - cov_ml),
        "timing_variance_led_ci95_ps2": interval(var_t_led),
        "timing_variance_ml_ci95_ps2": interval(var_t_ml),
        "timing_variance_led_minus_ml_ci95_ps2": interval(var_t_led - var_t_ml),
        "bootstrap_fraction_ml_covariance_nonpositive": float(np.mean(cov_ml <= 0)),
        "common_fwhm_ml_proxy_ci95_ps": interval(fwhm_ml),
        "ml_covariance_ci_excludes_zero": bool(
            np.quantile(cov_ml, 0.025) > 0 or np.quantile(cov_ml, 0.975) < 0
        ),
    }


def _plot_scatter(board_data: dict, output: Path, seed: int):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    colors = {
        "core": "0.55", "energy_only": "tab:orange",
        "timing_only": "tab:blue", "shared": "tab:red",
    }
    for row, state in enumerate(("led", "ml")):
        for col, board in enumerate(("UC", "FBK")):
            ax = axes[row, col]
            points = board_data[board][state]
            groups = board_data[board]["groups"]
            core_idx = np.flatnonzero(groups["core"])
            if len(core_idx) > 15000:
                core_idx = rng.choice(core_idx, 15000, replace=False)
            ax.scatter(points[core_idx, 0], points[core_idx, 1],
                       s=5, alpha=.13, color=colors["core"],
                       rasterized=True, label="Core (sample)")
            for group in ("energy_only", "timing_only", "shared"):
                idx = np.flatnonzero(groups[group])
                if len(idx):
                    ax.scatter(points[idx, 0], points[idx, 1],
                               s=17, alpha=.75, color=colors[group],
                               rasterized=True, label=f"{group} (n={len(idx)})")
            ax.axhline(0, color="0.7", lw=.7)
            ax.axvline(0, color="0.7", lw=.7)
            ax.set_title(f"{board} — {state.upper()}")
            ax.set_xlabel("Energy residual [ps]")
            ax.set_ylabel("Timing residual [ps]")
            ax.grid(alpha=.15)
            ax.legend(fontsize=7)
    fig.savefig(output / "core_and_outlier_populations.png", dpi=180)
    plt.close(fig)


def _plot_sensitivity(rows: list[dict], output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, board in zip(axes, ("UC", "FBK")):
        for state, style in (("LED", "--"), ("ML", "-")):
            subset = [r for r in rows if r["board"] == board
                      and r["state"] == state and r["population"] == "core"]
            subset.sort(key=lambda r: r["tail_fraction_per_channel"])
            ax.plot(
                [100*r["tail_fraction_per_channel"] for r in subset],
                [r["covariance_ps2"] for r in subset],
                marker="o", linestyle=style, label=state,
            )
        ax.axhline(0, color="0.5", lw=.8)
        ax.set_title(board)
        ax.set_xlabel("Extreme tail removed per channel [%]")
        ax.set_ylabel("Core covariance [ps²]")
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle("Sensitivity of core covariance to outlier definition")
    fig.savefig(output / "core_covariance_tail_sensitivity.png", dpi=180)
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def analyze(args):
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    boards = {}
    summary_rows = []
    bootstrap_rows = []
    rng = np.random.default_rng(args.seed)

    for board in ("UC", "FBK"):
        prefix = board.lower()
        energy_run = getattr(args, f"{prefix}_energy_run").expanduser().resolve()
        timing_run = getattr(args, f"{prefix}_timing_run").expanduser().resolve()
        cache_root = getattr(args, f"{prefix}_cache_root")
        artifact_root = getattr(args, f"{prefix}_artifact_root")
        if cache_root and artifact_root:
            raise ValueError(f"{board}: specify cache-root OR artifact-root, not both")
        root = (
            artifact_root.expanduser().resolve() if artifact_root else
            cache_root.expanduser().resolve() / "artifact_store" if cache_root else None
        )
        ids, led, ml, sources = _paired_board(energy_run, timing_run, root)
        board_data = {"ids": ids, "led": led, "ml": ml, "sources": sources}
        primary_groups = None
        thresholds_by_fraction = {}

        for fraction in sorted(set(args.tail_fractions)):
            groups, thresholds = _outlier_groups(ml, fraction)
            thresholds_by_fraction[str(fraction)] = thresholds
            if np.isclose(fraction, args.bootstrap_tail_fraction):
                primary_groups = groups
            for population in GROUPS:
                mask = groups[population]
                for state, points in (("LED", led), ("ML", ml)):
                    stats = _statistics(points[mask])
                    summary_rows.append({
                        "board": board,
                        "tail_fraction_per_channel": fraction,
                        "population": population,
                        "state": state,
                        **stats,
                    })

        if primary_groups is None:
            raise ValueError(
                "--bootstrap-tail-fraction must be included in --tail-fractions"
            )
        board_data["groups"] = primary_groups
        boards[board] = board_data

        for population in GROUPS:
            mask = primary_groups[population]
            result = _bootstrap_paired(
                led[mask], ml[mask], args.bootstrap, rng
            )
            bootstrap_rows.append({
                "board": board,
                "tail_fraction_per_channel": args.bootstrap_tail_fraction,
                "population": population,
                "n_events": int(mask.sum()),
                **result,
            })

        board_data["thresholds"] = thresholds_by_fraction
        print(
            f"{board}: matched={len(ids):,}; "
            f"core={int(primary_groups['core'].sum()):,}; "
            f"energy-only={int(primary_groups['energy_only'].sum()):,}; "
            f"timing-only={int(primary_groups['timing_only'].sum()):,}; "
            f"shared={int(primary_groups['shared'].sum()):,}",
            flush=True,
        )

    _write_csv(out / "core_covariance_summary.csv", summary_rows)
    _write_csv(out / "core_covariance_bootstrap.csv", bootstrap_rows)
    _plot_scatter(boards, out, args.seed)
    _plot_sensitivity(summary_rows, out)

    payload = {
        "outlier_definition": (
            "For each board, select the largest absolute ML residuals separately "
            "in energy and timing, using ceil(N*tail_fraction) events per channel. "
            "Groups are disjoint: core, energy_only, timing_only, shared."
        ),
        "primary_tail_fraction_per_channel": args.bootstrap_tail_fraction,
        "sensitivity_tail_fractions_per_channel": sorted(set(args.tail_fractions)),
        "same_event_membership_for_led_and_ml": True,
        "bootstrap": {
            "draws": args.bootstrap,
            "seed": args.seed,
            "unit": "paired event rows within each board and population",
            "selection_uncertainty": (
                "Intervals condition on fixed ML-defined outlier memberships; "
                "the empirical tail thresholds are not re-estimated in each draw."
            ),
        },
        "covariance_interpretation": {
            "model": "e_energy = c + epsilon_energy; e_timing = c + epsilon_timing",
            "required_assumptions": [
                "The common component enters both residual channels with unit loading.",
                "Channel-specific components are mutually independent and independent of c.",
                "The covariance is not dominated by shared reference, acquisition, or model artifacts.",
                "A Gaussian-equivalent FWHM is meaningful only if c is approximately Gaussian."
            ],
            "warning": (
                "sqrt(covariance) and 2.35482*sqrt(covariance) are conditional "
                "common-mode proxies, not a demonstrated intrinsic LYSO+SiPM limit "
                "or a validated lower bound on pair CTR."
            ),
            "no_sqrt2_conversion": True,
        },
        "boards": {
            board: {
                "n_matched_events": len(data["ids"]),
                "sources": data["sources"],
                "thresholds_by_tail_fraction": data["thresholds"],
            }
            for board, data in boards.items()
        },
        "outputs": [
            "core_covariance_summary.csv",
            "core_covariance_bootstrap.csv",
            "core_and_outlier_populations.png",
            "core_covariance_tail_sensitivity.png",
        ],
    }
    (out / "core_covariance_assumptions.json").write_text(
        json.dumps(_jsonable(payload), indent=2) + "\n", encoding="utf-8"
    )

    print("\nPRIMARY CORE RESULTS (ML-defined membership; LED/ML paired):")
    for row in summary_rows:
        if (row["tail_fraction_per_channel"] == args.bootstrap_tail_fraction
                and row["population"] == "core"):
            print(
                f"{row['board']} {row['state']}: N={row['n']:,}, "
                f"cov={row['covariance_ps2']:.4f} ps^2, "
                f"rho={row['pearson']:.5f}, "
                f"Var(T)={row['variance_timing_ps2']:.4f} ps^2, "
                f"common-FWHM proxy={row['common_fwhm_equivalent_proxy_ps']}",
                flush=True,
            )
    print(f"\nSaved covariance analysis: {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for board in ("uc", "fbk"):
        parser.add_argument(f"--{board}-energy-run", type=Path, required=True)
        parser.add_argument(f"--{board}-timing-run", type=Path, required=True)
        parser.add_argument(f"--{board}-cache-root", type=Path)
        parser.add_argument(f"--{board}-artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tail-fractions", type=float, nargs="+",
        default=[0.002, 0.005, 0.01],
        help="Per-channel ML extreme-tail fractions; default 0.2%, 0.5%, 1%.",
    )
    parser.add_argument(
        "--bootstrap-tail-fraction", type=float, default=0.002,
        help="Tail fraction receiving paired bootstrap CIs (default 0.002).",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 200:
        parser.error("--bootstrap must be >= 200")
    if any(not 0 < x < 0.5 for x in args.tail_fractions):
        parser.error("All tail fractions must be in (0, 0.5)")
    if not any(np.isclose(x, args.bootstrap_tail_fraction)
               for x in args.tail_fractions):
        parser.error("bootstrap-tail-fraction must appear in tail-fractions")
    analyze(args)


if __name__ == "__main__":
    main()
