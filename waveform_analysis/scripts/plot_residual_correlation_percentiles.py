"""Paired energy/timing blind residual correlation by shared severity percentile.

Reads saved predictions from Codex ArtifactStore (or legacy run-local pred.npz).
Never reads or reconstructs waveforms and never runs inference.
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
from waveform_analysis.data.storage import RunStore
from waveform_analysis.scripts.residual_exploratory_diagnostics import exploratory_diagnostics

BANDS = (0, 50, 80, 90, 95, 97, 99, 99.5, 99.8, 100)
CENTRAL = (50, 60, 70, 80, 90, 95, 97, 98, 99, 99.5, 99.8, 100)


def _read(run, artifact_root):
    store = RunStore(run, artifact_root=artifact_root)
    path = store.predictions_path
    legacy = run / "artifacts" / "pred.npz"
    if not path.is_file() and legacy.is_file():
        path = legacy
    if not path.is_file():
        raise FileNotFoundError(f"Blind predictions missing: {path} or {legacy}")
    with np.load(path, allow_pickle=False) as f:
        ids = np.asarray(f["event_id"], np.int64)
        led = np.asarray(f["led_residual_ps"], float)
        ml = np.asarray(f["corrected_ps"], float)
    if ids.ndim != 1 or led.shape != ids.shape or ml.shape != ids.shape:
        raise ValueError(f"Bad blind prediction dimensions: {path}")
    if len(np.unique(ids)) != len(ids) or not np.isfinite(led).all() or not np.isfinite(ml).all():
        raise ValueError(f"Ambiguous IDs or nonfinite residuals: {path}")
    return ids, led, ml, str(path)


def _rho_cov(points):
    n = len(points)
    if n < 3:
        return np.nan, np.nan, np.nan, np.nan
    c = np.cov(points, rowvar=False)
    sd = np.sqrt(np.diag(c))
    rho = c[0, 1] / (sd[0] * sd[1]) if np.all(sd > 0) else np.nan
    return float(rho), float(c[0, 1]), float(sd[0]), float(sd[1])


def _ci(points, draws, rng):
    if len(points) < 4:
        return np.nan, np.nan
    values = []
    for _ in range(draws):
        sample = points[rng.integers(len(points), size=len(points))]
        rho = _rho_cov(sample)[0]
        if np.isfinite(rho):
            values.append(rho)
    return (tuple(map(float, np.quantile(values, (0.025, 0.975))))
            if len(values) >= max(10, draws // 2) else (np.nan, np.nan))


def _rows(points, label, draws, rng):
    score = np.max(np.abs(points), axis=1)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), float)
    ranks[order] = 100 * (np.arange(len(score)) + 0.5) / len(score)
    rows = []
    settings = ([("band", a, b) for a, b in zip(BANDS[:-1], BANDS[1:])]
                + [("central", 0, b) for b in CENTRAL])
    for kind, lo, hi in settings:
        mask = (ranks >= lo) & (ranks < hi)
        data = points[mask]
        rho, cov, sig_e, sig_t = _rho_cov(data)
        low, high = _ci(data, draws, rng)
        rows.append(dict(
            state=label, kind=kind, percentile_low=lo, percentile_high=hi,
            n=len(data), rho=rho, rho_ci_low=low, rho_ci_high=high,
            covariance_ps2=cov, sigma_energy_ps=sig_e, sigma_timing_ps=sig_t,
            score_min_ps=float(score[mask].min()) if mask.any() else np.nan,
            score_max_ps=float(score[mask].max()) if mask.any() else np.nan,
        ))
    return rows


def _plot(rows, kind, destination):
    fig, (ax, count) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, constrained_layout=True,
        gridspec_kw={"height_ratios": [3, 1]})
    for state, name in (("led", "LED"), ("ml", "ML corrected")):
        subset = [r for r in rows if r["kind"] == kind and r["state"] == state]
        x = np.arange(len(subset)) if kind == "band" else np.array([r["percentile_high"] for r in subset])
        y = np.array([r["rho"] for r in subset])
        lower = np.array([r["rho_ci_low"] for r in subset])
        upper = np.array([r["rho_ci_high"] for r in subset])
        line = ax.plot(x, y, marker="o", label=name)[0]
        ax.fill_between(x, lower, upper, alpha=0.15, color=line.get_color())
        count.plot(x, [r["n"] for r in subset], marker="o", color=line.get_color(), label=name)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Pearson correlation rho (95% bootstrap CI)")
    ax.grid(alpha=0.2)
    ax.legend()
    count.set_ylabel("Events")
    count.grid(alpha=0.2)
    if kind == "band":
        subset = [r for r in rows if r["kind"] == kind and r["state"] == "led"]
        labels = [f'{r["percentile_low"]:g}-{r["percentile_high"]:g}' for r in subset]
        count.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
        count.set_xlabel("Shared-score percentile band [%]")
        fig.suptitle("Energy/timing correlation: disjoint severity bands")
    else:
        count.set_xlabel("Central population retained up to percentile [%]")
        fig.suptitle("Energy/timing correlation: progressively exclude outer tails")
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-run", type=Path, required=True)
    parser.add_argument("--timing-run", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cache_root and args.artifact_root:
        parser.error("Choose --cache-root OR --artifact-root")
    if args.bootstrap < 20:
        parser.error("--bootstrap must be >= 20")
    artifact_root = (
        args.artifact_root.expanduser().resolve() if args.artifact_root
        else args.cache_root.expanduser().resolve() / "artifact_store" if args.cache_root
        else None)
    energy = _read(args.energy_run.expanduser().resolve(), artifact_root)
    timing = _read(args.timing_run.expanduser().resolve(), artifact_root)
    ids, ai, bi = np.intersect1d(energy[0], timing[0], assume_unique=True, return_indices=True)
    if len(ids) < 20:
        raise ValueError(f"Too few paired blind events: {len(ids)}")
    rng = np.random.default_rng(args.seed)
    rows = []
    paired = {}
    for label, column in (("led", 1), ("ml", 2)):
        points = np.column_stack((energy[column][ai], timing[column][bi]))
        paired[label] = points
        rows.extend(_rows(points, label, args.bootstrap, rng))
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "correlation_by_percentile.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, "band", out / "correlation_disjoint_bands.png")
    _plot(rows, "central", out / "correlation_cumulative_central.png")
    exploratory_diagnostics(paired["led"], paired["ml"], out,
                            draws=args.bootstrap, seed=args.seed)
    (out / "summary.json").write_text(json.dumps({
        "n_common_blind_events": len(ids),
        "energy_predictions": energy[3], "timing_predictions": timing[3],
        "score": "max(abs(energy residual), abs(timing residual))",
        "percentile_definition": "rank on shared score, separately for LED and ML",
        "bootstrap_draws": args.bootstrap, "seed": args.seed,
        "bootstrap_condition": "fixed percentile-band memberships; boundaries not re-estimated",
        "caution": "conditioning on residual magnitude may itself affect correlation; not a causal or CTR-floor estimate",
    }, indent=2), encoding="utf-8")
    print(f"Paired events: {len(ids)} | outputs: {out}")
    print("Additional outputs: exploratory_diagnostics.json, exploratory_components.csv, "
          "covariance_and_difference_variance.png, paired_residual_hexbin_central99.png")


if __name__ == "__main__":
    main()
