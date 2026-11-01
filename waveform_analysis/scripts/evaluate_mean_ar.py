"""Standalone deterministic mean-conditioned AR suffix-reconstruction benchmark.

Run from repository root:
python -m waveform_analysis.scripts.evaluate_mean_ar --uc-results waveform_analysis/results/UC/benchmark_48V_final --fbk-results waveform_analysis/results/FBK/benchmark_48V_final --uc-cache-root ~/UC_cache --fbk-cache-root ~/FBK_cache --output-root ~/mean_ar --max-train-events 3000 --max-eval-events 256

This script NEVER retrains correction models or modifies the prepared caches.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup, _cache_root
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import mode_family, waveform_view, model_sample_mask


@dataclass
class ARFit:
    order: int
    coefficients: np.ndarray
    alpha: float
    stable: bool


def select_events(n: int, limit: int, seed: int) -> np.ndarray:
    if n <= 0 or limit < 1:
        raise ValueError("Expected nonempty events and a positive event limit")
    return np.sort(np.random.default_rng(seed).choice(n, min(n, limit), replace=False))


def waveform_statistics(train: np.ndarray, std_floor: float = 1e-4):
    mean = np.mean(train, axis=0, dtype=np.float64)
    sigma = np.std(train, axis=0, dtype=np.float64)
    sigma = np.maximum(sigma, std_floor)
    return mean, sigma


def make_training_rows(residual: np.ndarray, order: int, max_rows: int, seed: int):
    """Sample (event,time) positions; no complete event/time design matrix allocated."""
    n, length = residual.shape
    if length <= order:
        raise ValueError("AR order must be smaller than number of usable samples")
    positions = np.random.default_rng(seed).integers(
        0, n * (length - order), size=min(max_rows, n * (length - order)))
    events = positions // (length - order)
    times = order + positions % (length - order)
    offsets = np.arange(1, order + 1, dtype=np.int64)
    features = residual[events[:, None], times[:, None] - offsets[None, :]]
    target = residual[events, times]
    return features, target


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    gram = x.T @ x
    gram.flat[:: len(gram) + 1] += alpha
    return np.linalg.solve(gram, x.T @ y)


def is_stable(coefficients: np.ndarray) -> bool:
    # Companion matrix eigenvalues must lie strictly inside the unit circle.
    p = len(coefficients)
    matrix = np.zeros((p, p), dtype=np.float64)
    matrix[0] = coefficients
    if p > 1:
        matrix[1:, :-1] = np.eye(p - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(matrix))) < 1 - 1e-7)


def fit_one_ar(train: np.ndarray, validation: np.ndarray, order: int,
               alphas: list[float], max_rows: int, seed: int) -> ARFit:
    x, y = make_training_rows(train, order, max_rows, seed)
    vx, vy = make_training_rows(validation, order, max(1000, max_rows // 5), seed + 1)
    best = None
    for alpha in alphas:
        coef = ridge_fit(x, y, alpha)
        if not is_stable(coef):
            continue
        score = float(np.mean((vx @ coef - vy) ** 2))
        if best is None or score < best[0]:
            best = score, alpha, coef
    if best is None:
        raise ValueError(
            f"No stable AR({order}) candidate found. Increase Ridge penalties; "
            "an unstable recursive model must not be evaluated."
        )
    return ARFit(order, best[2], best[1], True)


def forecast_suffix(waves: np.ndarray, mean: np.ndarray, sigma: np.ndarray,
                    cutoff: int, fit: ARFit, standardized: bool) -> np.ndarray:
    """Vectorized over events; recursion uses only prefix and prior forecasts."""
    if cutoff < fit.order or cutoff >= waves.shape[1]:
        raise ValueError("Invalid cutoff relative to AR order or waveform length")
    residual = (waves[:, :cutoff] - mean[None, :cutoff])
    if standardized:
        residual = residual / sigma[None, :cutoff]
    # Only last p values are needed; rolling state ordered newest first.
    history = np.ascontiguousarray(residual[:, -fit.order:][:, ::-1])
    output = np.empty((len(waves), waves.shape[1] - cutoff), np.float64)
    for j, t in enumerate(range(cutoff, waves.shape[1])):
        next_residual = history @ fit.coefficients
        value = mean[t] + (sigma[t] * next_residual if standardized else next_residual)
        output[:, j] = np.clip(value, 0.0, 1.0)
        # The next step must use the clipped waveform, not the unbounded
        # latent prediction, so the autoregressive state stays consistent.
        effective = output[:, j] - mean[t]
        if standardized:
            effective /= sigma[t]
        history[:, 1:] = history[:, :-1].copy()
        history[:, 0] = effective
    return output


def choose_cutoffs(mean: np.ndarray, sigma: np.ndarray, minimum: int, count: int):
    """Densify near waveform rise; retain coverage of early, peak, and tail."""
    n = len(mean)
    if n - minimum < 3 or count < 3:
        raise ValueError("Not enough samples/cutoffs for evaluation")
    baseline = float(np.median(mean[: max(3, n // 20)]))
    peak = int(np.argmax(np.abs(mean - baseline)))
    change = np.abs(np.diff(mean, prepend=mean[0]))
    # Smooth derivative to avoid selecting a noisy single-sample discontinuity.
    kernel = np.ones(max(3, n // 200), dtype=float)
    kernel /= len(kernel)
    change = np.convolve(change, kernel, mode="same")
    rise = int(np.argmax(change[: max(peak + 1, 1)])) if peak > minimum else peak
    center = int(np.clip(rise, minimum, n - 2))
    # Dense cutoffs around rise, with a few earlier and later coverage points.
    dense = np.linspace(max(minimum, center - max(8, n // 12)),
                        min(n - 2, center + max(8, n // 12)),
                        max(3, count - 4))
    early = [minimum, int(np.clip(center // 2, minimum, n - 2))]
    late = [int(np.clip(peak, minimum, n - 2)),
            int(np.clip(peak + (n - peak) * .6, minimum, n - 2))]
    proposed = np.unique(np.clip(np.rint(np.r_[dense, early, late]).astype(int), minimum, n - 2))
    if len(proposed) < count:
        proposed = np.unique(np.r_[proposed, np.rint(np.linspace(minimum, n - 2, count)).astype(int)])
    if len(proposed) > count:
        # Preserve spatially separated anchors and choose dense remainder.
        anchors = set(early + late)
        chosen = sorted(anchors.intersection(set(proposed.tolist())))
        candidates = sorted(set(proposed.tolist()) - set(chosen), key=lambda x: abs(x-center))
        chosen += candidates[:max(0, count-len(chosen))]
        proposed = np.sort(np.asarray(chosen, dtype=int))
    return proposed.astype(int), {"baseline": baseline, "peak": peak, "rise": rise}


def save_plot(path: Path, time: np.ndarray, actual: np.ndarray, mean: np.ndarray,
              forecast: np.ndarray, cutoff: int, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(time, actual, label="True waveform", lw=1.2)
    ax.plot(time, mean, label="Population mean", lw=1.3)
    ax.plot(time[cutoff:], forecast, label="AR forecast", lw=1.3)
    ax.axvline(time[cutoff], ls="--", color="gray", lw=.8)
    ax.set(xlabel="Time (ps)", ylabel="Normalized voltage", title=title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def analyze_channel(train: np.ndarray, ev: np.ndarray, times: np.ndarray, transform,
                    output: Path, orders: list[int], n_cutoffs: int,
                    alpha_grid: list[float], max_rows: int, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    if train.shape[1] != ev.shape[1] or train.shape[1] != len(times):
        raise ValueError("Training/evaluation grids differ")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(ev)):
        raise ValueError("Non-finite waveform values")
    if np.min(train) < -1e-5 or np.max(train) > 1.00001 or np.min(ev) < -1e-5 or np.max(ev) > 1.00001:
        raise ValueError("Expected waveform inputs normalized to [0,1] by the existing cache")
    # Disjoint development fit/validation events for hyperparameter choice.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(train))
    n_val = max(1, int(round(.2 * len(train))))
    if len(train) < 10 or len(ev) < 1:
        raise ValueError("Need >=10 development events and >=1 evaluation event")
    subtrain = train[perm[n_val:]]
    valid = train[perm[:n_val]]
    mu, sig = waveform_statistics(subtrain)
    min_cutoff = max(orders) + 1
    cutoffs, anchors = choose_cutoffs(mu, sig, min_cutoff, n_cutoffs)
    fitted = {}
    for order in orders:
        for standardized in (False, True):
            label = "standardized_ar" if standardized else "mean_ar"
            residual = (subtrain - mu) / sig if standardized else subtrain - mu
            valres = (valid - mu) / sig if standardized else valid - mu
            fitted[label, order] = fit_one_ar(residual, valres, order, alpha_grid, max_rows,
                                              seed + order + 1000 * standardized)

    # Refit using all selected development events at selected regularization;
    # keep evaluation/blind events entirely untouched during fitting/selection.
    full_mu, full_sig = waveform_statistics(train)
    for (label, order), fit in list(fitted.items()):
        standardized = label == "standardized_ar"
        resid = ((train - full_mu) / full_sig if standardized else train - full_mu)
        x, y = make_training_rows(resid, order, max_rows, seed + 5 + order)
        coef = ridge_fit(x, y, fit.alpha)
        if is_stable(coef):
            fitted[label, order] = ARFit(order, coef, fit.alpha, True)
        # Otherwise retain development-subtrain fit with its respective statistics.
        # For consistency choose the subtrain statistics for such a case.
        else:
            raise ValueError(f"Refitted {label} AR({order}) unstable; cannot safely use full development mean")

    output_scale = float(np.asarray(transform.maximum).reshape(-1)[0] -
                         np.asarray(transform.minimum).reshape(-1)[0]) if transform is not None else float("nan")
    results = []
    positional = []
    for cutoff in cutoffs:
        truth = ev[:, cutoff:]
        base = np.broadcast_to(full_mu[None, cutoff:], truth.shape)
        for label, order in [("mean", 0)] + [(kind, p) for kind in ("mean_ar", "standardized_ar") for p in orders]:
            if label == "mean":
                pred = base
                alpha = None
            else:
                fit = fitted[label, order]
                pred = forecast_suffix(ev, full_mu, full_sig, int(cutoff), fit,
                                       label == "standardized_ar")
                alpha = fit.alpha
            error2 = (truth - pred) ** 2
            mse = float(np.mean(error2))
            rmse = float(np.sqrt(mse))
            mean_err = float(np.mean((truth-base)**2))
            results.append(dict(cutoff=int(cutoff), time_ps=float(times[cutoff]), method=label,
                                ar_order=order, ridge_alpha=alpha, n_eval=len(ev),
                                mse=mse, rmse=rmse, rmse_mv=rmse*output_scale,
                                mse_improvement_pct=100*(1-mse/mean_err) if mean_err>0 else None))
            for j, sqe in enumerate(np.mean(error2, axis=0)):
                positional.append((int(cutoff), label, order, int(cutoff)+j,
                                   float(times[cutoff+j]), float(sqe)))
            if label == "standardized_ar" and order == orders[0] and cutoff == cutoffs[len(cutoffs)//2]:
                save_plot(output/"example_reconstruction.png", times, ev[0], full_mu,
                          pred[0], int(cutoff), f"Standardized AR({order}) | cutoff {cutoff}")
    with (output/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (output/"position_errors.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cutoff","method","ar_order","sample","time_ps","mse"])
        writer.writerows(positional)
    np.savez_compressed(output/"fitted_models.npz", mean=full_mu, sigma=full_sig,
        time_ps=times, cutoffs=cutoffs, **{f"{label}_{order}_coef": fit.coefficients
            for (label,order),fit in fitted.items()})
    metadata = dict(n_train=len(train), n_eval=len(ev), n_samples=len(times),
                    cutoffs=cutoffs.tolist(), anchor_samples=anchors, orders=orders,
                    coefficients={f"{label}_{order}": dict(alpha=fit.alpha, stable=True)
                                  for (label,order),fit in fitted.items()},
                    normalized_input=True, voltage_range_mV=output_scale,
                    split="development_for_fit_and_selection__blind_for_evaluation")
    (output/"summary.json").write_text(json.dumps(metadata, indent=2))
    return results


def _pick_run(results_root: Path, mode: str, window: str):
    matches = []
    for run in discover_runs(results_root):
        config = resolve_run_config(run["directory"])
        if config["mode"] == mode and config["window_name"] == window:
            matches.append((run, config))
    if not matches:
        raise ValueError(f"No completed {mode}/{window} run beneath {results_root}")
    return matches[0]


def run(args):
    lookup = PreparedCacheLookup()
    for board, results, cache_override in (
        ("UC", args.uc_results, args.uc_cache_root),
        ("FBK", args.fbk_results, args.fbk_cache_root),
    ):
        for mode in ("energy_to_energy", "timing_to_timing"):
            run, config = _pick_run(results, mode, args.window)
            cache = _cache_root(config, run["directory"], cache_override,
                                roles=("development","blind"))
            development = lookup.find(cache, "development", config, run["manifest"])
            blind = lookup.find(cache, "blind", config, run["manifest"])
            family = mode_family(mode)
            dtime = np.asarray(getattr(development, family+"_time_ps"), dtype=float)
            btime = np.asarray(getattr(blind, family+"_time_ps"), dtype=float)
            if not np.array_equal(dtime, btime):
                raise ValueError(f"{board}/{family} development and blind time grids differ")
            # The prepared timing mask marks artificial excluded/cropped samples.
            # Restrict to a genuine contiguous interval; never train AR across a gap.
            mask = model_sample_mask(development, mode)
            if not np.array_equal(mask, model_sample_mask(blind, mode)):
                raise ValueError("Development/blind sample masks differ")
            keep = np.flatnonzero(mask)
            if len(keep) < 3 or np.any(np.diff(keep) != 1):
                raise ValueError("Usable sample mask is non-contiguous; cannot autoregress over missing gaps")
            time = dtime[keep]
            train_ids = select_events(development.n_events, args.max_train_events, args.seed)
            eval_ids = select_events(blind.n_events, args.max_eval_events, args.seed+1)
            train_pair = waveform_view(development, mode, train_ids).materialize()[:, :, keep]
            ev_pair = waveform_view(blind, mode, eval_ids).materialize()[:, :, keep]
            transform = getattr(development, family+"_transform")
            for detector in range(2):
                destination = args.output_root / board / family / f"detector_{detector+1}"
                print(f"Evaluating {board}/{family}/detector_{detector+1} | train={len(train_ids)} eval={len(eval_ids)}", flush=True)
                analyze_channel(np.asarray(train_pair[:, detector], float),
                                np.asarray(ev_pair[:, detector], float), time,
                                transform, destination, args.ar_orders,
                                args.n_cutoffs, args.alphas, args.max_fit_rows,
                                args.seed+detector)
            del train_pair, ev_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uc-results", type=Path, required=True)
    p.add_argument("--fbk-results", type=Path, required=True)
    p.add_argument("--uc-cache-root", type=Path, required=True)
    p.add_argument("--fbk-cache-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--window", default="wide")
    p.add_argument("--ar-orders", type=int, nargs="+", default=[8,16,32,64,128])
    p.add_argument("--alphas", type=float, nargs="+", default=[1e-3,1e-1,1.0,10.0,100.0])
    p.add_argument("--n-cutoffs", type=int, default=10)
    p.add_argument("--max-train-events", type=int, default=3000)
    p.add_argument("--max-eval-events", type=int, default=256)
    p.add_argument("--max-fit-rows", type=int, default=75000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if not args.ar_orders or min(args.ar_orders) < 1 or args.n_cutoffs < 5:
        p.error("Require positive AR orders and at least 5 cutoffs")
    if not args.alphas or min(args.alphas) <= 0:
        p.error("Ridge alphas must be positive")
    run(args)


if __name__ == "__main__":
    main()
