#!/usr/bin/env python3
"""Standalone deterministic prefix reconstruction on UC/FBK prepared caches.

Each board, mode and detector is fitted independently. Development waveforms
are used for fitting; blind waveforms only for evaluation. No XAI/model changes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from waveform_analysis.data.dataset import load_prepared_dataset
from waveform_analysis.data.named_cache import resolve_alias
from waveform_analysis.signal.sample_mask import control_clipping_sample_mask

METHODS = ("mean", "mean_ar", "standardized_ar")


def _aliases(root, mode, window, role, dataset_name=None):
    hits = sorted(Path(root).expanduser().resolve().glob(
        f"datasets/*/*/{mode}/{window}/{role}"
    ))
    if dataset_name:
        hits = [p for p in hits if p.parts[-4] == dataset_name]
    return hits


def load_pair(root, mode, window="wide", development_name=None, blind_name=None):
    """Locate one matching named-cache development/blind pair without refitting."""
    root = Path(root).expanduser().resolve()
    dev = _aliases(root, mode, window, "development", development_name)
    blind = _aliases(root, mode, window, "blind", blind_name)
    combinations = [(a, b) for a in dev for b in blind if a.parts[-5] == b.parts[-5]]
    if len(combinations) != 1:
        raise ValueError(
            f"{root}/{mode}/{window}: found {len(combinations)} matching development/"
            f"blind pairs; specify --development-name and --blind-name. "
            f"Development: {[str(p) for p in dev]}; blind: {[str(p) for p in blind]}"
        )
    dev_path, blind_path = combinations[0]
    training = load_prepared_dataset(resolve_alias(dev_path))
    evaluation = load_prepared_dataset(resolve_alias(blind_path))
    if (training.manifest.get("dataset_role") != "development" or
            evaluation.manifest.get("dataset_role") != "blind"):
        raise ValueError("Prepared roles disagree with named cache paths")
    x_train = getattr(training, f"{mode}_windows")
    x_eval = getattr(evaluation, f"{mode}_windows")
    t_train = getattr(training, f"{mode}_time_ps")
    t_eval = getattr(evaluation, f"{mode}_time_ps")
    transform = getattr(training, f"{mode}_transform")
    eval_transform = getattr(evaluation, f"{mode}_transform")
    if any(x is None for x in (x_train, x_eval, t_train, t_eval, transform, eval_transform)):
        raise ValueError(f"Missing prepared {mode} waveforms/time grid/transform")
    if (x_train.ndim != 3 or x_eval.ndim != 3 or x_train.shape[1] != 2 or
            x_eval.shape[1:] != x_train.shape[1:]):
        raise ValueError("Expected arrays [events, 2 detectors, samples] on identical grids")
    if not np.array_equal(t_train, t_eval):
        raise ValueError("Development and blind time grids differ")
    for attr in ("minimum", "maximum"):
        if not np.array_equal(getattr(transform, attr), getattr(eval_transform, attr)):
            raise ValueError("Development and blind normalized-voltage transforms differ")
    if mode == "timing":
        mask_alias = root / "sample_masks" / dev_path.parts[-5] / mode / window
        mask_file = resolve_alias(mask_alias) / "mask.npz"
        if mask_file.is_file():
            with np.load(mask_file) as data:
                valid = np.asarray(data["mask"], dtype=bool)
                if not np.array_equal(data["time_ps"], t_train):
                    raise ValueError("Timing sample mask uses a different time grid")
        else:
            # prepare-cache produces prepared waveforms but does not always
            # materialize the separate, control-fitted timing sample mask.
            # Derive the exact same mask without fitting on development/blind.
            control_alias = (root / "datasets" / dev_path.parts[-5] /
                             dev_path.parts[-5] / mode / window / "control")
            if not control_alias.exists():
                raise FileNotFoundError(
                    f"Timing mask missing and no control cache at {control_alias}")
            control = load_prepared_dataset(resolve_alias(control_alias))
            if control.manifest.get("dataset_role") != "control":
                raise ValueError("Invalid control dataset role for timing mask")
            if not np.array_equal(control.timing_time_ps, t_train):
                raise ValueError("Control/development timing grids differ")
            if control.timing_windows is None:
                raise ValueError("Control timing waveforms missing")
            control_transform = control.timing_transform
            if control_transform is None or any(
                not np.array_equal(getattr(control_transform, attr), getattr(transform, attr))
                for attr in ("minimum", "maximum")
            ):
                raise ValueError("Control/development timing transforms differ")
            valid = control_clipping_sample_mask(control.timing_windows)
            print(f"[{root.name}/timing] Derived control-based sample mask "
                  f"({np.count_nonzero(valid)}/{valid.size} retained); "
                  f"no stored sample mask was required.", flush=True)
    else:
        valid = np.ones(x_train.shape[-1], dtype=bool)
    if valid.shape != (x_train.shape[-1],):
        raise ValueError("Invalid timing sample mask shape")
    source = {
        "development_alias": str(dev_path),
        "blind_alias": str(blind_path),
        "control": dev_path.parts[-5],
        "development_fingerprint": training.manifest.get("fingerprint"),
        "blind_fingerprint": evaluation.manifest.get("fingerprint"),
    }
    return x_train, x_eval, np.asarray(t_train), transform, valid, source


def valid_intervals(mask, max_order=0):
    """All contiguous retained sample runs that can support the specified AR."""
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    intervals = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return [(int(a), int(b)) for a, b in intervals if b-a > max_order+1]


def active_interval(mask, mean, max_order):
    """For backwards compatibility: most prominent rising-edge valid interval."""
    intervals = valid_intervals(mask, max_order)
    if not intervals:
        raise ValueError("No unmasked interval long enough for maximum AR order")
    return max(intervals, key=lambda pair: float(np.max(np.diff(mean[pair[0]:pair[1]]))))


def dense_cutoffs(mean, interval, max_order, count=10):
    """Cutoffs densely around the population-mean rising edge, plus edges/tail."""
    lo, hi = interval
    first, last = lo + max_order, hi - 1
    if count < 1 or last-first+1 < count:
        raise ValueError("Insufficient valid samples for requested cutoffs")
    rise = lo + int(np.argmax(np.diff(mean[lo:hi])))
    width = hi-lo
    dense = np.linspace(rise-.02*width, rise+.06*width, max(1, count-3))
    candidates = [lo+.03*width, *dense, lo+.66*width, lo+.86*width]
    chosen = {int(np.clip(round(t), first, last)) for t in candidates}
    if len(chosen) < count:
        for t in np.linspace(first, last, max(count*12, 120)):
            chosen.add(int(round(t)))
            if len(chosen) >= count:
                break
    return np.asarray(sorted(chosen)[:count], dtype=np.int64)


def distributed_cutoffs(mean, mask, max_order, count):
    """Prefer rising-edge cutoffs; reserve some for valid post-mask segments."""
    intervals = valid_intervals(mask, max_order)
    if not intervals:
        raise ValueError("No retained region can support the maximum AR order")
    primary = active_interval(mask, mean, max_order)
    later = [pair for pair in intervals if pair[0] > primary[1]]
    # Cutoffs after cropped regions retain only their own contiguous history.
    secondary = min(len(later)*2, max(0, count//3))
    result = dense_cutoffs(mean, primary, max_order, count-secondary).tolist()
    if secondary:
        for j in range(secondary):
            pair = later[j % len(later)]
            start, stop = pair
            fractions = (.20, .65)
            cutoff = start + max_order + round(
                fractions[j // len(later) % 2]*(stop-start-max_order-1))
            result.append(int(min(cutoff, stop-1)))
    return np.asarray(sorted(set(result)), dtype=np.int64)


def local_forecast_end(mask, cutoff):
    """AR terminates at the first masked sample; it never bridges a gap."""
    mask = np.asarray(mask, bool)
    if cutoff < 0 or cutoff >= len(mask) or not mask[cutoff]:
        raise ValueError("Cutoff must be in a valid region")
    end = cutoff
    while end < len(mask) and mask[end]:
        end += 1
    return end


def fit_moments(train):
    values = np.asarray(train, dtype=np.float64)
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    positive = std[std > 0]
    floor = max(1e-5, .02 * float(np.median(positive)) if positive.size else 1e-5)
    return mean, np.maximum(std, floor), floor


def fit_ridge_orders(train, mean, std, interval, orders, alpha=1e-3,
                     stride=4, batch_events=8):
    """One maximal-order Gram accumulation, separate fits for each AR order.

    The same target rows are used across orders. Ridge alpha is relative to
    mean feature energy, making it insensitive to training-set size.
    """
    intervals = interval if isinstance(interval, list) else [interval]
    pmax = max(orders)
    fitted = {}
    for method in ("mean_ar", "standardized_ar"):
        gram = np.zeros((pmax, pmax), np.float64)
        cross = np.zeros(pmax, np.float64)
        n_rows = 0
        for offset in range(0, len(train), batch_events):
            values = np.asarray(train[offset:offset + batch_events], np.float64)
            residual = values - mean
            if method == "standardized_ar":
                residual /= std
            for lo, hi in intervals:
                if hi-lo <= pmax:
                    continue
                windows = sliding_window_view(
                    residual[:, lo:hi], pmax + 1, axis=1)[:, ::stride]
                design = windows[:, :, :-1][:, :, ::-1].reshape(-1, pmax)
                target = windows[:, :, -1].reshape(-1)
                gram += design.T @ design
                cross += design.T @ target
                n_rows += len(target)
        if not n_rows:
            raise ValueError("No AR training samples")
        models = {}
        for p in orders:
            block = gram[:p, :p]
            ridge_scale = float(np.trace(block) / p)
            penalty = alpha * max(ridge_scale, 1e-12)
            coeff = np.linalg.solve(block + penalty * np.eye(p), cross[:p])
            # np.roots([1,-a1,...,-ap]) returns characteristic eigenvalues.
            radius = float(max(abs(np.roots(np.r_[1., -coeff]))))
            models[p] = {
                "coefficients": coeff,
                "root_radius": radius,
                "stable": bool(radius < 1.),
                "ridge_penalty": penalty,
                "training_rows": n_rows,
            }
        fitted[method] = models
    return fitted


def forecast_suffix(prefix, mean, std, coeff, cutoff, end, standardized=False):
    """Predict recursively using strictly the prefix and already predicted samples."""
    prefix = np.asarray(prefix, np.float64)
    n_events, prefix_len = prefix.shape
    if prefix_len != cutoff:
        raise ValueError("Prefix length and cutoff disagree")
    p = len(coeff)
    if cutoff < p or end <= cutoff or end > len(mean):
        raise ValueError("Invalid cutoff, AR order, or suffix endpoint")
    history = np.clip(prefix[:, cutoff-p:cutoff].copy(), 0., 1.)
    predictions = np.empty((n_events, end-cutoff), np.float64)
    for t in range(cutoff, end):
        residual = history - mean[t-p:t]
        if standardized:
            residual /= std[t-p:t]
        next_residual = residual[:, ::-1] @ coeff
        values = mean[t] + (std[t] if standardized else 1.) * next_residual
        values = np.clip(values, 0., 1.)
        predictions[:, t-cutoff] = values
        if p > 1:
            history[:, :-1] = history[:, 1:]
        history[:, -1] = values
    return predictions


def damped_ar_forecast(ar_forecast, mean, time_ps, cutoff, half_life_ns):
    """Blend a raw recursive AR forecast toward the mean at long horizons.

    Decay is postprocessing ONLY. The AR state itself retains its original
    recursion, so varying half-life does not change AR coefficients/history.
    The first predicted sample has weight one and is never altered.
    """
    raw = np.asarray(ar_forecast, np.float64)
    if raw.ndim != 2:
        raise ValueError("Expected [events, forecast_samples]")
    if half_life_ns is None:
        return raw.copy()
    half_life_ns = float(half_life_ns)
    if not np.isfinite(half_life_ns) or half_life_ns <= 0:
        raise ValueError("Decay half-life in ns must be positive and finite")
    times = np.asarray(time_ps, np.float64)
    end = int(cutoff) + raw.shape[1]
    if times.ndim != 1 or cutoff < 0 or end > len(times):
        raise ValueError("Forecast range lies outside aligned time grid")
    elapsed_ns = (times[cutoff:end] - times[cutoff]) / 1000.
    if np.any(elapsed_ns < 0) or np.any(np.diff(elapsed_ns) <= 0):
        raise ValueError("Aligned forecast times must be strictly increasing")
    weight = np.exp2(-elapsed_ns / half_life_ns)
    baseline = np.asarray(mean, np.float64)[None, cutoff:end]
    return np.clip(baseline + weight[None, :] * (raw - baseline), 0., 1.)


def metric_row(truth, pred, mv_scale):
    error = np.asarray(pred, np.float64) - np.asarray(truth, np.float64)
    mse = float(np.mean(error**2))
    return {
        "mse_normalized": mse,
        "rmse_normalized": float(np.sqrt(mse)),
        "mae_normalized": float(np.mean(abs(error))),
        "rmse_mV": float(np.sqrt(mse) * mv_scale),
        "mae_mV": float(np.mean(abs(error)) * mv_scale),
    }


def choose_indices(total, limit, seed):
    if limit < 0:
        raise ValueError("Event limit must be nonnegative; zero means all")
    count = total if limit == 0 else min(limit, total)
    if count < 2:
        raise ValueError("At least two events required")
    return np.sort(np.random.default_rng(seed).choice(total, count, replace=False))


def plot_summary(rows, out, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"mean": "#333333", "mean_ar": "#1879B3",
              "standardized_ar": "#35956C"}
    order = max(r["order"] for r in rows)
    for method in METHODS:
        series = sorted((r for r in rows if r["method"] == method and
                         (r["order"] == order or method == "mean")
                         and r.get("half_life_ns") in (None, "none", "mean")),
                        key=lambda r: r["cutoff_sample"])
        ax.plot([r["cutoff_ns"] for r in series],
                [r["rmse_mV"] for r in series],
                marker="o", color=colors[method], label=method)
    ax.set(xlabel="Cutoff position (ns)", ylabel="Suffix RMSE (mV)", title=title)
    ax.grid(alpha=.3)
    ax.legend(title=f"AR order {order}")
    fig.tight_layout()
    fig.savefig(out / "rmse_vs_cutoff.png", dpi=160)
    plt.close(fig)


def plot_decay_comparison(rows, path, title, orders, half_lives_ns):
    """Compare full valid-suffix errors for standardized AR at fixed order."""
    order = max(orders)
    fig, ax = plt.subplots(figsize=(10, 5))
    choices = [
        ("mean", "mean", "Mean baseline"),
        ("standardized_ar", "none", f"AR({order}) no decay"),
    ] + [
        ("standardized_ar", float(h), f"AR({order}) half-life {h:g} ns")
        for h in half_lives_ns
    ]
    for method, half_life, label in choices:
        series = sorted(
            (row for row in rows
             if row["method"] == method and row["half_life_ns"] == half_life
             and (method == "mean" or row["order"] == order)),
            key=lambda row: row["cutoff_ns"],
        )
        if series:
            ax.plot([row["cutoff_ns"] for row in series],
                    [row["rmse_mV"] for row in series],
                    "-o", markersize=3, label=label)
    ax.set(xlabel="Aligned input cutoff (ns)",
           ylabel="Full valid-suffix reconstruction RMSE (mV)", title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def evaluate_one(board, mode, detector, x_train, x_eval, time_ps, transform,
                 mask, provenance, args):
    name = f"{board}/{mode}/detector_{detector+1}"
    out = args.output_root / board / mode / f"detector_{detector+1}"
    out.mkdir(parents=True, exist_ok=True)
    train_ids = choose_indices(len(x_train), args.max_train_events, args.seed)
    eval_ids = choose_indices(len(x_eval), args.max_eval_events, args.seed + 991)
    training = np.asarray(x_train[train_ids, detector, :], np.float64)
    evaluation = np.asarray(x_eval[eval_ids, detector, :], np.float64)
    mean, std, std_floor = fit_moments(training)
    intervals = valid_intervals(mask, max(args.ar_orders))
    cutoffs = distributed_cutoffs(mean, mask, max(args.ar_orders), args.n_cutoffs)
    fitted = fit_ridge_orders(training, mean, std, intervals, args.ar_orders,
                              alpha=args.ridge_alpha, stride=args.train_stride,
                              batch_events=args.batch_events)
    scale = np.asarray(transform.maximum - transform.minimum).reshape(-1)
    mv_scale = float(scale[detector] if scale.size >= 2 else scale[0])
    rows = []
    example = None
    for cutoff in cutoffs:
        cutoff = int(cutoff)
        stop = local_forecast_end(mask, cutoff)
        # All future samples are reconstructed: AR on the current valid run,
        # population mean across/after the first gap. Masked samples are not
        # scored, but later *valid* samples remain part of the full-suffix score.
        future_valid = np.asarray(mask[cutoff:], dtype=bool)
        truth = evaluation[:, cutoff:][:, future_valid]
        baseline_full = np.broadcast_to(
            np.clip(mean[cutoff:], 0., 1.),
            (len(evaluation), len(mean)-cutoff)).copy()
        baseline = baseline_full[:, future_valid]
        base = metric_row(truth, baseline, mv_scale)
        for order in args.ar_orders:
            for method in ("mean_ar", "standardized_ar"):
                model = fitted[method][order]
                prediction = forecast_suffix(
                    evaluation[:, :cutoff], mean, std, model["coefficients"],
                    cutoff, stop, standardized=(method == "standardized_ar")
                )
                # One raw AR recursion supports every decay horizon.
                for half_life in (None, *args.decay_half_lives_ns):
                    faded = damped_ar_forecast(
                        prediction, mean, time_ps, cutoff, half_life)
                    full_prediction = baseline_full.copy()
                    full_prediction[:, :stop-cutoff] = faded
                    metric = metric_row(
                        truth, full_prediction[:, future_valid], mv_scale)
                    n_local = min(stop-cutoff, args.local_horizon_samples)
                    local = metric_row(
                        evaluation[:, cutoff:cutoff+n_local],
                        faded[:, :n_local], mv_scale)
                    local_base = metric_row(
                        evaluation[:, cutoff:cutoff+n_local],
                        np.broadcast_to(mean[cutoff:cutoff+n_local],
                                        (len(evaluation), n_local)), mv_scale)
                    rows.append({
                        **metric, "method": method, "order": order,
                        "half_life_ns": ("none" if half_life is None
                                         else float(half_life)),
                        "cutoff_sample": cutoff,
                        "cutoff_ns": float(time_ps[cutoff]/1000),
                        "n_missing": int(np.count_nonzero(future_valid)),
                        "ar_predicted_samples": stop-cutoff,
                        "n_eval_events": len(eval_ids),
                        "local_rmse_mV": local["rmse_mV"],
                        "local_mse_improvement_percent": (
                            100 * (1 - local["mse_normalized"] /
                                   local_base["mse_normalized"])
                            if local_base["mse_normalized"] > 0 else 0.),
                        "first_step_jump_mV": float(np.mean(abs(
                            faded[:, 0] - evaluation[:, cutoff-1])) * mv_scale),
                        "natural_step_mV": float(np.mean(abs(
                            evaluation[:, cutoff] - evaluation[:, cutoff-1])) * mv_scale),
                        "mse_improvement_percent": (
                            100 * (1 - metric["mse_normalized"] /
                                   base["mse_normalized"])
                            if base["mse_normalized"] > 0 else 0.),
                        "stable": model["stable"],
                        "root_radius": model["root_radius"],
                    })
                # Retain the original example plot without changing its semantics.
                full_prediction = baseline_full.copy()
                full_prediction[:, :stop-cutoff] = prediction
                if (example is None and method == "standardized_ar" and
                        order == args.ar_orders[-1]):
                    example = (cutoff, evaluation[0].copy(), full_prediction[0].copy(),
                               baseline_full[0].copy(), int(eval_ids[0]))
                continue
                rows.append({
                    **metric, "method": method, "order": order,
                    "cutoff_sample": cutoff,
                    "cutoff_ns": float(time_ps[cutoff]/1000),
                    "n_missing": int(np.count_nonzero(future_valid)),
                    "ar_predicted_samples": stop-cutoff,
                    "n_eval_events": len(eval_ids),
                    "mse_improvement_percent": (
                        100 * (1 - metric["mse_normalized"]/base["mse_normalized"])
                        if base["mse_normalized"] > 0 else 0.),
                    "stable": model["stable"],
                    "root_radius": model["root_radius"],
                })
                if (example is None and method == "standardized_ar" and
                        order == args.ar_orders[-1]):
                    example = (cutoff, evaluation[0].copy(), full_prediction[0].copy(),
                               baseline_full[0].copy(), int(eval_ids[0]))
        rows.append({
            **base, "method": "mean", "order": 0, "half_life_ns": "mean",
            "local_rmse_mV": float("nan"), "local_mse_improvement_percent": 0.,
            "first_step_jump_mV": float(np.mean(abs(
                baseline_full[:, 0] - evaluation[:, cutoff-1])) * mv_scale),
            "natural_step_mV": float(np.mean(abs(
                evaluation[:, cutoff] - evaluation[:, cutoff-1])) * mv_scale),
            "cutoff_sample": cutoff, "cutoff_ns": float(time_ps[cutoff]/1000),
            "n_missing": int(np.count_nonzero(future_valid)),
            "ar_predicted_samples": stop-cutoff,
            "n_eval_events": len(eval_ids),
            "mse_improvement_percent": 0., "stable": True, "root_radius": 0.,
        })
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "model.json").write_text(json.dumps({
        "board": board, "mode": mode, "detector": detector+1,
        "source": provenance, "fit_role": "development", "eval_role": "blind",
        "train_events": len(train_ids), "eval_events": len(eval_ids),
        "train_indices": train_ids.tolist(), "eval_indices": eval_ids.tolist(),
        "mean": mean.tolist(), "std": std.tolist(), "std_floor": std_floor,
        "valid_intervals": [list(i) for i in valid_intervals(mask)],
        "valid_scored_samples": int(np.count_nonzero(mask)),
        "cutoffs": cutoffs.tolist(),
        "orders": args.ar_orders, "ridge_alpha": args.ridge_alpha,
        "train_stride": args.train_stride, "seed": args.seed,
        "decay_half_lives_ns": args.decay_half_lives_ns,
        "local_horizon_samples": args.local_horizon_samples,
        "models": {method: {str(p): {
            **{k:v for k,v in fitted[method][p].items() if k != "coefficients"},
            "coefficients": fitted[method][p]["coefficients"].tolist()
        } for p in args.ar_orders} for method in fitted},
    }, indent=2), encoding="utf-8")
    plot_summary(rows, out, name)
    plot_decay_comparison(rows, out / "decay_comparison.png", name,
                          args.ar_orders, args.decay_half_lives_ns)
    if example is not None:
        cutoff, original, prediction, baseline, event_id = example
        ns = np.asarray(time_ps) / 1000
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(ns, original, color="#222222", lw=1.2, label="Blind original")
        ax.plot(ns[cutoff:], baseline, color="#C98740",
                label="Mean replacement")
        ax.plot(ns[cutoff:], prediction, color="#2B9C74",
                label="Standardized AR")
        ax.axvline(ns[cutoff], color="gray", ls="--")
        ax.set(xlabel="Time (ns)", ylabel="Normalized voltage [0,1]",
               title=f"{name}, blind row {event_id}, cutoff {ns[cutoff]:.3f} ns")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(out / "waveform_example.png", dpi=160)
        plt.close(fig)
    print(f"[{name}] train={len(train_ids)} blind={len(eval_ids)} "
          f"valid_regions={valid_intervals(mask)} cutoffs={len(cutoffs)} "
          f"orders={args.ar_orders} -> {out}", flush=True)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uc-cache-root", type=Path, required=True)
    parser.add_argument("--fbk-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--window", default="wide")
    parser.add_argument("--development-name", help="Dataset label if ambiguous")
    parser.add_argument("--blind-name", help="Blind dataset label if ambiguous")
    parser.add_argument("--ar-orders", nargs="+", type=int,
                        default=[8, 16, 32, 64, 128])
    parser.add_argument("--n-cutoffs", type=int, default=10)
    parser.add_argument("--decay-half-lives-ns", nargs="+", type=float,
                        default=[0.5, 1., 2., 5., 10.],
                        help="Fade AR residual toward mean after these half-lives (ns)")
    parser.add_argument("--local-horizon-samples", type=int, default=100,
                        help="Also score the first N AR-predicted samples (default: 100)")
    parser.add_argument("--ridge-alpha", type=float, default=1e-3,
                        help="Ridge penalty relative to mean Gram diagonal")
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--batch-events", type=int, default=8)
    parser.add_argument("--max-train-events", type=int, default=512,
                        help="0 uses all development events")
    parser.add_argument("--max-eval-events", type=int, default=128,
                        help="0 uses all blind events")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.ar_orders = sorted(set(args.ar_orders))
    args.decay_half_lives_ns = sorted(set(args.decay_half_lives_ns))
    if (not args.decay_half_lives_ns or
            any(not np.isfinite(h) or h <= 0 for h in args.decay_half_lives_ns)):
        parser.error("Provide positive finite decay half-lives in ns")
    if args.local_horizon_samples < 1:
        parser.error("--local-horizon-samples must be positive")
    if not args.ar_orders or min(args.ar_orders) < 1:
        parser.error("AR orders must be positive")
    if args.ridge_alpha < 0 or args.train_stride < 1 or args.batch_events < 1:
        parser.error("Ridge alpha >= 0, stride and batch size >= 1 required")
    if args.n_cutoffs < 4:
        parser.error("At least 4 cutoffs required")
    return args


def main(argv=None):
    args = parse_args(argv)
    for board, root in (("UC", args.uc_cache_root), ("FBK", args.fbk_cache_root)):
        for mode in ("energy", "timing"):
            train, blind, time_ps, transform, mask, source = load_pair(
                root, mode, args.window, args.development_name, args.blind_name)
            for detector in (0, 1):
                evaluate_one(board, mode, detector, train, blind, time_ps,
                             transform, mask, source, args)
    print(f"All eight independent analyses written to {args.output_root}")


if __name__ == "__main__":
    main()
