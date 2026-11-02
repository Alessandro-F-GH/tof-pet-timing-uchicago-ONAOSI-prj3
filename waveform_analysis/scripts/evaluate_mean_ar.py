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

METHODS = ("mean", "mean_ar", "standardized_ar", "gru", "knn", "knn_boundary")


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



def make_gru_encoder_decoder(hidden):
    """GRU encodes the entire real prefix; MLP decodes arbitrary future times."""
    import torch
    from torch import nn

    class EncoderDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.GRU(2, hidden, batch_first=True)
            self.decoder = nn.Sequential(
                nn.Linear(hidden + 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        def forward(self, prefix_features, future_features):
            _, state = self.encoder(prefix_features)
            context = state[-1].unsqueeze(1).expand(
                -1, future_features.shape[1], -1)
            return self.decoder(
                torch.cat((context, future_features), dim=-1)).squeeze(-1)

    return EncoderDecoder()


def gru_features(values, indices, length):
    """Read observed voltage and relative time, without residual transforms."""
    return np.stack((
        np.asarray(values, dtype=np.float32),
        np.broadcast_to(indices.astype(np.float32) / max(length - 1, 1),
                        values.shape),
    ), axis=-1).astype(np.float32)


def gru_future_features(mean, positions, batch, length):
    """Future timestamp and population mean, without event-specific future."""
    return np.stack((
        np.broadcast_to(positions.astype(np.float32) /
                        max(length - 1, 1), (batch, len(positions))),
        np.broadcast_to(mean[positions], (batch, len(positions))),
    ), axis=-1).astype(np.float32)


def fit_gru_forecaster(training, mean, std, mask, cutoff, stop, args, seed):
    """Fit a cutoff-specific direct forecaster using only development events.

    Train on the complete valid future interval with the whole real prefix.
    Decoder directly regresses voltage, rather than residuals.
    """
    import torch

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    lo, _ = segment_for_cutoff_for_gru(mask, cutoff)
    if cutoff <= lo or stop <= cutoff:
        raise ValueError("GRU requires a nonempty prefix and forecast interval")
    length = len(mean)
    model = make_gru_encoder_decoder(args.gru_hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.gru_lr)
    prefix_positions = np.arange(lo, cutoff, dtype=np.int64)
    future_positions = np.arange(cutoff, stop, dtype=np.int64)
    y = np.asarray(training, np.float32)
    avg_variance = float(np.mean(std[future_positions] ** 2))
    variance_scale = max(avg_variance, 1e-6)
    mean32 = np.asarray(mean, np.float32)

    for epoch in range(args.gru_epochs):
        model.train()
        total = 0.
        for _ in range(args.gru_steps_per_epoch):
            ids = rng.integers(0, len(y), size=args.gru_batch_size)
            prefix = gru_features(
                y[ids[:, None], prefix_positions[None, :]],
                prefix_positions, length)
            future = gru_future_features(
                mean32, future_positions, len(ids), length)
            truth = torch.from_numpy(y[ids[:, None], future_positions[None, :]].copy())
            predicted_x = model(torch.from_numpy(prefix),
                                torch.from_numpy(future))
            loss = ((predicted_x - truth) ** 2).mean() / variance_scale
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            total += float(loss.detach())
        print(f"[GRU cutoff={cutoff}] epoch {epoch+1}/{args.gru_epochs} "
              f"scaled future MSE={total / args.gru_steps_per_epoch:.5f}",
              flush=True)
    model.eval()
    return model


def forecast_gru(evaluation, mean, std, mask, cutoff, end, model,
                 batch_size=32, future_batch=256):
    """Direct, nonautoregressive suffix inference with frozen encoder state."""
    import torch

    lo, _ = segment_for_cutoff_for_gru(mask, cutoff)
    if end <= cutoff or cutoff <= lo:
        raise ValueError("GRU forecast requires a nonempty valid prefix")
    length = len(mean)
    mean32 = np.asarray(mean, np.float32)
    std32 = np.asarray(std, np.float32)
    prefix_positions = np.arange(lo, cutoff, dtype=np.int64)
    output = np.empty((len(evaluation), end - cutoff), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for first in range(0, len(evaluation), batch_size):
            batch = np.asarray(evaluation[first:first + batch_size],
                               np.float32)
            prefix = gru_features(
                batch[:, prefix_positions], prefix_positions, length)
            _, state = model.encoder(torch.from_numpy(prefix))
            hidden = state[-1]
            for t0 in range(cutoff, end, future_batch):
                t1 = min(end, t0 + future_batch)
                positions = np.arange(t0, t1, dtype=np.int64)
                future = torch.from_numpy(gru_future_features(
                    mean32, positions, len(batch), length))
                context = hidden.unsqueeze(1).expand(
                    -1, len(positions), -1)
                predicted_x = model.decoder(
                    torch.cat((context, future), dim=-1)).squeeze(-1).numpy()
                output[first:first + len(batch), t0-cutoff:t1-cutoff] = (
                    np.clip(predicted_x, 0., 1.))
    return output


def segment_for_cutoff_for_gru(mask, cutoff):
    for lo, hi in valid_intervals(mask):
        if lo <= cutoff < hi:
            return lo, hi
    raise ValueError("Cutoff outside retained mask")


def forecast_knn(training, evaluation, mask, cutoff, end, k_values, batch_size=32):
    """Nearest development prefixes -> mean of neighbors' actual future.

    Distances are computed only over the contiguous observed segment preceding
    the cutoff. The blind suffix is never used to identify neighbors.
    """
    lo, _ = segment_for_cutoff_for_gru(mask, cutoff)
    if cutoff <= lo or end <= cutoff:
        raise ValueError("kNN needs an observed prefix and future interval")
    source = np.asarray(training[:, lo:cutoff], dtype=np.float64)
    target = np.asarray(evaluation[:, lo:cutoff], dtype=np.float64)
    k_max = max(k_values)
    if k_max > len(source):
        raise ValueError("kNN k exceeds number of development events")
    # Squared Euclidean distances; common prefix length makes scaling
    # unnecessary for rankings. Matrix products avoid 3-D allocations.
    source_sq = np.einsum("ij,ij->i", source, source)
    forecasts = {k: np.empty((len(target), end-cutoff),
                              dtype=np.float64) for k in k_values}
    for first in range(0, len(target), batch_size):
        batch = target[first:first+batch_size]
        dist2 = (np.einsum("ij,ij->i", batch, batch)[:, None]
                 + source_sq[None, :] - 2.0 * batch @ source.T)
        np.maximum(dist2, 0., out=dist2)
        neighbors = np.argsort(dist2, axis=1, kind="stable")[:, :k_max]
        future = training[neighbors, cutoff:end]
        for k in k_values:
            forecasts[k][first:first+len(batch)] = future[:, :k].mean(axis=1)
    return forecasts


def boundary_correct_knn(evaluation, knn_suffix, cutoff, time_ps,
                         tau_ns=0.5):
    """Short exponential offset correction for the kNN forecast at cutoff.

    Only uses the last observed target value and the first kNN future value.
    The first forecast equals the last observed value; the correction decays
    toward the ordinary kNN forecast. No blind suffix is accessed.
    """
    if cutoff < 1 or tau_ns <= 0:
        raise ValueError("Boundary correction requires cutoff >= 1 and tau_ns > 0")
    t = np.asarray(time_ps[cutoff:cutoff + knn_suffix.shape[1]], np.float64)
    weights = np.exp(-(t - t[0]) / (tau_ns * 1000.0))
    offset = evaluation[:, cutoff - 1] - knn_suffix[:, 0]
    return np.clip(knn_suffix + offset[:, None] * weights[None, :], 0., 1.)


def select_knn_k(training, mean, mask, cutoff, end, k_values,
                 batch_size=32):
    """Development-only leave-one-out selection by full valid-suffix MSE.

    Excludes each event from its neighbor pool. After the next mask, the
    forecast uses the development population mean, matching blind scoring.
    """
    lo, _ = segment_for_cutoff_for_gru(mask, cutoff)
    prefix = np.asarray(training[:, lo:cutoff], dtype=np.float64)
    n = len(prefix)
    candidates = sorted(set(k for k in k_values if 1 <= k < n))
    if not candidates:
        raise ValueError("No kNN candidate smaller than development event count")
    source_sq = np.einsum("ij,ij->i", prefix, prefix)
    future_valid = np.asarray(mask[cutoff:], dtype=bool)
    truth = np.asarray(training[:, cutoff:][:, future_valid], dtype=np.float64)
    baseline = np.broadcast_to(
        np.clip(mean[cutoff:], 0., 1.), (n, len(mean)-cutoff)).copy()
    total_squared = {k: 0. for k in candidates}
    for first in range(0, n, batch_size):
        last = min(n, first+batch_size)
        batch = prefix[first:last]
        dist2 = (np.einsum("ij,ij->i", batch, batch)[:, None]
                 + source_sq[None, :] - 2. * batch @ prefix.T)
        np.maximum(dist2, 0., out=dist2)
        dist2[np.arange(last-first), np.arange(first, last)] = np.inf
        neighbors = np.argsort(dist2, axis=1, kind="stable")[:, :max(candidates)]
        futures = training[neighbors, cutoff:end]
        for k in candidates:
            pred = baseline[first:last].copy()
            pred[:, :end-cutoff] = futures[:, :k].mean(axis=1)
            err = pred[:, future_valid] - truth[first:last]
            total_squared[k] += float(np.sum(err * err))
    denom = truth.size
    errors = {k: value / denom for k, value in total_squared.items()}
    # Smaller k wins exact ties to keep selection deterministic.
    best_k = min(candidates, key=lambda k: (errors[k], k))
    return best_k, errors


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
              "standardized_ar": "#35956C", "gru": "#8056B3", "knn": "#B35A7D", "knn_boundary": "#B88521"}
    ar_order = max(r["order"] for r in rows
                   if r["method"] in ("mean_ar", "standardized_ar"))
    for method in METHODS:
        series = sorted(
            (r for r in rows if r["method"] == method and
             (method in ("mean", "gru", "knn", "knn_boundary") or r["order"] == ar_order)),
            key=lambda r: r["cutoff_sample"])
        if not series:
            continue
        label = ("kNN (selected k)" if method == "knn" else
                 "Boundary-corrected kNN" if method == "knn_boundary" else method)
        ax.plot([r["cutoff_ns"] for r in series],
                [r["rmse_mV"] for r in series],
                marker="o", color=colors[method], label=label)
    ax.set(xlabel="Cutoff position (ns)", ylabel="Suffix RMSE (mV)", title=title)
    ax.grid(alpha=.3)
    ax.legend(title=f"AR order {ar_order}")
    fig.tight_layout()
    fig.savefig(out / "rmse_vs_cutoff.png", dpi=160)
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
    # A dedicated direct forecaster is fitted at each cutoff.
    # No blind data are used for training or checkpoint selection.
    gru_seed = args.seed + detector + (0 if board == "UC" else 100)
    scale = np.asarray(transform.maximum - transform.minimum).reshape(-1)
    mv_scale = float(scale[detector] if scale.size >= 2 else scale[0])
    rows = []
    examples = []
    gru_examples = {}
    knn_examples = {}
    boundary_examples = {}
    knn_selection = {}
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
                full_prediction = baseline_full.copy()
                full_prediction[:, :stop-cutoff] = prediction
                metric = metric_row(truth, full_prediction[:, future_valid], mv_scale)
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
                if method == "standardized_ar" and order == args.ar_orders[-1]:
                    examples.append((cutoff, stop, evaluation[0].copy(),
                                     full_prediction[0].copy(), baseline_full[0].copy(),
                                     int(eval_ids[0])))
        if args.with_knn:
            best_k, dev_mse = select_knn_k(
                training, mean, mask, cutoff, stop, args.knn_k,
                batch_size=args.knn_batch_size)
            knn_selection[str(cutoff)] = {
                "chosen_k": best_k,
                "development_loo_mse_by_k": {
                    str(k): float(mse) for k, mse in dev_mse.items()},
            }
            print(f"[{name}] cutoff {cutoff}: development LOO "
                  f"k={best_k}, MSE={dev_mse[best_k]:.6g}", flush=True)
            suffix = forecast_knn(
                training, evaluation, mask, cutoff, stop, [best_k],
                batch_size=args.knn_batch_size)[best_k]
            knn_full = baseline_full.copy()
            knn_full[:, :stop-cutoff] = suffix
            knn_metric = metric_row(
                truth, knn_full[:, future_valid], mv_scale)
            rows.append({
                **knn_metric, "method": "knn", "order": best_k,
                "cutoff_sample": cutoff,
                "cutoff_ns": float(time_ps[cutoff]/1000),
                "n_missing": int(np.count_nonzero(future_valid)),
                "ar_predicted_samples": stop-cutoff,
                "n_eval_events": len(eval_ids),
                "mse_improvement_percent": (
                    100 * (1 - knn_metric["mse_normalized"] /
                           base["mse_normalized"])
                    if base["mse_normalized"] > 0 else 0.),
                "stable": True, "root_radius": 0.,
            })
            knn_examples[cutoff] = (best_k, knn_full[0].copy())
            if args.with_boundary_knn:
                corrected = boundary_correct_knn(
                    evaluation, suffix, cutoff, time_ps[:stop],
                    tau_ns=args.boundary_tau_ns)
                corrected_full = baseline_full.copy()
                corrected_full[:, :stop-cutoff] = corrected
                corrected_metric = metric_row(
                    truth, corrected_full[:, future_valid], mv_scale)
                rows.append({
                    **corrected_metric, "method": "knn_boundary", "order": best_k,
                    "cutoff_sample": cutoff,
                    "cutoff_ns": float(time_ps[cutoff]/1000),
                    "n_missing": int(np.count_nonzero(future_valid)),
                    "ar_predicted_samples": stop-cutoff,
                    "n_eval_events": len(eval_ids),
                    "mse_improvement_percent": (
                        100 * (1 - corrected_metric["mse_normalized"] /
                               base["mse_normalized"])
                        if base["mse_normalized"] > 0 else 0.),
                    "stable": True, "root_radius": 0.,
                })
                boundary_examples[cutoff] = corrected_full[0].copy()
        if args.with_gru:
            import torch
            checkpoint = out / f"gru_cutoff_{cutoff:05d}.pt"
            config = {
                "architecture": "gru_direct_voltage_full_suffix_v2",
                "cutoff": cutoff, "stop": stop,
                "hidden": args.gru_hidden, "epochs": args.gru_epochs,
                "batch_size": args.gru_batch_size,
                "steps_per_epoch": args.gru_steps_per_epoch,
                "lr": args.gru_lr, "seed": gru_seed,
                "train_indices": train_ids.tolist(),
            }
            gru = None
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu",
                                   weights_only=True)
                if saved.get("config") == config:
                    gru = make_gru_encoder_decoder(args.gru_hidden)
                    gru.load_state_dict(saved["state_dict"])
                    gru.eval()
                    print(f"[{name}] GRU checkpoint reused: {checkpoint}",
                          flush=True)
            if gru is None:
                gru = fit_gru_forecaster(
                    training, mean, std, mask, cutoff, stop, args, gru_seed + cutoff)
                torch.save({"config": config, "state_dict": gru.state_dict()},
                           checkpoint)
            gru_suffix = forecast_gru(
                evaluation, mean, std, mask, cutoff, stop, gru,
                batch_size=args.gru_batch_size)
            gru_full = baseline_full.copy()
            gru_full[:, :stop-cutoff] = gru_suffix
            gru_metric = metric_row(truth, gru_full[:, future_valid], mv_scale)
            rows.append({
                **gru_metric, "method": "gru", "order": 0,
                "cutoff_sample": cutoff, "cutoff_ns": float(time_ps[cutoff]/1000),
                "n_missing": int(np.count_nonzero(future_valid)),
                "ar_predicted_samples": stop-cutoff,
                "n_eval_events": len(eval_ids),
                "mse_improvement_percent": (
                    100 * (1 - gru_metric["mse_normalized"] / base["mse_normalized"])
                    if base["mse_normalized"] > 0 else 0.),
                "stable": True, "root_radius": 0.,
            })
            gru_examples[cutoff] = gru_full[0].copy()
        rows.append({
            **base, "method": "mean", "order": 0,
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
        "boundary_knn": {"enabled": args.with_boundary_knn,
                         "tau_ns": args.boundary_tau_ns,
                         "first_sample_anchor": "last_observed_value",
                         "decay": "exponential"},
        "knn": {"enabled": args.with_knn, "k_values": args.knn_k,
                "distance": "euclidean_observed_contiguous_prefix",
                "aggregation": "uniform_mean",
                "selection": "development_leave_one_out_full_suffix_mse",
                "per_cutoff": knn_selection},
        "gru": ({"enabled": True, "architecture": "gru_direct_voltage_full_suffix_v2",
                 "per_cutoff": True, "hidden": args.gru_hidden,
                 "epochs": args.gru_epochs,
                 "steps_per_epoch": args.gru_steps_per_epoch,
                 "batch_size": args.gru_batch_size, "learning_rate": args.gru_lr,
                 "checkpoint_template": "gru_cutoff_XXXXX.pt"}
                if args.with_gru else {"enabled": False}),
        "models": {method: {str(p): {
            **{k:v for k,v in fitted[method][p].items() if k != "coefficients"},
            "coefficients": fitted[method][p]["coefficients"].tolist()
        } for p in args.ar_orders} for method in fitted},
    }, indent=2), encoding="utf-8")
    plot_summary(rows, out, name)
    example_dir = out / "waveform_examples"
    example_dir.mkdir(exist_ok=True)
    ns = np.asarray(time_ps) / 1000
    for cutoff, stop, original, prediction, baseline, event_id in examples:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(ns, original, color="#222222", lw=1.2, label="Blind original")
        ax.plot(ns[cutoff:], baseline, color="#C98740", label="Mean replacement")
        ax.plot(ns[cutoff:], prediction, color="#2B9C74",
                label=f"Standardized AR (order {args.ar_orders[-1]})")
        if cutoff in boundary_examples:
            ax.plot(ns[cutoff:], boundary_examples[cutoff],
                    color="#B88521", lw=1.2, label="Boundary-corrected kNN")
        if cutoff in gru_examples:
            ax.plot(ns[cutoff:], gru_examples[cutoff],
                    color="#8056B3", lw=1.2, label="GRU forecast")
        if args.with_knn:
            k, knn_prediction = knn_examples[cutoff]
            ax.plot(ns[cutoff:], knn_prediction,
                    color="#B35A7D", lw=1.2, label=f"kNN (k={k})")
        ax.axvline(ns[cutoff], color="gray", ls="--",
                   label=f"Cutoff {ns[cutoff]:.3f} ns")
        if stop < len(ns):
            ax.axvline(ns[stop], color="#8B8B8B", ls=":",
                       label="AR stops at mask boundary")
        ax.set(xlabel="Time (ns)", ylabel="Normalized voltage [0,1]",
               title=f"{name}, blind row {event_id}, cutoff {ns[cutoff]:.3f} ns")
        ax.legend()
        ax.grid(alpha=.25)
        fig.tight_layout()
        fig.savefig(example_dir / f"waveform_cutoff_{cutoff:05d}.png", dpi=160)
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
    parser.add_argument("--ridge-alpha", type=float, default=1e-3,
                        help="Ridge penalty relative to mean Gram diagonal")
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--batch-events", type=int, default=8)
    parser.add_argument("--max-train-events", type=int, default=512,
                        help="0 uses all development events")
    parser.add_argument("--max-eval-events", type=int, default=128,
                        help="0 uses all blind events")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-boundary-knn", action="store_true",
                        help="kNN with exponentially decaying boundary offset")
    parser.add_argument("--boundary-tau-ns", type=float, default=0.5,
                        help="Fixed decay time of boundary correction in ns")
    parser.add_argument("--with-knn", action="store_true",
                        help="Compare nearest development-prefix waveforms")
    parser.add_argument("--knn-k", nargs="+", type=int, default=[1, 5, 20],
                        help="Candidate neighbor counts selected separately per cutoff on development")
    parser.add_argument("--knn-batch-size", type=int, default=32)
    parser.add_argument("--with-gru", action="store_true",
                        help="Train a causal GRU forecaster on development data")
    parser.add_argument("--gru-hidden", type=int, default=32)
    parser.add_argument("--gru-epochs", type=int, default=8)
    parser.add_argument("--gru-steps-per-epoch", type=int, default=32)
    parser.add_argument("--gru-batch-size", type=int, default=32)
    parser.add_argument("--gru-lr", type=float, default=1e-3)
    args = parser.parse_args(argv)
    args.output_root = args.output_root.expanduser().resolve()
    args.ar_orders = sorted(set(args.ar_orders))
    if not args.ar_orders or min(args.ar_orders) < 1:
        parser.error("AR orders must be positive")
    if args.ridge_alpha < 0 or args.train_stride < 1 or args.batch_events < 1:
        parser.error("Ridge alpha >= 0, stride and batch size >= 1 required")
    if (min(args.gru_hidden, args.gru_epochs,
            args.gru_steps_per_epoch, args.gru_batch_size) < 1 or args.gru_lr <= 0):
        parser.error("GRU hyperparameters must be positive")
    args.knn_k = sorted(set(args.knn_k))
    if not args.knn_k or min(args.knn_k) < 1 or args.knn_batch_size < 1:
        parser.error("kNN neighbor counts and batch size must be positive")
    if args.boundary_tau_ns <= 0:
        parser.error("Boundary correction tau must be positive")
    if args.with_boundary_knn:
        args.with_knn = True
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
