"""Group original LED-aligned tails by saved ML predictions, without ML caches.

Run from repository root. Saved event_id is the zero-based ROOT entry index
(as used by prepared datasets), NOT the ROOT event_id branch. Only selected
entries are decoded and only their tail samples are accumulated. No filtering,
clipping, scaling, pulse selection or model fitting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from waveform_analysis.data.root_io import iterate_energy_chunks, energy_event_count
from waveform_analysis.signal.pulses import decode_oriented
from waveform_analysis.reporting.plotting import plot_context

BANDS = ((2, 12), (45, 55), (88, 98))
PACKAGE = Path(__file__).resolve().parents[1]


def prediction_groups(path, role, n_entries):
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["event_id"])
        pred = np.asarray(data["prediction_ps"], dtype=float)
        if "dataset_role" in data and str(data["dataset_role"].item()) != role:
            raise ValueError(f"{path}: prediction dataset role differs from --role")
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or pred.shape != ids.shape
            or ids.size < 2 or not np.isfinite(pred).all()
            or np.unique(ids).size != ids.size
            or ids.min() < 0 or ids.max() >= n_entries):
        raise ValueError(f"{path}: invalid predictions or ROOT entry indices")
    groups = []
    lookup = {}
    for g, (p0, p1) in enumerate(BANDS):
        lo, hi = np.percentile(pred, [p0, p1])
        selected = (pred >= lo) & (pred < hi)
        if not selected.any():
            raise ValueError(f"{path}: empty percentile band {p0}-{p1}")
        for event, value in zip(ids[selected], pred[selected]):
            lookup[int(event)] = (g, float(value))
        groups.append(dict(percentiles=[p0, p1], bounds_ps=[float(lo), float(hi)],
                           n_selected=int(selected.sum())))
    return lookup, groups


def aligned_tail(signal, dt_ns, threshold, time_ns):
    """First upward fixed absolute voltage crossing, linearly interpolated."""
    hits = np.flatnonzero((signal[:-1] < threshold) & (signal[1:] >= threshold))
    if not hits.size:
        return None
    i = int(hits[0])
    crossing = i + (threshold - signal[i]) / (signal[i + 1] - signal[i])
    x = crossing + time_ns / dt_ns
    if x[0] < 0 or x[-1] > signal.size - 1:
        return None
    lo = int(np.floor(x[0]))
    hi = min(signal.size, int(np.ceil(x[-1])) + 1)
    if not np.isfinite(signal[lo:hi]).all():
        return None
    return np.interp(x, np.arange(lo, hi), signal[lo:hi])


def aggregate(root, prediction_files, args):
    n_entries = energy_event_count(root)
    contexts = {}
    for family, path in prediction_files.items():
        lookup, groups = prediction_groups(path, args.role, n_entries)
        contexts[family] = dict(lookup=lookup, groups=groups, path=path)
    selected_entries = set().union(*(c["lookup"] for c in contexts.values()))
    entry = 0
    grid = None
    dt_ns = None
    for chunk in iterate_energy_chunks(
        root, energy_channels_one_based=(1, 2), timing_channels_one_based=(3, 4),
        step_size=256, entry_stop=max(selected_entries) + 1,
    ):
        for row in range(len(chunk.event_index)):
            event = entry
            entry += 1
            if event not in selected_entries:
                continue
            for family, c in contexts.items():
                group = c["lookup"].get(event)
                if group is None:
                    continue
                g, value = group
                prefix = "timing_" if family == "timing" else ""
                samples = chunk.timing_samples if prefix else chunk.samples
                gains = getattr(chunk, prefix + "vertical_gain_v_per_count")
                offsets = getattr(chunk, prefix + "vertical_offset_v")
                intervals = getattr(chunk, prefix + "horizontal_interval_s")
                steps = np.asarray(intervals[row], float) * 1e9
                if grid is None:
                    dt_ns = float(steps[0])
                    if not np.isfinite(dt_ns) or dt_ns <= 0:
                        raise ValueError("Invalid native sampling interval")
                    grid = args.tail_start_ns + dt_ns * np.arange(
                        int(np.floor((args.tail_end_ns - args.tail_start_ns) / dt_ns + 1e-8)) + 1)
                    for ctx in contexts.values():
                        ctx["sum"] = np.zeros((len(BANDS), 2, grid.size))
                        ctx["counts"] = np.zeros(len(BANDS), dtype=int)
                        ctx["pred_sum"] = np.zeros(len(BANDS))
                if not np.allclose(steps, dt_ns, rtol=1e-8, atol=0):
                    raise ValueError(f"{root}: inconsistent native sampling intervals")
                threshold = args.energy_threshold_mv if family == "energy" else args.timing_threshold_mv
                tails = []
                for d in range(2):
                    signal = decode_oriented(samples[d][row], gains[row, d], offsets[row, d], 1)
                    tails.append(aligned_tail(signal, dt_ns, threshold, grid))
                if any(t is None for t in tails):
                    continue
                c["sum"][g] += np.stack(tails)
                c["counts"][g] += 1
                c["pred_sum"][g] += value
    for family, c in contexts.items():
        if grid is None or np.any(c["counts"] == 0):
            raise ValueError(f"{root}: empty aligned group for {family}; check threshold and tail interval")
        c["mean"] = c["sum"] / c["counts"][:, None, None]
        for g, info in enumerate(c["groups"]):
            info["n_used"] = int(c["counts"][g])
            info["n_skipped"] = info["n_selected"] - info["n_used"]
            info["mean_prediction_ps"] = float(c["pred_sum"][g] / c["counts"][g])
    return grid, contexts


def save_result(board, family, grid, context, args, style):
    out = args.output_root / f"{board}_{family}"
    out.mkdir(parents=True, exist_ok=True)
    threshold = args.energy_threshold_mv if family == "energy" else args.timing_threshold_mv
    means = context["mean"]
    np.savez_compressed(out / "grouped_tails.npz", time_ns=grid, mean_mV=means,
                        counts=context["counts"],
                        prediction_bounds_ps=np.array([g["bounds_ps"] for g in context["groups"]]),
                        mean_prediction_ps=np.array([g["mean_prediction_ps"] for g in context["groups"]]))
    (out / "summary.json").write_text(json.dumps({
        "board": board, "family": family, "role": args.role,
        "predictions": str(context["path"]), "threshold_mV": threshold,
        "event_reference": "zero-based ROOT entry index",
        "alignment": "first upward fixed threshold crossing, linear interpolation, per detector",
        "preprocessing": "ADC calibration only; no clipping or filtering",
        "groups": context["groups"],
    }, indent=2), encoding="utf-8")
    colors = style["preprocessing"]["detector_colors"]
    for name, lo, hi in (("tail", args.tail_start_ns, args.tail_end_ns),
                         ("detail", args.zoom_start_ns, args.zoom_end_ns)):
        mask = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
        if mask.sum() < 2:
            raise ValueError("Plot interval contains fewer than two samples")
        with plot_context(style):
            fig, axes = plt.subplots(3, 1, sharex=True, sharey=True,
                                     figsize=(7.0, 5.2), constrained_layout=True)
            for g, ax in enumerate(axes):
                for d in range(2):
                    ax.plot(grid[mask], means[g, d, mask], color=colors[d],
                            lw=1.2, label=f"Detector {d+1}")
                info = context["groups"][g]
                p0, p1 = info["percentiles"]
                ax.set_title(f"P{p0}–P{p1} | Mean prediction: {info['mean_prediction_ps']:+.1f} ps")
                ax.margins(x=0)
                ax.grid(alpha=.2)
            axes[1].set_ylabel("Amplitude [mV]")
            axes[-1].set_xlabel("Time relative to LED [ns]")
            axes[0].legend(loc="best")
            fig.savefig(out / f"grouped_{name}.pdf", bbox_inches="tight")
            plt.close(fig)
    print(f"{board} {family}: {context['groups']} | {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uc-energy", type=Path)
    p.add_argument("--uc-timing", type=Path)
    p.add_argument("--fbk-energy", type=Path)
    p.add_argument("--fbk-timing", type=Path)
    p.add_argument("--uc-root", type=Path)
    p.add_argument("--fbk-root", type=Path)
    p.add_argument("--role", choices=("development", "blind"), default="development")
    p.add_argument("--energy-threshold-mv", type=float, default=10.)
    p.add_argument("--timing-threshold-mv", type=float, default=20.)
    p.add_argument("--tail-start-ns", type=float, default=12.)
    p.add_argument("--tail-end-ns", type=float, default=45.)
    p.add_argument("--zoom-start-ns", type=float, default=25.)
    p.add_argument("--zoom-end-ns", type=float, default=29.)
    p.add_argument("--output-root", type=Path, default=Path("fig/results/prediction_tails"))
    args = p.parse_args()
    if not (0 <= args.tail_start_ns <= args.zoom_start_ns < args.zoom_end_ns <= args.tail_end_ns
            and args.energy_threshold_mv > 0 and args.timing_threshold_mv > 0):
        p.error("Require positive thresholds and zoom contained inside tail")
    if not any((args.uc_energy, args.uc_timing, args.fbk_energy, args.fbk_timing)):
        p.error("Supply at least one saved prediction NPZ")
    style = json.loads((PACKAGE / "config/plots/default.json").read_text(encoding="utf-8"))
    run = "R1" if args.role == "development" else "R2"
    for board, date in (("UC", "10-03"), ("FBK", "10-05")):
        files = {f: getattr(args, f"{board.lower()}_{f}") for f in ("energy", "timing")}
        files = {f: path for f, path in files.items() if path is not None}
        if not files:
            continue
        root = getattr(args, f"{board.lower()}_root") or PACKAGE / "processed_data" / date / f"48V-{run}.root"
        grid, contexts = aggregate(root, files, args)
        for family, context in contexts.items():
            save_result(board, family, grid, context, args, style)


if __name__ == "__main__":
    main()
