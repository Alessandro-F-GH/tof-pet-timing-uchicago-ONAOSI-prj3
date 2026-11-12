"""Group SAVED ML predictions by the original waveform phase near a fixed frequency.

No ML inference, model loading, filtering, waveform modification, or refitting.
Uses exact development event-ID alignment and the same prepared original inputs.
Reports phase bins (equal angular widths) and circular-linear association.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.view import mode_family
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.scripts.analyze_waveform_fft import validate_grid
from waveform_analysis.scripts.analyze_oscillation_phase import (
    project_fixed_frequency, circular_relation)
from waveform_analysis.scripts.evaluate_conditional_knn import _prediction_payload


def circular_bins(phases, predictions, n_bins=12):
    """Equal-width phase bins; empty bins are reported explicitly."""
    phi = np.asarray(phases, np.float64)
    y = np.asarray(predictions, np.float64)
    if phi.shape != y.shape or phi.ndim != 1 or n_bins < 3:
        raise ValueError("Phase/prediction shapes or bin count invalid")
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    groups = np.clip(np.searchsorted(edges, phi, side="right") - 1,
                     0, n_bins - 1)
    records = []
    for j in range(n_bins):
        value = y[groups == j]
        records.append({
            "bin": j,
            "phase_center_deg": float(np.degrees((edges[j] + edges[j+1])/2)),
            "n": int(value.size),
            "mean_prediction_ps": float(np.mean(value)) if value.size else None,
            "std_prediction_ps": (float(np.std(value, ddof=1))
                                  if value.size > 1 else None),
            "sem_prediction_ps": (float(np.std(value, ddof=1)/np.sqrt(value.size))
                                  if value.size > 1 else None),
            "median_prediction_ps": float(np.median(value)) if value.size else None,
        })
    return records, groups


def plot_grouped(out, tag, records, phases, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    center = np.asarray([r["phase_center_deg"] for r in records])
    means = np.array([np.nan if r["mean_prediction_ps"] is None
                      else r["mean_prediction_ps"] for r in records])
    sem = np.array([np.nan if r["sem_prediction_ps"] is None
                    else r["sem_prediction_ps"] for r in records])
    counts = np.array([r["n"] for r in records])
    ax[0].errorbar(center, means, yerr=sem, fmt="o-", capsize=3,
                   label="Mean ± SEM (descriptive)")
    ax[0].axhline(np.mean(predictions), color="0.5", ls="--",
                  label="Overall mean")
    ax[0].set(xlim=(-180, 180), xlabel="Oscillation phase (degrees)",
              ylabel="Saved ML prediction (ps)", title=tag)
    ax[0].legend()
    ax[0].grid(alpha=.2)
    ax[1].bar(center, counts, width=330/len(records))
    ax[1].set(xlim=(-180, 180), xlabel="Oscillation phase (degrees)",
              ylabel="Events per bin")
    ax[1].grid(alpha=.2)
    fig.savefig(out / f"{tag}_phase_bins.png", dpi=170)
    fig.savefig(out / f"{tag}_phase_bins.pdf")
    plt.close(fig)


def run(args):
    if args.cache_root and args.artifact_root:
        raise ValueError("Choose one cache root")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    record = read_run(run_dir)
    config = resolve_run_config(run_dir)
    manifest = record["manifest"]
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find("development", config, manifest)
    family = mode_family(config["mode"])
    waveform = getattr(dataset, f"{family}_windows")
    time_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    if waveform is None or waveform.shape != (dataset.n_events, 2, len(time_ps)):
        raise ValueError("Unexpected prepared waveform shape")
    store = RunStore(run_dir, artifact_root=config["preprocessing"].get("artifact_root"))
    saved_path = store.development_predictions_path
    if not saved_path.is_file():
        saved_path = run_dir / "artifacts" / "development_pred.npz"
    predictions = _prediction_payload(saved_path, dataset, manifest)
    all_ids = np.asarray(dataset.event_index, np.int64)
    if args.max_events and args.max_events < dataset.n_events:
        rows = np.sort(np.random.default_rng(args.seed).choice(
            dataset.n_events, size=args.max_events, replace=False))
    else:
        rows = np.arange(dataset.n_events)
    time_ns = time_ps / 1000.
    windows = {"pre_LED": args.pre_window_ns, "tail": args.tail_window_ns}
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    transform = getattr(dataset, f"{family}_transform")
    minimum = np.asarray(transform.minimum, np.float64).reshape(2, 1)
    span = (np.asarray(transform.maximum, np.float64).reshape(2, 1)
            - minimum)
    if np.any(span <= 0):
        raise ValueError("Invalid physical waveform transformation")

    summary = {"model": manifest["model"], "run": str(run_dir),
               "source": str(saved_path), "dataset_role": "development",
               "n_selected": int(len(rows)), "frequency_GHz": args.frequency_GHz,
               "n_bins": args.phase_bins, "amplitude_percentile": args.min_amplitude_percentile,
               "no_inference": True, "waveform_modified": False, "results": {}}
    all_rows = []
    for win_label, limits in windows.items():
        idx = np.flatnonzero((time_ns >= limits[0] - 1e-9)
                             & (time_ns <= limits[1] + 1e-9))
        if idx.size < 16:
            raise ValueError(f"Not enough samples in {win_label}")
        for d in (0, 1):
            amp = np.empty(len(rows), np.float64)
            phase = np.empty(len(rows), np.float64)
            for start in range(0, len(rows), args.chunk_size):
                end = min(start + args.chunk_size, len(rows))
                input_waves = np.asarray(
                    waveform[rows[start:end], d, :], np.float64)
                input_waves = input_waves[:, idx] * span[d, 0] + minimum[d, 0]
                _, a, p = project_fixed_frequency(
                    input_waves, time_ns[idx], args.frequency_GHz)
                amp[start:end], phase[start:end] = a, p
            amplitude_floor = float(np.percentile(
                amp, args.min_amplitude_percentile))
            use = (np.isfinite(phase) & np.isfinite(amp)
                   & (amp >= amplitude_floor))
            if use.sum() < max(8, args.phase_bins):
                raise ValueError("Too few events after amplitude selection")
            y = predictions[rows][use]
            phi = phase[use]
            result, groups = circular_bins(phi, y, args.phase_bins)
            association = circular_relation(phi, y)
            tag = f"{win_label}_detector{d+1}"
            summary["results"][tag] = {
                "n_used": int(use.sum()), "amplitude_floor_mV": amplitude_floor,
                "mean_saved_prediction_ps": float(y.mean()),
                "phase_association": association, "bins": result,
            }
            plot_grouped(out, tag, result, phi, y)
            for i, j in enumerate(result):
                all_rows.append({"window": win_label, "detector": d+1, **j})
            print(f"PHASE GROUP | {tag} | used={use.sum()} "
                  f"| R2={association['r2_in_sample']:.5f} "
                  f"| sinusoid_amp={association['sinusoid_magnitude_ps']:.3f} ps",
                  flush=True)
            np.savez_compressed(
                out / f"{tag}_events.npz",
                event_id=all_ids[rows], saved_prediction_ps=predictions[rows],
                oscillation_amplitude_mV=amp, oscillation_phase_rad=phase,
                selected=use, phase_bin=np.where(
                    use, np.clip(np.searchsorted(
                        np.linspace(-np.pi, np.pi, args.phase_bins+1),
                        phase, side="right")-1, 0, args.phase_bins-1), -1))
    with (out / "phase_bins.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-events", type=int, default=10000,
                   help="0 selects all development events")
    p.add_argument("--frequency-GHz", type=float, default=2.5)
    p.add_argument("--phase-bins", type=int, default=12)
    p.add_argument("--min-amplitude-percentile", type=float, default=20.)
    p.add_argument("--pre-window-ns", type=float, nargs=2, default=(-2., 0.))
    p.add_argument("--tail-window-ns", type=float, nargs=2, default=(12., 45.))
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
