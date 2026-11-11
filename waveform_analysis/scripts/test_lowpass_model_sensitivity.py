"""Probe frozen MultiRocket-RDST sensitivity to low-pass filtering of one PET event.

Reuses the exact saved prepared waveform and trained feature transformer/model.
No retraining, no waveform alignment changes, no XAI imputation. The replay of
the unmodified signal MUST agree with its saved development prediction.

This is a model sensitivity (intervention on input), not evidence that an
oscillation is causally responsible for the detector's timing error.
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
from waveform_analysis.engine.train import load_fitted_model, release_training_memory
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.models import get_model
from waveform_analysis.scripts.evaluate_conditional_knn import _prediction_payload
from waveform_analysis.scripts.analyze_waveform_fft import validate_grid


def lowpass_prepared(pairs, *, sample_rate_GHz, cutoff_GHz,
                     order=4, detectors=(0, 1), clip=True):
    """SOS zero-phase low-pass on entire original prepared grid, one channel at a time.

    Filtering is performed in the prepared affine amplitude units; an LTI
    low-pass commutes with affine scaling because DC gain is unity.
    """
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(pairs, np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or x.shape[2] < 32:
        raise ValueError("Expected prepared [event, detector, time>=32] waveforms")
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite prepared input")
    if not (0 < cutoff_GHz < sample_rate_GHz / 2) or order < 1:
        raise ValueError("Cutoff must be in (0, Nyquist), order >=1")
    if not detectors or any(d not in (0, 1) for d in detectors):
        raise ValueError("Choose detector 1, detector 2 or both")
    sos = butter(order, cutoff_GHz, fs=sample_rate_GHz,
                 btype="lowpass", output="sos")
    out = x.copy()
    for detector in detectors:
        out[:, detector, :] = sosfiltfilt(
            sos, x[:, detector, :].astype(np.float64), axis=-1).astype(np.float32)
    preclip_max = float(np.max(np.abs(out[:, list(detectors), :] - np.clip(
        out[:, list(detectors), :], 0., 1.))))
    if clip:
        out[:, list(detectors), :] = np.clip(out[:, list(detectors), :], 0., 1.)
    return out, preclip_max


def plot_one_event(out, x, time_ns, predictions, cutoffs, detector_label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for d in range(2):
        axes[d].plot(time_ns, x[0, d], label="Original", linewidth=1.4)
        for cutoff, wave in cutoffs:
            axes[d].plot(time_ns, wave[0, d], label=f"LP {cutoff:g} GHz",
                         alpha=.85, linewidth=1)
        axes[d].set(ylabel=f"Detector {d+1} prepared units",
                    title=f"Detector {d+1} | low-pass: {detector_label}")
        axes[d].grid(alpha=.2)
        axes[d].legend(fontsize=8, ncol=3)
    fx = [float("inf")] + [float(c) for c, _ in cutoffs]
    yp = [float(predictions["original"])] + [
        float(predictions[f"{c:g}"]) for c, _ in cutoffs]
    axes[2].plot(range(len(fx)), yp, "o-")
    axes[2].set_xticks(range(len(fx)),
                       ["Original"] + [f"{f:g}" for f in fx[1:]])
    axes[2].set(xlabel="Low-pass cutoff (GHz)", ylabel="Model output (ps)")
    axes[2].grid(alpha=.2)
    axes[1].set_xlabel("Time relative to LED (ns)")
    fig.savefig(out / "single_event_lowpass.png", dpi=170)
    fig.savefig(out / "single_event_lowpass.pdf")
    plt.close(fig)


def run(args):
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Choose --cache-root or --artifact-root")
    if args.max_events < 1 or args.batch_size < 1:
        raise ValueError("Event count / batch size must be positive")
    cutoffs = sorted(set(map(float, args.cutoff_GHz)), reverse=True)
    if len(cutoffs) != len(args.cutoff_GHz):
        raise ValueError("Duplicate cutoff")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    run_record = read_run(run_dir)
    config = resolve_run_config(run_dir)
    manifest = run_record["manifest"]
    if manifest["model"] not in {"direct_multirocket_rdst", "shared_multirocket_rdst",
                                  "direct_multirocket", "shared_multirocket",
                                  "direct_rdst", "shared_rdst"}:
        raise ValueError(f"Expected MultiRocket/RDST model, got {manifest['model']}")
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find("development", config, manifest)
    family = mode_family(config["mode"])
    waveforms = getattr(dataset, f"{family}_windows")
    time_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    if waveforms is None or waveforms.shape != (dataset.n_events, 2, len(time_ps)):
        raise ValueError("Prepared waveform shape mismatch")
    fs = 1000. / dt_ps
    if any(not 0 < c < fs / 2 for c in cutoffs):
        raise ValueError(f"Cutoffs must lie strictly below Nyquist {fs/2:.4g} GHz")
    store = RunStore(run_dir, artifact_root=config["preprocessing"].get("artifact_root"))
    saved_path = store.development_predictions_path
    if not saved_path.is_file():
        saved_path = run_dir / "artifacts" / "development_pred.npz"
    saved_prediction = _prediction_payload(saved_path, dataset, manifest)
    ids = np.asarray(dataset.event_index, np.int64)
    if args.event_id is not None:
        rows = np.flatnonzero(ids == args.event_id)
        if len(rows) != 1:
            raise ValueError(f"Event ID {args.event_id} is not unique/present")
    else:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(dataset.n_events,
                                  min(args.max_events, dataset.n_events),
                                  replace=False))
    inputs = np.asarray(waveforms[rows], np.float32)
    original_saved = saved_prediction[rows]
    spec = get_model(manifest["model"])
    model_dir = tshap._resolve_tshap_model_directory(
        spec, store, run_dir, explicit_model_dir=True)
    fitted = load_fitted_model(
        spec, model_dir, run_record["best"]["parameters"], config)
    detector_labels = {"1": (0,), "2": (1,), "both": (0, 1)}
    channels = detector_labels[args.filter_detector]
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        baseline = tshap._predict_bounded(spec, fitted, inputs, args.batch_size)
        max_replay_delta = float(np.max(np.abs(baseline - original_saved)))
        if not np.allclose(baseline, original_saved, atol=1e-4, rtol=1e-6):
            raise ValueError(
                f"Frozen model replay mismatch, max={max_replay_delta:.6g} ps; "
                "do not compare filtered predictions to incompatible originals.")
        records, snapshots = [], []
        results = {}
        for cutoff in cutoffs:
            altered, preclip = lowpass_prepared(
                inputs, sample_rate_GHz=fs, cutoff_GHz=cutoff,
                order=args.order, detectors=channels)
            altered_prediction = tshap._predict_bounded(
                spec, fitted, altered, args.batch_size)
            delta = altered_prediction - baseline
            if not np.all(np.isfinite(delta)):
                raise ValueError("Nonfinite filtered model outputs")
            results[f"{cutoff:g}"] = altered_prediction
            snapshots.append((cutoff, altered[:1]))
            records.append({
                "cutoff_GHz": float(cutoff),
                "mean_delta_ps": float(delta.mean()),
                "std_delta_ps": float(delta.std()),
                "rmse_delta_ps": float(np.sqrt(np.mean(delta**2))),
                "mae_delta_ps": float(np.mean(np.abs(delta))),
                "max_abs_delta_ps": float(np.max(np.abs(delta))),
                "filter_preclip_excess": preclip,
                "pearson_original_vs_filtered": (
                    float(np.corrcoef(baseline, altered_prediction)[0, 1])
                    if len(rows) >= 3 and baseline.std() > 0
                    and altered_prediction.std() > 0 else None),
            })
            print(f"LOWPASS | cutoff={cutoff:g} GHz | "
                  f"model_delta_RMSE={records[-1]['rmse_delta_ps']:.3f} ps "
                  f"| mean={records[-1]['mean_delta_ps']:+.3f} ps "
                  f"| max={records[-1]['max_abs_delta_ps']:.3f} ps", flush=True)
        np.savez_compressed(
            out / "predictions.npz", event_id=ids[rows],
            original_prediction_ps=baseline, saved_prediction_ps=original_saved,
            **{f"lowpass_{c:g}_prediction_ps": results[f"{c:g}"] for c in cutoffs})
        plot_one_event(out, inputs[:1], time_ps/1000.,
                       {"original": baseline[0], **{k: v[0] for k,v in results.items()}},
                       snapshots, args.filter_detector)
        summary = {
            "reference_run": str(run_dir), "model": manifest["model"],
            "role": "development", "events": int(len(rows)),
            "event_ids": ids[rows].tolist() if len(rows) < 100 else None,
            "filter_detector": args.filter_detector,
            "cutoffs_GHz": cutoffs, "butterworth_order": args.order,
            "filter": "sosfiltfilt zero-phase on full prepared input; clipped [0,1]",
            "original_replay_max_abs_error_ps": max_replay_delta,
            "frozen_transform": True, "frozen_model": True,
            "sampling_GHz": fs, "statistics": records,
            "note": ("Filtering can shift or distort the LED rising edge and may "
                     "induce noncausal pre-ringing. Sensitivity to low-pass is "
                     "not evidence that kernels explicitly encode oscillation phase.")
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        with (out / "cutoff_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        return summary
    finally:
        release_training_memory()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--event-id", type=int,
                   help="Specific prepared development event ID")
    p.add_argument("--max-events", type=int, default=32,
                   help="Random deterministic cohort when --event-id is omitted")
    p.add_argument("--filter-detector", choices=("1","2","both"), default="both")
    p.add_argument("--cutoff-GHz", type=float, nargs="+",
                   default=[5., 3., 2., 1.5, 1.],
                   help="Low-pass -3dB frequencies (GHz); above/below 2.5GHz")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
