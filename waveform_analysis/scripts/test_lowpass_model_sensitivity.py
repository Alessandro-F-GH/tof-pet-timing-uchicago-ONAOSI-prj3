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



def temporal_filter_weights(time_ns, *, region="full",
                            baseline_end_ns=-1.0, tail_start_ns=12.0,
                            transition_ns=0.5):
    """Blend weights in [0,1], zero over the protected leading edge.

    Full: all samples filtered, to reproduce the original experiment.
    Baseline: samples before baseline_end_ns filtered, cosine fade to zero.
    Tail: samples after tail_start_ns filtered, cosine fade up to one.
    No direct multiplication of independently filtered snippets: filter
    the whole trace, blend only outside the protected interval.
    """
    t = np.asarray(time_ns, dtype=np.float64)
    if (t.ndim != 1 or t.size < 32 or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Expected finite monotonic prepared time axis")
    if region not in {"full", "baseline", "tail", "baseline_tail"}:
        raise ValueError("Invalid low-pass intervention region")
    if region == "full":
        return np.ones_like(t)
    if (not np.isfinite(baseline_end_ns) or not np.isfinite(tail_start_ns)
            or baseline_end_ns >= tail_start_ns or not np.isfinite(transition_ns)
            or transition_ns <= 0):
        raise ValueError("Baseline/tail boundaries need an ordered protected window and positive transition")
    if baseline_end_ns <= t[0] or tail_start_ns >= t[-1]:
        raise ValueError("Selected baseline and tail regions must fall inside the prepared time grid")
    if t[0] >= baseline_end_ns - transition_ns or t[-1] <= tail_start_ns + transition_ns:
        raise ValueError("Insufficient samples for the chosen fade transitions")
    weights = np.zeros_like(t)
    if region in {"baseline", "baseline_tail"}:
        # w=1 before (end-transition), w=0 from baseline_end onward.
        u = np.clip((baseline_end_ns - t) / transition_ns, 0., 1.)
        weights = np.maximum(weights, .5 - .5 * np.cos(np.pi * u))
    if region in {"tail", "baseline_tail"}:
        # w=0 up to tail_start, w=1 after (start+transition).
        u = np.clip((t - tail_start_ns) / transition_ns, 0., 1.)
        weights = np.maximum(weights, .5 - .5 * np.cos(np.pi * u))
    protected = (t >= baseline_end_ns) & (t <= tail_start_ns)
    if np.any(weights[protected] != 0.):
        raise AssertionError("Leading-edge protection mask unexpectedly nonzero")
    return weights

def lowpass_prepared(pairs, *, sample_rate_GHz, cutoff_GHz,
                     order=4, detectors=(0, 1), clip=True,
                     time_ns=None, region="full", baseline_end_ns=-1.,
                     tail_start_ns=12., transition_ns=.5):
    """Filter the full trace, then blend changes ONLY in selected regions.

    Filtering uses prepared affine amplitudes. Region baseline_tail preserves
    every sample between baseline_end_ns and tail_start_ns bit-for-bit.
    The filtering remains noncausal and modified samples are OOD inputs.
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
    if region == "full":
        weight = np.ones(x.shape[-1], dtype=np.float64)
    else:
        if time_ns is None:
            raise ValueError("time_ns required for selective filtering")
        if np.shape(time_ns) != (x.shape[-1],):
            raise ValueError("time_ns does not match waveform length")
        weight = temporal_filter_weights(
            time_ns, region=region, baseline_end_ns=baseline_end_ns,
            tail_start_ns=tail_start_ns, transition_ns=transition_ns)
    out = x.copy()
    preclip_max = 0.
    for detector in detectors:
        original = x[:, detector, :].astype(np.float64)
        filtered = sosfiltfilt(sos, original, axis=-1)
        blended = original + (filtered - original) * weight[None, :]
        # Do not change unselected time samples, including the protected edge.
        preclip_max = max(preclip_max, float(np.max(
            np.abs(blended - np.clip(blended, 0., 1.)))))
        if clip:
            blended = np.where(weight[None, :] > 0,
                               np.clip(blended, 0., 1.), original)
        out[:, detector, :] = blended.astype(np.float32)
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
    time_ns = time_ps / 1000.
    weights = temporal_filter_weights(
        time_ns, region=args.filter_region,
        baseline_end_ns=args.baseline_end_ns,
        tail_start_ns=args.tail_start_ns,
        transition_ns=args.transition_ns)
    protected_indices = (time_ns >= args.baseline_end_ns) & (time_ns <= args.tail_start_ns)
    if args.filter_region != "full" and not protected_indices.any():
        raise ValueError("No samples inside protected rising-edge interval")
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
        replay_delta = baseline - original_saved
        max_replay_delta = float(np.max(np.abs(replay_delta)))
        replay_rmse = float(np.sqrt(np.mean(replay_delta ** 2)))
        # Rocket/RDST floating-point transform replays can differ slightly
        # from the original batched inference. They need not be bit-exact.
        # All filter effects below are referenced to THIS replayed baseline,
        # never to the historical saved prediction.
        replay_tolerance_ps = 0.1
        if max_replay_delta > replay_tolerance_ps:
            raise ValueError(
                f"Frozen model replay differs from saved predictions by "
                f"{max_replay_delta:.6g} ps (RMSE {replay_rmse:.6g} ps), "
                f"exceeding {replay_tolerance_ps:g} ps. Check model provenance.")
        print(f"REPLAY | max_abs={max_replay_delta:.6f} ps "
              f"| RMSE={replay_rmse:.6f} ps "
              f"| tolerance={replay_tolerance_ps:.3f} ps", flush=True)
        records, snapshots = [], []
        results = {}
        for cutoff in cutoffs:
            altered, preclip = lowpass_prepared(
                inputs, sample_rate_GHz=fs, cutoff_GHz=cutoff,
                order=args.order, detectors=channels,
                time_ns=time_ns, region=args.filter_region,
                baseline_end_ns=args.baseline_end_ns,
                tail_start_ns=args.tail_start_ns,
                transition_ns=args.transition_ns)
            if args.filter_region != "full" and not np.array_equal(
                    altered[:, :, protected_indices], inputs[:, :, protected_indices]):
                raise AssertionError("Protected rising-edge samples changed after filtering")
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
                "protected_samples_exactly_unchanged": (
                    bool(np.array_equal(
                        altered[:, :, protected_indices],
                        inputs[:, :, protected_indices]))
                    if args.filter_region != "full" else None),
                "perturbed_input_rms_prepared_units": float(np.sqrt(
                    np.mean((altered.astype(np.float64) -
                             inputs.astype(np.float64)) ** 2))),
                "max_abs_input_delta_prepared_units": float(np.max(
                    np.abs(altered.astype(np.float64) -
                           inputs.astype(np.float64)))),
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
            "filter_region": args.filter_region,
            "protected_window_ns": [args.baseline_end_ns, args.tail_start_ns],
            "transition_ns": args.transition_ns,
            "protected_sample_count": int(protected_indices.sum()),
            "modified_sample_count": int((weights > 0).sum()),
            "filter": ("sosfiltfilt zero-phase applied to full prepared trace, "
                       "then cosine-blended only in selected regions; clipped "
                       "only on blended samples"),
            "original_replay_max_abs_error_ps": max_replay_delta,
            "original_replay_rmse_ps": replay_rmse,
            "replay_absolute_tolerance_ps": replay_tolerance_ps,
            "frozen_transform": True, "frozen_model": True,
            "sampling_GHz": fs, "statistics": records,
            "note": ("Selective filtering preserves the protected rising-edge "
                     "samples exactly, but the low-pass signal is computed "
                     "noncausally from the whole waveform; even a masked "
                     "intervention can be out of training distribution. "
                     "Sensitivity does not demonstrate oscillation-phase encoding.")
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
    p.add_argument("--filter-region",
                   choices=("full", "baseline", "tail", "baseline_tail"),
                   default="full",
                   help="Filter entire waveform (legacy) or only baseline/tail")
    p.add_argument("--baseline-end-ns", type=float, default=-1.0,
                   help="Protected rising edge starts here; baseline fades out before it")
    p.add_argument("--tail-start-ns", type=float, default=12.0,
                   help="Protected rising edge ends here; tail fades in after it")
    p.add_argument("--transition-ns", type=float, default=0.5,
                   help="Cosine transition width at filtered/unfiltered boundaries")
    p.add_argument("--order", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
