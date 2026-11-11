"""Rotate fitted 2.5-GHz oscillation phase in baseline/tail, replay frozen model.

This controlled input perturbation leaves the LED rising edge bit-identical.
It is diagnostic sensitivity, not a causal attribution or a claim that the
observed frequency carries predictive timing information.
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
from waveform_analysis.scripts.test_lowpass_model_sensitivity import temporal_filter_weights


def fit_local_oscillation(x, time_ns, frequency_GHz, fit_mask):
    """Least-squares sinusoid plus affine trend on unblended plateau samples.

    The trend is included as a nuisance in the fit but NEVER modified. Fit
    cos and sin coefficients are the physical oscillation component estimates.
    """
    t = np.asarray(time_ns, np.float64)
    y = np.asarray(x, np.float64)
    m = np.asarray(fit_mask, bool)
    if y.ndim != 2 or t.shape != (y.shape[1],) or m.shape != t.shape:
        raise ValueError("Invalid waveform/time/mask shapes")
    if m.sum() < 20 or (np.ptp(t[m]) * frequency_GHz < 2):
        raise ValueError("At least two cycles of the frequency required in fit window")
    theta = 2 * np.pi * frequency_GHz * t
    A = np.column_stack((np.ones(m.sum()), t[m] - np.mean(t[m]),
                         np.cos(theta[m]), np.sin(theta[m])))
    params, _, rank, _ = np.linalg.lstsq(A, y[:, m].T, rcond=None)
    if rank != 4:
        raise ValueError("Degenerate phase fit")
    cosine, sine = params[2], params[3]
    amplitude = np.hypot(cosine, sine)
    phase = np.arctan2(-sine, cosine)
    return amplitude, phase


def phase_rotate_prepared(pairs, time_ns, *, frequency_GHz=2.5,
                          phase_shift_deg=90., region="baseline_tail",
                          detectors=(0, 1), baseline_end_ns=-1.,
                          tail_start_ns=12., transition_ns=.5):
    """Replace ONLY the fitted narrowband component by a phase-rotated one.

    Estimate baseline and tail oscillations independently on the flat-weight
    portions of each region; preserve each fitted amplitude. Changes use
    cosine blending to zero before reaching the protected rising-edge span.
    The unmodified signal (phase shift=0) is returned exactly.
    """
    source = np.asarray(pairs, np.float32)
    t = np.asarray(time_ns, np.float64)
    if source.ndim != 3 or source.shape[1] != 2 or t.shape != (source.shape[2],):
        raise ValueError("Expected paired prepared waveforms and matching time axis")
    if region not in {"baseline", "tail", "baseline_tail"}:
        raise ValueError("Choose baseline, tail or baseline_tail")
    if not 0 < frequency_GHz < 0.5 / np.median(np.diff(t)):
        raise ValueError("Frequency outside Nyquist range")
    if not detectors or any(d not in (0, 1) for d in detectors):
        raise ValueError("Invalid selected detectors")
    angle = np.deg2rad(float(phase_shift_deg))
    result = source.copy()
    info = {}
    if np.isclose(np.mod(angle, 2*np.pi), 0, atol=1e-14):
        return result, info
    theta = 2 * np.pi * frequency_GHz * t
    for section in ("baseline", "tail"):
        if region != "baseline_tail" and region != section:
            continue
        weights = temporal_filter_weights(
            t, region=section, baseline_end_ns=baseline_end_ns,
            tail_start_ns=tail_start_ns, transition_ns=transition_ns)
        fit_mask = weights > (1 - 1e-12)
        for d in detectors:
            amp, phi = fit_local_oscillation(
                source[:, d, :], t, frequency_GHz, fit_mask)
            old = amp[:, None] * np.cos(theta[None, :] + phi[:, None])
            new = amp[:, None] * np.cos(theta[None, :] + phi[:, None] + angle)
            delta = weights[None, :] * (new - old)
            result[:, d, :] = (source[:, d, :].astype(np.float64) + delta).astype(np.float32)
            info[f"{section}_detector{d+1}_amplitude"] = amp
            info[f"{section}_detector{d+1}_phase_rad"] = phi
    protected = (t >= baseline_end_ns) & (t <= tail_start_ns)
    if not np.array_equal(result[:, :, protected], source[:, :, protected]):
        raise AssertionError("Protected rising-edge samples changed")
    return result, info


def save_plots(output, time_ns, original, examples, phases, prediction):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for d in range(2):
        axes[d].plot(time_ns, original[0, d], color="black",
                     lw=1.4, label="original")
        for angle, x in examples:
            axes[d].plot(time_ns, x[0, d], lw=.9, alpha=.7,
                         label=f"{angle:g} degrees")
        axes[d].set_ylabel(f"D{d+1} prepared amplitude")
        axes[d].grid(alpha=.2)
        axes[d].legend(ncol=4, fontsize=8)
    axes[2].plot(phases, prediction[0, :], "o-")
    zero_index = int(np.flatnonzero(phases == 0.)[0])
    axes[2].axhline(prediction[0, zero_index], ls="--", color="0.4")
    axes[2].set(xlabel="Rotation applied (degrees)",
                ylabel="Prediction (ps)", title="First selected event")
    axes[2].grid(alpha=.2)
    axes[1].set_xlabel("Time relative to LED (ns)")
    fig.savefig(output / "phase_rotation_event.png", dpi=170)
    fig.savefig(output / "phase_rotation_event.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    changes = prediction - prediction[:, [int(np.flatnonzero(phases == 0.)[0])]]
    ax.plot(phases, changes.T, lw=.5, alpha=.15)
    ax.plot(phases, changes.mean(axis=0), "o-", lw=2, label="Mean change")
    ax.plot(phases, np.sqrt(np.mean(changes**2, axis=0)), "o-",
            lw=2, label="RMSE change")
    ax.set(xlabel="Rotation applied (degrees)", ylabel="Prediction change (ps)",
           title="Phase-rotation sensitivity across events")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(output / "phase_rotation_summary.png", dpi=170)
    fig.savefig(output / "phase_rotation_summary.pdf")
    plt.close(fig)


def run(args):
    if args.cache_root and args.artifact_root:
        raise ValueError("Select one cache root")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    run_record = read_run(run_dir)
    config = resolve_run_config(run_dir)
    manifest = run_record["manifest"]
    tshap._apply_tshap_cache(
        config, cache_root=args.cache_root, artifact_root=args.artifact_root)
    dataset = PreparedCacheLookup().find("development", config, manifest)
    family = mode_family(config["mode"])
    waves = getattr(dataset, f"{family}_windows")
    time_ps, dt_ps = validate_grid(getattr(dataset, f"{family}_time_ps"))
    if waves is None or waves.shape != (dataset.n_events, 2, len(time_ps)):
        raise ValueError("Prepared waveform shape mismatch")
    time_ns = time_ps / 1000.
    ids = np.asarray(dataset.event_index, np.int64)
    if args.event_id is not None:
        rows = np.flatnonzero(ids == args.event_id)
        if len(rows) != 1:
            raise ValueError("Requested event not uniquely present")
    else:
        rng = np.random.default_rng(args.seed)
        rows = np.sort(rng.choice(dataset.n_events,
                                  min(args.max_events, dataset.n_events),
                                  replace=False))
    source = np.asarray(waves[rows], np.float32)
    store = RunStore(run_dir, artifact_root=config["preprocessing"].get("artifact_root"))
    saved_path = store.development_predictions_path
    if not saved_path.is_file():
        saved_path = run_dir / "artifacts" / "development_pred.npz"
    saved = _prediction_payload(saved_path, dataset, manifest)[rows]
    spec = get_model(manifest["model"])
    model_dir = tshap._resolve_tshap_model_directory(
        spec, store, run_dir, explicit_model_dir=True)
    fitted = load_fitted_model(
        spec, model_dir, run_record["best"]["parameters"], config)
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    channels = {"1": (0,), "2": (1,), "both": (0, 1)}[args.filter_detector]
    phases = np.array(sorted(set([0.] + list(map(float, args.phase_deg)))))
    preds = np.empty((len(rows), len(phases)), np.float64)
    examples = []
    amplitudes = {}
    try:
        original = tshap._predict_bounded(spec, fitted, source, args.batch_size)
        max_replay = float(np.max(np.abs(original - saved)))
        print(f"REPLAY | model={manifest['model']} | max_abs={max_replay:.5f} ps", flush=True)
        for j, angle in enumerate(phases):
            transformed, info = phase_rotate_prepared(
                source, time_ns, frequency_GHz=args.frequency_GHz,
                phase_shift_deg=angle, region=args.region, detectors=channels,
                baseline_end_ns=args.baseline_end_ns,
                tail_start_ns=args.tail_start_ns,
                transition_ns=args.transition_ns)
            if angle == 0:
                prediction = original.copy()
                if not np.array_equal(source, transformed):
                    raise AssertionError("Zero phase shift must preserve source exactly")
            else:
                prediction = tshap._predict_bounded(
                    spec, fitted, transformed, args.batch_size)
                for key, value in info.items():
                    amplitudes[key] = value
            preds[:, j] = prediction
            if angle in (-180., -90., 90., 180.):
                examples.append((angle, transformed[:1].copy()))
            diff = prediction - original
            print(f"PHASE | rotation={angle:+g} deg | RMSE={np.sqrt(np.mean(diff**2)):.3f} ps "
                  f"| mean={diff.mean():+.3f} ps | max={np.max(np.abs(diff)):.3f} ps",
                  flush=True)
        if np.any(~np.isfinite(preds)):
            raise ValueError("Nonfinite predictions")
        delta = preds - preds[:, [np.flatnonzero(phases == 0.)[0]]]
        metrics = []
        for j, angle in enumerate(phases):
            metrics.append({"phase_deg": float(angle),
                            "rmse_delta_ps": float(np.sqrt(np.mean(delta[:, j]**2))),
                            "mean_delta_ps": float(np.mean(delta[:, j])),
                            "mae_delta_ps": float(np.mean(np.abs(delta[:, j]))),
                            "max_abs_delta_ps": float(np.max(np.abs(delta[:, j])))})
        np.savez_compressed(
            out / "phase_sweep_predictions.npz",
            event_id=ids[rows], phase_deg=phases,
            prediction_ps=preds, original_saved_prediction_ps=saved,
            **amplitudes)
        with (out / "phase_sweep_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        summary = {
            "run": str(run_dir), "model": manifest["model"], "role": "development",
            "n_events": len(rows), "frequency_GHz": args.frequency_GHz,
            "region": args.region, "detector": args.filter_detector,
            "baseline_end_ns": args.baseline_end_ns,
            "tail_start_ns": args.tail_start_ns,
            "transition_ns": args.transition_ns,
            "phase_deg": phases.tolist(),
            "max_original_replay_mismatch_ps": max_replay,
            "estimator": "per-event least squares: constant + slope + cos + sin",
            "perturbation": "fixed fitted amplitude, phase rotated, cosine-tapered in selected regions",
            "protected_region": "unaltered prepared samples between baseline-end and tail-start",
            "caution": ("The intervention preserves only the estimated sinusoid's amplitude, "
                        "not the entire signal PSD or physical realizability. "
                        "Even a large response may reflect out-of-distribution sensitivity."),
            "metrics": metrics,
        }
        (out / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        save_plots(out, time_ns, source[:1], examples, phases, preds)
        return summary
    finally:
        release_training_memory()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--event-id", type=int)
    p.add_argument("--max-events", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--filter-detector", choices=("1", "2", "both"), default="both")
    p.add_argument("--frequency-GHz", type=float, default=2.5)
    p.add_argument("--region", choices=("baseline", "tail", "baseline_tail"),
                   default="baseline_tail")
    p.add_argument("--baseline-end-ns", type=float, default=-1.)
    p.add_argument("--tail-start-ns", type=float, default=12.)
    p.add_argument("--transition-ns", type=float, default=.5)
    p.add_argument("--phase-deg", type=float, nargs="+",
                   default=[-180, -135, -90, -45, 45, 90, 135, 180])
    return p


if __name__ == "__main__":
    run(parser().parse_args())
