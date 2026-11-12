"""Exploratory tail-phase correction to a saved low-pass Linear Ridge model.

Uses ORIGINAL, unfiltered *prepared ML wide windows* from main-branch XAI
cache. Exact join: filtered prediction event_id == cache event_index,
independently for development and blind. No model refit or modification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.reporting.stats import metric_values, paired_model_bootstrap


def _unique(ids, label):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if ids.size != np.unique(ids).size:
        raise ValueError(f"{label}: duplicate event_index/event_id")
    return ids


def _align(root_ids, prediction_ids, *, role):
    """Return the ROOT row of every prediction, preserving prediction order."""
    source = _unique(root_ids, f"{role} ROOT")
    requested = _unique(prediction_ids, f"{role} predictions")
    permutation = np.argsort(source)
    sorted_ids = source[permutation]
    locations = np.searchsorted(sorted_ids, requested)
    present = locations < sorted_ids.size
    present[present] &= sorted_ids[locations[present]] == requested[present]
    if not np.all(present):
        missing = requested[~present]
        raise ValueError(
            f"{role}: {missing.size} prediction IDs missing from ROOT, "
            f"first missing IDs={missing[:10].tolist()}"
        )
    return permutation[locations]


def _cached_wide(root, role, prediction_ids, expected_root, args):
    """Match saved wide, UNFILTERED ML inputs with low-pass Ridge predictions.

    The cache is original main-branch prepared-dataset format. Time is relative
    to each detector's own ML/LED alignment (energy_time_ps); never use a native
    ROOT acquisition clock or samples outside the model's input window.
    """
    root = Path(root).expanduser().resolve()
    candidates = []
    for directory in (root / "artifact_store" / "products" / "prepared-datasets").glob("*"):
        manifest_path = directory / "manifest.json"
        if not (directory / "energy_windows.npy").is_file() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source = Path(str(manifest.get("dataset_source", "")).replace("\\\\", "/")).name
        if (manifest.get("dataset_role") == role
                and manifest.get("mode") == "energy_to_energy"
                and manifest.get("window_ns") == {"start": -2.0, "end": 50.0}
                and source == Path(expected_root).name
                and int(manifest.get("subsampling", -1)) == 1):
            candidates.append((directory, manifest))
    if len(candidates) != 1:
        raise ValueError(
            f"{role}: expected exactly one wide unfiltered dataset from "
            f"{Path(expected_root).name} in {root}; found {len(candidates)}. "
            "Do not choose an ambiguous cache."
        )
    directory, manifest = candidates[0]
    ids = np.load(directory / "event_index.npy", mmap_mode="r")
    rows = _align(ids, prediction_ids, role=role)
    wave = np.load(directory / "energy_windows.npy", mmap_mode="r")
    with np.load(directory / "energy_transform.npz") as tr:
        minimum = np.asarray(tr["minimum"], np.float64)
        maximum = np.asarray(tr["maximum"], np.float64)
    time_ns = np.asarray(
        np.load(directory / "energy_time_ps.npy", mmap_mode="r"), np.float64
    ) * 1e-3
    if wave.ndim != 3 or wave.shape[1] != 2 or wave.shape[0] != len(ids):
        raise ValueError(f"{role}: malformed cached energy_windows array")
    if wave.shape[-1] != len(time_ns):
        raise ValueError(f"{role}: waveform and time axis lengths differ")
    if np.any(np.diff(time_ns) <= 0):
        raise ValueError(f"{role}: non-increasing ML relative time axis")
    if np.max(np.abs(np.diff(time_ns) - np.median(np.diff(time_ns)))) > 1e-7:
        raise ValueError(f"{role}: nonuniform ML sampling")
    selection = np.flatnonzero(
        (time_ns >= args.tail_start_ns - 1e-9)
        & (time_ns <= args.tail_end_ns + 1e-9)
    )
    if (selection.size < 16 or selection[0] == 0
            or selection[-1] >= wave.shape[2]):
        raise ValueError(
            f"{role}: requested tail [{args.tail_start_ns}, {args.tail_end_ns}] ns "
            "must lie strictly inside wide ML window"
        )
    if minimum.shape != (2, 1) or maximum.shape != (2, 1):
        raise ValueError(f"{role}: unsupported cache normalization shape")
    if np.any(maximum <= minimum):
        raise ValueError(f"{role}: invalid normalization limits")
    print(
        f"{role.upper()} CACHE | {directory} | source={manifest['dataset_source']} "
        f"| cache_n={len(ids)} pred_n={len(prediction_ids)} matched={len(rows)} "
        f"| ML_time_ns=[{time_ns[0]:.4f},{time_ns[-1]:.4f}] "
        f"| analysis_ns=[{time_ns[selection[0]]:.4f},{time_ns[selection[-1]]:.4f}]",
        flush=True,
    )
    # Memmap-backed cache: process batches; output follows prediction event_id order.
    result = np.empty((len(rows), 2), dtype=np.complex128)
    tail_times = time_ns[selection]
    t = tail_times - tail_times.mean()
    taper = np.hanning(t.size)
    reference = np.exp(-2j * np.pi * args.frequency_ghz * tail_times)
    # Linearly detrend in the time domain, then apply the Hann Fourier kernel.
    osc_kernel = taper * reference
    scale = taper.sum()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        x = np.asarray(wave[rows[start:end]], np.float64)[:, :, selection]
        x = minimum[None, :, :] + x * (maximum - minimum)[None, :, :]
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{role}: nonfinite original ML waveforms")
        mean = x.mean(axis=-1, keepdims=True)
        slope = np.einsum("bcn,n->bc", x, t) / np.dot(t, t)
        detrended = x - mean - slope[:, :, None] * t[None, None, :]
        result[start:end] = (
            2 / scale * np.einsum("bcn,n->bc", detrended, osc_kernel)
        )
    return result, str(directory)


def _features(coefficients, representation):
    amplitude = np.abs(coefficients)
    if not np.all(np.isfinite(amplitude)) or np.any(amplitude <= 1e-12):
        raise ValueError("Oscillation amplitude too small to define phase")
    unit = coefficients / amplitude
    if representation == "relative":
        delta = unit[:, 0] * np.conj(unit[:, 1])
        return np.column_stack((delta.real, delta.imag))
    if representation == "individual":
        return np.column_stack((
            unit[:, 0].real, unit[:, 0].imag,
            unit[:, 1].real, unit[:, 1].imag,
        ))
    raise ValueError(f"Unknown representation: {representation}")


def _predictions(store, role):
    path = (
        store.development_predictions_path
        if role == "development" else store.predictions_path
    )
    if not path.is_file():
        raise FileNotFoundError(f"{role}: missing saved Ridge predictions at {path}")
    with np.load(path, allow_pickle=False) as payload:
        result = {name: np.asarray(payload[name]) for name in payload.files}
    for field in ("event_id", "prediction_ps", "corrected_ps",
                  "led_residual_ps", "final_fit_fingerprint"):
        if field not in result:
            raise ValueError(f"{role}: missing prediction field {field}")
    n = len(_unique(result["event_id"], f"{role} predictions"))
    for field in ("prediction_ps", "corrected_ps", "led_residual_ps"):
        if result[field].shape != (n,) or not np.all(np.isfinite(result[field])):
            raise ValueError(f"{role}: invalid prediction array {field}")
    if not np.allclose(
        result["led_residual_ps"] - result["prediction_ps"],
        result["corrected_ps"], rtol=0, atol=1e-6,
    ):
        raise ValueError(f"{role}: inconsistent LED, Ridge and residual values")
    return result


def run(args):
    config = resolve_run_config(args.filtered_run)
    if config["model"]["name"] != "direct_linear_ridge" or config["mode"] != "energy_to_energy":
        raise ValueError("Input run must be energy_to_energy/direct_linear_ridge")
    if not config["preprocessing"].get("lowpass", {}).get("enabled", False):
        raise ValueError("Input Ridge run must use filtered waveforms")

    store = RunStore(
        args.filtered_run, artifact_root=config["preprocessing"]["artifact_root"]
    )
    development = _predictions(store, "development")
    blind = _predictions(store, "blind")
    if str(np.asarray(development["final_fit_fingerprint"]).item()) != str(
        np.asarray(blind["final_fit_fingerprint"]).item()
    ):
        raise ValueError("Development and blind predictions originate from different Ridge fits")

    dev_coeff, dev_cache = _cached_wide(
        args.cache_root, "development", development["event_id"],
        config["development"]["root_file"], args,
    )
    blind_coeff, blind_cache = _cached_wide(
        args.cache_root, "blind", blind["event_id"],
        config["blind"]["root_file"], args,
    )
    dev_x = _features(dev_coeff, args.representation)
    blind_x = _features(blind_coeff, args.representation)

    # These development predictions are in-sample predictions of final Ridge,
    # not CV OOF predictions: the test is exploratory, not an unbiased estimate.
    phase_model = Ridge(alpha=args.alpha, fit_intercept=True).fit(
        dev_x, development["corrected_ps"]
    )
    phase_delta_ps = np.asarray(phase_model.predict(blind_x), np.float64)
    baseline = blind["corrected_ps"]
    augmented = baseline - phase_delta_ps
    fit_cfg = config["fit"]
    baseline_metrics = metric_values(baseline, fit_cfg, seed=172)
    phase_metrics = metric_values(augmented, fit_cfg, seed=172)
    uncertainty = {
        metric: paired_model_bootstrap(
            blind["event_id"], baseline, blind["event_id"], augmented,
            fit_cfg, metric=metric, n_resamples=args.bootstrap,
            seed=333 + index,
        )
        for index, metric in enumerate(("ctr", "rmse"))
    }
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", {
        "exploratory_warning": "Phase fitted on in-sample development Ridge residuals; "
                               "previously inspected blind is not an untouched holdout.",
        "filtered_run": str(Path(args.filtered_run).resolve()),
        "original_ml_cache_development": dev_cache,
        "original_ml_cache_blind": blind_cache,
        "join_key": "prediction.event_id == cached_original_ML.event_index",
        "frequency_ghz": args.frequency_ghz,
        "tail_window_ns_relative_to_ML_anchor": [args.tail_start_ns, args.tail_end_ns],
        "phase_reference": "per_detector_ML_LED_anchor_t_equals_zero",
        "representation": args.representation,
        "alpha": args.alpha,
        "coefficients": phase_model.coef_.tolist(),
        "intercept_ps": float(phase_model.intercept_),
        "n_development": len(development["event_id"]),
        "n_blind": len(blind["event_id"]),
        "baseline": baseline_metrics,
        "phase_added": phase_metrics,
        "paired_bootstrap_baseline_minus_phase": uncertainty,
    })
    np.savez_compressed(
        out / "blind_predictions.npz",
        event_id=np.asarray(blind["event_id"], np.int64),
        filtered_ridge_prediction_ps=blind["prediction_ps"],
        phase_correction_ps=phase_delta_ps,
        baseline_corrected_residual_ps=baseline,
        phase_corrected_residual_ps=augmented,
        phase_ch1_rad=np.angle(blind_coeff[:, 0]),
        phase_ch2_rad=np.angle(blind_coeff[:, 1]),
        amplitude_ch1=np.abs(blind_coeff[:, 0]),
        amplitude_ch2=np.abs(blind_coeff[:, 1]),
    )
    print(
        f"BLIND | N={len(blind['event_id'])} | "
        f"CTR={baseline_metrics['ctr_ps']:.3f} -> {phase_metrics['ctr_ps']:.3f} ps "
        f"| RMSE={baseline_metrics['rmse_ps']:.3f} -> {phase_metrics['rmse_ps']:.3f} ps",
        flush=True,
    )
    for metric, item in uncertainty.items():
        print(
            f"PAIRED {metric.upper()} | baseline - phase = "
            f"{item['difference_ps']:.3f} +/- {item['bootstrap_std_ps']:.3f} ps",
            flush=True,
        )
    print(f"OUTPUT | {out}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-root", type=Path, default=Path.home() / "UC_cache")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--filtered-run", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--frequency-ghz", type=float, default=2.5)
    p.add_argument("--tail-start-ns", type=float, default=5.0)
    p.add_argument("--tail-end-ns", type=float, default=30.0)
    p.add_argument("--representation", choices=("relative", "individual"), default="relative")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=100)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
