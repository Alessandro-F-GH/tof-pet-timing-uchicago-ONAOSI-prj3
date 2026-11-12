"""Exploratory tail-phase correction to a saved low-pass Linear Ridge model.

Uses ORIGINAL, unfiltered samples from the ROOT input files, not legacy main
branch caches. Exact join: filtered prediction event_id == ROOT event_index,
independently for development and blind. No original-model refit or modification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.results import resolve_run_config
from waveform_analysis.data.root_io import iterate_energy_chunks
from waveform_analysis.data.storage import RunStore
from waveform_analysis.reporting.stats import metric_values, paired_model_bootstrap
from waveform_analysis.signal.pulses import decode_oriented, pulse_hits


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


def _phase_of_tail(waveform, *, dt_ns, time0_ns, frequency_ghz,
                   tail_start_ns, tail_end_ns, threshold_mv):
    """Absolute-acquisition-time Fourier coefficient from post-peak tail."""
    signal = np.asarray(waveform, dtype=np.float64)
    hits = pulse_hits(signal, threshold_mv, dt_ns * 1e-9)
    if not hits:
        raise ValueError("No pulse above the native energy trigger threshold")
    lead = int(hits[0].leading_index)
    # Same first-pulse convention as energy selection; determine the peak
    # in a bounded interval, not from the entire acquisition record.
    stop = min(signal.size, lead + max(2, int(np.ceil(10.0 / dt_ns))))
    peak = lead + int(np.argmax(signal[lead:stop]))
    first = peak + int(np.ceil(tail_start_ns / dt_ns - 1e-9))
    last = peak + int(np.floor(tail_end_ns / dt_ns + 1e-9)) + 1
    if first < 0 or last > signal.size or last - first < 16:
        raise ValueError("Post-peak tail interval does not fit native ROOT waveform")
    segment = signal[first:last]
    if not np.all(np.isfinite(segment)):
        raise ValueError("Nonfinite native samples in tail")
    n = segment.size
    local = np.arange(n, dtype=np.float64) * dt_ns
    # Remove linear tail drift; use tapered, fixed-frequency complex projection.
    slope, intercept = np.polyfit(local, segment, 1)
    detrended = segment - (slope * local + intercept)
    taper = np.hanning(n)
    complex_amplitude = 2 * np.dot(
        detrended * taper, np.exp(-2j * np.pi * frequency_ghz * local)
    ) / taper.sum()
    absolute_time = time0_ns + first * dt_ns
    return complex_amplitude * np.exp(
        -2j * np.pi * np.remainder(frequency_ghz * absolute_time, 1.0)
    )


def _phase_from_root(root_file, prediction_ids, config, role, args):
    """Stream raw ROOT, decode ONLY requested events, require exact ID coverage."""
    prediction_ids = _unique(prediction_ids, f"{role} predictions")
    results = np.empty((len(prediction_ids), 2), dtype=np.complex128)
    found = np.zeros(len(prediction_ids), dtype=bool)
    order = np.argsort(prediction_ids)
    wanted = prediction_ids[order]
    dataset = config[role]
    channels = dataset["channels"]
    energy_ch = tuple(map(int, channels["energy"]))
    polarity = tuple(map(int, channels.get("polarities", [1, 1])))
    thresholds = tuple(map(float, config["preprocessing"]["energy"]["trigger_threshold_mV"]))
    root_file = Path(root_file).resolve()

    scanned = 0
    for chunk in iterate_energy_chunks(
        root_file, energy_channels_one_based=energy_ch,
        timing_channels_one_based=None, step_size="128 MB", entry_stop=None,
    ):
        ids = _unique(chunk.event_index, f"{role} ROOT chunk")
        scanned += ids.size
        locations = np.searchsorted(wanted, ids)
        match = locations < wanted.size
        match[match] &= wanted[locations[match]] == ids[match]
        for local in np.flatnonzero(match):
            pos = int(order[locations[local]])
            if found[pos]:
                raise ValueError(f"{role}: duplicate ROOT event_index={prediction_ids[pos]}")
            for detector in range(2):
                signal = decode_oriented(
                    chunk.samples[detector][local],
                    chunk.vertical_gain_v_per_count[local, detector],
                    chunk.vertical_offset_v[local, detector],
                    polarity[detector],
                )
                results[pos, detector] = _phase_of_tail(
                    signal,
                    dt_ns=float(chunk.horizontal_interval_s[local, detector]) * 1e9,
                    time0_ns=float(chunk.horizontal_offset_s[local, detector]) * 1e9,
                    frequency_ghz=args.frequency_ghz,
                    tail_start_ns=args.tail_start_ns,
                    tail_end_ns=args.tail_end_ns,
                    threshold_mv=thresholds[detector],
                )
            found[pos] = True
    if not np.all(found):
        missing = prediction_ids[~found]
        raise ValueError(
            f"{role}: {missing.size}/{len(prediction_ids)} prediction events not in "
            f"{root_file}; first IDs={missing[:10].tolist()}"
        )
    if not np.all(np.isfinite(results)):
        raise ValueError(f"{role}: nonfinite Fourier coefficients")
    print(
        f"{role.upper()} ROOT JOIN | source={root_file} | scanned={scanned} "
        f"| requested={len(prediction_ids)} | matched={int(found.sum())} "
        f"| matched_by=event_index | order=prediction_event_id",
        flush=True,
    )
    return results


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

    dev_coeff = _phase_from_root(
        config["development"]["root_file"], development["event_id"],
        config, "development", args,
    )
    blind_coeff = _phase_from_root(
        config["blind"]["root_file"], blind["event_id"],
        config, "blind", args,
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
        "source_root_development": str(Path(config["development"]["root_file"]).resolve()),
        "source_root_blind": str(Path(config["blind"]["root_file"]).resolve()),
        "join_key": "prediction.event_id == ROOT.event_index",
        "frequency_ghz": args.frequency_ghz,
        "tail_window_ns_after_native_pulse_peak": [args.tail_start_ns, args.tail_end_ns],
        "phase_reference": "absolute_scope_acquisition_time",
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
