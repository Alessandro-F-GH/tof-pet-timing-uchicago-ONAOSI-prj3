"""Exploratory add-on: 2.5 GHz tail phase over filtered Linear Ridge predictions.

Read-only: does not refit Ridge, alter source caches or modify run artifacts.
Input native caches MUST be unfiltered; matches original ROOT event_index against
filtered prediction event_id independently for development and blind.
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


def _unique(ids, role):
    ids = np.asarray(ids, np.int64).reshape(-1)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{role}: duplicate event IDs")
    return ids


def _align(native_ids, prediction_ids, *, role, require_all=True):
    """Return native-cache row for each prediction ID, never assume row order."""
    native = _unique(native_ids, f"{role} native")
    predicted = _unique(prediction_ids, f"{role} predictions")
    sorter = np.argsort(native)
    ordered = native[sorter]
    positions = np.searchsorted(ordered, predicted)
    matched = (positions < len(ordered))
    matched[matched] &= (ordered[positions[matched]] == predicted[matched])
    if require_all and not np.all(matched):
        raise ValueError(
            f"{role}: {np.count_nonzero(~matched)}/{len(predicted)} prediction events "
            "missing from unfiltered native waveform cache; refusing partial matching"
        )
    return sorter[positions[matched]], matched


def _native_cache(directory, *, expected_source, role):
    directory = Path(directory).expanduser().resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    source = Path(manifest["source"]).expanduser().resolve()
    if source != Path(expected_source).expanduser().resolve():
        raise ValueError(
            f"{role}: native cache ROOT {source} differs from prediction ROOT "
            f"{Path(expected_source).resolve()}"
        )
    if manifest.get("mode") != "energy_to_energy":
        raise ValueError(f"{role}: requires native energy_to_energy cache")
    # A native cache built under the lowpass configuration has different source
    # semantics. The provenance is supplied explicitly via the unfiltered run
    # and compared below; no heuristic on the alias name is trusted.
    paths = {
        name: directory / f"energy_{name}.npy"
        for name in ("windows_mV", "window_start_time_s", "sample_interval_s",
                     "rising_start")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{role}: missing native cache component {path}")
    native = {
        "event_id": np.load(directory / "event_index.npy", mmap_mode="r"),
        **{key: np.load(path, mmap_mode="r") for key, path in paths.items()},
        "manifest": manifest,
        "directory": directory,
    }
    n = len(native["event_id"])
    if (native["windows_mV"].shape[:2] != (n, 2)
            or any(native[name].shape != (n, 2)
                   for name in ("window_start_time_s", "sample_interval_s", "rising_start"))):
        raise ValueError(f"{role}: inconsistent native cache shapes")
    return native


def _validate_unfiltered(native, original_run_config, role):
    if original_run_config["mode"] != "energy_to_energy":
        raise ValueError("Unfiltered run must use energy_to_energy")
    if original_run_config["preprocessing"].get("lowpass", {}).get("enabled", False):
        raise ValueError("Reference run preprocessing has lowpass enabled: expected original waveforms")
    if Path(native["manifest"]["source"]).resolve() != Path(
        original_run_config[role]["root_file"]
    ).resolve():
        raise ValueError(f"{role}: original-run ROOT differs from native cache ROOT")
    # Native cache manifest stores selection provenance, not preprocessing
    # config. Verify through original prepared dataset's explicit native ref,
    # if supplied separately, rather than mistaking filenames for provenance.


def _tail_phase(native, rows, *, frequency_ghz, start_ns, end_ns):
    """Oscillation at fixed f in post-onset tail; phase on absolute scope time."""
    out = np.empty((len(rows), 2), dtype=np.complex128)
    waves = native["windows_mV"]
    starts = native["window_start_time_s"]
    intervals = native["sample_interval_s"]
    rising = native["rising_start"]
    if end_ns <= start_ns or start_ns < 0:
        raise ValueError("Require nonnegative tail interval with end > start")
    for i, row in enumerate(rows):
        for detector in range(2):
            dt_ns = float(intervals[row, detector]) * 1e9
            if not np.isfinite(dt_ns) or dt_ns <= 0:
                raise ValueError(f"Invalid sample interval at native row {row}")
            onset = int(rising[row, detector])
            first = onset + int(np.ceil(start_ns / dt_ns - 1e-9))
            last = onset + int(np.floor(end_ns / dt_ns + 1e-9)) + 1
            if first < 0 or last > waves.shape[2] or last - first < 16:
                raise ValueError(
                    f"Tail not contained in native waveform at event_id="
                    f"{int(native['event_id'][row])}, detector={detector + 1}; "
                    "adjust --tail-end-ns to fit the native materialized window"
                )
            segment = np.asarray(waves[row, detector, first:last], np.float64)
            if not np.all(np.isfinite(segment)):
                raise ValueError(f"Nonfinite native signal at row {row}")
            n = len(segment)
            local_t_ns = np.arange(n, dtype=np.float64) * dt_ns
            # Remove baseline and linear tail drift before narrowband projection.
            slope, intercept = np.polyfit(local_t_ns, segment, 1)
            detrended = segment - (slope * local_t_ns + intercept)
            taper = np.hanning(n)
            amplitude = (2 / taper.sum()) * np.dot(
                detrended * taper, np.exp(-2j * np.pi * frequency_ghz * local_t_ns)
            )
            # Absolute scope coordinate, not an LED-aligned input coordinate.
            start_ns_absolute = float(starts[row, detector]) * 1e9 + first * dt_ns
            phase_angle = np.remainder(frequency_ghz * start_ns_absolute, 1.0)
            out[i, detector] = amplitude * np.exp(-2j * np.pi * phase_angle)
    if not np.all(np.isfinite(out)):
        raise ValueError("Nonfinite phase coefficients")
    return out


def _features(coefficients, representation):
    abs_values = np.abs(coefficients)
    if np.any(abs_values <= 1e-12):
        raise ValueError("Near-zero oscillation amplitude, phase is undefined")
    unit = coefficients / abs_values
    if representation == "relative":
        # Relative channel phase preserves the calibrated time coordinate of
        # each detector; no absolute-phase shortcut is used.
        delta = unit[:, 0] * np.conj(unit[:, 1])
        return np.column_stack([np.real(delta), np.imag(delta)])
    if representation == "individual":
        return np.column_stack([
            np.real(unit[:, 0]), np.imag(unit[:, 0]),
            np.real(unit[:, 1]), np.imag(unit[:, 1]),
        ])
    raise ValueError(f"Unknown phase representation {representation}")


def _predictions(store, role):
    path = store.development_predictions_path if role == "development" else store.predictions_path
    if not path.is_file():
        raise FileNotFoundError(
            f"{role}: filtered model prediction cache missing: {path}. "
            "Development predictions are generated only by completed runs."
        )
    with np.load(path, allow_pickle=False) as file:
        data = {k: np.asarray(file[k]) for k in file.files}
    for field in ("event_id", "prediction_ps", "corrected_ps", "led_residual_ps",
                  "analysis_protocol_identity", "final_fit_fingerprint"):
        if field not in data:
            raise ValueError(f"{role}: prediction payload lacks {field}")
    n = len(data["event_id"])
    if any(np.asarray(data[k]).shape != (n,) for k in
           ("prediction_ps", "corrected_ps", "led_residual_ps")):
        raise ValueError(f"{role}: prediction payload shapes differ")
    if not np.allclose(
        data["led_residual_ps"] - data["prediction_ps"],
        data["corrected_ps"], rtol=0, atol=1e-6,
    ):
        raise ValueError(f"{role}: predictions and corrected target inconsistent")
    return data


def _inputs(role, cache_path, prediction, original_config, filtered_config, *,
            frequency_ghz, start_ns, end_ns):
    native = _native_cache(
        cache_path, expected_source=filtered_config[role]["root_file"], role=role,
    )
    _validate_unfiltered(native, original_config, role)
    rows, mask = _align(native["event_id"], prediction["event_id"], role=role)
    if not np.all(mask):
        raise AssertionError("Unmatched native events")
    coeff = _tail_phase(native, rows, frequency_ghz=frequency_ghz,
                        start_ns=start_ns, end_ns=end_ns)
    print(
        f"{role.upper()} JOIN | predictions={len(prediction['event_id'])} "
        f"| native={len(native['event_id'])} | matched={len(rows)} "
        f"| exact_source={native['manifest']['source']} | native_cache={native['directory']}",
        flush=True,
    )
    return coeff


def run(args):
    filtered = resolve_run_config(args.filtered_run)
    original = resolve_run_config(args.original_run)
    if filtered["model"]["name"] != "direct_linear_ridge":
        raise ValueError("Filtered run must be direct_linear_ridge")
    if filtered["mode"] != "energy_to_energy":
        raise ValueError("Only energy_to_energy is supported")
    if not filtered["preprocessing"].get("lowpass", {}).get("enabled", False):
        raise ValueError("Filtered Ridge run does not enable lowpass")
    if original["preprocessing"].get("lowpass", {}).get("enabled", False):
        raise ValueError("Original run unexpectedly applies lowpass")
    for role in ("development", "blind"):
        if Path(filtered[role]["root_file"]).resolve() != Path(
            original[role]["root_file"]
        ).resolve():
            raise ValueError(f"{role}: original and filtered model use different ROOT files")

    store = RunStore(args.filtered_run, artifact_root=filtered["preprocessing"]["artifact_root"])
    dev = _predictions(store, "development")
    blind = _predictions(store, "blind")
    final_fit = str(np.asarray(dev["final_fit_fingerprint"]).item())
    if final_fit != str(np.asarray(blind["final_fit_fingerprint"]).item()):
        raise ValueError("Development and blind predictions refer to different final fits")
    if not np.all(np.isfinite(dev["corrected_ps"])) or not np.all(np.isfinite(blind["corrected_ps"])):
        raise ValueError("Nonfinite filtered Ridge residuals")
    dev_coeff = _inputs(
        "development", args.native_development, dev, original, filtered,
        frequency_ghz=args.frequency_ghz, start_ns=args.tail_start_ns,
        end_ns=args.tail_end_ns,
    )
    blind_coeff = _inputs(
        "blind", args.native_blind, blind, original, filtered,
        frequency_ghz=args.frequency_ghz, start_ns=args.tail_start_ns,
        end_ns=args.tail_end_ns,
    )
    dev_x = _features(dev_coeff, args.representation)
    blind_x = _features(blind_coeff, args.representation)

    # Development predictions are IN-SAMPLE final-Ridge predictions.
    # Thus the development fit is exploratory; blind evaluation is also
    # exploratory if previous experiments have already inspected that blind.
    correction = Ridge(alpha=args.alpha, fit_intercept=True).fit(
        dev_x, dev["corrected_ps"]
    )
    added = np.asarray(correction.predict(blind_x), dtype=np.float64)
    base_residual = blind["corrected_ps"]
    corrected = base_residual - added
    params = filtered["fit"]
    base = metric_values(base_residual, params, seed=172)
    augmented = metric_values(corrected, params, seed=172)
    bootstrap = {
        m: paired_model_bootstrap(
            blind["event_id"], base_residual, blind["event_id"], corrected,
            params, metric=m, n_resamples=args.bootstrap,
            seed=333 + i,
        )
        for i, m in enumerate(("ctr", "rmse"))
    }
    result = {
        "note": "Exploratory: phase correction trained on in-sample development "
                "predictions of the final filtered Ridge model; blind may have "
                "been inspected in prior experiments.",
        "filtered_run": str(Path(args.filtered_run).resolve()),
        "unfiltered_run": str(Path(args.original_run).resolve()),
        "native_development": str(Path(args.native_development).resolve()),
        "native_blind": str(Path(args.native_blind).resolve()),
        "frequency_ghz": args.frequency_ghz,
        "tail_relative_to_native_rising_start_ns": [
            args.tail_start_ns, args.tail_end_ns,
        ],
        "phase_reference": "native_scope_absolute_time",
        "feature_representation": args.representation,
        "alpha": args.alpha,
        "coefficients": correction.coef_.tolist(),
        "intercept_ps": float(correction.intercept_),
        "training_events": len(dev["event_id"]),
        "blind_events": len(blind["event_id"]),
        "baseline": base,
        "phase_added": augmented,
        "paired_bootstrap_baseline_minus_phase": bootstrap,
    }
    out = Path(args.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", result)
    np.savez_compressed(
        out / "blind_predictions.npz",
        event_id=np.asarray(blind["event_id"], dtype=np.int64),
        filtered_ridge_prediction_ps=blind["prediction_ps"],
        phase_correction_ps=added,
        corrected_residual_ps=corrected,
        baseline_corrected_residual_ps=base_residual,
        phase_ch1_rad=np.angle(blind_coeff[:, 0]),
        phase_ch2_rad=np.angle(blind_coeff[:, 1]),
        amplitude_ch1=np.abs(blind_coeff[:, 0]),
        amplitude_ch2=np.abs(blind_coeff[:, 1]),
    )
    print(
        f"BLIND | N={len(blind['event_id'])} | CTR "
        f"{base['ctr_ps']:.3f} -> {augmented['ctr_ps']:.3f} ps "
        f"| RMSE {base['rmse_ps']:.3f} -> {augmented['rmse_ps']:.3f} ps",
        flush=True,
    )
    for key, value in bootstrap.items():
        print(
            f"PAIRED {key.upper()} | baseline-phase="
            f"{value['difference_ps']:.3f} ± {value['bootstrap_std_ps']:.3f} ps",
            flush=True,
        )
    print(f"OUTPUT | {out}", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filtered-run", type=Path, required=True,
                   help="Completed lowpass direct_linear_ridge run directory")
    p.add_argument("--original-run", type=Path, required=True,
                   help="Original unfiltered energy run, for ROOT/protocol provenance")
    p.add_argument("--native-development", type=Path, required=True,
                   help="Original unfiltered native-waveforms artifact directory")
    p.add_argument("--native-blind", type=Path, required=True,
                   help="Original unfiltered native-waveforms artifact directory")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--frequency-ghz", type=float, default=2.5)
    p.add_argument("--tail-start-ns", type=float, default=10.0)
    p.add_argument("--tail-end-ns", type=float, default=40.0)
    p.add_argument("--representation", choices=("relative", "individual"),
                   default="relative")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--bootstrap", type=int, default=100)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
