"""Evaluate a digital CFD correction on existing LED-aligned ML prepared cache.

No ROOT reads, new absolute timestamps, ML inference or ML refit.
The local LED and CFD crossings are linearly interpolated on the *same*
prepared input. Development-only CFD fraction selection; Blind is evaluated
once with frozen fraction and an optional development-only constant offset.

Saved development/blind NPZs are aligned to the prepared event_id and include
prediction_ps, corrected_ps, led_residual_ps and validity masks. They can be
compared to frozen ML predictions using event IDs, never array position.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, model_target, model_sample_mask
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.engine.cfd_led_correction import prepared_cfd_corrections
from waveform_analysis.reporting.stats import metric_values, blind_event_bootstrap
from waveform_analysis.scripts.plot_conditional_target_heatmaps import baseline_indices


def _json(path, object_):
    path.write_text(json.dumps(object_, indent=2, allow_nan=False), encoding="utf-8")


def _crossings(dataset, mode, fractions, baseline_window_ns, chunk_size):
    family = mode_family(mode)
    waves = getattr(dataset, f"{family}_windows")
    times = np.asarray(getattr(dataset, f"{family}_time_ps"), dtype=np.float64)
    transform = getattr(dataset, f"{family}_transform")
    if waves is None or transform is None:
        raise ValueError("Prepared waveform or saved physical transform missing")
    if waves.shape != (dataset.n_events, 2, times.size):
        raise ValueError("Invalid prepared waveform dimensions")
    if mode != "energy_to_energy":
        # Saturation can distort peak-based CFD on the timing channels;
        # explicitly fail rather than silently misinterpret a plateau.
        raise ValueError("CFD benchmark currently supports energy_to_energy only")
    mask = model_sample_mask(dataset, mode)
    if not mask.all():
        raise ValueError("CFD needs full rising-edge prepared grid")
    inds = baseline_indices(times, *baseline_window_ns)
    threshold = float(dataset.manifest["fixed_led_threshold_mV"])
    scale = np.asarray(transform.maximum - transform.minimum,
                       dtype=np.float64).reshape(2, -1)
    if scale.shape != (2, 1):
        raise ValueError("Unexpected physical amplitude scaling shape")
    correction, led_crossing, peak_times = prepared_cfd_corrections(
        waves, times, fractions=fractions, baseline_indices=inds,
        led_threshold_mV=threshold, physical_scale_mV=scale[:, 0],
        chunk_size=chunk_size)
    return correction, led_crossing, peak_times


def _score(target, correction, fit, seed):
    good = np.isfinite(correction) & np.isfinite(target)
    if good.sum() < 5:
        raise ValueError("Fewer than five valid CFD events")
    # Center the absolute residual using development only, like a conventional
    # per-method global timing calibration. This is not waveform supervision.
    offset = float(np.mean(target[good] - correction[good]))
    corrected = target[good] - correction[good] - offset
    m = metric_values(corrected, fit, seed=seed)
    return m, offset, int(good.sum())


def _write_predictions(path, dataset, mode, corrections, fraction, offset,
                       *, role, model_dir):
    target = np.asarray(model_target(dataset, mode), dtype=np.float64)
    valid = np.isfinite(corrections) & np.isfinite(target)
    prediction = np.asarray(corrections, np.float64) + float(offset)
    prediction[~valid] = np.nan
    corrected = target - prediction
    np.savez_compressed(
        path,
        event_id=np.asarray(dataset.event_index, dtype=np.int64),
        prediction_ps=prediction,
        corrected_ps=corrected,
        led_residual_ps=target,
        valid=valid,
        fraction=np.asarray(fraction),
        development_offset_ps=np.asarray(offset),
        dataset_role=np.asarray(role),
        method=np.asarray("prepared_digital_cfd_led_correction"),
        reference_model_run=np.asarray(str(model_dir)),
        analysis_protocol_identity=np.asarray(
            dataset.manifest["analysis_protocol_identity"]),
        event_population_identity=np.asarray(
            dataset.manifest["event_population_identity"]),
        control_mode_fingerprint=np.asarray(
            dataset.manifest["control_mode_fingerprint"]),
        fixed_led_threshold_mV=np.asarray(
            dataset.manifest["fixed_led_threshold_mV"]),
        led_control_mean_ps=np.asarray(dataset.manifest["led_control_mean_ps"]),
        true_tof_ps=np.asarray(dataset.manifest["true_tof_ps"]),
        target_definition=np.asarray(dataset.manifest["target_definition"]),
    )
    return valid, corrected, target


def run(args):
    if args.cache_root and args.artifact_root:
        raise ValueError("Choose --cache-root OR --artifact-root")
    fractions = np.asarray(args.fractions, dtype=np.float64)
    if not np.all(np.isfinite(fractions)) or np.unique(fractions).size != len(fractions):
        raise ValueError("Fractions must be unique finite values")
    if args.chunk_size < 1 or args.bootstrap < 1:
        raise ValueError("chunk-size/bootstrap must be positive")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    out = args.output_root.expanduser().resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError("CFD output must be outside the ML model run")
    if not (out / "development_predictions.npz").exists() and out.exists() and any(out.iterdir()):
        raise ValueError("Output directory exists and is not a CFD benchmark; choose a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    run_data = read_run(run_dir)
    config = resolve_run_config(run_dir)
    tshap._apply_tshap_cache(config, cache_root=args.cache_root,
                            artifact_root=args.artifact_root)
    manifest = run_data["manifest"]
    lookup = PreparedCacheLookup()
    development = lookup.find("development", config, manifest)
    if config["mode"] != "energy_to_energy":
        raise ValueError("Only energy_to_energy is supported (peak saturation for timing)")
    fit = dict(config["fit"])
    target = np.asarray(model_target(development, config["mode"]), dtype=np.float64)
    baseline = tuple(float(z) for z in args.baseline_window_ns)
    corrections, led_local, peak_times = _crossings(
        development, config["mode"], fractions, baseline, args.chunk_size)
    common = np.isfinite(corrections).all(axis=1)
    if common.sum() < 5:
        raise ValueError("No sufficient common CFD population across scanned fractions")
    if common.mean() < args.minimum_coverage:
        raise ValueError(
            f"Common valid CFD coverage {common.mean():.1%} is below "
            f"minimum {args.minimum_coverage:.1%}. Change scan fractions; "
            "do not use different populations to rank candidates.")
    candidates = []
    for j, fraction in enumerate(fractions):
        # Rank fractions using EXACTLY the same development event population.
        measured, offset, n = _score(
            target[common], corrections[common, j], fit,
            semantic_seed(args.seed, "cfd", float(fraction)))
        candidates.append({
            "fraction": float(fraction), "ctr_ps": measured["ctr_ps"],
            "rmse_ps": measured["rmse_ps"], "n_common": n,
            "coverage_fraction_specific": float(np.isfinite(corrections[:, j]).mean()),
        })
    selected = min(candidates, key=lambda d: (d[args.metric + "_ps"], d["fraction"]))
    chosen = float(selected["fraction"])
    selected_index = int(np.flatnonzero(np.isclose(fractions, chosen, rtol=0, atol=1e-12))[0])
    correction_dev = corrections[:, selected_index]
    dev_metric, offset, _ = _score(
        target, correction_dev, fit, semantic_seed(args.seed, "cfd", "dev"))
    valid_dev, residual_dev, led_dev = _write_predictions(
        out / "development_predictions.npz", development, config["mode"],
        correction_dev, chosen, offset, role="development", model_dir=run_dir)
    del corrections, led_local, peak_times
    print(f"SELECTED | fraction={chosen:g} | common coverage={common.mean():.2%} | "
          f"development CTR={dev_metric['ctr_ps']:.3f} ps | "
          f"development RMSE={dev_metric['rmse_ps']:.3f} ps", flush=True)
    # Only now open the Blind prepared cache; no selection based on Blind.
    blind = lookup.find("blind", config, manifest)
    for key in ("mode", "control_mode_fingerprint", "fixed_led_threshold_mV",
                "led_control_mean_ps", "target_definition"):
        if blind.manifest[key] != development.manifest[key]:
            raise ValueError(f"Development/Blind calibration mismatch: {key}")
    if not np.array_equal(
            getattr(development, f"{mode_family(config['mode'])}_time_ps"),
            getattr(blind, f"{mode_family(config['mode'])}_time_ps")):
        raise ValueError("Development/Blind prepared time grids differ")
    correction_blind, _, _ = _crossings(
        blind, config["mode"], [chosen], baseline, args.chunk_size)
    valid_blind, residual_blind, led_blind = _write_predictions(
        out / "blind_predictions.npz", blind, config["mode"],
        correction_blind[:, 0], chosen, offset, role="blind", model_dir=run_dir)
    if valid_blind.sum() < 5:
        raise ValueError("Fewer than five valid Blind CFD events")
    bootstrap, draws = blind_event_bootstrap(
        residual_blind[valid_blind], led_blind[valid_blind], fit,
        n_resamples=args.bootstrap, seed=semantic_seed(args.seed, "cfd", "blind"))
    np.savez_compressed(out / "blind_bootstrap.npz", **draws)
    summary = {
        "method": "prepared_digital_cfd_led_correction",
        "reference_model_dir": str(run_dir),
        "mode": config["mode"],
        "development_cache": str(development.directory),
        "blind_cache": str(blind.directory),
        "fraction_selection": "development_only_common_valid_population",
        "metric": args.metric,
        "fraction_scan": candidates,
        "fraction": chosen,
        "development_centering_offset_ps": offset,
        "centering": "single development-only mean corrected residual; no waveform labels used to define fraction crossing",
        "baseline_window_ns": list(baseline),
        "development": {
            "n": development.n_events,
            "n_valid": int(valid_dev.sum()),
            "coverage": float(valid_dev.mean()),
            "metrics_on_selected_development": dev_metric,
        },
        "blind": {
            "n": blind.n_events, "n_valid": int(valid_blind.sum()),
            "coverage": float(valid_blind.mean()),
            "metrics": bootstrap,
        },
        "saved_predictions": {
            "development": "development_predictions.npz",
            "blind": "blind_predictions.npz",
        },
        "sign_convention": ("prediction=(LED1-CFD1)-(LED2-CFD2) "
                            "+development_offset; corrected=LED_target-prediction"),
        "invalid_events": "NaN predictions with explicit valid mask; never imputed",
        "caveat": ("Peak and crossings derive from clipped LED-aligned prepared input; "
                   "not guaranteed identical to native-waveform CFD."),
    }
    _json(out / "summary.json", summary)
    print(f"BLIND | fraction={chosen:g} | valid={valid_blind.sum()}/{blind.n_events} "
          f"({valid_blind.mean():.2%}) | CTR={bootstrap['ctr_ps']:.3f} ps | "
          f"RMSE={bootstrap['rmse_ps']:.3f} ps | "
          f"LED CTR={bootstrap['led_ctr_ps']:.3f} ps", flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--fractions", nargs="+", type=float,
                   default=[round(v, 2) for v in np.arange(.05, .51, .05)])
    p.add_argument("--metric", choices=("ctr", "rmse"), default="ctr")
    p.add_argument("--baseline-window-ns", nargs=2, type=float, default=(-1.9, -1.0))
    p.add_argument("--minimum-coverage", type=float, default=.9)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
