"""Development-selected multi-threshold LED correction on existing ML prepared cache.

Uses interpolated rising-edge crossings at >=3 absolute mV levels (<=50 mV),
eventwise OLS t(V) intercepts, and a development-only global offset.
No ROOT reads, absolute timestamp reconstruction or ML model inference.
Blind is opened only after selecting thresholds on development.

--compare-model-dir reads saved ML development/Blind predictions and reports
paired output correlation, agreement and optional paired physical metrics.
The comparison NEVER participates in threshold selection.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import read_run, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, model_target
from waveform_analysis.engine import tshap_explainer as tshap
from waveform_analysis.engine.multithreshold_led import (
    multithreshold_corrections, validate_threshold_sets)
from waveform_analysis.reporting.stats import (
    metric_values, blind_event_bootstrap)
from waveform_analysis.scripts.plot_conditional_target_heatmaps import (
    baseline_indices)


DEFAULT_SETS = ((10., 15., 20.), (10., 20., 30.),
                (15., 25., 35.), (20., 30., 40.),
                (20., 35., 50.), (10., 20., 30., 40.),
                (10., 20., 30., 40., 50.))


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False),
                          encoding="utf-8")


def _load_cache(run_dir, role, *, cache_root, artifact_root):
    config = resolve_run_config(run_dir)
    manifest = read_run(run_dir)["manifest"]
    tshap._apply_tshap_cache(config, cache_root=cache_root,
                            artifact_root=artifact_root)
    return PreparedCacheLookup().find(role, config, manifest), config, manifest


def _corrections(dataset, mode, groups, baseline_window, crossing_window, chunk):
    family = mode_family(mode)
    waves = getattr(dataset, f"{family}_windows")
    times = getattr(dataset, f"{family}_time_ps")
    transform = getattr(dataset, f"{family}_transform")
    if waves is None or times is None or transform is None:
        raise ValueError("Prepared waveform, time grid, or input transform missing")
    if waves.shape != (dataset.n_events, 2, len(times)):
        raise ValueError("Unexpected prepared waveform shape")
    if mode != "energy_to_energy":
        raise ValueError("Multi-threshold currently supports energy_to_energy only")
    baseline = baseline_indices(np.asarray(times, np.float64),
                                *baseline_window)
    scale = np.asarray(transform.maximum - transform.minimum,
                       dtype=np.float64).reshape(2, -1)
    if scale.shape != (2, 1):
        raise ValueError("Invalid saved vertical scale dimensions")
    return multithreshold_corrections(
        waves, times, threshold_sets=groups, baseline_indices=baseline,
        led_threshold_mV=float(dataset.manifest["fixed_led_threshold_mV"]),
        physical_scale_mV=scale[:, 0],
        crossing_window_ns=crossing_window, chunk_size=chunk)


def _score(y, prediction, fit, seed):
    valid = np.isfinite(prediction) & np.isfinite(y)
    if valid.sum() < 5:
        raise ValueError("Fewer than five valid multi-threshold events")
    offset = float(np.mean(y[valid] - prediction[valid]))
    corrected = y[valid] - prediction[valid] - offset
    metrics = metric_values(corrected, fit, seed=seed)
    return metrics, offset, int(valid.sum())


def _save(path, dataset, mode, raw, offset, thresholds, role, diagnostics):
    target = np.asarray(model_target(dataset, mode), np.float64)
    valid = np.isfinite(raw) & np.isfinite(target)
    pred = np.asarray(raw, dtype=np.float64).copy() + float(offset)
    pred[~valid] = np.nan
    residual = target - pred
    np.savez_compressed(
        path,
        event_id=np.asarray(dataset.event_index, np.int64),
        prediction_ps=pred, corrected_ps=residual,
        led_residual_ps=target, valid=valid,
        thresholds_mV=np.asarray(thresholds, np.float64),
        development_offset_ps=np.asarray(offset),
        dataset_role=np.asarray(role),
        method=np.asarray("prepared_multithreshold_led"),
        analysis_protocol_identity=np.asarray(
            dataset.manifest["analysis_protocol_identity"]),
        event_population_identity=np.asarray(
            dataset.manifest["event_population_identity"]),
        control_mode_fingerprint=np.asarray(
            dataset.manifest["control_mode_fingerprint"]),
        fixed_led_threshold_mV=np.asarray(
            dataset.manifest["fixed_led_threshold_mV"]),
        led_control_mean_ps=np.asarray(
            dataset.manifest["led_control_mean_ps"]),
        true_tof_ps=np.asarray(dataset.manifest["true_tof_ps"]),
        target_definition=np.asarray(dataset.manifest["target_definition"]),
        led_local_crossing_ps=diagnostics["led_crossing_ps"],
        extrapolated_time_ps=diagnostics["intercept_ps"],
        fit_slope_ps_per_mV=diagnostics["slope_ps_per_mV"],
        fit_rmse_ps=diagnostics["fit_rmse_ps"])
    return valid, residual, target


def _load_ml_saved(run_dir, role, *, cache_root, artifact_root):
    dataset, config, manifest = _load_cache(
        run_dir, role, cache_root=cache_root, artifact_root=artifact_root)
    store = RunStore(run_dir, artifact_root=config["preprocessing"].get("artifact_root"))
    if role == "blind":
        # The existing validated loader handles both new artifact-store outputs
        # and old batch runs with artifacts/pred.npz (no embedded fingerprints).
        # For legacy data it checks the completed run stages, exact event IDs,
        # and the full saved LED residual vector against the prepared dataset.
        prediction, _ = tshap._load_materialized_local_predictions(
            store, dataset, manifest, config["mode"])
        ids = np.asarray(dataset.event_index, np.int64)
        path = store.predictions_path
        if not path.is_file():
            legacy = run_dir / "artifacts" / "pred.npz"
            if not legacy.is_file():
                raise FileNotFoundError("Validated Blind predictions have no source file")
            path = legacy
        pred = np.asarray(prediction, np.float64)
    else:
        path = store.development_predictions_path
        if not path.is_file():
            legacy = run_dir / "artifacts" / "development_pred.npz"
            if legacy.is_file():
                path = legacy
        if not path.is_file():
            raise FileNotFoundError(f"Saved ML {role} predictions missing: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = ("event_id", "prediction_ps", "final_fit_fingerprint",
                        "analysis_protocol_identity")
            if any(name not in archive for name in required):
                raise ValueError(f"ML development prediction artifact lacks provenance: {path}")
            expected_fp = manifest.get("stage_fingerprints", {}).get("final_fit")
            if (not expected_fp or
                    str(archive["final_fit_fingerprint"].item()) != str(expected_fp)):
                raise ValueError("Saved ML predictions are from a different final fit")
            ids = np.asarray(archive["event_id"], np.int64)
            pred = np.asarray(archive["prediction_ps"], np.float64)
    prepared_ids = np.asarray(dataset.event_index, np.int64)
    if (ids.ndim != 1 or ids.shape != pred.shape
            or len(np.unique(ids)) != len(ids)
            or not np.array_equal(np.sort(ids), np.sort(prepared_ids))):
        raise ValueError("ML saved event IDs do not match their prepared cache")
    if not np.all(np.isfinite(pred)):
        raise ValueError("ML saved predictions contain nonfinite values")
    return dataset, config, ids, pred, path


def _compare_saved(mt_dataset, mt_raw, mt_offset, mode, model_dirs,
                   role, *, cache_root, artifact_root, fit, seed,
                   reference_channels):
    """Paired comparisons; identifiers and target calibration are checked."""
    mt_ids = np.asarray(mt_dataset.event_index, np.int64)
    mt_pred = np.asarray(mt_raw, np.float64) + float(mt_offset)
    mt_targets = np.asarray(model_target(mt_dataset, mode), np.float64)
    if len(np.unique(mt_ids)) != mt_ids.size:
        raise ValueError("Duplicate event IDs in multi-threshold reference")
    results = []
    for model_dir in model_dirs:
        directory = tshap._resolve_model_run_directory(model_dir)
        other, config, ids, pred, path = _load_ml_saved(
            directory, role, cache_root=cache_root, artifact_root=artifact_root)
        if config["mode"] != mode:
            raise ValueError(f"Comparison mode mismatch: {directory}")
        if config[role].get("channels") != reference_channels:
            # Distinct detector channels must not be treated as the same event.
            raise ValueError("ML comparison channel definitions disagree")
        for name in ("control_mode_fingerprint", "target_definition"):
            if other.manifest[name] != mt_dataset.manifest[name]:
                raise ValueError(f"Comparison {name} mismatch: {directory}")
        for name in ("true_tof_ps", "led_control_mean_ps",
                     "fixed_led_threshold_mV"):
            if not np.isclose(float(other.manifest[name]),
                              float(mt_dataset.manifest[name]),
                              rtol=0, atol=1e-3):
                raise ValueError(f"Comparison calibration mismatch: {name}")
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        positions = np.searchsorted(sorted_ids, mt_ids)
        present = positions < sorted_ids.size
        present[present] &= sorted_ids[positions[present]] == mt_ids[present]
        left_rows = np.flatnonzero(present)
        right_rows = order[positions[present]]
        left_y = mt_targets[left_rows]
        right_y = np.asarray(model_target(other, mode), np.float64)[right_rows]
        if left_y.size and np.max(np.abs(left_y - right_y)) > 1e-3:
            raise ValueError("Matched prepared targets disagree between runs")
        good = (np.isfinite(mt_pred[left_rows]) & np.isfinite(pred[right_rows])
                & np.isfinite(left_y) & np.isfinite(right_y))
        a, b = mt_pred[left_rows[good]], pred[right_rows[good]]
        y = left_y[good]
        if a.size < 5:
            raise ValueError(f"Not enough valid common comparison events: {directory}")
        diff = a - b
        corr = (float(np.corrcoef(a, b)[0, 1])
                if np.std(a) > 0 and np.std(b) > 0 else None)
        # Physical residual evaluation is diagnostic, never used for selecting
        # threshold sets. Both methods use exactly the same paired population.
        mt_metric = metric_values(y - a, fit,
                                  seed=semantic_seed(seed, role, "mt", str(directory)))
        ml_metric = metric_values(y - b, fit,
                                  seed=semantic_seed(seed, role, "ml", str(directory)))
        results.append({
            "run_dir": str(directory), "predictions_source": str(path),
            "role": role, "common_events": int(a.size),
            "mt_only_or_invalid_events": int(len(mt_ids) - a.size),
            "ml_only_events": int(len(ids) - a.size),
            "prediction_pearson_r": corr,
            "mt_prediction_mean_ps": float(a.mean()),
            "ml_prediction_mean_ps": float(b.mean()),
            "mt_prediction_std_ps": float(a.std()),
            "ml_prediction_std_ps": float(b.std()),
            "mt_minus_ml_bias_ps": float(diff.mean()),
            "mt_minus_ml_rmse_ps": float(np.sqrt(np.mean(diff**2))),
            "mt_minus_ml_mae_ps": float(np.mean(np.abs(diff))),
            "paired_mt_ctr_ps": mt_metric["ctr_ps"],
            "paired_mt_rmse_ps": mt_metric["rmse_ps"],
            "paired_ml_ctr_ps": ml_metric["ctr_ps"],
            "paired_ml_rmse_ps": ml_metric["rmse_ps"],
        })
    return results


def _comparison_runs(explicit, roots):
    """Find saved run directories under one or more result roots, recursively."""
    direct = [tshap._resolve_model_run_directory(p) for p in explicit]
    discovered = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Comparison model root not found: {root}")
        manifest = root / "metadata" / "manifest.json"
        if manifest.is_file():
            discovered.append(root)
        else:
            discovered.extend(
                p.parent.parent for p in root.rglob("metadata/manifest.json"))
    # Preserve explicit order and stable discovered order. No duplicate models.
    unique = {}
    for path in direct + sorted(discovered):
        unique[str(path.resolve())] = path.resolve()
    if roots and not discovered:
        raise ValueError("No saved ML runs found under --compare-model-root")
    return list(unique.values()), {str(p.resolve()) for p in direct}


def _compare_all(dataset, raw, offset, mode, paths, explicit_paths, role,
                 *, cache_root, artifact_root, fit, seed, reference_channels):
    results, skipped = [], []
    for path in paths:
        try:
            result = _compare_saved(
                dataset, raw, offset, mode, [path], role,
                cache_root=cache_root, artifact_root=artifact_root,
                fit=fit, seed=seed, reference_channels=reference_channels)
            results.extend(result)
        except (ValueError, FileNotFoundError, KeyError, RuntimeError) as exc:
            if str(path.resolve()) in explicit_paths:
                raise
            skipped.append({"run_dir": str(path), "role": role,
                            "reason": f"{type(exc).__name__}: {exc}"})
    return results, skipped


def _saved_mt(dataset, mode, path, role):
    """Validate saved MT against current prepared cache without recomputing."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing precomputed multi-threshold predictions: {path}")
    with np.load(path, allow_pickle=False) as stored:
        required = ("event_id", "prediction_ps", "led_residual_ps", "valid",
                    "dataset_role", "method", "analysis_protocol_identity",
                    "control_mode_fingerprint", "target_definition",
                    "fixed_led_threshold_mV", "led_control_mean_ps",
                    "true_tof_ps", "development_offset_ps")
        if missing := set(required) - set(stored.files):
            raise ValueError(f"MT prediction file missing {sorted(missing)}")
        if str(stored["method"].item()) != "prepared_multithreshold_led":
            raise ValueError("Not a multi-threshold prediction file")
        if str(stored["dataset_role"].item()) != role:
            raise ValueError("MT prediction dataset role mismatch")
        for key in ("analysis_protocol_identity", "control_mode_fingerprint",
                    "target_definition"):
            if str(stored[key].item()) != str(dataset.manifest[key]):
                raise ValueError(f"MT saved metadata differs from cache: {key}")
        for key in ("fixed_led_threshold_mV", "led_control_mean_ps",
                    "true_tof_ps"):
            if not np.isclose(float(stored[key]), float(dataset.manifest[key]),
                              rtol=0, atol=1e-6):
                raise ValueError(f"MT saved calibration differs: {key}")
        ids = np.asarray(stored["event_id"], np.int64)
        preds = np.asarray(stored["prediction_ps"], np.float64)
        led = np.asarray(stored["led_residual_ps"], np.float64)
        valid = np.asarray(stored["valid"], bool)
        offset = float(stored["development_offset_ps"])
    reference = np.asarray(dataset.event_index, np.int64)
    target = np.asarray(model_target(dataset, mode), np.float64)
    if (not np.array_equal(ids, reference) or
            preds.shape != target.shape or led.shape != target.shape or
            valid.shape != target.shape):
        raise ValueError("MT saved events mismatch prepared population")
    if not np.allclose(led, target, rtol=0, atol=1e-6):
        raise ValueError("MT LED targets mismatch prepared population")
    if not np.array_equal(valid, np.isfinite(preds)):
        raise ValueError("MT valid mask mismatches prediction finiteness")
    return preds - offset, offset


def _write_comparison_csv(output, results):
    if not results:
        return
    fields = list(results[0])
    with (output / "model_correlations.csv").open("w", newline="",
                                                   encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

def run(args):
    if args.cache_root is not None and args.artifact_root is not None:
        raise ValueError("Choose --cache-root OR --artifact-root")
    groups = validate_threshold_sets(
        [tuple(map(float, group.split(","))) for group in args.threshold_sets])
    compare_paths, explicit_paths = _comparison_runs(
        args.compare_model_dir, args.compare_model_root)
    if args.compare_only_root is not None:
        out = args.compare_only_root.expanduser().resolve()
        summary_file = out / "summary.json"
        if not summary_file.is_file():
            raise FileNotFoundError(f"Existing multi-threshold summary missing: {summary_file}")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if summary.get("method") != "prepared_multithreshold_led":
            raise ValueError("Compare-only directory does not contain MT results")
        origin = tshap._resolve_model_run_directory(args.model_dir)
        if Path(summary["reference_model_dir"]).resolve() != origin:
            raise ValueError("Original MT reference run differs from --model-dir")
        results, failures = {}, {}
        for role in ("development", "blind"):
            dataset, config, _ = _load_cache(
                origin, role, cache_root=args.cache_root,
                artifact_root=args.artifact_root)
            if config["mode"] != summary["mode"]:
                raise ValueError("Saved MT comparison mode mismatch")
            raw, offset = _saved_mt(
                dataset, config["mode"], out / f"{role}_predictions.npz", role)
            if not np.isclose(offset, float(summary["development_offset_ps"]),
                              rtol=0, atol=1.e-7):
                raise ValueError("Saved MT offset disagrees with summary")
            results[role], failures[role] = _compare_all(
                dataset, raw, offset, config["mode"], compare_paths,
                explicit_paths, role, cache_root=args.cache_root,
                artifact_root=args.artifact_root, fit=config["fit"],
                seed=args.seed, reference_channels=config[role].get("channels"))
        summary["comparison_development"] = results["development"]
        summary["comparison_blind"] = results["blind"]
        summary["comparison_skipped_development"] = failures["development"]
        summary["comparison_skipped_blind"] = failures["blind"]
        _write_json(summary_file, summary)
        _write_comparison_csv(out, results["blind"])
        for row in results["blind"]:
            print(f"CORRELATION | {Path(row['run_dir']).name} | "
                  f"Pearson r={row['prediction_pearson_r']} | "
                  f"common={row['common_events']}", flush=True)
        for row in failures["blind"]:
            print(f"SKIPPED | {row['run_dir']} | {row['reason']}", flush=True)
        print(f"COMPARED | {len(results['blind'])} models | "
              f"skipped={len(failures['blind'])} | {out}", flush=True)
        return summary
    if args.chunk_size < 1 or args.bootstrap < 1:
        raise ValueError("chunk-size/bootstrap must be positive")
    run_dir = tshap._resolve_model_run_directory(args.model_dir)
    out = args.output_root.expanduser().resolve()
    if out == run_dir or run_dir in out.parents:
        raise ValueError("Output must be outside reference ML run")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)

    development, config, _ = _load_cache(
        run_dir, "development", cache_root=args.cache_root,
        artifact_root=args.artifact_root)
    mode = config["mode"]
    if mode != "energy_to_energy":
        raise ValueError("Initially limited to energy_to_energy")
    y_dev = np.asarray(model_target(development, mode), np.float64)
    fit = config["fit"]
    baseline = tuple(map(float, args.baseline_window_ns))
    crossing = tuple(map(float, args.crossing_window_ns))
    corrections, diagnostic = _corrections(
        development, mode, groups, baseline, crossing, args.chunk_size)
    # All threshold sets are compared on exactly the same development events.
    common = np.all(np.isfinite(corrections), axis=1) & np.isfinite(y_dev)
    coverage = float(common.mean())
    if common.sum() < 5 or coverage < args.minimum_coverage:
        raise ValueError(
            f"Common development coverage {coverage:.2%} below "
            f"{args.minimum_coverage:.2%}; narrow threshold sets")
    rows = []
    for j, group in enumerate(groups):
        metric, _, _ = _score(
            y_dev[common], corrections[common, j], fit,
            semantic_seed(args.seed, "mt", "candidate", j))
        rows.append({"thresholds_mV": ",".join(map(str, group)),
                     "ctr_ps": metric["ctr_ps"], "rmse_ps": metric["rmse_ps"],
                     "common_events": int(common.sum()),
                     "individual_coverage": float(np.isfinite(corrections[:, j]).mean())})
    best = min(range(len(rows)), key=lambda j: (
        rows[j][f"{args.metric}_ps"], groups[j]))
    chosen = groups[best]
    raw = corrections[:, best]
    metric_dev, offset, _ = _score(
        y_dev, raw, fit, semantic_seed(args.seed, "mt", "development"))
    fit_dev = {k: (v[:, :, best] if v.ndim == 3 else v)
               for k, v in diagnostic.items() if k != "crossing_levels_mV"}
    valid_dev, residual_dev, led_dev = _save(
        out / "development_predictions.npz", development, mode,
        raw, offset, chosen, "development", fit_dev)
    compare_development, skipped_development = _compare_all(
        development, raw, offset, mode, compare_paths, explicit_paths,
        "development", cache_root=args.cache_root, artifact_root=args.artifact_root,
        fit=fit, seed=args.seed,
        reference_channels=config["development"].get("channels"))
    print(f"SELECTED | thresholds={chosen} mV | development CTR="
          f"{metric_dev['ctr_ps']:.3f} ps | RMSE={metric_dev['rmse_ps']:.3f} ps "
          f"| coverage={valid_dev.mean():.2%}", flush=True)
    del corrections, diagnostic

    # Blind data is not loaded until every parameter has been frozen.
    blind, blind_cfg, _ = _load_cache(
        run_dir, "blind", cache_root=args.cache_root,
        artifact_root=args.artifact_root)
    for key in ("mode", "control_mode_fingerprint", "target_definition"):
        if blind.manifest[key] != development.manifest[key]:
            raise ValueError(f"Blind calibration mismatch: {key}")
    for key in ("true_tof_ps", "led_control_mean_ps",
                "fixed_led_threshold_mV"):
        if not np.isclose(float(blind.manifest[key]),
                          float(development.manifest[key]), atol=1e-3, rtol=0):
            raise ValueError(f"Blind calibration mismatch: {key}")
    if not np.array_equal(getattr(blind, "energy_time_ps"),
                          getattr(development, "energy_time_ps")):
        raise ValueError("Development/Blind prepared grids differ")
    blind_corr, blind_diag = _corrections(
        blind, mode, [chosen], baseline, crossing, args.chunk_size)
    fit_blind = {k: (v[:, :, 0] if v.ndim == 3 else v)
                 for k, v in blind_diag.items() if k != "crossing_levels_mV"}
    valid_blind, residual_blind, led_blind = _save(
        out / "blind_predictions.npz", blind, mode, blind_corr[:, 0],
        offset, chosen, "blind", fit_blind)
    if valid_blind.sum() < 5:
        raise ValueError("Fewer than five finite Blind multi-threshold events")
    bootstrap, draws = blind_event_bootstrap(
        residual_blind[valid_blind], led_blind[valid_blind], fit,
        n_resamples=args.bootstrap,
        seed=semantic_seed(args.seed, "mt", "blind"))
    np.savez_compressed(out / "blind_bootstrap.npz", **draws)
    compare_blind, skipped_blind = _compare_all(
        blind, blind_corr[:, 0], offset, mode, compare_paths, explicit_paths,
        "blind", cache_root=args.cache_root, artifact_root=args.artifact_root,
        fit=fit, seed=args.seed,
        reference_channels=config["blind"].get("channels"))
    with (out / "threshold_scan.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "method": "prepared_multithreshold_led", "reference_model_dir": str(run_dir),
        "mode": mode, "metric": args.metric,
        "threshold_scan": rows, "selected_thresholds_mV": list(chosen),
        "selection": "development_only_common_valid_population",
        "development_offset_ps": float(offset),
        "development_cache": str(development.directory),
        "blind_cache": str(blind.directory),
        "baseline_window_ns": list(baseline),
        "crossing_window_ns": list(crossing),
        "development": {"n": int(development.n_events),
                        "valid": int(valid_dev.sum()),
                        "coverage": float(valid_dev.mean()),
                        "metrics": metric_dev},
        "blind": {"n": int(blind.n_events),
                  "valid": int(valid_blind.sum()),
                  "coverage": float(valid_blind.mean()),
                  "metrics": bootstrap},
        "comparison_development": compare_development,
        "comparison_blind": compare_blind,
        "comparison_skipped_development": skipped_development,
        "comparison_skipped_blind": skipped_blind,
        "sign": "prediction=(LED1-t0_1)-(LED2-t0_2)+dev_offset; corrected=LED_target-prediction",
        "warning": ("Prepared waveforms may be clipped; OLS extrapolation is "
                    "linear only locally and does not isolate physical causes."),
    }
    _write_json(out / "summary.json", summary)
    _write_comparison_csv(out, compare_blind)
    print(f"BLIND | CTR={bootstrap['ctr_ps']:.3f} ps | "
          f"RMSE={bootstrap['rmse_ps']:.3f} ps | "
          f"LED CTR={bootstrap['led_ctr_ps']:.3f} ps "
          f"| coverage={valid_blind.mean():.2%}", flush=True)
    for result in compare_blind:
        print(f"CORRELATION | {Path(result['run_dir']).name} | "
              f"Pearson r={result['prediction_pearson_r']} | "
              f"common={result['common_events']} | "
              f"MT minus ML RMSE={result['mt_minus_ml_rmse_ps']:.3f} ps", flush=True)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Reference run identifying frozen prepared data and fit settings")
    p.add_argument("--compare-model-dir", type=Path, nargs="*", default=[],
                   help="Saved ML run(s) to compare on development and blind")
    p.add_argument("--compare-model-root", type=Path, nargs="*", default=[],
                   help="Search recursively for all saved model runs in these directories")
    p.add_argument("--compare-only-root", type=Path,
                   help="Reuse completed MT predictions and update correlations only; "
                        "must point to an existing MT output directory")
    p.add_argument("--cache-root", type=Path)
    p.add_argument("--artifact-root", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--threshold-sets", nargs="+",
                   default=[",".join(str(v) for v in g) for g in DEFAULT_SETS],
                   help="Comma-separated mV thresholds per candidate; <=50, >=3 levels")
    p.add_argument("--metric", choices=("ctr", "rmse"), default="ctr")
    p.add_argument("--baseline-window-ns", nargs=2, type=float,
                   default=(-1.9, -1.0))
    p.add_argument("--crossing-window-ns", nargs=2, type=float,
                   default=(-.8, 2.0))
    p.add_argument("--minimum-coverage", type=float, default=.9)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
