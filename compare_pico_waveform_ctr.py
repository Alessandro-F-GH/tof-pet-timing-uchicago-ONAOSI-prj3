#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
WAVEFORM_ROOT = REPO_ROOT / "waveform_analysis"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WAVEFORM_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVEFORM_ROOT))

from utils.config import config_copy, load_config
from utils.pipeline import build_selection, extract_features, load_features, save_features
from utils_fit import choose_best, fit_delta_times_integer_fs
from utils_fit.outliers import robust_mad_filter
from utils_fit.plotting import plot_ctr_histogram

INVALID_TIME_FS = np.iinfo(np.int64).min
FS_PER_PS = 1000.0

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "figure.titlesize": 20,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Pico-TDC timing-channel CTR with oscilloscope adaptive LED. "
            "CTR is the Gaussian-equivalent shortest configured-coverage interval; final uncertainties are event-bootstrap standard deviations."
        )
    )
    parser.add_argument("--pico-summary", required=True, type=Path)
    parser.add_argument("--scope-root-folder", required=True, type=Path)
    parser.add_argument("--scope-config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("ctr_pico_vs_scope"))
    parser.add_argument("--root-pattern", default="*.root")
    parser.add_argument("--threshold-selection-stage", choices=("blind", "validation"), default="blind")
    parser.add_argument("--blind-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--voltage-pattern", default=r"(?P<voltage>\d+(?:\.\d+)?)V")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--pico-timing-threshold-mv", type=float, default=40.0)
    parser.add_argument("--pico-acquisition-mode", default=None)
    parser.add_argument("--allow-legacy-pico-summary", action="store_true")
    parser.add_argument("--scope-min-led-threshold-mv", type=float, default=20.0)
    parser.add_argument("--scope-led-outlier-z", type=float, default=4.0)
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=100,
        help="Number of event bootstrap resamples for each final Pico/scope FWHM point.",
    )
    parser.add_argument(
        "--bootstrap-min-success-fraction",
        type=float,
        default=0.9,
        help="Require at least this fraction of bootstrap FWHM estimates to succeed.",
    )
    return parser.parse_args()


def _stable_seed(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(item) for item in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def _voltage(path: Path, pattern: str) -> float:
    match = re.search(pattern, path.name)
    if not match:
        raise ValueError(f"Cannot infer voltage from {path.name!r} using {pattern!r}")
    try:
        return float(match.group("voltage"))
    except (IndexError, KeyError):
        return float(match.group(1))


def _split_selected(selected: np.ndarray, *, blind_fraction: float, validation_fraction: float, seed: int):
    indices = np.flatnonzero(np.asarray(selected, dtype=bool))
    if indices.size < 5:
        raise RuntimeError("Too few selected events to build comparison split")
    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)
    n_blind = min(max(1, int(round(indices.size * blind_fraction))), indices.size - 2)
    blind = np.sort(order[:n_blind])
    development = order[n_blind:]
    n_validation = min(max(1, int(round(development.size * validation_fraction))), development.size - 1)
    validation = np.sort(development[:n_validation])
    train = np.sort(development[n_validation:])
    return train, validation, blind


def _bootstrap_ctr(
    delta_fs: np.ndarray,
    *,
    fit_config: dict[str, Any],
    method: str,
    parameter: float,
    n_bootstrap: int,
    seed: int,
    min_success_fraction: float,
) -> dict[str, Any]:
    """Bootstrap an already-selected cohort using the canonical robust CTR."""
    values = np.asarray(delta_fs, dtype=np.int64).reshape(-1)
    if values.size < int(fit_config.get("min_events", 10)):
        raise RuntimeError(f"Too few selected events for bootstrap: {values.size}")
    if n_bootstrap < 2:
        raise ValueError("--bootstrap-samples must be >= 2")
    rng = np.random.default_rng(seed)
    ctrs: list[float] = []
    for _ in range(int(n_bootstrap)):
        sample = values[rng.integers(0, values.size, size=values.size)]
        result = fit_delta_times_integer_fs(
            sample,
            method=method,
            parameter=float(parameter),
            n_total=int(values.size),
            n_selected=int(values.size),
            config=fit_config,
        )
        if result.success and np.isfinite(result.ctr_ps):
            ctrs.append(float(result.ctr_ps))
    minimum_success = math.ceil(float(min_success_fraction) * int(n_bootstrap))
    if len(ctrs) < minimum_success:
        raise RuntimeError(
            f"Only {len(ctrs)}/{n_bootstrap} bootstrap FWHM estimates succeeded for {method}; need at least {minimum_success}."
        )
    ctr_array = np.asarray(ctrs, dtype=np.float64)
    return {
        "ctr_mean_ps": float(np.mean(ctr_array)),
        "ctr_bootstrap_std_ps": float(np.std(ctr_array, ddof=1)),
        "bootstrap_samples_requested": int(n_bootstrap),
        "bootstrap_samples_successful": int(ctr_array.size),
        "bootstrap_ctr_min_ps": float(np.min(ctr_array)),
        "bootstrap_ctr_max_ps": float(np.max(ctr_array)),
    }


def _scope_delta_after_rejection(features, selected_indices, threshold_index, *, outlier_z):
    a = np.asarray(features["t_led_a_fs"], dtype=np.int64)[selected_indices, threshold_index]
    b = np.asarray(features["t_led_b_fs"], dtype=np.int64)[selected_indices, threshold_index]
    valid = (a != INVALID_TIME_FS) & (b != INVALID_TIME_FS)
    delta_fs = a[valid] - b[valid]
    if delta_fs.size < 3:
        raise RuntimeError("Too few valid oscilloscope LED pairs")
    rejection = robust_mad_filter(delta_fs.astype(np.float64) / FS_PER_PS, enabled=True, zscore_limit=float(outlier_z))
    filtered = delta_fs[rejection.mask]
    return filtered, {
        "n_before_outlier": int(delta_fs.size),
        "n_after_outlier": int(filtered.size),
        "n_outlier_rejected": int(rejection.rejected),
        "outlier_center_ps": float(rejection.center),
        "outlier_sigma_ps": float(rejection.robust_sigma),
        "outlier_limit_ps": float(rejection.max_distance),
    }


def _fit_scope_threshold(features, selected_indices, threshold_index, threshold_mV, fit_config, *, method, outlier_z):
    delta_fs, rejection_meta = _scope_delta_after_rejection(features, selected_indices, threshold_index, outlier_z=outlier_z)
    result = fit_delta_times_integer_fs(
        delta_fs,
        method=method,
        parameter=float(threshold_mV),
        n_total=int(features["event_id"].size),
        n_selected=int(delta_fs.size),
        config=fit_config,
    )
    return result, delta_fs, rejection_meta


def _scope_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = config_copy(load_config(args.scope_config))
    roots = sorted(args.scope_root_folder.glob(args.root_pattern))
    if not roots:
        raise RuntimeError(f"No ROOT files match {args.root_pattern!r} in {args.scope_root_folder}")
    cache_root = args.output / "scope_feature_cache"
    histogram_root = args.output / "scope_led_histograms"
    cache_root.mkdir(parents=True, exist_ok=True)
    histogram_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for root_file in roots:
        voltage = _voltage(root_file, args.voltage_pattern)
        cache = cache_root / f"{root_file.stem}.npz"
        if args.reuse_features and cache.is_file():
            features = load_features(cache, cfg, root_file)
        else:
            features = extract_features(root_file, cfg)
            save_features(cache, features)
        selection = build_selection(features, cfg)
        _, validation, blind = _split_selected(
            selection.selected,
            blind_fraction=args.blind_fraction,
            validation_fraction=args.validation_fraction,
            seed=_stable_seed(args.seed, "split", root_file.name),
        )
        all_thresholds = np.asarray(features["led_thresholds_mV"], dtype=np.float64)
        threshold_indices = np.flatnonzero(all_thresholds >= float(args.scope_min_led_threshold_mv))
        thresholds = all_thresholds[threshold_indices]
        if thresholds.size == 0:
            raise RuntimeError(f"No LED thresholds >= {args.scope_min_led_threshold_mv:g} mV for {root_file.name}")
        score_indices = blind if args.threshold_selection_stage == "blind" else validation
        score_results = []
        for original_index, threshold_mV in zip(threshold_indices, thresholds):
            result, _, _ = _fit_scope_threshold(
                features, score_indices, int(original_index), float(threshold_mV), cfg["fit"],
                method=f"LED selection ({args.threshold_selection_stage})", outlier_z=args.scope_led_outlier_z,
            )
            score_results.append(result)
        chosen = choose_best(score_results)
        if chosen is None:
            raise RuntimeError(f"No successful LED threshold FWHM estimate for {root_file.name}")
        chosen_local = int(np.argmin(np.abs(thresholds - chosen.parameter)))
        chosen_original_index = int(threshold_indices[chosen_local])
        final_result, final_delta_fs, rejection_meta = _fit_scope_threshold(
            features, blind, chosen_original_index, float(chosen.parameter), cfg["fit"],
            method="Oscilloscope LED blind", outlier_z=args.scope_led_outlier_z,
        )
        if not final_result.success:
            raise RuntimeError(f"Final scope LED FWHM failed for {root_file.name}: {final_result.message}")
        bootstrap = _bootstrap_ctr(
            final_delta_fs,
            fit_config=cfg["fit"],
            method="Oscilloscope LED bootstrap",
            parameter=float(chosen.parameter),
            n_bootstrap=args.bootstrap_samples,
            seed=_stable_seed(args.seed, "scope-bootstrap", root_file.name, chosen.parameter),
            min_success_fraction=args.bootstrap_min_success_fraction,
        )
        plot_ctr_histogram(
            final_result,
            histogram_root / f"{root_file.stem}.png",
            dpi=int(cfg.get("plot", {}).get("dpi", 180)),
            title=f"{root_file.stem} · adaptive LED · {chosen.parameter:g} mV · blind",
        )
        rows.append({
            "source_file": root_file.name,
            "voltage_V": float(voltage),
            "threshold_selection_stage": args.threshold_selection_stage,
            "threshold_mV": float(chosen.parameter),
            "min_threshold_mV": float(args.scope_min_led_threshold_mv),
            "nominal_ctr_ps": float(final_result.ctr_ps),
            **bootstrap,
            **rejection_meta,
            "fit_metric": "gaussian_equivalent_shortest_coverage_interval",
        })
        print(
            f"[scope][{root_file.name}] V={voltage:g} V | best LED threshold={chosen.parameter:g} mV | "
            f"retained={rejection_meta['n_after_outlier']}/{rejection_meta['n_before_outlier']} | "
            f"bootstrap CTR={bootstrap['ctr_mean_ps']:.2f} ± {bootstrap['ctr_bootstrap_std_ps']:.2f} ps"
        )
    return rows


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _choose_best_pico_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _read_summary_rows(args.pico_summary)
    if not rows:
        raise RuntimeError(f"No rows in {args.pico_summary}")
    if not args.allow_legacy_pico_summary:
        valid_metric = "gaussian_equivalent_shortest_coverage_interval"
        for row in rows:
            if str(row.get("fit_metric", "")) != valid_metric:
                raise RuntimeError("Pico summary contains rows not generated with the robust shortest-coverage CTR.")
    candidates: dict[float, list[dict[str, str]]] = {}
    for row in rows:
        if args.pico_acquisition_mode is not None and row.get("AcquisitionMode") != args.pico_acquisition_mode:
            continue
        try:
            voltage = float(row["Voltage"]); threshold = float(row["T_th"]); ctr = float(row["CTR_ps"])
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isclose(threshold, float(args.pico_timing_threshold_mv), rtol=0.0, atol=1e-9) or not np.isfinite(ctr):
            continue
        candidates.setdefault(voltage, []).append(row)
    if not candidates:
        raise RuntimeError(f"No Pico rows at T_th={args.pico_timing_threshold_mv:g} mV")
    chosen = []
    for voltage, voltage_rows in sorted(candidates.items()):
        best = min(voltage_rows, key=lambda row: float(row["CTR_ps"]))
        chosen.append(best)
        if len(voltage_rows) > 1:
            print(f"[pico][{voltage:g} V] {len(voltage_rows)} rows -> using {best.get('run_id','')} (CTR={float(best['CTR_ps']):.2f} ps)")
    return chosen


def _find_selection_csv(run_dir: Path) -> Path:
    direct = run_dir / "csv" / "selection.csv"
    if direct.is_file():
        return direct
    matches = sorted((run_dir / "csv").glob("selection.*"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise RuntimeError(f"No selection table found under {run_dir / 'csv'}")
    raise RuntimeError(f"Multiple selection tables found under {run_dir / 'csv'}")


def _find_toa_lsb_ps(state_path: Path) -> float:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    found: list[float] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "toa_lsb_ps" in value:
                try:
                    number = float(value["toa_lsb_ps"])
                    if np.isfinite(number) and number > 0:
                        found.append(number)
                except (TypeError, ValueError):
                    pass
            for child in value.values(): walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    walk(state)
    if not found:
        raise RuntimeError(f"Could not find toa_lsb_ps in {state_path}")
    rounded = {round(value, 12) for value in found}
    if len(rounded) > 1:
        raise RuntimeError(f"Inconsistent toa_lsb_ps values in {state_path}: {sorted(rounded)}")
    return float(found[0])


def _load_pico_selected_delta_fs(summary_path: Path, run_id: str, *, outlier_z: float):
    output_root = summary_path.resolve().parent
    run_dir = output_root / "analysis" / run_id
    selection_path = _find_selection_csv(run_dir)
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing Janus state file: {state_path}")
    toa_lsb_ps = _find_toa_lsb_ps(state_path)
    with selection_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected_delta_lsb = []
    for row in rows:
        try:
            if not (bool(int(row["duration_selected"])) and bool(int(row["alignment_selected"]))):
                continue
            selected_delta_lsb.append(int(row["time_b_lsb"]) - int(row["time_a_lsb"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Malformed Janus selection row in {selection_path}") from exc
    if len(selected_delta_lsb) < 3:
        raise RuntimeError(f"Too few selected Pico events for {run_id}")
    delta_ps = np.asarray(selected_delta_lsb, dtype=np.float64) * toa_lsb_ps
    rejection = robust_mad_filter(delta_ps, enabled=True, zscore_limit=float(outlier_z))
    selected_ps = delta_ps[rejection.mask]
    selected_fs = np.rint(selected_ps * FS_PER_PS).astype(np.int64)
    return selected_fs, {
        "toa_lsb_ps": float(toa_lsb_ps),
        "n_before_outlier": int(delta_ps.size),
        "n_after_outlier": int(selected_fs.size),
        "n_outlier_rejected": int(rejection.rejected),
        "outlier_center_ps": float(rejection.center),
        "outlier_sigma_ps": float(rejection.robust_sigma),
        "outlier_limit_ps": float(rejection.max_distance),
    }


def _pico_rows(args: argparse.Namespace, fit_config: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in _choose_best_pico_rows(args):
        run_id = str(row["run_id"]); voltage = float(row["Voltage"]); threshold = float(row["T_th"])
        delta_fs, rejection_meta = _load_pico_selected_delta_fs(args.pico_summary, run_id, outlier_z=4.0)
        bootstrap = _bootstrap_ctr(
            delta_fs,
            fit_config=fit_config,
            method="Pico-TDC bootstrap",
            parameter=threshold,
            n_bootstrap=args.bootstrap_samples,
            seed=_stable_seed(args.seed, "pico-bootstrap", run_id),
            min_success_fraction=args.bootstrap_min_success_fraction,
        )
        output.append({
            "run_id": run_id,
            "voltage_V": voltage,
            "timing_threshold_mV": threshold,
            "nominal_ctr_ps": float(row["CTR_ps"]),
            **bootstrap,
            **rejection_meta,
            "fit_metric": "gaussian_equivalent_shortest_coverage_interval",
        })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def _paired_rows(pico, scope):
    scope_by_voltage = {float(row["voltage_V"]): row for row in scope}
    rows = []
    for p in pico:
        voltage = float(p["voltage_V"]); s = scope_by_voltage.get(voltage)
        if s is None: continue
        pico_mean = float(p["ctr_mean_ps"]); scope_mean = float(s["ctr_mean_ps"])
        pico_err = float(p["ctr_bootstrap_std_ps"]); scope_err = float(s["ctr_bootstrap_std_ps"])
        combined_error = math.sqrt(pico_err**2 + scope_err**2)
        z = (pico_mean - scope_mean) / combined_error if combined_error > 0 else float("nan")
        rows.append({
            "voltage_V": voltage,
            "pico_run_id": p["run_id"],
            "pico_threshold_mV": p["timing_threshold_mV"],
            "scope_threshold_mV": s["threshold_mV"],
            "pico_ctr_mean_ps": pico_mean,
            "pico_bootstrap_std_ps": pico_err,
            "scope_ctr_mean_ps": scope_mean,
            "scope_bootstrap_std_ps": scope_err,
            "difference_mean_ps": pico_mean - scope_mean,
            "combined_bootstrap_error_ps": combined_error,
            "difference_over_combined_error": z,
        })
    return rows


def _plot_comparison(paired, path: Path, *, pico_threshold_mV: float, threshold_selection_stage: str) -> None:
    del threshold_selection_stage
    if not paired:
        raise RuntimeError("No voltages are shared by Pico and oscilloscope results")
    ordered = sorted(paired, key=lambda row: float(row["voltage_V"]))
    voltage = np.asarray([float(row["voltage_V"]) for row in ordered])
    pico_mean = np.asarray([float(row["pico_ctr_mean_ps"]) for row in ordered])
    scope_mean = np.asarray([float(row["scope_ctr_mean_ps"]) for row in ordered])
    z = np.asarray([float(row["difference_over_combined_error"]) for row in ordered])
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True, gridspec_kw={"height_ratios": [3.0, 1.0]})
    ax.plot(voltage, pico_mean, marker="o", alpha=0.7, linestyle="none", markersize=15, label=f"Pico-TDC · T_th={pico_threshold_mV:g} mV")
    ax.plot(voltage, scope_mean, marker="s", alpha=0.7, linestyle="none", markersize=15, label="Oscilloscope adaptive LED")
    ax.set_ylabel("Robust CTR [ps]", fontsize=18); ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=15)
    residual_ax.axhline(0.0, linewidth=1.2); residual_ax.axhline(1.0, linewidth=1.2, linestyle="--"); residual_ax.axhline(-1.0, linewidth=1.2, linestyle="--")
    residual_ax.plot(voltage, z, marker="+", linestyle="none", markersize=14, markeredgewidth=2.0)
    residual_ax.set_xlabel("Bias voltage [V]", fontsize=18); residual_ax.set_ylabel("Difference in sigma", fontsize=17); residual_ax.grid(alpha=0.3)
    fig.suptitle("Pico-TDC vs oscilloscope", fontsize=20); fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples < 2:
        raise ValueError("--bootstrap-samples must be >= 2")
    if not 0.0 < args.bootstrap_min_success_fraction <= 1.0:
        raise ValueError("--bootstrap-min-success-fraction must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    scope_cfg = config_copy(load_config(args.scope_config))
    scope = _scope_rows(args)
    pico = _pico_rows(args, scope_cfg["fit"])
    paired = _paired_rows(pico, scope)
    _write_csv(args.output / "oscilloscope_adaptive_led.csv", scope)
    _write_csv(args.output / "pico_tdc.csv", pico)
    _write_csv(args.output / "paired_comparison.csv", paired)
    _plot_comparison(paired, args.output / "ctr_vs_voltage_bootstrap.png", pico_threshold_mV=args.pico_timing_threshold_mv, threshold_selection_stage=args.threshold_selection_stage)
    metadata = {
        "ctr_metric": "gaussian_equivalent_shortest_coverage_interval",
        "coverage_fraction": float(scope_cfg["fit"].get("coverage_fraction", 0.90)),
        "core_bin_width_ps": float(scope_cfg["fit"].get("bin_width_ps", 5.0)),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_error_definition": "sample standard deviation (ddof=1) of successful bootstrap robust-CTR estimates",
        "seed": args.seed,
    }
    with (args.output / "comparison_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    print(f"Wrote comparison to: {args.output}")


if __name__ == "__main__":
    main()
