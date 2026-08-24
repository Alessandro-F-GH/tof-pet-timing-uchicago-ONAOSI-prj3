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
from utils_fit.plotting import plot_gaussian_fit

INVALID_TIME_FS = np.iinfo(np.int64).min
FS_PER_PS = 1000.0

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Pico-TDC timing-channel CTR with oscilloscope adaptive LED. "
            "Final uncertainties are event-bootstrap standard deviations."
        )
    )
    parser.add_argument("--pico-summary", required=True, type=Path)
    parser.add_argument("--scope-root-folder", required=True, type=Path)
    parser.add_argument("--scope-config", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("ctr_pico_vs_scope"))
    parser.add_argument("--root-pattern", default="*.root")

    parser.add_argument(
        "--threshold-selection-stage",
        choices=("blind", "validation"),
        default="blind",
        help=(
            "blind = oracle threshold selection and final evaluation on blind; "
            "validation = choose threshold on validation, evaluate on blind."
        ),
    )
    parser.add_argument("--blind-fraction", type=float, default=0.2)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260813)

    parser.add_argument(
        "--voltage-pattern",
        default=r"(?P<voltage>\d+(?:\.\d+)?)V",
    )
    parser.add_argument("--reuse-features", action="store_true")

    parser.add_argument(
        "--pico-timing-threshold-mv",
        type=float,
        default=40.0,
        help="Only Pico-TDC results at this timing threshold are used. Default: 40 mV.",
    )
    parser.add_argument(
        "--pico-acquisition-mode",
        default=None,
        help="Optional exact AcquisitionMode filter.",
    )
    parser.add_argument(
        "--allow-legacy-pico-summary",
        action="store_true",
    )

    parser.add_argument(
        "--scope-min-led-threshold-mv",
        type=float,
        default=20.0,
        help="Minimum LED threshold included in oscilloscope adaptive scan. Default: 20 mV.",
    )
    parser.add_argument(
        "--scope-led-outlier-z",
        type=float,
        default=4.0,
        help="Median/MAD LED outlier cut, applied once before bootstrap. Default: 4.",
    )

    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=300,
        help="Number of event bootstrap resamples for each final Pico/scope point.",
    )
    parser.add_argument(
        "--bootstrap-min-success-fraction",
        type=float,
        default=0.9,
        help="Require at least this fraction of bootstrap Gaussian fits to succeed.",
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


def _split_selected(
    selected: np.ndarray,
    *,
    blind_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(np.asarray(selected, dtype=bool))
    if indices.size < 5:
        raise RuntimeError("Too few selected events to build comparison split")

    rng = np.random.default_rng(seed)
    order = rng.permutation(indices)

    n_blind = max(1, int(round(indices.size * blind_fraction)))
    n_blind = min(n_blind, indices.size - 2)
    blind = np.sort(order[:n_blind])

    development = order[n_blind:]
    n_validation = max(1, int(round(development.size * validation_fraction)))
    n_validation = min(n_validation, development.size - 1)
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
    """Bootstrap an already-selected timing cohort.

    IMPORTANT: no outlier rejection occurs here. `delta_fs` is frozen before
    this function is called. Each bootstrap sample is formed by sampling events
    with replacement, then applying only the common Gaussian fit.
    """
    values = np.asarray(delta_fs, dtype=np.int64).reshape(-1)
    if values.size < int(fit_config.get("min_events", 10)):
        raise RuntimeError(
            f"Too few selected events for bootstrap: {values.size}"
        )
    if n_bootstrap < 2:
        raise ValueError("--bootstrap-samples must be >= 2")

    rng = np.random.default_rng(seed)
    ctrs: list[float] = []

    for _ in range(int(n_bootstrap)):
        sample_indices = rng.integers(0, values.size, size=values.size)
        sample = values[sample_indices]
        fit = fit_delta_times_integer_fs(
            sample,
            method=method,
            parameter=float(parameter),
            n_total=int(values.size),
            n_selected=int(values.size),
            config=fit_config,
        )
        if fit.success and np.isfinite(fit.ctr_ps):
            ctrs.append(float(fit.ctr_ps))

    minimum_success = math.ceil(float(min_success_fraction) * int(n_bootstrap))
    if len(ctrs) < minimum_success:
        raise RuntimeError(
            f"Only {len(ctrs)}/{n_bootstrap} bootstrap fits succeeded for {method}; "
            f"need at least {minimum_success}."
        )

    ctr_array = np.asarray(ctrs, dtype=np.float64)
    return {
        "ctr_mean_ps": float(np.mean(ctr_array)),
        # User-requested bootstrap error: std among bootstrap CTR estimates.
        "ctr_bootstrap_std_ps": float(np.std(ctr_array, ddof=1)),
        "bootstrap_samples_requested": int(n_bootstrap),
        "bootstrap_samples_successful": int(ctr_array.size),
        "bootstrap_ctr_min_ps": float(np.min(ctr_array)),
        "bootstrap_ctr_max_ps": float(np.max(ctr_array)),
    }


def _scope_delta_after_rejection(
    features: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    threshold_index: int,
    *,
    outlier_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    a = np.asarray(features["t_led_a_fs"], dtype=np.int64)[
        selected_indices, threshold_index
    ]
    b = np.asarray(features["t_led_b_fs"], dtype=np.int64)[
        selected_indices, threshold_index
    ]

    valid = (a != INVALID_TIME_FS) & (b != INVALID_TIME_FS)
    delta_fs = a[valid] - b[valid]

    if delta_fs.size < 3:
        raise RuntimeError("Too few valid oscilloscope LED pairs")

    # Apply exactly once. Bootstrap is later performed from filtered_delta_fs.
    rejection = robust_mad_filter(
        delta_fs.astype(np.float64) / FS_PER_PS,
        enabled=True,
        zscore_limit=float(outlier_z),
    )
    filtered_delta_fs = delta_fs[rejection.mask]

    return filtered_delta_fs, {
        "n_before_outlier": int(delta_fs.size),
        "n_after_outlier": int(filtered_delta_fs.size),
        "n_outlier_rejected": int(rejection.rejected),
        "outlier_center_ps": float(rejection.center),
        "outlier_sigma_ps": float(rejection.robust_sigma),
        "outlier_limit_ps": float(rejection.max_distance),
    }


def _fit_scope_threshold(
    features: dict[str, np.ndarray],
    selected_indices: np.ndarray,
    threshold_index: int,
    threshold_mV: float,
    fit_config: dict[str, Any],
    *,
    method: str,
    outlier_z: float,
):
    delta_fs, rejection_meta = _scope_delta_after_rejection(
        features,
        selected_indices,
        threshold_index,
        outlier_z=outlier_z,
    )
    fit = fit_delta_times_integer_fs(
        delta_fs,
        method=method,
        parameter=float(threshold_mV),
        n_total=int(features["event_id"].size),
        n_selected=int(delta_fs.size),
        config=fit_config,
    )
    return fit, delta_fs, rejection_meta


def _scope_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    cfg = config_copy(load_config(args.scope_config))
    roots = sorted(args.scope_root_folder.glob(args.root_pattern))
    if not roots:
        raise RuntimeError(
            f"No ROOT files match {args.root_pattern!r} in {args.scope_root_folder}"
        )

    cache_root = args.output / "scope_feature_cache"
    fit_plot_root = args.output / "scope_led_fits"
    cache_root.mkdir(parents=True, exist_ok=True)
    fit_plot_root.mkdir(parents=True, exist_ok=True)

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
        threshold_mask = all_thresholds >= float(args.scope_min_led_threshold_mv)
        threshold_indices = np.flatnonzero(threshold_mask)
        thresholds = all_thresholds[threshold_mask]

        if thresholds.size == 0:
            raise RuntimeError(
                f"No LED thresholds >= {args.scope_min_led_threshold_mv:g} mV "
                f"for {root_file.name}; available={all_thresholds.tolist()}"
            )

        score_indices = (
            blind if args.threshold_selection_stage == "blind" else validation
        )

        score_fits = []
        # Threshold selection uses the nominal common fit after one 4σ rejection.
        # We do NOT bootstrap every candidate threshold.
        for original_index, threshold_mV in zip(threshold_indices, thresholds):
            fit, _, _ = _fit_scope_threshold(
                features,
                score_indices,
                int(original_index),
                float(threshold_mV),
                cfg["fit"],
                method=f"LED selection ({args.threshold_selection_stage})",
                outlier_z=args.scope_led_outlier_z,
            )
            score_fits.append(fit)

        chosen = choose_best(score_fits)
        if chosen is None:
            raise RuntimeError(f"No successful LED threshold fit for {root_file.name}")

        chosen_local = int(np.argmin(np.abs(thresholds - chosen.parameter)))
        chosen_original_index = int(threshold_indices[chosen_local])

        final_indices = blind
        final_fit, final_delta_fs, rejection_meta = _fit_scope_threshold(
            features,
            final_indices,
            chosen_original_index,
            float(chosen.parameter),
            cfg["fit"],
            method="Oscilloscope LED blind",
            outlier_z=args.scope_led_outlier_z,
        )
        if not final_fit.success:
            raise RuntimeError(
                f"Final scope LED fit failed for {root_file.name}: {final_fit.message}"
            )

        bootstrap = _bootstrap_ctr(
            final_delta_fs,
            fit_config=cfg["fit"],
            method="Oscilloscope LED bootstrap",
            parameter=float(chosen.parameter),
            n_bootstrap=args.bootstrap_samples,
            seed=_stable_seed(
                args.seed, "scope-bootstrap", root_file.name, chosen.parameter
            ),
            min_success_fraction=args.bootstrap_min_success_fraction,
        )

        plot_gaussian_fit(
            final_fit,
            fit_plot_root / f"{root_file.stem}.png",
            dpi=int(cfg.get("plot", {}).get("dpi", 180)),
            title=(
                f"{root_file.stem} · adaptive LED · "
                f"{chosen.parameter:g} mV · blind"
            ),
        )

        row = {
            "source_file": root_file.name,
            "voltage_V": float(voltage),
            "threshold_selection_stage": args.threshold_selection_stage,
            "threshold_mV": float(chosen.parameter),
            "min_threshold_mV": float(args.scope_min_led_threshold_mv),
            "nominal_ctr_ps": float(final_fit.ctr_ps),
            **bootstrap,
            **rejection_meta,
            "fit_metric": "common_bin_integrated_gaussian_all_events",
        }
        rows.append(row)

        print(
            f"[scope][{root_file.name}] V={voltage:g} V | "
            f"best LED threshold={chosen.parameter:g} mV | "
            f"4σ retained={rejection_meta['n_after_outlier']}/"
            f"{rejection_meta['n_before_outlier']} | "
            f"bootstrap CTR={bootstrap['ctr_mean_ps']:.2f} ± "
            f"{bootstrap['ctr_bootstrap_std_ps']:.2f} ps"
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
        valid_metric = "common_bin_integrated_gaussian_all_events"
        for row in rows:
            if str(row.get("fit_metric", "")) != valid_metric:
                raise RuntimeError(
                    "Pico summary contains rows not generated with the unified fitter."
                )

    candidates: dict[float, list[dict[str, str]]] = {}

    for row in rows:
        if (
            args.pico_acquisition_mode is not None
            and row.get("AcquisitionMode") != args.pico_acquisition_mode
        ):
            continue
        try:
            voltage = float(row["Voltage"])
            threshold = float(row["T_th"])
            ctr = float(row["CTR_ps"])
        except (KeyError, TypeError, ValueError):
            continue

        if not np.isclose(
            threshold,
            float(args.pico_timing_threshold_mv),
            rtol=0.0,
            atol=1e-9,
        ):
            continue
        if not np.isfinite(ctr):
            continue

        candidates.setdefault(voltage, []).append(row)

    if not candidates:
        raise RuntimeError(
            f"No Pico rows at T_th={args.pico_timing_threshold_mv:g} mV"
        )

    # User-requested duplicate policy:
    # same voltage + requested T_th -> keep the nominally best (smallest) CTR run.
    chosen: list[dict[str, str]] = []
    for voltage, voltage_rows in sorted(candidates.items()):
        best = min(voltage_rows, key=lambda row: float(row["CTR_ps"]))
        chosen.append(best)
        if len(voltage_rows) > 1:
            print(
                f"[pico][{voltage:g} V] {len(voltage_rows)} rows at "
                f"T_th={args.pico_timing_threshold_mv:g} mV -> "
                f"using best run {best.get('run_id', '')} "
                f"(nominal CTR={float(best['CTR_ps']):.2f} ps)"
            )
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
    raise RuntimeError(
        f"Multiple selection tables found under {run_dir / 'csv'}: "
        + ", ".join(str(path.name) for path in matches)
    )


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
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(state)
    if not found:
        raise RuntimeError(f"Could not find toa_lsb_ps in {state_path}")

    rounded = {round(value, 12) for value in found}
    if len(rounded) > 1:
        raise RuntimeError(
            f"Inconsistent toa_lsb_ps values in {state_path}: {sorted(rounded)}"
        )
    return float(found[0])


def _load_pico_selected_delta_fs(
    summary_path: Path,
    run_id: str,
    *,
    outlier_z: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load event-level Pico timing differences for the chosen run.

    The Janus selection table already contains time_a_lsb/time_b_lsb plus the
    duration/alignment masks. We reconstruct the same final selected cohort,
    apply the same 4σ LED rejection ONCE, then freeze the resulting events for
    bootstrap.
    """
    output_root = summary_path.resolve().parent
    run_dir = output_root / "analysis" / run_id
    selection_path = _find_selection_csv(run_dir)
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise RuntimeError(f"Missing Janus state file: {state_path}")

    toa_lsb_ps = _find_toa_lsb_ps(state_path)

    with selection_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f"Empty selection table: {selection_path}")

    selected_delta_lsb: list[int] = []
    for row in rows:
        try:
            duration_selected = bool(int(row["duration_selected"]))
            alignment_selected = bool(int(row["alignment_selected"]))
            if not (duration_selected and alignment_selected):
                continue
            time_a = int(row["time_a_lsb"])
            time_b = int(row["time_b_lsb"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Malformed Janus selection row in {selection_path}"
            ) from exc

        # Measurements.timing_lsb is time_b - time_a in the Janus pipeline.
        selected_delta_lsb.append(time_b - time_a)

    if len(selected_delta_lsb) < 3:
        raise RuntimeError(f"Too few selected Pico events for {run_id}")

    delta_ps = np.asarray(selected_delta_lsb, dtype=np.float64) * toa_lsb_ps

    # Same 4σ rejection as the final Janus fit. Applied once before bootstrap.
    rejection = robust_mad_filter(
        delta_ps,
        enabled=True,
        zscore_limit=float(outlier_z),
    )
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


def _pico_rows(
    args: argparse.Namespace,
    fit_config: dict[str, Any],
) -> list[dict[str, Any]]:
    selected_rows = _choose_best_pico_rows(args)
    output: list[dict[str, Any]] = []

    # Janus uses the same 4σ value introduced in the unified analysis.
    pico_outlier_z = 4.0

    for row in selected_rows:
        run_id = str(row["run_id"])
        voltage = float(row["Voltage"])
        threshold = float(row["T_th"])

        delta_fs, rejection_meta = _load_pico_selected_delta_fs(
            args.pico_summary,
            run_id,
            outlier_z=pico_outlier_z,
        )

        bootstrap = _bootstrap_ctr(
            delta_fs,
            fit_config=fit_config,
            method="Pico-TDC bootstrap",
            parameter=threshold,
            n_bootstrap=args.bootstrap_samples,
            seed=_stable_seed(args.seed, "pico-bootstrap", run_id),
            min_success_fraction=args.bootstrap_min_success_fraction,
        )

        output_row = {
            "run_id": run_id,
            "voltage_V": voltage,
            "timing_threshold_mV": threshold,
            "nominal_ctr_ps": float(row["CTR_ps"]),
            **bootstrap,
            **rejection_meta,
            "fit_metric": "common_bin_integrated_gaussian_all_events",
        }
        output.append(output_row)

        print(
            f"[pico][{run_id}] V={voltage:g} V | T_th={threshold:g} mV | "
            f"4σ retained={rejection_meta['n_after_outlier']}/"
            f"{rejection_meta['n_before_outlier']} | "
            f"bootstrap CTR={bootstrap['ctr_mean_ps']:.2f} ± "
            f"{bootstrap['ctr_bootstrap_std_ps']:.2f} ps"
        )

    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_rows(
    pico: list[dict[str, Any]],
    scope: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scope_by_voltage = {float(row["voltage_V"]): row for row in scope}
    rows: list[dict[str, Any]] = []

    for p in pico:
        voltage = float(p["voltage_V"])
        s = scope_by_voltage.get(voltage)
        if s is None:
            continue

        pico_mean = float(p["ctr_mean_ps"])
        scope_mean = float(s["ctr_mean_ps"])
        pico_err = float(p["ctr_bootstrap_std_ps"])
        scope_err = float(s["ctr_bootstrap_std_ps"])

        # Propagation is done ONLY here, between the two independent bootstrap
        # estimates, exactly as requested.
        combined_error = math.sqrt(pico_err**2 + scope_err**2)
        z = (
            (pico_mean - scope_mean) / combined_error
            if combined_error > 0.0
            else float("nan")
        )

        rows.append(
            {
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
            }
        )
    return rows


def _plot_comparison(
    paired: list[dict[str, Any]],
    path: Path,
    *,
    pico_threshold_mV: float,
    threshold_selection_stage: str,
) -> None:
    if not paired:
        raise RuntimeError(
            "No voltages are shared by Pico and oscilloscope results"
        )

    ordered = sorted(
        paired,
        key=lambda row: float(row["voltage_V"]),
    )

    voltage = np.asarray(
        [float(row["voltage_V"]) for row in ordered]
    )
    pico_mean = np.asarray(
        [float(row["pico_ctr_mean_ps"]) for row in ordered]
    )
    scope_mean = np.asarray(
        [float(row["scope_ctr_mean_ps"]) for row in ordered]
    )
    z = np.asarray(
        [
            float(row["difference_over_combined_error"])
            for row in ordered
        ]
    )

    fig, (ax, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(11.0, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.0]},
    )

    # Mean CTR only — no error bars
    ax.plot(
        voltage,
        pico_mean,
        marker="o",
        linestyle="none",
        markersize=10,
        label=f"Pico-TDC · T_th={pico_threshold_mV:g} mV",
    )

    ax.plot(
        voltage,
        scope_mean,
        marker="s",
        linestyle="none",
        markersize=10,
        label="Oscilloscope adaptive LED",
    )

    ax.set_ylabel(
        "CTR [ps]",
        fontsize=18,
    )
    ax.tick_params(
        axis="both",
        labelsize=15,
    )
    ax.grid(alpha=0.3)

    ax.legend(
        loc="lower left",
        fontsize=15,
    )

    residual_ax.axhline(
        0.0,
        linewidth=1.2,
    )
    residual_ax.axhline(
        1.0,
        linewidth=1.2,
        linestyle="--",
    )
    residual_ax.axhline(
        -1.0,
        linewidth=1.2,
        linestyle="--",
    )

    residual_ax.plot(
        voltage,
        z,
        marker="+",
        linestyle="none",
        markersize=14,
        markeredgewidth=2.0,
    )

    residual_ax.set_xlabel(
        "Bias voltage [V]",
        fontsize=18,
    )

    residual_ax.set_ylabel(
        r"Difference in sigma",
        fontsize=17,
    )

    residual_ax.tick_params(
        axis="both",
        labelsize=15,
    )

    residual_ax.grid(alpha=0.3)

    fig.suptitle(
        f"Pico-TDC vs oscilloscope",
        fontsize=20,
    )

    fig.tight_layout()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.bootstrap_samples < 2:
        raise ValueError("--bootstrap-samples must be >= 2")
    if not 0.0 < args.bootstrap_min_success_fraction <= 1.0:
        raise ValueError("--bootstrap-min-success-fraction must be in (0, 1]")
    if args.scope_led_outlier_z <= 0:
        raise ValueError("--scope-led-outlier-z must be positive")
    if args.scope_min_led_threshold_mv < 0:
        raise ValueError("--scope-min-led-threshold-mv must be non-negative")

    args.output.mkdir(parents=True, exist_ok=True)

    scope_cfg = config_copy(load_config(args.scope_config))

    scope = _scope_rows(args)
    pico = _pico_rows(args, scope_cfg["fit"])
    paired = _paired_rows(pico, scope)

    _write_csv(args.output / "oscilloscope_adaptive_led.csv", scope)
    _write_csv(args.output / "pico_tdc.csv", pico)
    _write_csv(args.output / "paired_comparison.csv", paired)

    _plot_comparison(
        paired,
        args.output / "ctr_vs_voltage_bootstrap.png",
        pico_threshold_mV=args.pico_timing_threshold_mv,
        threshold_selection_stage=args.threshold_selection_stage,
    )

    metadata = {
        "pico_timing_threshold_mV": args.pico_timing_threshold_mv,
        "pico_duplicate_policy": (
            "For duplicate rows at the same voltage and requested timing threshold, "
            "choose the run with minimum nominal CTR_ps, then bootstrap that run."
        ),
        "scope_min_led_threshold_mV": args.scope_min_led_threshold_mv,
        "scope_led_outlier_z": args.scope_led_outlier_z,
        "outlier_bootstrap_policy": (
            "4-sigma robust MAD rejection is applied exactly once to the final "
            "selected event cohort; bootstrap resamples only from retained events."
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_error_definition": (
            "sample standard deviation (ddof=1) of successful bootstrap CTR fits"
        ),
        "plotted_central_value": "mean of successful bootstrap CTR fits",
        "residual_definition": (
            "(mean_pico - mean_scope) / "
            "sqrt(std_bootstrap_pico^2 + std_bootstrap_scope^2)"
        ),
        "threshold_selection_stage": args.threshold_selection_stage,
        "seed": args.seed,
    }

    with (args.output / "comparison_metadata.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Wrote comparison to: {args.output}")
    print(f"Plot: {args.output / 'ctr_vs_voltage_bootstrap.png'}")


if __name__ == "__main__":
    main()
