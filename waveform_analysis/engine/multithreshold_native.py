"""Native-clock MT pickoff with Control-only threshold selection.

Operate on persisted filtered native energy windows, NOT LED-aligned ML arrays.
Match Development and Blind event IDs to the frozen ML prepared populations.
The output is a directly measured native time difference, not a re-interpolated
LED correction. All candidate selection and offset calibration are Control-only.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from waveform_analysis.data.named_cache import native_alias, cache_root_from_config
from waveform_analysis.data.preprocessing import load_preprocessed
from waveform_analysis.data.timing_adapter import led_grid
from waveform_analysis.engine.multithreshold_led import linear_pickoff
from waveform_analysis.reporting.stats import metric_values

CANDIDATE_LEVELS_MV = (10., 20., 30., 50., 100.)


def threshold_combinations(levels=CANDIDATE_LEVELS_MV):
    levels = tuple(float(v) for v in levels)
    if len(levels) != 5 or not np.all(np.isfinite(levels)) or not all(
            a < b for a, b in zip(levels, levels[1:])):
        raise ValueError("Expected five strictly increasing finite candidate thresholds")
    return tuple(combinations(levels, 3))


def load_native_energy(config, role):
    if config["mode"] != "energy_to_energy":
        raise ValueError("Native MT requires energy_to_energy")
    cache_root = cache_root_from_config(config)
    alias = native_alias(cache_root, config[role], config["control"], config["mode"])
    if not (alias / "manifest.json").is_file():
        raise FileNotFoundError(f"Native filtered energy cache unavailable: {alias}")
    native = load_preprocessed(
        alias, config[role]["root_file"], None, config[role],
        config["preprocessing"], config["mode"],
    )
    if native.manifest.get("time_reference") != "absolute_native_acquisition_time":
        raise ValueError("Native MT requires an absolute acquisition time reference")
    if native.energy_windows_mV is None or native.energy_window_start_time_s is None:
        raise ValueError("Native energy waveform/time arrays are missing")
    return native


def native_positions(native, event_ids):
    requested = np.asarray(event_ids, np.int64).reshape(-1)
    available = np.asarray(native.event_index, np.int64).reshape(-1)
    if (np.unique(requested).size != requested.size or
            np.unique(available).size != available.size):
        raise ValueError("Duplicate native or prepared event IDs")
    order = np.argsort(available)
    positions = np.searchsorted(available[order], requested)
    present = positions < available.size
    if present.any():
        present[present] &= available[order[positions[present]]] == requested[present]
    if not np.all(present):
        raise ValueError(f"Native cache misses {np.count_nonzero(~present)} ML events")
    return order[positions]


def candidate_pickoffs(native, indices, groups, *, baseline_window_ns):
    """Return per-event/detector crossing and intercepts in absolute ps.

    All five thresholds are computed together using the original LED kernel
    and baseline definition, then reused for all 10 triples.
    """
    rows = np.asarray(indices, np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= native.n_events):
        raise ValueError("Invalid native event positions")
    levels = tuple(sorted({float(v) for group in groups for v in group}))
    crossings = led_grid(
        native, "energy", rows, np.asarray(levels, float),
        baseline_window_ns=baseline_window_ns,
    )
    picks = np.full((rows.size, len(groups)), np.nan, dtype=np.float64)
    intercepts = np.full((rows.size, 2, len(groups)), np.nan, dtype=np.float64)
    slopes = np.full_like(intercepts, np.nan)
    fit_rmse = np.full_like(intercepts, np.nan)
    for j, group in enumerate(groups):
        chosen = [levels.index(float(level)) for level in group]
        t0, slope, spread = linear_pickoff(crossings[:, :, chosen], group)
        picks[:, j] = t0[:, 0] - t0[:, 1]
        intercepts[:, :, j] = t0
        slopes[:, :, j] = slope
        fit_rmse[:, :, j] = spread
    return picks, {
        "crossing_levels_mV": np.asarray(levels, np.float64),
        "crossing_times_ps": crossings,
        "intercept_ps": intercepts,
        "slope_ps_per_mV": slopes,
        "fit_rmse_ps": fit_rmse,
    }


def select_control_thresholds(native, fit_cfg, *, levels, minimum_coverage, seed,
                              baseline_window_ns, true_tof_ps):
    """Freeze one 3-threshold group by minimum Control CTR, then high coverage.

    A candidate with too few valid crossings is ineligible. No Development or
    Blind target is used to choose thresholds or the global correction offset.
    """
    groups = threshold_combinations(levels)
    positions = np.arange(native.n_events, dtype=np.int64)
    picks, diagnostics = candidate_pickoffs(
        native, positions, groups, baseline_window_ns=baseline_window_ns)
    scan = []
    for j, group in enumerate(groups):
        residual = picks[:, j] - float(true_tof_ps)
        valid = np.isfinite(residual)
        coverage = float(valid.mean()) if valid.size else 0.
        eligible = int(valid.sum()) >= 5 and coverage >= minimum_coverage
        ctr = (metric_values(residual[valid] - np.mean(residual[valid]),
                             fit_cfg, seed=seed)["ctr_ps"] if eligible else float("inf"))
        scan.append({
            "thresholds_mV": list(group),
            "ctr_ps": float(ctr) if np.isfinite(ctr) else None,
            "coverage": coverage, "valid_events": int(valid.sum()),
            "total_events": int(residual.size),
            "eligible": bool(eligible),
            "offset_ps": float(np.mean(residual[valid])) if eligible else None,
        })
    viable = [r for r in scan if r["eligible"] and r["ctr_ps"] is not None]
    if not viable:
        raise ValueError("No MT candidate meets minimum Control coverage")
    best = min(viable, key=lambda r: (r["ctr_ps"], -r["coverage"], r["thresholds_mV"]))
    return best, scan


def frozen_native_pickoff(native, prepared, thresholds, *, baseline_window_ns):
    positions = native_positions(native, prepared.event_index)
    picks, diag = candidate_pickoffs(
        native, positions, [thresholds], baseline_window_ns=baseline_window_ns)
    return picks[:, 0], diag, positions
