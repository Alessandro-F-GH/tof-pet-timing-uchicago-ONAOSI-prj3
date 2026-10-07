from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import numpy as np


METRICS = ("ctr", "rmse")


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def sample_std(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.std(values, ddof=1)) if values.size >= 2 else 0.0


def tolerance_for_fold(value, completed_folds: int) -> float | None:
    if isinstance(value, dict):
        raw = value.get(str(int(completed_folds)))
        return None if raw is None else float(raw)
    return float(value)


def _metric_key(metric: str) -> str:
    metric = str(metric).lower()
    if metric not in METRICS:
        raise ValueError(f"selection metric must be one of {METRICS}")
    return f"{metric}_ps"


def aggregate_candidate(fold_rows: list[dict], total_folds: int, *, pruned: bool, decision=None) -> dict:
    ordered = sorted(fold_rows, key=lambda row: int(row["fold_id"]))
    ctr = [float(row["ctr_ps"]) for row in ordered]
    rmse = [float(row["rmse_ps"]) for row in ordered]
    result = {
        "completed_folds": len(ordered),
        "total_folds": int(total_folds),
        "evaluated_fold_ids": [int(row["fold_id"]) for row in ordered],
        "ctr_mean_ps": float(np.mean(ctr)) if ctr else float("nan"),
        "ctr_std_ps": sample_std(ctr),
        "rmse_mean_ps": float(np.mean(rmse)) if rmse else float("nan"),
        "rmse_std_ps": sample_std(rmse),
        "pruned": bool(pruned),
        "partial_statistics": bool(pruned or len(ordered) < int(total_folds)),
        "pruning_fold": None,
        "pruning_reason": None,
        "pruning_tolerance_ps": None,
        "candidate_vs_incumbent_degradation_ps": None,
        "candidate_vs_led_degradation_ps": None,
        "incumbent_candidate_id": None,
    }
    if decision is not None and decision.pruned:
        result.update({
            "pruning_fold": int(decision.fold_id),
            "pruning_reason": str(decision.reason),
            "pruning_tolerance_ps": float(decision.tolerance_ps),
            "candidate_vs_incumbent_degradation_ps": decision.incumbent_degradation_ps,
            "candidate_vs_led_degradation_ps": decision.led_degradation_ps,
            "incumbent_candidate_id": decision.incumbent_candidate_id,
        })
    return result


@dataclass(frozen=True)
class PruningDecision:
    pruned: bool
    fold_id: int | None = None
    reason: str | None = None
    tolerance_ps: float | None = None
    incumbent_degradation_ps: float | None = None
    led_degradation_ps: float | None = None
    incumbent_candidate_id: str | None = None


def pruning_decision(
    candidate_rows: list[dict],
    *,
    metric: str,
    pruning: dict,
    total_folds: int,
    incumbent_candidate_id: str | None = None,
    incumbent_rows: list[dict] | None = None,
    force_complete: bool = False,
) -> PruningDecision:
    if force_complete or not bool(pruning.get("enabled", False)):
        return PruningDecision(False)
    completed = len(candidate_rows)
    if completed < int(pruning["min_folds_before_prune"]) or completed >= int(total_folds):
        return PruningDecision(False)

    key = _metric_key(metric)
    fold_ids = [int(row["fold_id"]) for row in candidate_rows]
    candidate_mean = float(np.mean([float(row[key]) for row in candidate_rows]))
    led_mean = float(np.mean([float(row[f"led_{key}"]) for row in candidate_rows]))
    led_degradation = candidate_mean - led_mean

    if bool(pruning.get("prune_if_worse_than_led", False)):
        tolerance = tolerance_for_fold(pruning["led_max_degradation_ps"], completed)
        if tolerance is not None and led_degradation > tolerance:
            return PruningDecision(
                True,
                fold_ids[-1],
                "led",
                float(tolerance),
                None,
                float(led_degradation),
                incumbent_candidate_id,
            )

    if incumbent_candidate_id is None or incumbent_rows is None:
        return PruningDecision(False)

    by_fold = {int(row["fold_id"]): row for row in incumbent_rows}
    if any(fold_id not in by_fold for fold_id in fold_ids):
        raise RuntimeError("incumbent is missing a fold completed by the candidate")
    incumbent_mean = float(np.mean([float(by_fold[fold_id][key]) for fold_id in fold_ids]))
    degradation = candidate_mean - incumbent_mean
    tolerance = tolerance_for_fold(pruning["max_degradation_ps"], completed)
    if tolerance is not None and degradation > tolerance:
        return PruningDecision(
            True,
            fold_ids[-1],
            "incumbent",
            float(tolerance),
            float(degradation),
            float(led_degradation),
            incumbent_candidate_id,
        )
    return PruningDecision(False)


def best_complete_candidate(candidate_rows: list[dict], metric: str) -> dict | None:
    key = f"{str(metric).lower()}_mean_ps"
    eligible = [
        row for row in candidate_rows
        if not _as_bool(row.get("pruned", False))
        and int(row.get("completed_folds", 0)) == int(row.get("total_folds", -1))
        and math.isfinite(float(row.get(key, float("nan"))))
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda row: (float(row[key]), str(row["candidate_id"])))


def evaluate_candidate(
    *,
    candidate_id: str,
    parameters: dict,
    folds,
    evaluate_fold: Callable,
    metric: str,
    pruning: dict,
    existing_fold_rows: list[dict] | None = None,
    persisted_candidate: dict | None = None,
    incumbent_candidate_id: str | None = None,
    incumbent_rows: list[dict] | None = None,
    force_complete: bool = False,
    persist_fold: Callable[[dict], None] | None = None,
) -> tuple[dict, list[dict]]:
    total_folds = len(folds)
    if persisted_candidate is not None and _as_bool(persisted_candidate.get("pruned", False)):
        rows = sorted(existing_fold_rows or [], key=lambda row: int(row["fold_id"]))
        return dict(persisted_candidate), rows

    rows_by_id = {int(row["fold_id"]): dict(row) for row in (existing_fold_rows or [])}
    decision = None
    for fold in folds:
        fold_id = int(fold.fold_id)
        if fold_id not in rows_by_id:
            row = dict(evaluate_fold(parameters, fold))
            row.update({"candidate_id": candidate_id, "fold_id": fold_id})
            rows_by_id[fold_id] = row
            if persist_fold is not None:
                persist_fold(row)

        ordered = [rows_by_id[index] for index in sorted(rows_by_id) if index <= fold_id]
        decision = pruning_decision(
            ordered,
            metric=metric,
            pruning=pruning,
            total_folds=total_folds,
            incumbent_candidate_id=incumbent_candidate_id,
            incumbent_rows=incumbent_rows,
            force_complete=force_complete,
        )
        if decision.pruned:
            break

    rows = [rows_by_id[index] for index in sorted(rows_by_id)]
    summary = aggregate_candidate(rows, total_folds, pruned=bool(decision and decision.pruned), decision=decision)
    summary.update({"candidate_id": candidate_id, "parameters": dict(parameters)})
    return summary, rows
