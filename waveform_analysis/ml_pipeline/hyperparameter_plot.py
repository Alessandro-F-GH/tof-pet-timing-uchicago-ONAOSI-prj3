from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _varied(candidates):
    keys = []
    for candidate in candidates.values():
        for key in candidate:
            if key not in keys:
                keys.append(key)
    return [
        key
        for key in keys
        if len({json.dumps(candidate.get(key), sort_keys=True) for candidate in candidates.values()}) > 1
    ]


def _numeric(values):
    return all(
        isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value))
        for value in values
    )


def plot_hyperparameter_validation(
    results,
    candidates,
    output_path,
    logger=None,
    *,
    metric="rmse_ps",
    metric_label="RMSE",
):
    varied = _varied(candidates)
    if not varied:
        return None
    rows = [row for row in results if row.get("phase") == "hyperparameter_validation"]
    if not rows:
        return None

    scores = {}
    for candidate_id in candidates:
        match = next(
            (
                row
                for row in rows
                if row.get("candidate_id") == candidate_id
                and str(row.get(metric, "")) not in ("", "nan")
            ),
            None,
        )
        if match is not None:
            value = float(match[metric])
            if np.isfinite(value):
                scores[candidate_id] = value
    if not scores:
        return None

    numeric = [
        key
        for key in varied
        if _numeric([candidates[candidate_id].get(key) for candidate_id in candidates])
    ]
    if numeric:
        counts = {
            key: len({candidates[candidate_id].get(key) for candidate_id in candidates})
            for key in numeric
        }
        xkey = sorted(numeric, key=lambda key: (-counts[key], varied.index(key)))[0]
    else:
        xkey = varied[0]

    series_keys = [key for key in varied if key != xkey]
    groups = {}
    for candidate_id, score in scores.items():
        candidate = candidates[candidate_id]
        label = (
            ", ".join(f"{key}={candidate.get(key)}" for key in series_keys)
            if series_keys
            else "candidates"
        )
        groups.setdefault(label, []).append((candidate.get(xkey), score, candidate_id))

    fig, ax = plt.subplots()
    for label, values in groups.items():
        values = sorted(
            values,
            key=lambda value: (
                float(value[0]) if isinstance(value[0], (int, float)) else str(value[0]),
                value[2],
            ),
        )
        ax.plot(
            [value[0] for value in values],
            [value[1] for value in values],
            marker="o",
            label=label if series_keys else None,
        )

    ax.set_xlabel(xkey)
    ax.set_ylabel(f"Fixed-validation {metric_label} [ps]")
    if series_keys:
        ax.legend()

    xs = [candidates[candidate_id].get(xkey) for candidate_id in scores]
    if _numeric(xs):
        values = np.asarray(xs, float)
        if np.all(values > 0):
            ax.set_xscale("log")
        elif logger:
            logger.warning(
                "Hyperparameter %s contains zero/negative values; using linear x scale",
                xkey,
            )

    if logger:
        logger.info(
            "Hyperparameter plot uses fixed-validation RMSE for one result per candidate; no CTR selection and no replica averaging"
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
