from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .common import load_artifact_array
from .plot_style import (
    DOUBLE_COLUMN,
    LABELS,
    clean_axis,
    panel_label,
    paper_context,
    save_figure,
)


def _model_output(run: Path, dataset: str, model: str, stage: str) -> np.ndarray | None:
    return load_artifact_array(run, dataset, model, stage, "model_output_ps")


def _residual(run: Path, dataset: str, model: str, stage: str) -> np.ndarray | None:
    return load_artifact_array(run, dataset, model, stage, "residuals_ps")


def _target_from_artifacts(
    run: Path,
    dataset: str,
    model: str,
    stage: str,
) -> np.ndarray | None:
    prediction = _model_output(run, dataset, model, stage)
    residual = _residual(run, dataset, model, stage)
    if prediction is None or residual is None:
        return None
    if prediction.shape != residual.shape:
        raise ValueError(
            f"{dataset}/{model}/{stage}: prediction/residual shape mismatch "
            f"{prediction.shape} != {residual.shape}"
        )
    return prediction + residual


def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"Correlation arrays have different shapes: {x.shape} != {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    n = int(np.count_nonzero(finite))
    if n < 2:
        return float("nan"), n
    x = x[finite]
    y = y[finite]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def _display_limits(arrays: list[np.ndarray]) -> tuple[float, float]:
    finite = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            finite.append(values)
    if not finite:
        return -1.0, 1.0
    pooled = np.concatenate(finite)
    low, high = np.quantile(pooled, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high):
        return -1.0, 1.0
    if high <= low:
        margin = max(abs(float(low)) * 0.05, 1.0)
        return float(low - margin), float(high + margin)
    margin = 0.06 * float(high - low)
    return float(low - margin), float(high + margin)


def plot_prediction_vs_target(
    output: Path,
    run: Path,
    manifest: dict[str, Any],
    dataset: str,
    model: str,
    label: str,
    paths: list[Path],
) -> None:
    import matplotlib.pyplot as plt

    stages: list[tuple[str, np.ndarray, np.ndarray]] = []
    for stage in ("train", "test"):
        prediction = _model_output(run, dataset, model, stage)
        target = _target_from_artifacts(run, dataset, model, stage)
        if prediction is None or target is None:
            continue
        stages.append((stage, target, prediction))
    if not stages:
        return

    limits = _display_limits([
        array for _, target, prediction in stages for array in (target, prediction)
    ])
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=DOUBLE_COLUMN if len(stages) > 1 else (3.45, 3.2),
        squeeze=False,
    )
    for panel_index, (ax, (stage, target, prediction)) in enumerate(zip(axes[0], stages)):
        finite = np.isfinite(target) & np.isfinite(prediction)
        x = target[finite]
        y = prediction[finite]
        correlation, n = _pearson(target, prediction)
        ax.scatter(x, y, s=5, alpha=0.24, rasterized=True, color="#0072B2")
        ax.plot(limits, limits, color="#7F7F7F", ls="--", lw=0.9)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Target [ps]")
        ax.set_ylabel("Prediction [ps]")
        clean_axis(ax, grid="both")
        panel_label(ax, f"({chr(97 + panel_index)})")
        ax.text(
            0.98, 0.04,
            f"$r$ = {correlation:.3f}\n$n$ = {n}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )
    fig.tight_layout(w_pad=0.8)
    target_path = save_figure(fig, output / f"prediction_vs_target_{dataset}.pdf")
    plt.close(fig)
    paths.append(target_path)


def _stage_output_matrix(
    run: Path,
    dataset: str,
    models: list[str],
    stage: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    available: list[tuple[str, np.ndarray]] = []
    for model in models:
        values = _model_output(run, dataset, model, stage)
        if values is not None:
            available.append((model, values))
    if len(available) < 2:
        return [], np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.int64)
    lengths = {values.size for _, values in available}
    if len(lengths) != 1:
        detail = {model: int(values.size) for model, values in available}
        raise ValueError(f"{dataset}/{stage}: model-output lengths differ: {detail}")
    names = [model for model, _ in available]
    n_models = len(names)
    corr = np.full((n_models, n_models), np.nan, dtype=np.float64)
    counts = np.zeros((n_models, n_models), dtype=np.int64)
    for i, (_, left) in enumerate(available):
        for j, (_, right) in enumerate(available):
            value, n = _pearson(left, right)
            corr[i, j] = value
            counts[i, j] = n
    return names, corr, counts


def _model_output_correlation_rows(
    run: Path,
    dataset: str,
    models: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in ("train", "test"):
        names, matrix, counts = _stage_output_matrix(run, dataset, models, stage)
        for i, left in enumerate(names):
            for j in range(i):
                rows.append(
                    {
                        "dataset": dataset,
                        "stage": stage,
                        "model_a": names[j],
                        "model_b": left,
                        "pearson_r": float(matrix[i, j]),
                        "n": int(counts[i, j]),
                    }
                )
    return rows


def plot_model_output_correlations(
    plot_output: Path,
    run: Path,
    dataset: str,
    models: list[str],
    labels: dict[str, str],
    paths: list[Path],
) -> None:
    import matplotlib.pyplot as plt

    stages = []
    for stage in ("train", "test"):
        names, matrix, counts = _stage_output_matrix(run, dataset, models, stage)
        if len(names) > 2:
            stages.append((stage, names, matrix, counts))
    if not stages:
        return

    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=DOUBLE_COLUMN if len(stages) > 1 else (3.45, 3.1),
        squeeze=False,
        constrained_layout=True,
    )
    image = None
    for panel_index, (ax, (_stage, names, matrix, _counts)) in enumerate(zip(axes[0], stages)):
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        shown = np.ma.array(matrix, mask=mask)
        image = ax.imshow(
            shown,
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
            interpolation="nearest",
        )
        display_names = [labels.get(name, name) for name in names]
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(display_names, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticklabels(display_names)
        ax.set_aspect("equal", adjustable="box")
        for i in range(len(names)):
            for j in range(i + 1):
                value = matrix[i, j]
                if np.isfinite(value):
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) >= 0.55 else "black",
                    )
        panel_label(ax, f"({chr(97 + panel_index)})")
        ax.grid(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
        cbar.set_label("Correlation [–]")
    target_path = save_figure(
        fig,
        plot_output / f"model_output_correlation_{dataset}.pdf",
    )
    plt.close(fig)
    paths.append(target_path)


def make_model_output_reports(
    run_dir: str | Path,
    plot_output_dir: str | Path,
    *,
    labels: dict[str, str] | None = None,
) -> list[Path]:
    run = Path(run_dir).resolve()
    plot_root = Path(plot_output_dir).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    labels = dict(labels or LABELS)
    paths: list[Path] = []
    correlation_rows: list[dict[str, Any]] = []

    with paper_context():
        datasets = list((manifest.get("datasets") or {}).keys())
        for dataset in datasets:
            artifact_dir = run / "artifacts" / dataset
            discovered_models = {
                path.name[: -len("_test_model_output_ps.npy")]
                for path in artifact_dir.glob("*_test_model_output_ps.npy")
                if path.name.endswith("_test_model_output_ps.npy")
            }
            discovered_models.update(
                path.name[: -len("_train_model_output_ps.npy")]
                for path in artifact_dir.glob("*_train_model_output_ps.npy")
                if path.name.endswith("_train_model_output_ps.npy")
            )
            models = [name for name in labels if name in discovered_models]
            models.extend(sorted(discovered_models - set(models)))
            for model in models:
                plot_prediction_vs_target(
                    plot_root / model,
                    run,
                    manifest,
                    dataset,
                    model,
                    labels.get(model, model),
                    paths,
                )
            if len(models) > 1:
                correlation_rows.extend(
                    _model_output_correlation_rows(run, dataset, models)
                )
            if len(models) > 2:
                plot_model_output_correlations(
                    plot_root / "correlations",
                    run,
                    dataset,
                    models,
                    labels,
                    paths,
                )

    correlation_path = run / "csv" / "model_output_correlations.csv"
    if correlation_rows:
        correlation_path.parent.mkdir(parents=True, exist_ok=True)
        with correlation_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["dataset", "stage", "model_a", "model_b", "pearson_r", "n"],
            )
            writer.writeheader()
            writer.writerows(correlation_rows)
    elif correlation_path.is_file():
        correlation_path.unlink()
    return paths
