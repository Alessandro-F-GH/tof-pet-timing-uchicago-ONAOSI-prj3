from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import KFold

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.models import get_model
from waveform_analysis.ml_pipeline.sample_mask import (
    apply_sample_mask,
    dataset_training_sample_mask,
)
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.view import model_target, waveform_view


MODEL_SETTINGS: dict[str, dict[str, Any]] = {
    "linear_svr": {
        "kind": "repository",
        "repository_model": "linear_svr",
        "parameters": {"C": 0.1, "epsilon_ps": 10.0},
        "config": {
            "loss": "epsilon_insensitive",
            "tolerance": 0.01,
            "max_iterations": 10000,
            "dual": "auto",
        },
    },
    "difference_knn_k2": {
        "kind": "repository",
        "repository_model": "difference_knn",
        "parameters": {"n_neighbors": 2, "weights": "distance"},
        "config": {"n_jobs": -1},
    },
    "difference_knn_k50": {
        "kind": "repository",
        "repository_model": "difference_knn",
        "parameters": {"n_neighbors": 50, "weights": "distance"},
        "config": {"n_jobs": -1},
    },
    "minirocket": {
        "kind": "minirocket",
        "parameters": {
            "n_kernels": 10000,
            "max_dilations_per_kernel": 32,
            "n_jobs": -1,
        },
        "config": {},
    },
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate regression instance hardness from prepared waveform datasets using "
            "fixed Linear-SVR, two k-NN scales, and MiniROCKET regressors on d(t)=s1(t)-s2(t)."
        )
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        nargs="+",
        required=True,
        help="One or more prepared dataset directories containing manifest.json and splits.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("instance_hardness"),
        help="Output root (default: ./instance_hardness)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Training-only out-of-fold splits (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260912,
        help="Deterministic OOF seed (default: 20260912)",
    )
    parser.add_argument(
        "--plot-format",
        choices=("pdf", "png"),
        default="pdf",
        help="Plot file format (default: pdf)",
    )
    return parser


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _difference_pair(pair: np.ndarray) -> np.ndarray:
    """Expose only d(t)=s1(t)-s2(t) while preserving the repository model API."""
    values = np.asarray(pair, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected [event, detector=2, sample], got {values.shape}")
    difference = np.ascontiguousarray(values[:, 0, :] - values[:, 1, :], dtype=np.float32)
    return np.stack([difference, np.zeros_like(difference)], axis=1)


def _instance_hardness(target: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, float]:
    """Regression IH from a pool of regressors.

    IH_i = 1 - mean_j exp(-(y_i-yhat_ji)^2 / gamma)
    gamma = mean_i y_i^2
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != y.size:
        raise ValueError(f"Predictions must be [model, event], got {pred.shape}")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("Instance hardness requires finite targets and predictions")

    gamma = float(np.mean(y**2))
    if not np.isfinite(gamma) or gamma <= np.finfo(np.float64).eps:
        raise ValueError("Target signal power is zero; instance hardness is undefined")

    normalized_squared_error = (pred - y[None, :]) ** 2 / gamma
    return 1.0 - np.mean(np.exp(-normalized_squared_error), axis=0), gamma


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.mean(
            np.abs(
                np.asarray(prediction, dtype=np.float64)
                - np.asarray(target, dtype=np.float64)
            )
        )
    )


def _fit_predict_fold(
    model_name: str,
    pair: np.ndarray,
    target: np.ndarray,
    fit_index: np.ndarray,
    predict_index: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    settings = MODEL_SETTINGS[model_name]
    kind = str(settings["kind"])

    if kind == "repository":
        spec = get_model(str(settings["repository_model"]))
        artifact = spec.fit(
            dict(settings["parameters"]),
            np.asarray(pair[fit_index], dtype=np.float32),
            np.asarray(target[fit_index], dtype=np.float64),
            seed=int(seed),
            config=dict(settings["config"]),
            validation_x=None,
            validation_target=None,
        )
        return np.asarray(
            spec.predict(artifact, pair[predict_index]),
            dtype=np.float64,
        ).reshape(-1)

    if kind == "minirocket":
        try:
            from aeon.regression.convolution_based import MiniRocketRegressor
        except ImportError as exc:
            raise ImportError(
                "MiniROCKET hardness analysis requires aeon; install waveform_analysis/requirements.txt"
            ) from exc

        difference = np.asarray(pair[:, 0, :], dtype=np.float32)
        if difference.shape[1] < 9:
            raise ValueError(
                f"MiniROCKET requires at least 9 time samples, got {difference.shape[1]}"
            )
        parameters = dict(settings["parameters"])
        model = MiniRocketRegressor(
            n_kernels=int(parameters["n_kernels"]),
            max_dilations_per_kernel=int(parameters["max_dilations_per_kernel"]),
            random_state=int(seed),
            n_jobs=int(parameters["n_jobs"]),
        )
        model.fit(difference[fit_index], np.asarray(target[fit_index], dtype=np.float64))
        return np.asarray(model.predict(difference[predict_index]), dtype=np.float64).reshape(-1)

    raise ValueError(f"Unknown hardness model kind: {kind!r}")


def _oof_predictions(
    model_name: str,
    pair: np.ndarray,
    target: np.ndarray,
    *,
    folds: int,
    split_seed: int,
    model_seed: int,
) -> np.ndarray:
    n_events = int(target.size)
    if folds < 2 or folds > n_events:
        raise ValueError(f"--folds must lie in [2, {n_events}], got {folds}")

    predictions = np.full(n_events, np.nan, dtype=np.float64)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(split_seed))

    for fold, (fit_index, predict_index) in enumerate(
        splitter.split(np.arange(n_events)),
        start=1,
    ):
        predictions[predict_index] = _fit_predict_fold(
            model_name,
            pair,
            target,
            np.asarray(fit_index, dtype=np.int64),
            np.asarray(predict_index, dtype=np.int64),
            seed=semantic_seed(model_seed, model_name, "oof", fold),
        )

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(f"{model_name}: OOF prediction did not cover every training event")
    return predictions


def _dataset_label(dataset) -> str:
    source = str(dataset.manifest.get("source", "")).strip()
    if source:
        return Path(source).stem
    return dataset.directory.name


def _voltage(dataset) -> float:
    training = np.asarray(dataset.training, dtype=np.int64)
    values = np.asarray(dataset.bias_voltage_V, dtype=np.float64)[training]
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _quantile_trend(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4:
        return np.asarray([]), np.asarray([])

    edges = np.unique(
        np.quantile(x, np.linspace(0.0, 1.0, min(int(bins), x.size) + 1))
    )
    if edges.size < 3:
        return np.asarray([]), np.asarray([])

    centers, medians = [], []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & (x <= right if index == edges.size - 2 else x < right)
        if np.any(mask):
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def _hardness_plot(
    path: Path,
    x: np.ndarray,
    hardness: np.ndarray,
    *,
    xlabel: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(x, hardness, s=10, alpha=0.28, label="training event")
    trend_x, trend_y = _quantile_trend(x, hardness)
    if trend_x.size:
        ax.plot(
            trend_x,
            trend_y,
            marker="o",
            linewidth=2.0,
            label="quantile-bin median",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Regression instance hardness")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyze_prepared_dataset(
    prepared_dir: Path,
    output_dir: Path,
    *,
    folds: int,
    seed: int,
    plot_format: str,
) -> list[Path]:
    dataset = load_prepared_dataset(prepared_dir)
    mode = str(dataset.manifest["mode"])
    training = np.asarray(dataset.training, dtype=np.int64)
    if training.size < folds:
        raise ValueError(
            f"{prepared_dir}: only {training.size} training events for {folds} folds"
        )

    sample_mask = dataset_training_sample_mask(dataset, mode)
    view = waveform_view(dataset, mode, training)
    pair = _difference_pair(apply_sample_mask(view.materialize(), sample_mask))
    target = np.asarray(model_target(dataset, mode)[training], dtype=np.float64)
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{prepared_dir}: training target contains non-finite values")

    label = _dataset_label(dataset)
    voltage = _voltage(dataset)

    predictions = {}
    summaries = []
    for model_name in MODEL_SETTINGS:
        model_prediction = _oof_predictions(
            model_name,
            pair,
            target,
            folds=folds,
            split_seed=semantic_seed(seed, label, "hardness_folds"),
            model_seed=semantic_seed(seed, label, model_name),
        )
        predictions[model_name] = model_prediction
        summaries.append(
            {
                "dataset": label,
                "voltage_V": voltage,
                "mode": mode,
                "model": model_name,
                "parameters_json": str(MODEL_SETTINGS[model_name]["parameters"]),
                "oof_rmse_ps": _rmse(target, model_prediction),
                "oof_mae_ps": _mae(target, model_prediction),
                "folds": int(folds),
                "training_events": int(training.size),
                "input_samples": int(pair.shape[-1]),
            }
        )

    prediction_matrix = np.stack(
        [predictions[name] for name in MODEL_SETTINGS],
        axis=0,
    )
    hardness, gamma = _instance_hardness(target, prediction_matrix)
    ensemble_prediction = np.mean(prediction_matrix, axis=0)
    summaries.append(
        {
            "dataset": label,
            "voltage_V": voltage,
            "mode": mode,
            "model": "equal_mean_ensemble",
            "parameters_json": str({"members": list(MODEL_SETTINGS)}),
            "oof_rmse_ps": _rmse(target, ensemble_prediction),
            "oof_mae_ps": _mae(target, ensemble_prediction),
            "folds": int(folds),
            "training_events": int(training.size),
            "input_samples": int(pair.shape[-1]),
        }
    )

    event_index = np.asarray(dataset.event_index, dtype=np.int64)[training]
    rows = []
    for position in range(training.size):
        row: dict[str, Any] = {
            "dataset": label,
            "voltage_V": voltage,
            "mode": mode,
            "training_position": int(position),
            "dataset_row": int(training[position]),
            "event_index": int(event_index[position]),
            "target_ps": float(target[position]),
            "abs_target_ps": float(abs(target[position])),
            "instance_hardness": float(hardness[position]),
            "gamma_target_power_ps2": gamma,
            "ensemble_prediction_ps": float(ensemble_prediction[position]),
            "ensemble_abs_error_ps": float(
                abs(ensemble_prediction[position] - target[position])
            ),
        }
        for model_name, model_prediction in predictions.items():
            prediction = float(model_prediction[position])
            squared_error = (prediction - float(target[position])) ** 2
            row[f"{model_name}_prediction_ps"] = prediction
            row[f"{model_name}_abs_error_ps"] = math.sqrt(squared_error)
            row[f"{model_name}_ih_similarity"] = math.exp(-squared_error / gamma)
        rows.append(row)

    dataset_output = output_dir / label
    event_csv = dataset_output / "instance_hardness.csv"
    summary_csv = dataset_output / "model_oof_summary.csv"
    arrays_path = dataset_output / "instance_hardness.npz"
    target_plot = dataset_output / f"instance_hardness_vs_target.{plot_format}"
    abs_target_plot = (
        dataset_output / f"instance_hardness_vs_abs_target.{plot_format}"
    )

    _write_csv(event_csv, rows)
    _write_csv(summary_csv, summaries)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        target_ps=target,
        instance_hardness=hardness,
        ensemble_prediction_ps=ensemble_prediction,
        **{
            f"{name}_prediction_ps": values
            for name, values in predictions.items()
        },
    )

    title_suffix = f"{voltage:g} V" if np.isfinite(voltage) else label
    _hardness_plot(
        target_plot,
        target,
        hardness,
        xlabel=r"Training target $y_{\mathrm{target}}$ [ps]",
        title=f"Regression instance hardness · {title_suffix}",
    )
    _hardness_plot(
        abs_target_plot,
        np.abs(target),
        hardness,
        xlabel=r"$|y_{\mathrm{target}}|$ [ps]",
        title=f"Regression instance hardness vs target magnitude · {title_suffix}",
    )

    return [
        event_csv,
        summary_csv,
        arrays_path,
        target_plot,
        abs_target_plot,
    ]


def run(
    prepared_dirs: list[Path],
    output_dir: Path,
    *,
    folds: int = 3,
    seed: int = 20260912,
    plot_format: str = "pdf",
) -> list[Path]:
    output = output_dir.resolve()
    generated = []
    for prepared_dir in prepared_dirs:
        generated.extend(
            analyze_prepared_dataset(
                prepared_dir.resolve(),
                output,
                folds=folds,
                seed=seed,
                plot_format=plot_format,
            )
        )
    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in run(
        args.prepared_dir,
        args.output_dir,
        folds=args.folds,
        seed=args.seed,
        plot_format=args.plot_format,
    ):
        print(path)


if __name__ == "__main__":
    main()
