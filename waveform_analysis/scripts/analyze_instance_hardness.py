from __future__ import annotations

import argparse
import copy
import csv
import json
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
    apply_sample_mask_to_time,
    dataset_training_sample_mask,
)
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.view import model_target, waveform_view


DEFAULT_MODELS = ("linear_svr", "difference_knn", "difference_shapelet")
EXCLUDED_MODELS = {"cnn", "cnn_2d"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate regression instance hardness on study training events using out-of-fold "
            "predictions from the repository's non-CNN difference-signal regressors."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: <run-dir>/instance_hardness)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Model pool (default: linear_svr difference_knn difference_shapelet)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional dataset names to analyze (default: every non-concatenated dataset in the study)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Out-of-fold splits over the training population (default: 3 for efficiency)",
    )
    parser.add_argument(
        "--inner-validation-fraction",
        type=float,
        default=0.15,
        help=(
            "Fraction of each outer-fold training population reserved only for early stopping of "
            "models that require validation, currently difference_shapelet (default: 0.15)"
        ),
    )
    parser.add_argument(
        "--plot-format",
        choices=("pdf", "png"),
        default="pdf",
        help="Plot file format (default: pdf)",
    )
    parser.add_argument(
        "--shapelet-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device used by difference_shapelet fits (default: auto)",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _difference_pair(pair: np.ndarray) -> np.ndarray:
    """Represent the input only through d(t)=s1(t)-s2(t), while preserving the model API.

    Repository difference models internally evaluate channel_0 - channel_1.
    Supplying [d(t), 0] therefore guarantees that every model receives exactly
    the same difference signal and has no access to the individual waveforms.
    """
    values = np.asarray(pair, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected waveform pair [event, detector=2, sample], got {values.shape}")
    difference = np.ascontiguousarray(values[:, 0, :] - values[:, 1, :], dtype=np.float32)
    return np.stack([difference, np.zeros_like(difference)], axis=1)


def _instance_hardness(target: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, float]:
    """Torquette et al. regression IH using squared error and target signal power.

    IH_i = 1 - mean_j exp(-(y_i-yhat_ji)^2 / gamma)
    gamma = mean_i y_i^2
    """
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    pred = np.asarray(predictions, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != y.size:
        raise ValueError(f"Predictions must be [model, event], got {pred.shape} for {y.size} targets")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("Instance hardness requires finite targets and predictions")
    gamma = float(np.mean(y**2))
    if not np.isfinite(gamma) or gamma <= np.finfo(np.float64).eps:
        raise ValueError("Target signal power is zero; regression instance hardness is undefined")
    normalized_squared_error = (pred - y[None, :]) ** 2 / gamma
    similarity = np.exp(-normalized_squared_error)
    return 1.0 - np.mean(similarity, axis=0), gamma


def _rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean(error**2)))


def _mae(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def _model_config(manifest: dict[str, Any], model_name: str) -> dict[str, Any]:
    configured = ((manifest.get("config") or {}).get("models") or {}).get(model_name)
    if isinstance(configured, dict):
        return copy.deepcopy(configured)

    model_path = Path(__file__).resolve().parents[1] / "config" / "model_spaces" / f"{model_name}.json"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"{model_name} is not configured in the study and no default model space exists at {model_path}"
        )
    config = _read_json(model_path)
    if str(config.get("model", model_name)) != model_name:
        raise ValueError(f"{model_path} does not declare model={model_name!r}")
    return config


def _runtime_config(
    model_config: dict[str, Any],
    *,
    input_time_ps: np.ndarray,
    output_limit_ps: float | None,
    shapelet_device: str,
) -> dict[str, Any]:
    runtime = copy.deepcopy(model_config)
    runtime["_input_time_ps"] = np.asarray(input_time_ps, dtype=np.float64)
    runtime["_prediction_max_abs_ps"] = output_limit_ps
    if "training" in runtime and isinstance(runtime["training"], dict):
        if str(runtime.get("model", "")) == "difference_shapelet":
            runtime["training"]["device"] = str(shapelet_device)
    return runtime


def _predict(spec, artifact, pair: np.ndarray, output_limit_ps: float | None) -> np.ndarray:
    values = np.asarray(spec.predict(artifact, pair), dtype=np.float64).reshape(-1)
    if output_limit_ps is not None:
        values = np.clip(values, -float(output_limit_ps), float(output_limit_ps))
    return values


def _fit(
    model_name: str,
    parameters: dict[str, Any],
    model_config: dict[str, Any],
    pair: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
    input_time_ps: np.ndarray,
    output_limit_ps: float | None,
    validation_pair: np.ndarray | None,
    validation_target: np.ndarray | None,
    shapelet_device: str,
):
    spec = get_model(model_name)
    runtime = _runtime_config(
        model_config,
        input_time_ps=input_time_ps,
        output_limit_ps=output_limit_ps,
        shapelet_device=shapelet_device,
    )
    return spec.fit(
        parameters,
        np.asarray(pair, dtype=np.float32),
        np.asarray(target, dtype=np.float64),
        seed=int(seed),
        config=runtime,
        validation_x=None if validation_pair is None else np.asarray(validation_pair, dtype=np.float32),
        validation_target=None
        if validation_target is None
        else np.asarray(validation_target, dtype=np.float64),
    )


def _select_parameters(
    run: Path,
    dataset_name: str,
    model_name: str,
    model_config: dict[str, Any],
    train_pair: np.ndarray,
    train_target: np.ndarray,
    validation_pair: np.ndarray,
    validation_target: np.ndarray,
    *,
    seed: int,
    input_time_ps: np.ndarray,
    output_limit_ps: float | None,
    shapelet_device: str,
) -> tuple[dict[str, Any], float]:
    """Select model-only hyperparameters on the study validation split.

    Target-range candidates are intentionally not used: this analysis trains every
    regressor on the complete training population.
    """
    del run
    spec = get_model(model_name)
    candidates = list(spec.candidates(model_config))
    if not candidates:
        raise ValueError(f"{model_name}: empty candidate space")

    best: tuple[float, str, dict[str, Any]] | None = None
    errors = []
    for index, raw in enumerate(candidates):
        parameters = dict(raw)
        parameters.pop("target_abs_max_ps", None)
        try:
            artifact = _fit(
                model_name,
                parameters,
                model_config,
                train_pair,
                train_target,
                seed=semantic_seed(seed, dataset_name, model_name, "parameter_selection", index),
                input_time_ps=input_time_ps,
                output_limit_ps=output_limit_ps,
                validation_pair=validation_pair,
                validation_target=validation_target,
                shapelet_device=shapelet_device,
            )
            prediction = _predict(spec, artifact, validation_pair, output_limit_ps)
            score = _rmse(validation_target, prediction)
            key = (score, json.dumps(parameters, sort_keys=True), parameters)
            if best is None or key[:2] < best[:2]:
                best = key
        except Exception as exc:
            errors.append(f"{parameters}: {type(exc).__name__}: {exc}")

    if best is None:
        raise RuntimeError(f"{dataset_name}/{model_name}: every parameter candidate failed: {'; '.join(errors)}")
    return best[2], float(best[0])


def _inner_split(indices: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64)
    if values.size < 4:
        raise ValueError("Need at least four outer-training events for an inner early-stopping split")
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("--inner-validation-fraction must lie in (0, 0.5)")
    shuffled = np.random.default_rng(int(seed)).permutation(values)
    n_validation = max(1, min(values.size - 1, int(round(values.size * float(fraction)))))
    return np.sort(shuffled[n_validation:]), np.sort(shuffled[:n_validation])


def _oof_predictions(
    dataset_name: str,
    model_name: str,
    parameters: dict[str, Any],
    model_config: dict[str, Any],
    pair: np.ndarray,
    target: np.ndarray,
    *,
    folds: int,
    seed: int,
    input_time_ps: np.ndarray,
    output_limit_ps: float | None,
    inner_validation_fraction: float,
    shapelet_device: str,
) -> np.ndarray:
    n = int(target.size)
    if folds < 2 or folds > n:
        raise ValueError(f"--folds must lie in [2, {n}], got {folds}")

    spec = get_model(model_name)
    predictions = np.full(n, np.nan, dtype=np.float64)
    splitter = KFold(n_splits=int(folds), shuffle=True, random_state=int(seed))

    for fold, (outer_train, outer_test) in enumerate(splitter.split(np.arange(n)), start=1):
        fit_positions = np.asarray(outer_train, dtype=np.int64)
        early_positions: np.ndarray | None = None

        if model_name == "difference_shapelet":
            fit_positions, early_positions = _inner_split(
                fit_positions,
                inner_validation_fraction,
                semantic_seed(seed, dataset_name, model_name, "inner_validation", fold),
            )

        artifact = _fit(
            model_name,
            parameters,
            model_config,
            pair[fit_positions],
            target[fit_positions],
            seed=semantic_seed(seed, dataset_name, model_name, "oof", fold),
            input_time_ps=input_time_ps,
            output_limit_ps=output_limit_ps,
            validation_pair=None if early_positions is None else pair[early_positions],
            validation_target=None if early_positions is None else target[early_positions],
            shapelet_device=shapelet_device,
        )
        predictions[outer_test] = _predict(spec, artifact, pair[outer_test], output_limit_ps)

    if not np.all(np.isfinite(predictions)):
        raise RuntimeError(f"{dataset_name}/{model_name}: OOF prediction did not cover every training event")
    return predictions


def _voltage(dataset) -> float:
    values = np.asarray(dataset.bias_voltage_V, dtype=np.float64)
    training = np.asarray(dataset.training, dtype=np.int64)
    finite = values[training][np.isfinite(values[training])]
    return float(np.median(finite)) if finite.size else float("nan")


def _quantile_trend(x: np.ndarray, y: np.ndarray, bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 4:
        return np.asarray([]), np.asarray([])
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, min(bins, x.size) + 1)))
    if edges.size < 3:
        return np.asarray([]), np.asarray([])
    centers, medians = [], []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (x >= left) & (x <= right if index == edges.size - 2 else x < right)
        if np.any(mask):
            centers.append(float(np.median(x[mask])))
            medians.append(float(np.median(y[mask])))
    return np.asarray(centers), np.asarray(medians)


def _hardness_plot(path: Path, x: np.ndarray, hardness: np.ndarray, xlabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.scatter(x, hardness, s=10, alpha=0.28, label="training event")
    trend_x, trend_y = _quantile_trend(x, hardness)
    if trend_x.size:
        ax.plot(trend_x, trend_y, marker="o", linewidth=2.0, label="quantile-bin median")
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


def _dataset_names(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    if bool(manifest.get("concatenate_datasets", False)):
        raise ValueError("Instance-hardness analysis currently requires an ordinary per-voltage study")
    available = list((manifest.get("datasets") or {}).keys())
    if requested:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(f"Unknown dataset(s): {missing}")
        return list(requested)
    return available


def _validate_models(names: list[str]) -> list[str]:
    if not names:
        raise ValueError("At least one model is required")
    forbidden = sorted(set(names) & EXCLUDED_MODELS)
    if forbidden:
        raise ValueError(f"CNN models are intentionally excluded from this analysis: {forbidden}")
    unsupported = sorted(set(names) - set(DEFAULT_MODELS))
    if unsupported:
        raise ValueError(
            f"This script is intentionally restricted to repository difference-signal regressors "
            f"{list(DEFAULT_MODELS)}; unsupported: {unsupported}"
        )
    return list(dict.fromkeys(names))


def analyze_dataset(
    run: Path,
    manifest: dict[str, Any],
    dataset_name: str,
    models: list[str],
    *,
    folds: int,
    inner_validation_fraction: float,
    shapelet_device: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    dataset_info = manifest["datasets"][dataset_name]
    dataset = load_prepared_dataset(dataset_info["prepared_dir"])

    training = np.asarray(dataset.training, dtype=np.int64)
    validation = np.asarray(dataset.validation, dtype=np.int64)
    if training.size < folds:
        raise ValueError(f"{dataset_name}: only {training.size} training events for {folds} folds")
    if validation.size < 1:
        raise ValueError(f"{dataset_name}: validation split is empty")

    sample_mask = dataset_training_sample_mask(dataset, mode)
    train_view = waveform_view(dataset, mode, training)
    validation_view = waveform_view(dataset, mode, validation)
    train_pair = _difference_pair(apply_sample_mask(train_view.materialize(), sample_mask))
    validation_pair = _difference_pair(apply_sample_mask(validation_view.materialize(), sample_mask))
    input_time_ps = apply_sample_mask_to_time(train_view.time_ps, sample_mask)

    all_target = model_target(dataset, mode)
    train_target = np.asarray(all_target[training], dtype=np.float64)
    validation_target = np.asarray(all_target[validation], dtype=np.float64)
    if not np.all(np.isfinite(train_target)) or not np.all(np.isfinite(validation_target)):
        raise ValueError(f"{dataset_name}: target contains non-finite values")

    output_limit_raw = ((manifest.get("config") or {}).get("ml_output") or {}).get("max_abs_ps")
    output_limit = None if output_limit_raw is None else float(output_limit_raw)
    base_seed = int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))

    model_predictions: dict[str, np.ndarray] = {}
    model_summary: list[dict[str, Any]] = []
    for model_name in models:
        model_config = _model_config(manifest, model_name)
        parameters, validation_rmse = _select_parameters(
            run,
            dataset_name,
            model_name,
            model_config,
            train_pair,
            train_target,
            validation_pair,
            validation_target,
            seed=base_seed,
            input_time_ps=input_time_ps,
            output_limit_ps=output_limit,
            shapelet_device=shapelet_device,
        )
        prediction = _oof_predictions(
            dataset_name,
            model_name,
            parameters,
            model_config,
            train_pair,
            train_target,
            folds=folds,
            seed=semantic_seed(base_seed, dataset_name, model_name, "hardness_oof"),
            input_time_ps=input_time_ps,
            output_limit_ps=output_limit,
            inner_validation_fraction=inner_validation_fraction,
            shapelet_device=shapelet_device,
        )
        model_predictions[model_name] = prediction
        model_summary.append(
            {
                "dataset": dataset_name,
                "voltage_V": _voltage(dataset),
                "model": model_name,
                "parameters_json": json.dumps(parameters, sort_keys=True),
                "validation_rmse_ps": validation_rmse,
                "oof_rmse_ps": _rmse(train_target, prediction),
                "oof_mae_ps": _mae(train_target, prediction),
                "folds": int(folds),
                "training_events": int(training.size),
                "input_samples": int(train_pair.shape[-1]),
            }
        )

    prediction_matrix = np.stack([model_predictions[name] for name in models], axis=0)
    hardness, gamma = _instance_hardness(train_target, prediction_matrix)
    ensemble_prediction = np.mean(prediction_matrix, axis=0)
    model_summary.append(
        {
            "dataset": dataset_name,
            "voltage_V": _voltage(dataset),
            "model": "equal_mean_ensemble",
            "parameters_json": json.dumps({"members": models}),
            "validation_rmse_ps": float("nan"),
            "oof_rmse_ps": _rmse(train_target, ensemble_prediction),
            "oof_mae_ps": _mae(train_target, ensemble_prediction),
            "folds": int(folds),
            "training_events": int(training.size),
            "input_samples": int(train_pair.shape[-1]),
        }
    )

    event_index = np.asarray(dataset.event_index, dtype=np.int64)[training]
    voltage = _voltage(dataset)
    event_rows = []
    for position in range(training.size):
        row: dict[str, Any] = {
            "dataset": dataset_name,
            "voltage_V": voltage,
            "training_position": int(position),
            "dataset_row": int(training[position]),
            "event_index": int(event_index[position]),
            "target_ps": float(train_target[position]),
            "abs_target_ps": float(abs(train_target[position])),
            "instance_hardness": float(hardness[position]),
            "gamma_target_power_ps2": gamma,
            "ensemble_prediction_ps": float(ensemble_prediction[position]),
            "ensemble_abs_error_ps": float(abs(ensemble_prediction[position] - train_target[position])),
        }
        for model_name in models:
            prediction = float(model_predictions[model_name][position])
            squared_error = (prediction - float(train_target[position])) ** 2
            row[f"{model_name}_prediction_ps"] = prediction
            row[f"{model_name}_abs_error_ps"] = math.sqrt(squared_error)
            row[f"{model_name}_ih_similarity"] = math.exp(-squared_error / gamma)
        event_rows.append(row)

    arrays = {
        "target_ps": train_target,
        "instance_hardness": hardness,
        "ensemble_prediction_ps": ensemble_prediction,
        **{f"{name}_prediction_ps": values for name, values in model_predictions.items()},
    }
    return event_rows, model_summary, arrays


def run(
    run_dir: Path,
    output_dir: Path | None = None,
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    folds: int = 3,
    inner_validation_fraction: float = 0.15,
    plot_format: str = "pdf",
    shapelet_device: str = "auto",
) -> list[Path]:
    run_path = run_dir.resolve()
    output = (output_dir or (run_path / "instance_hardness")).resolve()
    manifest = _read_json(run_path / "manifest.json")
    model_pool = _validate_models(list(models or DEFAULT_MODELS))
    dataset_pool = _dataset_names(manifest, datasets)

    all_events: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    generated: list[Path] = []

    for dataset_name in dataset_pool:
        event_rows, summary_rows, arrays = analyze_dataset(
            run_path,
            manifest,
            dataset_name,
            model_pool,
            folds=folds,
            inner_validation_fraction=inner_validation_fraction,
            shapelet_device=shapelet_device,
        )
        all_events.extend(event_rows)
        all_summary.extend(summary_rows)

        event_csv = output / "csv" / f"instance_hardness_{dataset_name}.csv"
        _write_csv(event_csv, event_rows)
        generated.append(event_csv)

        npz_path = output / "arrays" / f"instance_hardness_{dataset_name}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, **arrays)
        generated.append(npz_path)

        voltage = float(event_rows[0]["voltage_V"]) if event_rows else float("nan")
        label = f"{voltage:g} V" if np.isfinite(voltage) else dataset_name
        target_plot = output / "plots" / f"instance_hardness_vs_target_{dataset_name}.{plot_format}"
        abs_target_plot = output / "plots" / f"instance_hardness_vs_abs_target_{dataset_name}.{plot_format}"
        _hardness_plot(
            target_plot,
            arrays["target_ps"],
            arrays["instance_hardness"],
            r"Training target $y_{\mathrm{target}}$ [ps]",
            f"Regression instance hardness · {label}",
        )
        _hardness_plot(
            abs_target_plot,
            np.abs(arrays["target_ps"]),
            arrays["instance_hardness"],
            r"$|y_{\mathrm{target}}|$ [ps]",
            f"Regression instance hardness vs target magnitude · {label}",
        )
        generated.extend([target_plot, abs_target_plot])

    events_csv = output / "instance_hardness_all_events.csv"
    summary_csv = output / "model_oof_summary.csv"
    _write_csv(events_csv, all_events)
    _write_csv(summary_csv, all_summary)
    generated.extend([events_csv, summary_csv])
    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in run(
        args.run_dir,
        args.output_dir,
        models=args.models,
        datasets=args.datasets,
        folds=args.folds,
        inner_validation_fraction=args.inner_validation_fraction,
        plot_format=args.plot_format,
        shapelet_device=args.shapelet_device,
    ):
        print(path)


if __name__ == "__main__":
    main()
