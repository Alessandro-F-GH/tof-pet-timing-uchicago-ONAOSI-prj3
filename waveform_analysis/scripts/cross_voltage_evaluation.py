from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.models import get_model
from waveform_analysis.ml_pipeline.models.cnn import CNNArtifact, SharedScorerCNN
from waveform_analysis.ml_pipeline.models.cnn_2d import CNN2DArtifact, JointPairCNN2D
from waveform_analysis.ml_pipeline.models.difference_knn import DifferenceKNNArtifact
from waveform_analysis.ml_pipeline.models.difference_shapelet import (
    DifferenceShapeletArtifact,
    DifferenceShapeletRegressor,
)
from waveform_analysis.ml_pipeline.models.linear_svr import LinearSVRArtifact
from waveform_analysis.ml_pipeline.reporting import LABELS, MODEL_ORDER
from waveform_analysis.ml_pipeline.sample_mask import apply_sample_mask
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.stats import ctr_estimate
from waveform_analysis.ml_pipeline.view import corrected_timing_residual, model_target, waveform_view


@dataclass
class LoadedModel:
    name: str
    artifact: Any
    sample_mask: np.ndarray | None
    output_max_abs_ps: float | None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate already-trained per-voltage models on the blind/test sets of every other voltage "
            "in the same completed study."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed per-voltage study directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Destination directory (default: <run-dir>/cross_voltage)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Models to evaluate (default: every trained ML model found in the study)",
    )
    parser.add_argument(
        "--diagonal-tolerance-ps",
        type=float,
        default=0.1,
        help="Maximum allowed difference between reloaded-model and stored diagonal CTR (default: 0.1 ps)",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dataset_voltage(dataset: str, rows: list[dict[str, str]]) -> float:
    values = [
        _float(row.get("voltage_V"))
        for row in rows
        if row.get("dataset") == dataset and np.isfinite(_float(row.get("voltage_V")))
    ]
    return float(np.median(values)) if values else float("nan")


def _ordered_models(names: set[str]) -> list[str]:
    ordered = [name for name in MODEL_ORDER if name in names and name not in {"led", "cfd"}]
    ordered.extend(sorted(names - set(ordered) - {"led", "cfd"}))
    return ordered


def _study_layout(run: Path, requested_models: list[str] | None):
    manifest = _read_json(run / "manifest.json")
    if bool(manifest.get("concatenate_datasets", False)):
        raise ValueError("Cross-voltage evaluation requires an ordinary per-voltage study, not a concatenated study")

    rows = _read_results(run / "csv" / "results.csv")
    test_rows = [row for row in rows if row.get("stage") == "test"]
    datasets = sorted(
        {row["dataset"] for row in test_rows if row.get("dataset")},
        key=lambda name: (_dataset_voltage(name, rows), name),
    )
    if len(datasets) < 2:
        raise ValueError("Cross-voltage evaluation requires at least two voltage datasets")

    trained = {
        child.name
        for dataset in datasets
        for child in (run / "models" / dataset).iterdir()
        if child.is_dir()
    }
    available = {
        name
        for name in trained
        if all((run / "models" / dataset / name).is_dir() for dataset in datasets)
    }
    if requested_models:
        missing = sorted(set(requested_models) - available)
        if missing:
            raise ValueError(f"Requested models are not trained for every voltage: {missing}")
        models = list(requested_models)
    else:
        models = _ordered_models(available)
    if not models:
        raise ValueError("No common trained ML models were found across all voltage datasets")

    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    return manifest, rows, datasets, models, mode


def _torch_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _sample_mask(model_dir: Path, metadata: dict[str, Any]) -> np.ndarray | None:
    name = metadata.get("sample_mask_file")
    if not name:
        return None
    path = model_dir / str(name)
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved sample mask: {path}")
    return np.asarray(np.load(path), dtype=bool).reshape(-1)


def _output_limit(metadata: dict[str, Any], manifest: dict[str, Any]) -> float | None:
    training = metadata.get("training") or {}
    value = training.get("output_max_abs_ps")
    if value is None:
        value = ((manifest.get("config") or {}).get("ml_output") or {}).get("max_abs_ps")
    if value is None:
        return None
    return float(value)


def _load_model(run: Path, dataset: str, model_name: str, manifest: dict[str, Any]) -> LoadedModel:
    model_dir = run / "models" / dataset / model_name
    metadata = _read_json(model_dir / "metadata.json")
    sample_mask = _sample_mask(model_dir, metadata)
    model_config = ((manifest.get("config") or {}).get("models") or {}).get(model_name) or {}
    training_metadata = dict(metadata.get("training") or {})

    if model_name == "linear_svr":
        artifact = LinearSVRArtifact(joblib.load(model_dir / "model.joblib"), training_metadata)
    elif model_name == "difference_knn":
        artifact = DifferenceKNNArtifact(joblib.load(model_dir / "model.joblib"), training_metadata)
    elif model_name == "cnn":
        checkpoint = _torch_checkpoint(model_dir / "model.pt")
        model = SharedScorerCNN(model_config.get("architecture", {}))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        artifact = CNNArtifact(model, "cpu", dict(checkpoint.get("metadata") or training_metadata))
    elif model_name == "cnn_2d":
        checkpoint = _torch_checkpoint(model_dir / "model.pt")
        model = JointPairCNN2D(model_config.get("architecture", {}))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        artifact = CNN2DArtifact(model, "cpu", dict(checkpoint.get("metadata") or training_metadata))
    elif model_name == "difference_shapelet":
        checkpoint = _torch_checkpoint(model_dir / "model.pt")
        archive = model_dir / "learned_shapelets.npz"
        if not archive.is_file():
            raise FileNotFoundError(f"Missing learned shapelet archive: {archive}")
        initial_shapelets = []
        starts = []
        with np.load(archive) as data:
            input_time_ps = np.asarray(data.get("input_time_ps", []), dtype=np.float64)
            group = 0
            while f"group_{group}_shapelets" in data:
                initial_shapelets.append(np.asarray(data[f"group_{group}_shapelets"], dtype=np.float32))
                starts.append(np.asarray(data[f"group_{group}_starts_samples"], dtype=np.int64))
                group += 1
        architecture = model_config.get("architecture", {})
        model = DifferenceShapeletRegressor(
            initial_shapelets,
            starts,
            dense_units=[int(value) for value in architecture.get("dense_units", [32, 16])],
            dropout=float(architecture.get("dropout", 0.05)),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        artifact = DifferenceShapeletArtifact(
            model=model,
            input_time_ps=input_time_ps,
            device="cpu",
            metadata=dict(checkpoint.get("metadata") or training_metadata),
        )
    else:
        raise ValueError(
            f"Cross-voltage loader does not yet support model {model_name!r}. "
            "Add a loader matching that model's persisted artifact format."
        )

    return LoadedModel(
        name=model_name,
        artifact=artifact,
        sample_mask=sample_mask,
        output_max_abs_ps=_output_limit(metadata, manifest),
    )


def _predict(loaded: LoadedModel, dataset, mode: str) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(dataset.test, dtype=np.int64)
    pair = waveform_view(dataset, mode, indices).materialize()
    if loaded.sample_mask is not None:
        if loaded.sample_mask.size != pair.shape[-1]:
            raise ValueError(
                f"{loaded.name}: training-voltage sample mask has {loaded.sample_mask.size} samples "
                f"but prediction-voltage waveform has {pair.shape[-1]}"
            )
        pair = apply_sample_mask(pair, loaded.sample_mask)
    prediction = np.asarray(get_model(loaded.name).predict(loaded.artifact, pair), dtype=np.float64).reshape(-1)
    if loaded.output_max_abs_ps is not None:
        prediction = np.clip(
            prediction,
            -float(loaded.output_max_abs_ps),
            float(loaded.output_max_abs_ps),
        )
    target = model_target(dataset, mode)[indices]
    return prediction, corrected_timing_residual(target, prediction)


def _stored_metric(rows, dataset: str, model: str) -> tuple[float, float]:
    row = next(
        (
            item
            for item in rows
            if item.get("dataset") == dataset
            and item.get("method") == model
            and item.get("stage") == "test"
        ),
        None,
    )
    if row is None:
        raise ValueError(f"Missing stored blind-test metric for {dataset}/{model}")
    return _float(row.get("ctr_ps")), _float(row.get("ctr_uncertainty_ps"))


def _check_prepared_compatibility(prepared: dict[str, Any], mode: str) -> None:
    family = str(mode).split("_to_", 1)[0]
    reference_name = next(iter(prepared))
    reference = prepared[reference_name]
    reference_time = (
        np.asarray(reference.energy_time_ps, dtype=np.float64)
        if family == "energy"
        else np.asarray(reference.timing_time_ps, dtype=np.float64)
    )
    for name, dataset in prepared.items():
        time = (
            np.asarray(dataset.energy_time_ps, dtype=np.float64)
            if family == "energy"
            else np.asarray(dataset.timing_time_ps, dtype=np.float64)
        )
        if time.shape != reference_time.shape or not np.allclose(time, reference_time, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Prepared waveform grids are incompatible between {reference_name} and {name}; "
                "cross-voltage application requires the same prepared time axis."
            )


def _evaluate(
    run: Path,
    manifest: dict[str, Any],
    rows: list[dict[str, str]],
    datasets: list[str],
    models: list[str],
    mode: str,
    diagonal_tolerance_ps: float,
):
    fit_config = dict((manifest.get("config") or {}).get("fit") or {})
    base_seed = int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0))
    prepared = {
        name: load_prepared_dataset(manifest["datasets"][name]["prepared_dir"])
        for name in datasets
    }
    _check_prepared_compatibility(prepared, mode)
    records = []

    for model_name in models:
        trained = {
            train_dataset: _load_model(run, train_dataset, model_name, manifest)
            for train_dataset in datasets
        }
        for train_dataset in datasets:
            for predict_dataset in datasets:
                _prediction, residual = _predict(trained[train_dataset], prepared[predict_dataset], mode)
                finite = residual[np.isfinite(residual)]
                if finite.size != residual.size:
                    raise ValueError(
                        f"{model_name}: non-finite residuals for train={train_dataset}, predict={predict_dataset}"
                    )
                estimate = ctr_estimate(
                    finite,
                    fit_config,
                    seed=semantic_seed(
                        base_seed,
                        "cross_voltage",
                        model_name,
                        train_dataset,
                        predict_dataset,
                    ),
                    bootstrap=True,
                )
                recomputed_ctr = float(estimate.ctr_ps)
                recomputed_uncertainty = float(estimate.ctr_error_ps)

                diagonal = train_dataset == predict_dataset
                if diagonal:
                    stored_ctr, stored_uncertainty = _stored_metric(rows, predict_dataset, model_name)
                    difference = abs(recomputed_ctr - stored_ctr)
                    if not np.isfinite(stored_ctr) or difference > float(diagonal_tolerance_ps):
                        raise RuntimeError(
                            f"Diagonal consistency check failed for {model_name}/{train_dataset}: "
                            f"reloaded CTR={recomputed_ctr:.6g} ps, stored CTR={stored_ctr:.6g} ps, "
                            f"|difference|={difference:.6g} ps > tolerance={diagonal_tolerance_ps:g} ps"
                        )
                    ctr = stored_ctr
                    uncertainty = stored_uncertainty
                else:
                    ctr = recomputed_ctr
                    uncertainty = recomputed_uncertainty

                records.append(
                    {
                        "model": model_name,
                        "train_dataset": train_dataset,
                        "train_voltage_V": _dataset_voltage(train_dataset, rows),
                        "predict_dataset": predict_dataset,
                        "predict_voltage_V": _dataset_voltage(predict_dataset, rows),
                        "ctr_ps": ctr,
                        "ctr_uncertainty_ps": uncertainty,
                        "n_test": int(residual.size),
                        "diagonal": diagonal,
                        "recomputed_ctr_ps": recomputed_ctr,
                        "recomputed_ctr_uncertainty_ps": recomputed_uncertainty,
                    }
                )
    return records


def _write_long_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def _measurement(ctr: float, uncertainty: float) -> str:
    if not np.isfinite(ctr):
        return ""
    if not np.isfinite(uncertainty):
        return f"{ctr:.2f}"
    return f"{ctr:.2f} ± {uncertainty:.2f}"


def _matrix(records: list[dict[str, Any]], model: str):
    model_rows = [row for row in records if row["model"] == model]
    train_voltages = sorted({float(row["train_voltage_V"]) for row in model_rows})
    predict_voltages = sorted({float(row["predict_voltage_V"]) for row in model_rows})
    ctr = np.full((len(train_voltages), len(predict_voltages)), np.nan)
    uncertainty = np.full_like(ctr, np.nan)
    lookup = {
        (float(row["train_voltage_V"]), float(row["predict_voltage_V"])): row
        for row in model_rows
    }
    for i, train_voltage in enumerate(train_voltages):
        for j, predict_voltage in enumerate(predict_voltages):
            row = lookup[(train_voltage, predict_voltage)]
            ctr[i, j] = float(row["ctr_ps"])
            uncertainty[i, j] = float(row["ctr_uncertainty_ps"])
    return train_voltages, predict_voltages, ctr, uncertainty


def _write_matrix_csv(path: Path, train_voltages, predict_voltages, ctr, uncertainty) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["V_train \\ V_predict", *[f"{value:g} V" for value in predict_voltages]])
        for i, train_voltage in enumerate(train_voltages):
            writer.writerow(
                [
                    f"{train_voltage:g} V",
                    *[_measurement(ctr[i, j], uncertainty[i, j]) for j in range(len(predict_voltages))],
                ]
            )


def _plot_matrix(path: Path, model: str, train_voltages, predict_voltages, ctr, uncertainty) -> None:
    import matplotlib.pyplot as plt

    fig_width = max(6.6, 1.25 * len(predict_voltages) + 2.5)
    fig_height = max(5.5, 1.05 * len(train_voltages) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(ctr, aspect="equal", interpolation="nearest")
    ax.set_xticks(np.arange(len(predict_voltages)))
    ax.set_yticks(np.arange(len(train_voltages)))
    ax.set_xticklabels([f"{value:g} V" for value in predict_voltages])
    ax.set_yticklabels([f"{value:g} V" for value in train_voltages])
    ax.set_xlabel("Prediction / blind-test voltage")
    ax.set_ylabel("Training voltage")

    midpoint = 0.5 * (float(np.nanmin(ctr)) + float(np.nanmax(ctr)))
    for i in range(len(train_voltages)):
        for j in range(len(predict_voltages)):
            value = ctr[i, j]
            error = uncertainty[i, j]
            text_color = "white" if np.isfinite(value) and value > midpoint else "black"
            ax.text(
                j,
                i,
                _measurement(value, error),
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold" if i == j else "normal",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("CTR [ps]")
    ax.set_title(f"{LABELS.get(model, model)} · cross-voltage blind-test CTR")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run(
    run_dir: Path,
    output_dir: Path | None = None,
    requested_models: list[str] | None = None,
    diagonal_tolerance_ps: float = 0.1,
) -> list[Path]:
    run_path = run_dir.resolve()
    output = (output_dir or (run_path / "cross_voltage")).resolve()
    manifest, rows, datasets, models, mode = _study_layout(run_path, requested_models)
    records = _evaluate(
        run_path,
        manifest,
        rows,
        datasets,
        models,
        mode,
        diagonal_tolerance_ps,
    )

    generated = []
    long_csv = output / "cross_voltage_results.csv"
    _write_long_csv(long_csv, records)
    generated.append(long_csv)

    for model in models:
        train_voltages, predict_voltages, ctr, uncertainty = _matrix(records, model)
        matrix_csv = output / f"cross_voltage_{model}.csv"
        matrix_pdf = output / f"cross_voltage_{model}.pdf"
        _write_matrix_csv(matrix_csv, train_voltages, predict_voltages, ctr, uncertainty)
        _plot_matrix(matrix_pdf, model, train_voltages, predict_voltages, ctr, uncertainty)
        generated.extend([matrix_csv, matrix_pdf])
    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in run(
        args.run_dir,
        args.output_dir,
        args.models,
        args.diagonal_tolerance_ps,
    ):
        print(path)


if __name__ == "__main__":
    main()
