#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.view import waveform_view

CLASS_NAMES = np.asarray(["left", "center", "right"])
CLASS_IDS = np.asarray([0, 1, 2], dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a BORF-based three-class classifier to predict which Linear-SVR residual peak "
            "an event belongs to. Labels are left / center / right, with the center interval defined "
            "in a JSON configuration. BORF and the classifier are fitted on train only and evaluated "
            "on the frozen blind/test split."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory containing SVR artifacts.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("waveform_analysis/config/borf_svr_peak_classifier.json"),
        help="BORF classifier JSON configuration.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Study dataset name. Repeat for multiple datasets. Default: all datasets with train/test SVR residuals.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/svr_peak_borf_classifier/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _load_borf():
    try:
        from aeon.transformations.collection.dictionary_based import BORF
    except ImportError as exc:
        raise ImportError(
            "BORF requires the aeon package. Install waveform_analysis/requirements.txt "
            "or run `pip install aeon`."
        ) from exc
    return BORF


def _residual_path(run: Path, dataset: str, method: str, stage: str) -> Path:
    return run / "artifacts" / dataset / f"{method}_{stage}_residuals_ps.npy"


def _split_indices(run: Path, dataset: str, stage: str) -> np.ndarray:
    path = run / "splits" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split artifact: {path}")
    key = "training" if stage == "train" else "test"
    with np.load(path) as split:
        return np.asarray(split[key], dtype=np.int64)


def _labels(residual_ps: np.ndarray, center_interval_ps: tuple[float, float]) -> np.ndarray:
    low, high = center_interval_ps
    residual = np.asarray(residual_ps, dtype=np.float64).reshape(-1)
    labels = np.full(residual.size, -1, dtype=np.int64)
    finite = np.isfinite(residual)
    labels[finite & (residual < low)] = 0
    labels[finite & (residual >= low) & (residual <= high)] = 1
    labels[finite & (residual > high)] = 2
    return labels


def _waveforms(dataset, mode: str, indices: np.ndarray) -> np.ndarray:
    """Normalized paired waveform representation [event, detector=2, time]."""
    pair = waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64)).materialize(dtype=np.float32)
    if pair.ndim != 3 or pair.shape[1] != 2:
        raise ValueError(f"Expected paired waveforms [event, detector=2, time], got {pair.shape}")
    return np.ascontiguousarray(pair, dtype=np.float32)


def _write_predictions(
    path: Path,
    dataset,
    indices: np.ndarray,
    residual: np.ndarray,
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray | None,
) -> None:
    fields = [
        "position",
        "prepared_index",
        "event_index",
        "bias_voltage_V",
        "svr_residual_ps",
        "true_class",
        "predicted_class",
        "correct",
        "p_left",
        "p_center",
        "p_right",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for position, prepared_index in enumerate(np.asarray(indices, dtype=np.int64)):
            row = {
                "position": int(position),
                "prepared_index": int(prepared_index),
                "event_index": int(dataset.event_index[prepared_index]),
                "bias_voltage_V": float(dataset.bias_voltage_V[prepared_index]),
                "svr_residual_ps": float(residual[position]),
                "true_class": str(CLASS_NAMES[int(truth[position])]),
                "predicted_class": str(CLASS_NAMES[int(predicted[position])]),
                "correct": bool(int(truth[position]) == int(predicted[position])),
                "p_left": float("nan"),
                "p_center": float("nan"),
                "p_right": float("nan"),
            }
            if probabilities is not None:
                for class_id, class_name in enumerate(CLASS_NAMES):
                    row[f"p_{class_name}"] = float(probabilities[position, class_id])
            writer.writerow(row)


def _plot_confusion(path: Path, matrix: np.ndarray, *, dataset_name: str, normalized: bool) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    display.plot(ax=ax, cmap="Blues", values_format=".2f" if normalized else "d", colorbar=False)
    title = "Normalized blind/test confusion matrix" if normalized else "Blind/test confusion matrix"
    ax.set_title(f"{dataset_name} · BORF classifier · {title}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    return {str(name): int(np.count_nonzero(labels == class_id)) for class_id, name in enumerate(CLASS_NAMES)}


def analyse_dataset(
    run: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    dataset_name: str,
    output_root: Path,
) -> None:
    method = str(config.get("svr_method", "linear_svr"))
    interval = config.get("center_interval_ps")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("config.center_interval_ps must be [LOW_PS, HIGH_PS]")
    center_interval = (float(interval[0]), float(interval[1]))
    if not np.isfinite(center_interval).all() or center_interval[0] >= center_interval[1]:
        raise ValueError("center_interval_ps must contain finite LOW < HIGH")

    train_residual_path = _residual_path(run, dataset_name, method, "train")
    test_residual_path = _residual_path(run, dataset_name, method, "test")
    if not train_residual_path.is_file():
        raise FileNotFoundError(
            f"Missing training SVR residuals: {train_residual_path}. "
            "This classifier requires the same fitted SVR run to have persisted train residuals."
        )
    if not test_residual_path.is_file():
        raise FileNotFoundError(f"Missing blind/test SVR residuals: {test_residual_path}")

    train_residual = np.asarray(np.load(train_residual_path), dtype=np.float64).reshape(-1)
    test_residual = np.asarray(np.load(test_residual_path), dtype=np.float64).reshape(-1)
    train_indices = _split_indices(run, dataset_name, "train")
    test_indices = _split_indices(run, dataset_name, "test")
    if train_residual.size != train_indices.size:
        raise ValueError(f"Train residual/split length mismatch: {train_residual.size} != {train_indices.size}")
    if test_residual.size != test_indices.size:
        raise ValueError(f"Test residual/split length mismatch: {test_residual.size} != {test_indices.size}")

    entry = manifest["datasets"][dataset_name]
    dataset = load_prepared_dataset(entry["prepared_dir"])
    mode = str(manifest.get("mode") or manifest["config"]["mode"])

    train_labels_all = _labels(train_residual, center_interval)
    test_labels_all = _labels(test_residual, center_interval)
    train_valid = train_labels_all >= 0
    test_valid = test_labels_all >= 0
    if np.count_nonzero(train_valid) < 3:
        raise ValueError("Too few finite training residuals")
    train_labels = train_labels_all[train_valid]
    test_labels = test_labels_all[test_valid]
    missing_train_classes = set(CLASS_IDS.tolist()) - set(np.unique(train_labels).tolist())
    if missing_train_classes:
        raise ValueError(
            f"Training labels do not contain all three classes; missing "
            f"{[CLASS_NAMES[c] for c in sorted(missing_train_classes)]}. Change center_interval_ps."
        )

    train_waveforms = _waveforms(dataset, mode, train_indices[train_valid])
    test_waveforms = _waveforms(dataset, mode, test_indices[test_valid])
    if not np.all(np.isfinite(train_waveforms)) or not np.all(np.isfinite(test_waveforms)):
        raise ValueError("BORF classifier currently requires finite prepared waveform samples")

    BORF = _load_borf()
    borf_config = dict(config.get("borf") or {})
    borf = BORF(**borf_config)
    print(
        f"{dataset_name}: fitting BORF on training paired waveforms "
        f"{train_waveforms.shape} with center interval [{center_interval[0]:g}, {center_interval[1]:g}] ps"
    )
    train_features = borf.fit_transform(train_waveforms, train_labels)
    test_features = borf.transform(test_waveforms)

    classifier_config = dict(config.get("classifier") or {})
    classifier = LogisticRegression(
        C=float(classifier_config.get("C", 1.0)),
        class_weight=classifier_config.get("class_weight", "balanced"),
        solver=str(classifier_config.get("solver", "saga")),
        max_iter=int(classifier_config.get("max_iter", 2000)),
        tol=float(classifier_config.get("tol", 1e-4)),
        multi_class="auto",
        random_state=int(((manifest.get("config") or {}).get("validation") or {}).get("seed", 0)),
        n_jobs=-1,
    )
    classifier.fit(train_features, train_labels)
    predicted = np.asarray(classifier.predict(test_features), dtype=np.int64)

    probabilities = None
    if hasattr(classifier, "predict_proba"):
        raw_probability = np.asarray(classifier.predict_proba(test_features), dtype=np.float64)
        probabilities = np.zeros((test_labels.size, 3), dtype=np.float64)
        for column, class_id in enumerate(classifier.classes_):
            probabilities[:, int(class_id)] = raw_probability[:, column]

    matrix = confusion_matrix(test_labels, predicted, labels=CLASS_IDS)
    matrix_normalized = confusion_matrix(test_labels, predicted, labels=CLASS_IDS, normalize="true")
    report = classification_report(
        test_labels,
        predicted,
        labels=CLASS_IDS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    output = output_root / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({"borf": borf, "classifier": classifier}, output / "borf_classifier.joblib")
    np.save(output / "confusion_matrix.npy", matrix)
    np.save(output / "confusion_matrix_normalized.npy", matrix_normalized)
    _plot_confusion(output / "confusion_matrix.pdf", matrix, dataset_name=dataset_name, normalized=False)
    _plot_confusion(
        output / "confusion_matrix_normalized.pdf",
        matrix_normalized,
        dataset_name=dataset_name,
        normalized=True,
    )
    _write_predictions(
        output / "test_predictions.csv",
        dataset,
        test_indices[test_valid],
        test_residual[test_valid],
        test_labels,
        predicted,
        probabilities,
    )

    summary = {
        "dataset": dataset_name,
        "mode": mode,
        "svr_method": method,
        "target_definition": {
            "left": f"SVR residual < {center_interval[0]} ps",
            "center": f"{center_interval[0]} ps <= SVR residual <= {center_interval[1]} ps",
            "right": f"SVR residual > {center_interval[1]} ps",
            "center_interval_ps": list(center_interval),
        },
        "data_protocol": {
            "borf_fit": "training split only",
            "classifier_fit": "training split only",
            "blind_test_used_for_training": False,
            "input": "normalized paired prepared waveforms [event, detector=2, time]",
        },
        "borf": borf_config,
        "classifier": classifier_config,
        "train": {
            "n": int(train_labels.size),
            "class_counts": _class_counts(train_labels),
            "borf_features": [int(v) for v in train_features.shape],
        },
        "test": {
            "n": int(test_labels.size),
            "class_counts": _class_counts(test_labels),
            "accuracy": float(accuracy_score(test_labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)),
            "confusion_matrix": matrix.tolist(),
            "confusion_matrix_normalized": matrix_normalized.tolist(),
            "classification_report": report,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8")

    print(f"{dataset_name}: blind/test accuracy={summary['test']['accuracy']:.4f}")
    print(f"{dataset_name}: blind/test balanced accuracy={summary['test']['balanced_accuracy']:.4f}")
    print(matrix)
    print(f"outputs: {output}")


def main() -> None:
    args = parse_args()
    run = args.run_dir.resolve()
    config = _read_json(args.config.resolve())
    manifest = _read_json(run / "manifest.json")
    method = str(config.get("svr_method", "linear_svr"))

    available = []
    for dataset_name in (manifest.get("datasets") or {}):
        if _residual_path(run, dataset_name, method, "train").is_file() and _residual_path(
            run, dataset_name, method, "test"
        ).is_file():
            available.append(dataset_name)
    if args.dataset:
        requested = set(args.dataset)
        datasets = [name for name in available if name in requested]
        missing = requested - set(datasets)
        if missing:
            raise FileNotFoundError(
                f"Requested datasets are missing train/test {method} residual artifacts: {sorted(missing)}"
            )
    else:
        datasets = available
    if not datasets:
        raise FileNotFoundError(f"No datasets contain both train and test residual artifacts for {method}")

    output_root = (args.output_dir or run / "svr_peak_borf_classifier").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        analyse_dataset(run, manifest, config, dataset_name, output_root)


if __name__ == "__main__":
    main()
