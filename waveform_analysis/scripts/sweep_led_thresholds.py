from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.config import load_config
from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.study import run_study


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a study at fixed LED thresholds and compare validation CTR, "
            "ML CTR, and relative ML improvement. Threshold selection uses validation "
            "only; blind/test CTR is reported only for the validation-selected threshold."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Base experiment JSON")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help=(
            "Fixed LED thresholds in mV. Default: values from "
            "standard_methods.led_thresholds_mV in the base config."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subset of configured ML models to train; default: every model in the base config",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Sweep output root (default: <base study output>_led_threshold_sweep)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite threshold run directories and threshold-specific prepared caches",
    )
    return parser


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p") + "mV"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _threshold_config(
    base_config: dict[str, Any],
    *,
    threshold_mV: float,
    output_root: Path,
    requested_models: list[str] | None,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    threshold = float(threshold_mV)
    config["standard_methods"]["led_thresholds_mV"] = [threshold]

    if requested_models:
        missing = sorted(set(requested_models) - set(config["models"]))
        if missing:
            raise ValueError(f"Requested model(s) are not configured: {missing}")
        config["models"] = {
            name: config["models"][name]
            for name in requested_models
        }

    label = _threshold_label(threshold)
    config["experiment"]["name"] = (
        f"{config['experiment'].get('name', 'study')}_led_{label}"
    )
    config["experiment"]["output_dir"] = str(
        (output_root / "runs" / label).resolve()
    )

    # The prepared dataset depends on the LED threshold because the LED crossing,
    # coincidence population, anchor, alignment and ML target all depend on it.
    # Give each threshold its own cache while reusing threshold-independent
    # event-selection and waveform-preprocessing caches.
    config["preprocessing"]["prepared_dir"] = str(
        (output_root / "prepared" / label).resolve()
    )
    return config


def _run_thresholds(
    base_config: dict[str, Any],
    thresholds: list[float],
    output_root: Path,
    requested_models: list[str] | None,
    *,
    overwrite: bool,
):
    runs: dict[float, Path] = {}
    failures = []

    for threshold in thresholds:
        config = _threshold_config(
            base_config,
            threshold_mV=threshold,
            output_root=output_root,
            requested_models=requested_models,
        )
        run_dir = Path(config["experiment"]["output_dir"])
        prepared_dir = Path(config["preprocessing"]["prepared_dir"])

        if overwrite and prepared_dir.exists():
            shutil.rmtree(prepared_dir)

        print(f"\n=== LED threshold {threshold:g} mV ===")
        try:
            completed = run_study(
                config,
                overwrite=overwrite,
                rebuild_preprocessing=False,
            )
        except Exception as exc:
            failures.append(
                {
                    "threshold_mV": float(threshold),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"FAILED {threshold:g} mV: {type(exc).__name__}: {exc}")
            continue

        runs[float(threshold)] = Path(completed).resolve()

    return runs, failures


def _validation_payload(run: Path):
    manifest = _read_json(run / "manifest.json")
    datasets = {}
    for dataset_name, dataset_manifest in manifest["datasets"].items():
        prepared = load_prepared_dataset(dataset_manifest["prepared_dir"])
        validation = np.asarray(prepared.validation, dtype=np.int64)
        event_ids = np.asarray(prepared.event_index[validation], dtype=np.int64)
        datasets[dataset_name] = {
            "prepared": prepared,
            "validation_event_ids": event_ids,
        }
    return manifest, datasets


def _common_validation_events(payloads, dataset_name: str) -> np.ndarray:
    event_sets = []
    for _threshold, (_manifest, datasets) in payloads.items():
        if dataset_name not in datasets:
            continue
        event_sets.append(set(map(int, datasets[dataset_name]["validation_event_ids"])))
    if not event_sets:
        return np.asarray([], dtype=np.int64)
    common = set.intersection(*event_sets)
    return np.asarray(sorted(common), dtype=np.int64)


def _validation_residuals(
    run: Path,
    dataset_name: str,
    method: str,
    event_ids: np.ndarray,
    common_event_ids: np.ndarray,
) -> np.ndarray:
    path = run / "artifacts" / dataset_name / f"{method}_validation_residuals_ps.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation residuals: {path}")
    residual = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if residual.size != event_ids.size:
        raise ValueError(
            f"{dataset_name}/{method}: validation residual count {residual.size} "
            f"does not match validation event count {event_ids.size}"
        )
    lookup = {int(event): index for index, event in enumerate(event_ids)}
    try:
        positions = np.asarray(
            [lookup[int(event)] for event in common_event_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"{dataset_name}/{method}: common validation-event mapping failed"
        ) from exc
    return residual[positions]


def _test_ctr(rows, dataset: str, method: str) -> float:
    row = next(
        (
            item
            for item in rows
            if item.get("dataset") == dataset
            and item.get("method") == method
            and item.get("stage") == "test"
        ),
        None,
    )
    if row is None:
        return float("nan")
    try:
        return float(row["ctr_ps"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _analyze_validation(
    runs: dict[float, Path],
    base_config: dict[str, Any],
):
    payloads = {
        threshold: _validation_payload(run)
        for threshold, run in runs.items()
    }
    dataset_names = sorted(
        set.intersection(
            *[
                set(datasets)
                for _manifest, datasets in payloads.values()
            ]
        )
    )
    if not dataset_names:
        raise RuntimeError("No dataset is common to all successful threshold runs")

    fit_config = dict(base_config.get("fit") or {})
    seed = int(base_config["validation"]["seed"])
    model_names = list(next(iter(payloads.values()))[0]["config"]["models"])

    threshold_rows = []
    summary_rows = []

    for dataset_name in dataset_names:
        common_events = _common_validation_events(payloads, dataset_name)
        minimum = int(fit_config.get("min_events", 100))
        if common_events.size < minimum:
            raise RuntimeError(
                f"{dataset_name}: only {common_events.size} validation events are common "
                f"to every successful LED threshold; need at least {minimum}"
            )

        per_threshold = {}
        for threshold, run in sorted(runs.items()):
            manifest, datasets = payloads[threshold]
            event_ids = datasets[dataset_name]["validation_event_ids"]
            method_ctrs = {}
            for method in ["led", *model_names]:
                residual = _validation_residuals(
                    run,
                    dataset_name,
                    method,
                    event_ids,
                    common_events,
                )
                if not np.all(np.isfinite(residual)):
                    raise ValueError(
                        f"{dataset_name}/{method}/{threshold:g} mV contains non-finite "
                        "common-validation residuals"
                    )
                estimate = fit_ctr_ps(
                    residual,
                    fit_config,
                    seed=semantic_seed(
                        seed,
                        "led_threshold_sweep",
                        dataset_name,
                        method,
                        f"{threshold:g}",
                    ),
                    bootstrap=False,
                )
                method_ctrs[method] = float(estimate.ctr_ps)

            led_ctr = method_ctrs["led"]
            per_threshold[threshold] = method_ctrs
            prepared = datasets[dataset_name]["prepared"]
            family = str(manifest["mode"]).split("_to_", 1)[0]
            retained_validation = int(prepared.validation.size)
            input_events = int(prepared.manifest.get("n_input_events", prepared.n_events))
            retained_events = int(prepared.n_events)

            for model_name in model_names:
                model_ctr = method_ctrs[model_name]
                improvement = 100.0 * (led_ctr - model_ctr) / led_ctr
                threshold_rows.append(
                    {
                        "dataset": dataset_name,
                        "threshold_mV": float(threshold),
                        "model": model_name,
                        "common_validation_events": int(common_events.size),
                        "threshold_validation_events": retained_validation,
                        "threshold_retained_events": retained_events,
                        "threshold_input_events": input_events,
                        "led_validation_ctr_ps": led_ctr,
                        "model_validation_ctr_ps": model_ctr,
                        "relative_improvement_pct": improvement,
                        "development_led_ctr_ps": float(
                            prepared.manifest["led_development_ctr_ps"][family]
                        ),
                        "development_led_efficiency": float(
                            prepared.manifest["led_development_efficiency"][family]
                        ),
                    }
                )

        led_best_threshold = min(
            per_threshold,
            key=lambda threshold: (
                per_threshold[threshold]["led"],
                threshold,
            ),
        )
        led_best_ctr = per_threshold[led_best_threshold]["led"]

        for model_name in model_names:
            best_threshold = min(
                per_threshold,
                key=lambda threshold: (
                    per_threshold[threshold][model_name],
                    threshold,
                ),
            )
            model_validation_ctr = per_threshold[best_threshold][model_name]
            led_same_threshold = per_threshold[best_threshold]["led"]
            validation_improvement = (
                100.0
                * (led_same_threshold - model_validation_ctr)
                / led_same_threshold
            )

            selected_run = runs[best_threshold]
            test_rows = _read_results(selected_run / "csv" / "results.csv")
            blind_led_ctr = _test_ctr(test_rows, dataset_name, "led")
            blind_model_ctr = _test_ctr(test_rows, dataset_name, model_name)
            blind_improvement = (
                100.0 * (blind_led_ctr - blind_model_ctr) / blind_led_ctr
                if np.isfinite(blind_led_ctr)
                and blind_led_ctr != 0.0
                and np.isfinite(blind_model_ctr)
                else float("nan")
            )

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "common_validation_events": int(common_events.size),
                    "best_led_only_threshold_mV": float(led_best_threshold),
                    "best_led_only_validation_ctr_ps": float(led_best_ctr),
                    "best_ml_paired_threshold_mV": float(best_threshold),
                    "validation_led_ctr_at_ml_threshold_ps": float(led_same_threshold),
                    "validation_model_ctr_ps": float(model_validation_ctr),
                    "validation_relative_improvement_pct": float(
                        validation_improvement
                    ),
                    "blind_led_ctr_at_selected_threshold_ps": float(blind_led_ctr),
                    "blind_model_ctr_ps": float(blind_model_ctr),
                    "blind_relative_improvement_pct": float(blind_improvement),
                    "threshold_selected_using": "common_validation_model_ctr",
                    "blind_used_for_threshold_selection": False,
                }
            )

    return threshold_rows, summary_rows


def _plot_threshold_curves(
    output_dir: Path,
    threshold_rows: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    generated = []
    keys = sorted({(row["dataset"], row["model"]) for row in threshold_rows})
    for dataset, model in keys:
        rows = sorted(
            [
                row
                for row in threshold_rows
                if row["dataset"] == dataset and row["model"] == model
            ],
            key=lambda row: float(row["threshold_mV"]),
        )
        threshold = np.asarray([row["threshold_mV"] for row in rows], dtype=float)
        led_ctr = np.asarray([row["led_validation_ctr_ps"] for row in rows], dtype=float)
        model_ctr = np.asarray([row["model_validation_ctr_ps"] for row in rows], dtype=float)
        improvement = np.asarray(
            [row["relative_improvement_pct"] for row in rows],
            dtype=float,
        )

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.plot(threshold, led_ctr, marker="o", label="LED")
        ax.plot(threshold, model_ctr, marker="o", label=model)
        ax.set_xlabel("Fixed LED threshold [mV]")
        ax.set_ylabel("Common-validation CTR [ps]")
        ax.set_title(f"{dataset} · {model} · CTR vs LED threshold")
        ax.grid(True, alpha=0.2)
        ax.legend()
        fig.tight_layout()
        ctr_path = output_dir / f"ctr_vs_led_threshold_{dataset}_{model}.pdf"
        ctr_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(ctr_path)
        plt.close(fig)
        generated.append(ctr_path)

        fig, ax = plt.subplots(figsize=(7.6, 5.0))
        ax.plot(threshold, improvement, marker="o")
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("Fixed LED threshold [mV]")
        ax.set_ylabel("Relative CTR improvement [%]")
        ax.set_title(f"{dataset} · {model} · ML improvement vs LED threshold")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        improvement_path = (
            output_dir / f"relative_improvement_vs_led_threshold_{dataset}_{model}.pdf"
        )
        fig.savefig(improvement_path)
        plt.close(fig)
        generated.append(improvement_path)

    return generated


def run(
    config_path: Path,
    thresholds: list[float] | None,
    requested_models: list[str] | None,
    output_dir: Path | None,
    *,
    overwrite: bool,
) -> list[Path]:
    base_config = load_config(config_path)
    configured_thresholds = [
        float(value)
        for value in base_config["standard_methods"]["led_thresholds_mV"]
    ]
    thresholds = configured_thresholds if thresholds is None else [float(v) for v in thresholds]
    thresholds = sorted(set(thresholds))
    if not thresholds:
        raise ValueError("At least one LED threshold is required")
    if any((not np.isfinite(value) or value <= 0.0) for value in thresholds):
        raise ValueError("LED thresholds must be finite and positive")

    if output_dir is None:
        base_output = Path(base_config["experiment"]["output_dir"])
        output_root = base_output.with_name(base_output.name + "_led_threshold_sweep")
    else:
        output_root = output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runs, failures = _run_thresholds(
        base_config,
        thresholds,
        output_root,
        requested_models,
        overwrite=overwrite,
    )
    failure_csv = output_root / "csv" / "failed_thresholds.csv"
    if failures:
        _write_csv(failure_csv, failures)

    if not runs:
        raise RuntimeError("Every LED-threshold run failed")

    threshold_rows, summary_rows = _analyze_validation(runs, base_config)
    threshold_csv = output_root / "csv" / "threshold_comparison.csv"
    summary_csv = output_root / "csv" / "selected_thresholds.csv"
    _write_csv(threshold_csv, threshold_rows)
    _write_csv(summary_csv, summary_rows)

    generated = [threshold_csv, summary_csv]
    if failures:
        generated.append(failure_csv)
    generated.extend(
        _plot_threshold_curves(
            output_root / "plots",
            threshold_rows,
        )
    )

    print("\nValidation-selected LED thresholds:")
    for row in summary_rows:
        print(
            f"{row['dataset']} | {row['model']} | "
            f"LED-only best={row['best_led_only_threshold_mV']:g} mV | "
            f"ML-paired best={row['best_ml_paired_threshold_mV']:g} mV | "
            f"validation CTR={row['validation_model_ctr_ps']:.3f} ps | "
            f"improvement={row['validation_relative_improvement_pct']:.2f}% | "
            f"blind CTR={row['blind_model_ctr_ps']:.3f} ps"
        )

    return generated


def main() -> None:
    args = _parser().parse_args()
    for path in run(
        args.config,
        args.thresholds,
        args.models,
        args.output_dir,
        overwrite=args.overwrite,
    ):
        print(path)


if __name__ == "__main__":
    main()
