from __future__ import annotations

import copy
import csv
import gc
from pathlib import Path
from typing import Any

import numpy as np

from utils_fit import fit_ctr_ps

from .common import canonical_json, voltage_from_name
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .plot_style import (
    LABELS,
    SINGLE_COLUMN,
    clean_axis,
    model_style,
    paper_context,
    save_figure,
)
from .sample_mask import dataset_training_sample_mask
from .splits import semantic_seed
from .train import predict_indices, search_model, selected_model
from .view import calibrated_led, corrected_timing_residual, model_target


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _threshold_label(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p") + "mV"


def _threshold_config(
    base_config: dict[str, Any],
    threshold_mV: float,
    output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    threshold = float(threshold_mV)
    label = _threshold_label(threshold)
    config["standard_methods"]["led_thresholds_mV"] = [threshold]
    config["preprocessing"]["prepared_dir"] = str(
        Path(base_config["preprocessing"]["prepared_dir"]).resolve()
        / "_led_threshold_scan"
        / label
    )
    config["experiment"]["output_dir"] = str(
        (output_dir / "diagnostics" / label).resolve()
    )
    return config


def _paired_blind_improvement(
    led_residual: np.ndarray,
    model_residual: np.ndarray,
    fit_config: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float, int]:
    led = np.asarray(led_residual, dtype=np.float64).reshape(-1)
    model = np.asarray(model_residual, dtype=np.float64).reshape(-1)
    if led.shape != model.shape:
        raise ValueError("Paired blind bootstrap requires aligned residual arrays")
    finite = np.isfinite(led) & np.isfinite(model)
    led = led[finite]
    model = model[finite]
    if led.size < 2:
        raise ValueError("Paired blind bootstrap requires at least two events")

    led_ctr = fit_ctr_ps(led, fit_config, bootstrap=False).ctr_ps
    model_ctr = fit_ctr_ps(model, fit_config, bootstrap=False).ctr_ps
    central = 100.0 * (led_ctr - model_ctr) / led_ctr
    rng = np.random.default_rng(int(seed))
    bootstrap = []
    for _ in range(int(samples)):
        indices = rng.integers(0, led.size, size=led.size)
        try:
            reference = fit_ctr_ps(
                led[indices], fit_config, bootstrap=False
            ).ctr_ps
            corrected = fit_ctr_ps(
                model[indices], fit_config, bootstrap=False
            ).ctr_ps
        except ValueError:
            continue
        if np.isfinite(reference) and reference > 0 and np.isfinite(corrected):
            bootstrap.append(100.0 * (reference - corrected) / reference)
    uncertainty = (
        float(np.std(bootstrap, ddof=1))
        if len(bootstrap) > 1
        else float("nan")
    )
    return float(central), uncertainty, len(bootstrap)


def _plot_threshold_scan(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib.pyplot as plt

    generated: list[Path] = []
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with paper_context():
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            subset = sorted(
                [row for row in rows if str(row["dataset"]) == dataset],
                key=lambda row: float(row["threshold_mV"]),
            )
            threshold = np.asarray(
                [float(row["threshold_mV"]) for row in subset], dtype=float
            )
            led = np.asarray(
                [float(row["led_blind_ctr_ps"]) for row in subset], dtype=float
            )
            led_err = np.asarray(
                [float(row["led_blind_ctr_uncertainty_ps"]) for row in subset],
                dtype=float,
            )
            mlp = np.asarray(
                [float(row["mlp_blind_ctr_ps"]) for row in subset], dtype=float
            )
            mlp_err = np.asarray(
                [float(row["mlp_blind_ctr_uncertainty_ps"]) for row in subset],
                dtype=float,
            )
            improvement = np.asarray(
                [float(row["relative_improvement_percent"]) for row in subset],
                dtype=float,
            )
            improvement_err = np.asarray(
                [
                    float(row["paired_bootstrap_uncertainty_percent"])
                    for row in subset
                ],
                dtype=float,
            )

            fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
            ax.errorbar(
                threshold,
                led,
                yerr=np.where(np.isfinite(led_err), led_err, 0.0),
                capsize=2.5,
                label="LED",
                **model_style("led"),
            )
            ax.errorbar(
                threshold,
                mlp,
                yerr=np.where(np.isfinite(mlp_err), mlp_err, 0.0),
                capsize=2.5,
                label=LABELS.get("mlp", "MLP"),
                **model_style("mlp"),
            )
            ax.set_xlabel("LED threshold [mV]")
            ax.set_ylabel("Blind CTR [ps]")
            ax.legend(loc="best")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            target = save_figure(
                fig,
                plot_dir / f"blind_ctr_vs_led_threshold_{dataset}.pdf",
            )
            plt.close(fig)
            generated.append(target)

            fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
            ax.errorbar(
                threshold,
                improvement,
                yerr=np.where(
                    np.isfinite(improvement_err), improvement_err, 0.0
                ),
                capsize=2.5,
                **model_style("mlp"),
            )
            ax.axhline(0.0, color="#7F7F7F", linestyle=":", linewidth=0.9)
            ax.set_xlabel("LED threshold [mV]")
            ax.set_ylabel("Blind CTR improvement [%]")
            clean_axis(ax, grid="y")
            fig.tight_layout()
            target = save_figure(
                fig,
                plot_dir
                / f"blind_relative_improvement_vs_led_threshold_{dataset}.pdf",
            )
            plt.close(fig)
            generated.append(target)
    return generated


def run_blind_led_threshold_scan(
    preprocessed: list[Any],
    config: dict[str, Any],
    output_dir: Path,
    logger,
    progress,
    *,
    rebuild: bool,
    resume: bool = False,
) -> dict[str, Any]:
    model_name = "mlp"
    thresholds = sorted(
        {float(value) for value in config["standard_methods"]["led_thresholds_mV"]}
    )
    target_voltage = float(config["experiment"]["voltage_V"])
    mode = str(config["mode"])
    fit_config = dict(config.get("fit") or {})
    bootstrap_samples = int(fit_config.get("bootstrap_samples", 0))
    seed = int(config["validation"]["seed"])

    sources = []
    for source in preprocessed:
        dataset_name = Path(source.manifest["source"]).stem
        voltage = voltage_from_name(dataset_name)
        if np.isfinite(voltage) and np.isclose(
            voltage, target_voltage, rtol=0.0, atol=1e-9
        ):
            sources.append(source)
    if not sources:
        raise RuntimeError(
            f"No preprocessed dataset matches threshold-scan voltage {target_voltage:g} V"
        )

    csv_path = output_dir / "csv" / "threshold_scan.csv"
    selection_csv = output_dir / "csv" / "hyperparameter_selection.csv"
    rows = _read_csv(csv_path) if resume else []
    selection_rows = _read_csv(selection_csv) if resume else []
    completed = {
        (str(row.get("dataset")), float(row.get("threshold_mV")))
        for row in rows
        if row.get("dataset") and row.get("threshold_mV") not in {None, ""}
    }

    logger.info(
        "Threshold scan | mode=%s | voltage=%g V | thresholds=%d | model=%s",
        mode,
        target_voltage,
        len(thresholds),
        LABELS.get(model_name, model_name),
    )

    for source in sources:
        dataset_name = Path(source.manifest["source"]).stem
        for threshold in thresholds:
            key = (dataset_name, threshold)
            artifact_dir = (
                output_dir / "artifacts" / dataset_name / _threshold_label(threshold)
            )
            residual_path = artifact_dir / "mlp_blind_residuals_ps.npy"
            led_path = artifact_dir / "led_blind_residuals_ps.npy"
            output_path = artifact_dir / "mlp_blind_model_output_ps.npy"
            if (
                resume
                and key in completed
                and residual_path.is_file()
                and led_path.is_file()
                and output_path.is_file()
            ):
                progress.complete(
                    "threshold_scan",
                    f"{dataset_name} | {threshold:g} mV",
                    note="resume",
                    announce=False,
                )
                continue

            candidate_config = _threshold_config(config, threshold, output_dir)
            candidate_config["models"] = {"mlp": config["models"]["mlp"]}
            with progress.task(
                "threshold_scan",
                f"{dataset_name} | {threshold:g} mV",
                announce_start=False,
                announce_finish=False,
            ):
                dataset = prepare_ml_dataset(
                    source,
                    candidate_config,
                    rebuild=rebuild,
                    logger=logger,
                    log_summary=False,
                    write_diagnostics=False,
                )
                sample_mask = dataset_training_sample_mask(dataset, mode)
                spec = get_model(model_name)
                search = search_model(
                    spec,
                    config["models"][model_name],
                    candidate_config,
                    dataset,
                    mode,
                    seed=semantic_seed(
                        seed,
                        dataset_name,
                        mode,
                        model_name,
                        "threshold_scan",
                        f"{threshold:g}",
                    ),
                    dataset_name=dataset_name,
                    sample_mask=sample_mask,
                    logger=logger,
                )
                fitted = selected_model(search)
                blind = np.asarray(dataset.test, dtype=np.int64)
                prediction, _time, _pair = predict_indices(
                    spec, fitted, dataset, mode, blind
                )
                target = model_target(dataset, mode)
                model_residual = corrected_timing_residual(
                    target[blind], prediction
                )
                led_residual = calibrated_led(dataset, mode)[blind]

                led_fit = fit_ctr_ps(
                    np.asarray(led_residual, dtype=np.float64),
                    fit_config,
                    seed=semantic_seed(
                        seed,
                        dataset_name,
                        "led",
                        "threshold_scan",
                        f"{threshold:g}",
                    ),
                    bootstrap=True,
                )
                model_fit = fit_ctr_ps(
                    np.asarray(model_residual, dtype=np.float64),
                    fit_config,
                    seed=semantic_seed(
                        seed,
                        dataset_name,
                        model_name,
                        "threshold_scan",
                        f"{threshold:g}",
                    ),
                    bootstrap=True,
                )
                improvement, improvement_error, paired_successful = (
                    _paired_blind_improvement(
                        led_residual,
                        model_residual,
                        fit_config,
                        samples=bootstrap_samples,
                        seed=semantic_seed(
                            seed,
                            dataset_name,
                            model_name,
                            "threshold_scan_paired",
                            f"{threshold:g}",
                        ),
                    )
                )

                artifact_dir.mkdir(parents=True, exist_ok=True)
                np.save(residual_path, np.asarray(model_residual, dtype=np.float64))
                np.save(led_path, np.asarray(led_residual, dtype=np.float64))
                np.save(output_path, np.asarray(prediction, dtype=np.float64))

                rows = [
                    row
                    for row in rows
                    if not (
                        str(row.get("dataset")) == dataset_name
                        and row.get("threshold_mV") not in {None, ""}
                        and np.isclose(
                            float(row["threshold_mV"]),
                            threshold,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                ]
                selection_rows = [
                    row
                    for row in selection_rows
                    if not (
                        str(row.get("dataset")) == dataset_name
                        and row.get("threshold_mV") not in {None, ""}
                        and np.isclose(
                            float(row["threshold_mV"]),
                            threshold,
                            rtol=0.0,
                            atol=1e-12,
                        )
                    )
                ]
                selection_rows.append(
                    {
                        "dataset": dataset_name,
                        "voltage_V": target_voltage,
                        "mode": mode,
                        "threshold_mV": threshold,
                        "model": model_name,
                        "selection_population": "validation",
                        "selection_metric": "validation_ctr",
                        "validation_ctr_ps": float(search.best.score),
                        "selected_parameters_json": canonical_json(
                            search.best.candidate
                        ),
                        "refit_after_selection": False,
                    }
                )
                rows.append(
                    {
                        "dataset": dataset_name,
                        "voltage_V": target_voltage,
                        "mode": mode,
                        "threshold_mV": threshold,
                        "model": model_name,
                        "final_population": "blind",
                        "blind_events": int(blind.size),
                        "led_blind_ctr_ps": float(led_fit.ctr_ps),
                        "led_blind_ctr_uncertainty_ps": float(
                            led_fit.ctr_error_ps
                        ),
                        "mlp_blind_ctr_ps": float(model_fit.ctr_ps),
                        "mlp_blind_ctr_uncertainty_ps": float(
                            model_fit.ctr_error_ps
                        ),
                        "relative_improvement_percent": improvement,
                        "paired_bootstrap_uncertainty_percent": improvement_error,
                        "paired_bootstrap_successful": paired_successful,
                    }
                )
                _write_csv(selection_csv, selection_rows)
                _write_csv(csv_path, rows)
                search.best.artifact = None
                del fitted
                gc.collect()

            logger.info(
                "Threshold result | %s | %g mV | validation CTR=%.3f ps | "
                "blind MLP CTR=%.3f ± %.3f ps | improvement=%.2f ± %.2f%%",
                dataset_name,
                threshold,
                float(search.best.score),
                float(model_fit.ctr_ps),
                float(model_fit.ctr_error_ps),
                improvement,
                improvement_error,
            )

    rows.sort(
        key=lambda row: (
            str(row.get("dataset")),
            float(row.get("threshold_mV", 0.0)),
        )
    )
    selection_rows.sort(
        key=lambda row: (
            str(row.get("dataset")),
            float(row.get("threshold_mV", 0.0)),
        )
    )
    _write_csv(csv_path, rows)
    _write_csv(selection_csv, selection_rows)
    _plot_threshold_scan(output_dir, rows)
    return {
        "experiment_type": "threshold_scan",
        "model": model_name,
        "voltage_V": target_voltage,
        "mode": mode,
        "candidate_thresholds_mV": thresholds,
        "hyperparameter_selection_population": "validation",
        "hyperparameter_selection_metric": "validation_ctr",
        "final_evaluation_population": "blind",
        "threshold_selected_from_scan": False,
        "refit_after_validation_selection": False,
        "ctr_uncertainty": "event_bootstrap",
        "relative_improvement_uncertainty": "paired_event_bootstrap",
        "output_dir": str(output_dir.resolve()),
    }
