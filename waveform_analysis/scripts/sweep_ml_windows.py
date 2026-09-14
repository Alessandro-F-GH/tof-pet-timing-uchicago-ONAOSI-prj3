from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.common import atomic_json, canonical_json
from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.models import get_model
from waveform_analysis.ml_pipeline.reporting import LABELS
from waveform_analysis.ml_pipeline.sample_mask import (
    SAMPLE_CONSTANT_FRACTION,
    dataset_training_sample_mask,
)
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.train import predict_indices, search_model, selected_model
from waveform_analysis.ml_pipeline.view import (
    calibrated_led,
    corrected_timing_residual,
    model_target,
    target_family,
    waveform_view,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain configured ML models with progressively truncated waveform inputs. "
            "The prepared dataset is reused from a completed base study, so the LED "
            "threshold, timing target, event population, calibration, and train/validation/"
            "blind split are identical for every window. Only the temporal ML-input mask changes."
        )
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Completed base study directory containing manifest.json",
    )
    parser.add_argument(
        "--right-limits",
        nargs="+",
        type=float,
        required=True,
        metavar="NS",
        help="Right limits of the ML input window in ns, e.g. 1 10 20 40",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subset of models from the base study; default: all configured models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root; default: <base_run>_window_sweep",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing non-empty window-sweep output directory",
    )
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _logger(output_root: Path) -> logging.Logger:
    logger = logging.getLogger(f"window-sweep:{output_root}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_root / "window_sweep.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _window_label(right_ns: float) -> str:
    return f"{float(right_ns):g}".replace("-", "m").replace(".", "p") + "ns"


def _time_axis_ps(dataset, mode: str) -> np.ndarray:
    training = np.asarray(dataset.training, dtype=np.int64)
    if training.size == 0:
        raise ValueError("Training split is empty")
    return np.asarray(
        waveform_view(dataset, mode, training[:1]).time_ps,
        dtype=np.float64,
    ).reshape(-1)


def _requested_models(config: dict[str, Any], requested: list[str] | None) -> list[str]:
    configured = list(config["models"])
    if requested is None:
        return configured
    missing = sorted(set(requested) - set(configured))
    if missing:
        raise ValueError(f"Requested model(s) are not configured in the base run: {missing}")
    return list(dict.fromkeys(requested))


def _validate_limits(
    right_limits_ns: list[float],
    *,
    start_ns: float,
    available_end_ns: float,
    time_ps: np.ndarray,
) -> list[float]:
    limits = sorted(set(float(value) for value in right_limits_ns))
    if not limits:
        raise ValueError("At least one right limit is required")
    if any(not np.isfinite(value) for value in limits):
        raise ValueError("Window right limits must be finite")
    if any(value <= start_ns for value in limits):
        bad = [value for value in limits if value <= start_ns]
        raise ValueError(
            f"Every right limit must exceed the fixed left limit {start_ns:g} ns; got {bad}"
        )

    if time_ps.size > 1:
        dt_ns = float(np.median(np.diff(time_ps))) / 1000.0
    else:
        dt_ns = 0.0
    tolerance_ns = max(1e-9, 0.51 * abs(dt_ns))
    too_large = [value for value in limits if value > available_end_ns + tolerance_ns]
    if too_large:
        raise ValueError(
            "Requested right limit exceeds the prepared base window. "
            f"Available end is about {available_end_ns:g} ns; got {too_large}. "
            "Prepare the base study with a window at least as wide as the largest scan value."
        )
    return limits


def _fit_blind_ctr(
    residual_ps: np.ndarray,
    fit_config: dict[str, Any],
    *,
    seed: int,
):
    values = np.asarray(residual_ps, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("Blind residuals contain non-finite values")
    return fit_ctr_ps(values, fit_config, seed=int(seed), bootstrap=True)


def _plot_dataset(
    output_path: Path,
    dataset_name: str,
    rows: list[dict[str, Any]],
    *,
    left_limit_ns: float,
) -> None:
    import matplotlib.pyplot as plt

    subset = [row for row in rows if row["dataset"] == dataset_name]
    if not subset:
        return

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    model_names = list(dict.fromkeys(row["model"] for row in subset))
    for model_name in model_names:
        model_rows = sorted(
            [row for row in subset if row["model"] == model_name],
            key=lambda row: float(row["right_limit_ns"]),
        )
        x = np.asarray([row["right_limit_ns"] for row in model_rows], dtype=float)
        y = np.asarray([row["blind_ctr_ps"] for row in model_rows], dtype=float)
        yerr = np.asarray(
            [row["blind_ctr_uncertainty_ps"] for row in model_rows],
            dtype=float,
        )
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            label=LABELS.get(model_name, model_name),
        )

    led_ctr = float(subset[0]["led_blind_ctr_ps"])
    led_err = float(subset[0]["led_blind_ctr_uncertainty_ps"])
    if np.isfinite(led_ctr):
        ax.axhline(led_ctr, linestyle="--", linewidth=1.2, label="LED")
        if np.isfinite(led_err) and led_err > 0.0:
            ax.axhspan(led_ctr - led_err, led_ctr + led_err, alpha=0.12)

    ax.set_xlabel("ML window right limit [ns]")
    ax.set_ylabel("Blind-test CTR [ps]")
    ax.set_title(f"{dataset_name} · fixed left limit {left_limit_ns:g} ns")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run(
    base_run: Path,
    right_limits_ns: list[float],
    requested_models: list[str] | None = None,
    output_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> list[Path]:
    base_run = base_run.expanduser().resolve()
    manifest_path = base_run / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Base study manifest not found: {manifest_path}")

    manifest = _read_json(manifest_path)
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("Base study manifest does not contain the resolved config")
    datasets_manifest = manifest.get("datasets")
    if not isinstance(datasets_manifest, dict) or not datasets_manifest:
        raise ValueError("Base study manifest contains no datasets")

    mode = str(manifest.get("mode", config.get("mode", "")))
    if not mode:
        raise ValueError("Base study manifest does not define the channel mode")
    models = _requested_models(config, requested_models)
    seed = int(config["validation"]["seed"])
    fit_config = dict(config.get("fit") or {})

    if output_dir is None:
        output_root = base_run.with_name(base_run.name + "_window_sweep")
    else:
        output_root = output_dir.expanduser().resolve()

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    logger = _logger(output_root)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    dataset_plot_info: dict[str, float] = {}
    family = target_family(mode)

    logger.info("Base study | %s", base_run)
    logger.info("Mode | %s", mode)
    logger.info("Models | %s", ", ".join(models))
    logger.info(
        "Window scan policy | reuse prepared dataset and fixed LED target; only temporal input mask changes"
    )

    for dataset_name, dataset_info in datasets_manifest.items():
        prepared_dir = Path(dataset_info["prepared_dir"]).expanduser().resolve()
        dataset = load_prepared_dataset(prepared_dir)
        time_ps = _time_axis_ps(dataset, mode)
        base_constant_mask = dataset_training_sample_mask(dataset, mode)
        if base_constant_mask.size != time_ps.size:
            raise ValueError(
                f"{dataset_name}: training sample mask has {base_constant_mask.size} entries "
                f"but time axis has {time_ps.size}"
            )

        ml_input = dataset.manifest.get("ml_input") or config.get("ml_input") or {}
        configured_window = ml_input.get("window_ns") or {}
        start_ns = float(configured_window.get("start", np.min(time_ps) / 1000.0))
        available_end_ns = float(np.max(time_ps) / 1000.0)
        limits = _validate_limits(
            right_limits_ns,
            start_ns=start_ns,
            available_end_ns=available_end_ns,
            time_ps=time_ps,
        )
        dataset_plot_info[dataset_name] = start_ns

        led_threshold_mV = float(dataset.manifest["led_threshold_mV"][family])
        test_indices = np.asarray(dataset.test, dtype=np.int64)
        target = model_target(dataset, mode)
        led_residual = calibrated_led(dataset, mode)[test_indices]
        led_result = _fit_blind_ctr(
            led_residual,
            fit_config,
            seed=semantic_seed(seed, dataset_name, mode, "led", "test"),
        )

        logger.info(
            "Dataset %s | LED=%.6g mV | prepared window=[%.6g, %.6g] ns | "
            "constant-mask retained=%d/%d | blind=%d",
            dataset_name,
            led_threshold_mV,
            float(np.min(time_ps) / 1000.0),
            available_end_ns,
            int(np.count_nonzero(base_constant_mask)),
            int(base_constant_mask.size),
            int(test_indices.size),
        )

        for right_ns in limits:
            # Use a half-sample numerical tolerance so a requested boundary that is
            # nominally on the acquisition grid is not lost to floating-point noise.
            dt_ps = float(np.median(np.diff(time_ps))) if time_ps.size > 1 else 0.0
            tolerance_ps = max(1e-6, 0.51 * abs(dt_ps))
            temporal_mask = time_ps <= float(right_ns) * 1000.0 + tolerance_ps
            combined_mask = np.asarray(base_constant_mask & temporal_mask, dtype=bool)
            retained = int(np.count_nonzero(combined_mask))
            temporal_count = int(np.count_nonzero(temporal_mask))

            if retained < 2:
                failures.append(
                    {
                        "dataset": dataset_name,
                        "right_limit_ns": float(right_ns),
                        "model": "*",
                        "error": "Temporal + constant mask leaves fewer than two input samples",
                    }
                )
                logger.warning(
                    "%s | right=%g ns | skipped: only %d retained input samples",
                    dataset_name,
                    right_ns,
                    retained,
                )
                continue

            effective_right_ns = float(np.max(time_ps[combined_mask]) / 1000.0)
            mask_path = output_root / "masks" / dataset_name / f"{_window_label(right_ns)}.npy"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask_path, combined_mask)

            logger.info(
                "%s | right=%g ns | effective_right=%.6g ns | retained=%d/%d samples",
                dataset_name,
                right_ns,
                effective_right_ns,
                retained,
                int(base_constant_mask.size),
            )

            for model_name in models:
                model_config = config["models"][model_name]
                spec = get_model(model_name)
                try:
                    search = search_model(
                        spec,
                        model_config,
                        config,
                        dataset,
                        mode,
                        # Deliberately identical to the ordinary study seed: the
                        # temporal window, not a new random initialization policy,
                        # is the experimental variable.
                        seed=semantic_seed(seed, dataset_name, mode, model_name, "search"),
                        dataset_name=dataset_name,
                        sample_mask=combined_mask,
                        logger=logger,
                    )
                    fitted = selected_model(search)
                    prediction, _masked_time, _masked_pair = predict_indices(
                        spec,
                        fitted,
                        dataset,
                        mode,
                        test_indices,
                    )
                    residual = corrected_timing_residual(
                        target[test_indices],
                        prediction,
                    )
                    ctr = _fit_blind_ctr(
                        residual,
                        fit_config,
                        # Reusing the ordinary test seed also gives the same bootstrap
                        # resampling stream across window sizes for a given model.
                        seed=semantic_seed(seed, dataset_name, mode, model_name, "test"),
                    )

                    improvement = 100.0 * (led_result.ctr_ps - ctr.ctr_ps) / led_result.ctr_ps
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "mode": mode,
                            "model": model_name,
                            "led_threshold_mV": led_threshold_mV,
                            "window_start_ns": start_ns,
                            "right_limit_ns": float(right_ns),
                            "effective_right_limit_ns": effective_right_ns,
                            "window_width_ns": float(right_ns - start_ns),
                            "grid_samples_within_window": temporal_count,
                            "input_samples_after_constant_and_temporal_mask": retained,
                            "input_samples_full_prepared_window": int(base_constant_mask.size),
                            "constant_mask_fraction": SAMPLE_CONSTANT_FRACTION,
                            "validation_ctr_ps": float(search.best.score),
                            "selected_parameters_json": canonical_json(search.best.candidate),
                            "blind_ctr_ps": float(ctr.ctr_ps),
                            "blind_ctr_uncertainty_ps": float(ctr.ctr_error_ps),
                            "bootstrap_samples": int(ctr.bootstrap_samples),
                            "bootstrap_successful": int(ctr.bootstrap_successful),
                            "blind_events": int(ctr.n_valid),
                            "led_blind_ctr_ps": float(led_result.ctr_ps),
                            "led_blind_ctr_uncertainty_ps": float(led_result.ctr_error_ps),
                            "relative_improvement_vs_led_pct": float(improvement),
                            "target_recomputed": False,
                            "split_recomputed": False,
                        }
                    )
                    logger.info(
                        "%s | %s | right=%g ns | validation CTR=%.3f ps | "
                        "blind CTR=%.3f ± %.3f ps | improvement=%.2f%%",
                        dataset_name,
                        LABELS.get(model_name, model_name),
                        right_ns,
                        float(search.best.score),
                        float(ctr.ctr_ps),
                        float(ctr.ctr_error_ps),
                        float(improvement),
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "dataset": dataset_name,
                            "right_limit_ns": float(right_ns),
                            "model": model_name,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    logger.exception(
                        "%s | %s | right=%g ns | FAILED",
                        dataset_name,
                        model_name,
                        right_ns,
                    )
                finally:
                    # Long CNN scans otherwise retain Python-side references longer
                    # than necessary even though each window is an independent fit.
                    gc.collect()

    if not rows:
        failure_csv = output_root / "csv" / "failed_windows.csv"
        if failures:
            _write_csv(failure_csv, failures)
        raise RuntimeError("No window-scan model evaluation completed successfully")

    result_csv = output_root / "csv" / "window_scan.csv"
    _write_csv(result_csv, rows)
    generated = [result_csv]

    if failures:
        failure_csv = output_root / "csv" / "failed_windows.csv"
        _write_csv(failure_csv, failures)
        generated.append(failure_csv)

    plot_paths = []
    for dataset_name, left_limit_ns in dataset_plot_info.items():
        plot_path = output_root / "plots" / f"blind_ctr_vs_window_{dataset_name}.pdf"
        _plot_dataset(
            plot_path,
            dataset_name,
            rows,
            left_limit_ns=left_limit_ns,
        )
        if plot_path.is_file():
            plot_paths.append(plot_path)
    generated.extend(plot_paths)

    atomic_json(
        output_root / "manifest.json",
        {
            "analysis": "ml_window_right_limit_sweep",
            "base_run": str(base_run),
            "mode": mode,
            "models": models,
            "requested_right_limits_ns": sorted(set(map(float, right_limits_ns))),
            "target_policy": (
                "reuse base prepared dataset; LED threshold, calibration, target, event population, "
                "and train/validation/blind split remain fixed"
            ),
            "input_policy": (
                "training-derived 99% constant-sample mask intersected with time <= requested right limit"
            ),
            "model_selection": "validation CTR independently within each temporal window",
            "blind_policy": "blind split used only after model selection for each window",
            "ctr_metric": manifest.get("ctr_metric"),
            "ctr_coverage_fraction": manifest.get("ctr_coverage_fraction"),
            "ctr_bootstrap_samples": int(fit_config.get("bootstrap_samples", 0)),
            "ctr_uncertainty": "event-bootstrap standard deviation of canonical CTR",
            "successful_points": len(rows),
            "failed_points": len(failures),
        },
    )
    generated.append(output_root / "manifest.json")

    logger.info("Window sweep complete | %s", output_root)
    logger.info("Successful model/window points | %d", len(rows))
    logger.info("Failed model/window points | %d", len(failures))
    return generated


def main() -> None:
    args = _parser().parse_args()
    generated = run(
        args.run,
        args.right_limits,
        args.models,
        args.output_dir,
        overwrite=args.overwrite,
    )
    print("\nGenerated:")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
