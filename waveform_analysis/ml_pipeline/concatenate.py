from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from utils_fit import fit_ctr_ps

from .common import atomic_json, canonical_hash, voltage_from_name
from .config import mode_family
from .dataset import DATASET_FORMAT_VERSION, PreparedDataset, load_prepared_dataset


def _same_array(left: np.ndarray, right: np.ndarray, label: str) -> None:
    if np.asarray(left).shape != np.asarray(right).shape or not np.allclose(
        np.asarray(left), np.asarray(right), rtol=1e-9, atol=1e-12
    ):
        raise ValueError(f"Cannot concatenate datasets with different {label}")


def _copy_windows(datasets: list[PreparedDataset], family: str, target: Path) -> None:
    arrays = [dataset.energy_windows if family == "energy" else dataset.timing_windows for dataset in datasets]
    if any(array is None for array in arrays):
        raise ValueError(f"Cannot concatenate: {family} ML windows are missing")
    shapes = [np.asarray(array).shape for array in arrays]
    if any(shape[1:] != shapes[0][1:] for shape in shapes[1:]):
        raise ValueError(f"Cannot concatenate datasets with different {family} ML-window shapes")
    total = sum(shape[0] for shape in shapes)
    output = open_memmap(target, mode="w+", dtype=np.float32, shape=(total, *shapes[0][1:]))
    offset = 0
    for array in arrays:
        n = int(np.asarray(array).shape[0])
        output[offset : offset + n] = np.asarray(array, dtype=np.float32)
        offset += n
    output.flush()
    del output


def _concat_optional(datasets: list[PreparedDataset], attribute: str) -> np.ndarray | None:
    arrays = [getattr(dataset, attribute) for dataset in datasets]
    if all(array is None for array in arrays):
        return None
    if any(array is None for array in arrays):
        raise ValueError(f"Cannot concatenate: {attribute} is missing from only some datasets")
    return np.concatenate([np.asarray(array) for array in arrays], axis=0)


def concatenated_fingerprint(datasets: list[PreparedDataset], config: dict[str, Any], name: str) -> str:
    return canonical_hash(
        {
            "kind": "concatenated_prepared_dataset",
            "format_version": DATASET_FORMAT_VERSION,
            "name": str(name),
            "mode": str(config["mode"]),
            "fixed_led_threshold_mV": float(config["experiment"]["fixed_led_threshold_mV"]),
            "true_tof_ps": float(config["data"]["true_tof_ps"]),
            "fit": config.get("fit"),
            "ml_input": config.get("ml_input"),
            "sources": [dataset.manifest.get("fingerprint") for dataset in datasets],
        }
    )


def concatenate_prepared_datasets(
    datasets: list[PreparedDataset],
    directory: str | Path,
    config: dict[str, Any],
    *,
    name: str,
    rebuild: bool = False,
    logger=None,
) -> PreparedDataset:
    """Materialize one ML dataset from separately prepared bias-voltage datasets.

    Each source keeps its already frozen development/test assignment. Training,
    validation and blind-test indices are concatenated separately, while the LED
    calibration and slide target are recomputed globally on the concatenated
    training population. The configured fixed LED threshold is therefore common
    to every source voltage and one model is trained/evaluated for the full pool.
    """
    if len(datasets) < 2:
        raise ValueError("Concatenated experiment requires at least two prepared datasets")

    mode = str(config["mode"])
    family = mode_family(mode)
    experiment = config["experiment"]
    fixed_led = float(experiment["fixed_led_threshold_mV"])
    true_tof = float(config["data"]["true_tof_ps"])
    output = Path(directory).resolve()
    fingerprint = concatenated_fingerprint(datasets, config, name)

    if output.is_dir() and not rebuild:
        try:
            existing = load_prepared_dataset(output)
        except Exception as exc:
            raise ValueError(f"Concatenated prepared dataset is stale or unreadable: {output}: {exc}") from exc
        if existing.manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                f"Concatenated prepared dataset is stale: {output}. Re-run with preprocessing rebuild enabled."
            )
        if logger is not None:
            logger.info("Reusing concatenated ML dataset %s | %s", name, output)
        return existing
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for dataset in datasets:
        if str(dataset.manifest.get("mode")) != mode:
            raise ValueError("Cannot concatenate prepared datasets from different modes")
        threshold = float(dataset.manifest["led_threshold_mV"][family])
        if not np.isclose(threshold, fixed_led, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{Path(dataset.manifest['source']).stem}: prepared LED threshold {threshold:g} mV "
                f"does not match experiment.fixed_led_threshold_mV={fixed_led:g} mV"
            )

    first = datasets[0]
    first_time = first.energy_time_ps if family == "energy" else first.timing_time_ps
    first_transform = first.energy_transform if family == "energy" else first.timing_transform
    if first_time is None or first_transform is None:
        raise ValueError(f"First dataset has incomplete {family} ML input metadata")
    for dataset in datasets[1:]:
        time = dataset.energy_time_ps if family == "energy" else dataset.timing_time_ps
        transform = dataset.energy_transform if family == "energy" else dataset.timing_transform
        if time is None or transform is None:
            raise ValueError(f"Prepared dataset has incomplete {family} ML input metadata")
        _same_array(first_time, time, f"{family} time grid")
        _same_array(first_transform.minimum, transform.minimum, f"{family} normalization minimum")
        _same_array(first_transform.maximum, transform.maximum, f"{family} normalization maximum")

    sizes = [dataset.n_events for dataset in datasets]
    offsets = np.cumsum([0, *sizes[:-1]], dtype=np.int64)
    training = np.concatenate(
        [np.asarray(dataset.training, dtype=np.int64) + offset for dataset, offset in zip(datasets, offsets)]
    )
    validation = np.concatenate(
        [np.asarray(dataset.validation, dtype=np.int64) + offset for dataset, offset in zip(datasets, offsets)]
    )
    test = np.concatenate(
        [np.asarray(dataset.test, dtype=np.int64) + offset for dataset, offset in zip(datasets, offsets)]
    )
    development = np.concatenate([training, validation])
    total = int(sum(sizes))

    _copy_windows(datasets, family, output / f"{family}_windows.npy")
    np.save(output / f"{family}_time_ps.npy", np.asarray(first_time, dtype=np.float64))
    np.savez_compressed(
        output / f"{family}_transform.npz",
        minimum=np.asarray(first_transform.minimum, dtype=np.float32),
        maximum=np.asarray(first_transform.maximum, dtype=np.float32),
    )

    bias_voltage = np.concatenate([np.asarray(dataset.bias_voltage_V, dtype=np.float64) for dataset in datasets])
    original_event_index = np.concatenate([np.asarray(dataset.event_index, dtype=np.int64) for dataset in datasets])
    source_dataset = np.concatenate(
        [np.full(dataset.n_events, Path(dataset.manifest["source"]).stem, dtype="U128") for dataset in datasets]
    )
    np.save(output / "event_index.npy", np.arange(total, dtype=np.int64))
    np.save(output / "bias_voltage_V.npy", bias_voltage)
    np.save(output / "source_event_index.npy", original_event_index)
    np.save(output / "source_dataset.npy", source_dataset)
    np.savez_compressed(output / "splits.npz", training=training, validation=validation, test=test)

    led = _concat_optional(datasets, f"{family}_led_time_ps")
    anchor = _concat_optional(datasets, f"{family}_anchor_time_ps")
    anchor_offset = _concat_optional(datasets, f"{family}_anchor_offset_ps")
    if led is None or anchor is None or anchor_offset is None:
        raise ValueError(f"Cannot concatenate: {family} LED/anchor arrays are incomplete")
    np.save(output / f"{family}_led_time_ps.npy", np.asarray(led, dtype=np.float64))
    np.save(output / f"{family}_anchor_time_ps.npy", np.asarray(anchor, dtype=np.float64))
    np.save(output / f"{family}_anchor_offset_ps.npy", np.asarray(anchor_offset, dtype=np.float64))

    led_pair = np.asarray(led[:, 0] - led[:, 1], dtype=np.float64)
    delta_delta = np.asarray(anchor_offset[:, 0] - anchor_offset[:, 1], dtype=np.float64)
    mean_led = float(np.mean(led_pair[training]))
    calibration_bias = mean_led - true_tof
    target = led_pair - delta_delta - true_tof - calibration_bias
    np.save(output / f"{family}_target_ps.npy", target)

    development_ctr = fit_ctr_ps(
        led_pair[development] - true_tof,
        config["fit"],
        seed=int(config["validation"]["seed"]),
        bootstrap=False,
    ).ctr_ps

    source_rows = []
    for dataset, offset in zip(datasets, offsets):
        source_name = Path(dataset.manifest["source"]).stem
        source_rows.append(
            {
                "dataset": source_name,
                "source": dataset.manifest["source"],
                "prepared_dir": str(dataset.directory),
                "fingerprint": dataset.manifest.get("fingerprint"),
                "offset": int(offset),
                "n_events": dataset.n_events,
                "training": int(dataset.training.size),
                "validation": int(dataset.validation.size),
                "test": int(dataset.test.size),
                "voltage_V": voltage_from_name(source_name),
            }
        )

    manifest = {
        "format_version": DATASET_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "dataset_name": str(name),
        "source": f"concatenated::{name}",
        "concatenated": True,
        "source_datasets": source_rows,
        "source_dataset_file": "source_dataset.npy",
        "source_event_index_file": "source_event_index.npy",
        "mode": mode,
        "true_tof_ps": true_tof,
        "n_input_events": total,
        "n_events": total,
        "split": {
            "training": int(training.size),
            "validation": int(validation.size),
            "test": int(test.size),
        },
        "led_threshold_mV": {family: fixed_led},
        "led_threshold_policy": "fixed_by_concatenated_experiment",
        "led_development_ctr_ps": {family: float(development_ctr)},
        "led_development_coverage": {family: int(development.size)},
        "led_development_efficiency": {family: 1.0},
        "led_training_mean_ps": {family: mean_led},
        "calibration_bias_ps": {family: calibration_bias},
        "cfd_fraction": {},
        "cfd_development_ctr_ps": {},
        "ctr_selection_metric": "fixed_bin_histogram_fwhm",
        "ctr_bin_width_ps": float(config["fit"]["bin_width_ps"]),
        "ml_input": config["ml_input"],
        "normalization": first.manifest["normalization"],
        "target_definition": "delta_t_led - delta_delta_anchor - true_tof - global_concatenated_calibration_bias",
        "anchor_definition": first.manifest.get("anchor_definition"),
        "anchor_offset_definition": first.manifest.get("anchor_offset_definition"),
        "corrected_definition": "target - paired_model_prediction",
        "time_reference": first.manifest.get("time_reference"),
    }
    atomic_json(output / "manifest.json", manifest)
    if logger is not None:
        logger.info(
            "Concatenated ML dataset %s | sources=%d | train=%d validation=%d test=%d | fixed LED=%.6g mV",
            name,
            len(datasets),
            training.size,
            validation.size,
            test.size,
            fixed_led,
        )
    return load_prepared_dataset(output)
