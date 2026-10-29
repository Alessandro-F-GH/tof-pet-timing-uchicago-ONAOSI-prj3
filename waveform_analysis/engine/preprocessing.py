from __future__ import annotations
import json, shutil
from pathlib import Path
from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.engine.control_preprocessing import fit_control_artifact
from waveform_analysis.data.preprocessing import preprocess_selected
from waveform_analysis.engine.event_selection import apply_selection_rules
from waveform_analysis.data.preparation import prepare_ml_dataset
from waveform_analysis.data.sample_mask import attach_control_sample_mask
from waveform_analysis.reporting.preprocessing_plots import (
    configure_plotting,
    plot_materialized_event,
)

_SELECTION_DIAGNOSTICS = (
    "photopeak_selection.png",
    "baseline_noise.png",
    "baseline_clipping.png",
    "timing_tot_selection.png",
    "selection_summary.csv",
)


def fit_control(config, *, rebuild=False, logger=None):
    configure_plotting(config["plot_config"])
    modes = tuple(config.get("_control_modes") or [config["mode"]])
    fit_by_mode = config.get("_control_fit_by_mode") or {config["mode"]: config["fit"]}
    return fit_control_artifact(
        config["control"]["root_file"],
        config["control"],
        config["preprocessing"],
        config["fit"],
        fit_by_mode=fit_by_mode,
        modes=modes,
        cache_root=config["preprocessing"]["cache_dir"],
        rebuild=rebuild,
        logger=logger,
    )


def apply_frozen_preprocessing(
    config, role, control_artifact, *, rebuild=False, logger=None
):
    role = str(role)
    if role not in {"control", "development", "blind"}:
        raise ValueError(f"unknown dataset role {role}")
    configure_plotting(config["plot_config"])
    dataset_cfg = config[role]
    cache_root = Path(config["preprocessing"]["cache_dir"])
    selection = apply_selection_rules(
        dataset_cfg["root_file"],
        dataset_cfg,
        config["preprocessing"],
        control_artifact["selection_rules"],
        config["mode"],
        cache_dir=cache_root / f"{role}_selection",
        rebuild=rebuild,
        logger=logger,
    )
    native = preprocess_selected(
        dataset_cfg["root_file"],
        selection,
        dataset_cfg,
        config["preprocessing"],
        config["mode"],
        cache_dir=cache_root / f"{role}_native",
        rebuild=rebuild,
        logger=logger,
    )
    return selection, native


def prepare_role_dataset(
    config, role, control_artifact, *, rebuild=False, logger=None, allow_empty=False
):
    cache = config.get('_prepared_datasets')
    key = canonical_hash({
        'role': role, 'allow_empty': allow_empty, 'dataset': config[role], 'mode': config['mode'],
        'window': config['window_ns'], 'ml_input': config['ml_input'],
        'preprocessing': config['preprocessing'], 'control': control_artifact,
    }) if cache is not None else None
    if cache is not None and not rebuild and key in cache:
        return cache[key]
    selection, native = apply_frozen_preprocessing(
        config, role, control_artifact, rebuild=rebuild, logger=logger
    )
    prepared = prepare_ml_dataset(
        native,
        control_artifact,
        config,
        dataset_config=config[role],
        dataset_role=role,
        allow_empty=allow_empty,
        cache_dir=Path(config["preprocessing"]["cache_dir"]) / f"{role}_ml",
        rebuild=rebuild,
        logger=logger,
    )
    control_prepared = None
    if config["mode"] == "timing_to_timing":
        if role == "control":
            control_prepared = prepared
        else:
            _, control_native = apply_frozen_preprocessing(
                config, "control", control_artifact, rebuild=rebuild, logger=None
            )
            control_prepared = prepare_ml_dataset(
                control_native, control_artifact, config, dataset_config=config["control"],
                dataset_role="control", cache_dir=Path(config["preprocessing"]["cache_dir"]) / "control_ml",
                rebuild=rebuild,
            )
    prepared = attach_control_sample_mask(
        prepared, control_prepared,
        cache_root=Path(config["preprocessing"]["cache_dir"]),
        artifact_root=Path(config["preprocessing"]["artifact_root"]),
        logger=logger
    )
    result = selection, native, prepared
    if cache is not None:
        cache[key] = result
    return result


def publish_preprocessing_diagnostics(
    config, role, control_artifact, batch_root, *, force=False, logger=None
):
    selection, native = apply_frozen_preprocessing(
        config, role, control_artifact, rebuild=False, logger=None
    )
    mode = "energy" if config["mode"] == "energy_to_energy" else "timing"
    out = Path(batch_root).resolve() / "preprocessing" / str(role) / mode
    metadata = out / "manifest.json"
    fingerprint = str(selection.manifest["fingerprint"])
    plot_fingerprint = canonical_hash(config["plot_config"])
    if metadata.is_file() and not force:
        try:
            existing = json.loads(metadata.read_text(encoding="utf-8"))
            if (
                existing.get("selection_fingerprint") == fingerprint
                and existing.get("control_fingerprint")
                == control_artifact["fingerprint"]
                and existing.get("plot_fingerprint") == plot_fingerprint
            ):
                return out
        except (OSError, json.JSONDecodeError):
            pass
    if out.exists():
        shutil.rmtree(out)
    plots_dir = out / "plots"
    tables_dir = out / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _SELECTION_DIAGNOSTICS:
        source = Path(selection.directory) / name
        if source.is_file():
            destination = (
                tables_dir if source.suffix.lower() == ".csv" else plots_dir
            ) / name
            shutil.copy2(source, destination)
            copied.append(str(destination.relative_to(out)))
    family = str(selection.manifest["family"])
    wave = plot_materialized_event(
        native,
        family,
        config[role]["channels"][family],
        plots_dir / "waveform.png",
        f"{role.capitalize()} selected {family}-channel event",
    )
    if wave is not None:
        copied.append(str(Path(wave).relative_to(out)))
    atomic_json(
        metadata,
        {
            "dataset_role": role,
            "source_dataset": str(Path(config[role]["root_file"]).resolve()),
            "control_source_dataset": str(
                Path(config["control"]["root_file"]).resolve()
            ),
            "control_fingerprint": control_artifact["fingerprint"],
            "selection_rules_fingerprint": selection.manifest.get("rules_fingerprint"),
            "selection_fingerprint": fingerprint,
            "n_raw": selection.manifest.get("n_raw"),
            "n_selected": selection.manifest.get("n_selected"),
            "stage_counts": selection.manifest.get("stage_counts"),
            "rules_fitted_on": "control",
            "plot_fingerprint": plot_fingerprint,
            "files": copied,
        },
    )
    return out
