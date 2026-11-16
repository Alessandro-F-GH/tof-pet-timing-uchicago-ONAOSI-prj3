"""Post-training TSHAP with frozen, globally benchmark-selected imputers.

Does not call fit/tune or open raw ROOT files. Frozen imputers are loaded
once per board/channel and their identities are validated against Development.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import shutil
import tempfile
from types import MethodType

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.prepared_cache import PreparedCacheLookup
from waveform_analysis.data.results import batch_root, resolve_run_config
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import mode_family, waveform_view
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.imputer_selection import FrozenImputers
from waveform_analysis.engine.train import (
    load_fitted_model, predict_indices, release_training_memory, saved_model_complete,
)
from waveform_analysis.engine import tshap_explainer as base
from waveform_analysis.models import get_model

METHOD_VERSION = 1


def _configure_frozen_explainer(explainer, frozen, predict):
    """Reuse the original TSHAP interpolation/aggregation, replace only scoring.

    The same frozen imputers are used for both complementary coalitions.
    For reference draws, take the mean prediction over saved Control references.
    """
    empty = frozen.empty_references()
    empty_score = float(np.mean(base._predict_bounded(
        predict[0], predict[1], empty, predict[2]
    )))

    def calculate(self, fnc, input_sample, baselines, positions):
        sample = np.asarray(input_sample, dtype=np.float32)
        full = float(np.asarray(fnc(sample[None, ...]))[0])
        nearest = frozen.knn.search(sample) if frozen.knn is not None else None
        scores = []
        for position in positions:
            start = int(position[1])
            width = int(position[2]) if len(position) == 3 else self.window_length
            if width != int(frozen.metadata["window_samples"]):
                raise ValueError("Frozen benchmark does not cover partial TSHAP windows")
            matches = np.flatnonzero(frozen.starts == start)
            if len(matches) != 1:
                raise ValueError("TSHAP window is absent from frozen benchmark")
            index = int(matches[0])
            kept = frozen.reconstruct(sample, "keep_only", index, nearest=nearest)
            removed = frozen.reconstruct(sample, "remove_window", index, nearest=nearest)
            v_keep = float(np.mean(fnc(kept)))
            v_remove = float(np.mean(fnc(removed)))
            scores.append(base._two_feature_shapley(full, empty_score, v_keep, v_remove))
        self.last_window_positions = [(-1, int(p[1])) for p in positions]
        self.last_window_scores = np.asarray(scores, dtype=np.float64)
        return self.last_window_scores.copy()

    explainer._calculate_window_attrib_scores = MethodType(calculate, explainer)
    return explainer


def _save_simple_local_plot(destination, time_ps, waveform_mv, starts, width,
                            raw_scores, event_id, group):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.asarray(time_ps, np.float64) / 1000.
    dt = np.median(np.diff(t))
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(t, waveform_mv[0], label="Detector 1")
    axes[0].plot(t, waveform_mv[1], label="Detector 2")
    axes[0].legend()
    axes[0].set_ylabel("Amplitude (mV)")
    axes[1].bar(t[starts] + width * dt / 2, raw_scores, width=width * dt)
    axes[1].axhline(0, linewidth=.5)
    axes[1].set_ylabel("Paired Shapley score (ps)")
    axes[1].set_xlabel("Time (ns)")
    fig.suptitle(f"{group} | event {event_id}")
    fig.tight_layout()
    filename = f"{group}__event_{int(event_id)}.png"
    fig.savefig(destination / filename, dpi=160)
    plt.close(fig)
    return filename


def explain_frozen(
    results=None, *, model_dir=None, models=None, ml_window=None, mode=None,
    cache_root=None, output_root=None, imputer_root, max_events=100,
    window_samples=20, stride_samples=20, predict_batch_size=64,
    seed=42, overwrite=False, local=False, events_per_group=2,
    extreme_margin=.05, logger=None,
):
    logger = logger or logging.getLogger(__name__)
    runs, source = base._tshap_run_inputs(results, model_dir)
    if not runs:
        raise ValueError("No complete saved model runs")
    output_root = Path(output_root or "xai").expanduser().resolve()
    source = Path(source).resolve()
    if output_root == source or output_root.is_relative_to(source):
        raise ValueError("XAI output must be outside model results")
    selection_root = Path(imputer_root).expanduser().resolve()
    lookup = PreparedCacheLookup()
    outputs = []
    selected = 0
    for run in runs:
        manifest = run["manifest"]
        if models is not None and manifest["model"] not in models:
            continue
        directory = Path(run["directory"])
        config = resolve_run_config(directory)
        base._apply_tshap_cache(config, cache_root=cache_root)
        chosen_window = base._effective_tshap_window(ml_window, model_dir=model_dir)
        if not base._matches_tshap_window(config, chosen_window) or not base._matches_tshap_mode(config, mode):
            continue
        if config.get("window_name") != "wide":
            raise ValueError("Frozen benchmark currently supports wide-window models only")
        selected += 1
        channel = mode_family(config["mode"])
        board = base._find_board_label(directory)
        if board == "Unknown board":
            raise ValueError(f"Cannot infer UC or FBK board from {directory}")
        development = lookup.find("development", config, manifest)
        blind = lookup.find("blind", config, manifest)
        time_ps = np.asarray(getattr(blind, f"{channel}_time_ps"), dtype=np.float64)
        folder = (selection_root / board / channel / "wide" /
                  f"w{window_samples}_s{stride_samples}")
        frozen = FrozenImputers(
            folder, development_identity=development.manifest["analysis_protocol_identity"],
            window_samples=window_samples, stride_samples=stride_samples,
        )
        if frozen.metadata["board"] != board or frozen.metadata["mode"] != channel:
            raise ValueError("Frozen imputer board/mode mismatch")
        if frozen.arrays["fit_reference"].shape[-1] != len(time_ps):
            raise ValueError("Frozen imputer length does not match Blind waveform")
        target_starts, _, _ = base._valid_window_starts(
            len(time_ps), window_samples, stride_samples, frozen.retained,
        )
        if not np.array_equal(target_starts, frozen.starts):
            raise ValueError("Frozen benchmark windows do not match XAI")
        spec = get_model(manifest["model"])
        run_store = RunStore(directory)
        if not saved_model_complete(spec, run_store.model_dir):
            raise FileNotFoundError(f"Missing saved model: {run_store.model_dir}")
        destination = ((base._local_explanation_destination(output_root, directory)
                        if local else base._explanation_destination(output_root, directory))
                       .parent / ("tshap_frozen_local" if local else "tshap_frozen"))
        identity = dict(
            version=METHOD_VERSION, method="frozen_selected_imputer_tshap",
            model=spec.name, board=board, mode=channel,
            final_fit=manifest["stage_fingerprints"]["final_fit"],
            blind=manifest["dataset_populations"]["blind"]["analysis_protocol_identity"],
            imputer_selection=frozen.metadata,
            seed=int(seed), window_samples=int(window_samples),
            stride_samples=int(stride_samples),
            explained_events=(int(events_per_group) * len(base.LOCAL_GROUP_LABELS)
                              if local else int(max_events)),
        )
        fingerprint = canonical_hash(identity)
        if not local and base._complete(destination, fingerprint) and not overwrite:
            outputs.append(destination)
            continue
        if local and base._complete_local(
            destination, fingerprint, len(base.LOCAL_GROUP_LABELS) * events_per_group
        ) and not overwrite:
            outputs.append(destination)
            continue

        predictions = targets = None
        if local:
            predictions, targets = base._load_materialized_local_predictions(
                run_store, blind, manifest, config["mode"],
            )
            items, improvement, _ = base._select_local_events(
                predictions, targets, events_per_group=events_per_group,
                extreme_margin=extreme_margin,
            )
            indices = np.asarray([int(item["index"]) for item in items], np.int64)
            groups = [item["group"] for item in items]
        else:
            indices = base._choose_indices(blind.n_events, max_events,
                                            base._selection_seed(int(seed)))
            groups = None
        waves = waveform_view(blind, config["mode"], indices).materialize()
        transformer = getattr(blind, f"{channel}_transform")
        if transformer is None:
            raise ValueError("Missing frozen physical waveform scale")
        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tshap-frozen-", dir=parent))
        try:
            with diagnostic_random_state():
                fitted = load_fitted_model(
                    spec, run_store.model_dir, run["best"]["parameters"], config,
                )
                try:
                    if fitted.sample_mask is not None and not np.array_equal(
                        np.asarray(fitted.sample_mask, dtype=bool), frozen.retained
                    ):
                        raise ValueError("Frozen imputer mask differs from saved model")
                    predict = lambda wave: base._predict_bounded(
                        spec, fitted, wave, predict_batch_size,
                    )
                    original = predict_indices(
                        spec, fitted, blind, config["mode"], indices,
                        chunk_size=predict_batch_size,
                    )
                    replay = predict(waves)
                    if not np.allclose(original, replay, atol=1e-4, rtol=1e-6):
                        raise ValueError("Frozen TSHAP did not reproduce saved model inference")
                    explainer = base._new_explainer(
                        window_samples, stride_samples, False,
                        predict_batch_size, sample_mask=frozen.retained,
                    )
                    explainer = _configure_frozen_explainer(
                        explainer, frozen, (spec, fitted, predict_batch_size),
                    )
                    maps = np.empty_like(waves, dtype=np.float64)
                    scores = np.empty((len(waves), 1, len(frozen.starts)), np.float64)
                    reference = frozen.empty_references()[:1]
                    for i, pair in enumerate(waves):
                        result, _ = explainer.explain(pair[None], reference, predict)
                        maps[i] = result[0]
                        starts, raw = base._raw_window_data(explainer)
                        if not np.array_equal(starts, frozen.starts):
                            raise ValueError("Frozen TSHAP window positions drifted")
                        scores[i] = raw
                        logger.info("Frozen TSHAP | %s | %d/%d",
                                    spec.name, i+1, len(waves))
                finally:
                    del fitted
                    release_training_memory()

            physical = np.asarray(transformer.inverse(waves), np.float64)
            event_ids = np.asarray(blind.event_index[indices], dtype=np.int64)
            if local:
                plot_names = []
                with (tmp / "events.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["group", "event_id", "dataset_index", "target_ps", "prediction_ps"])
                    for i, group in enumerate(groups):
                        filename = _save_simple_local_plot(
                            tmp, time_ps, physical[i], frozen.starts, window_samples,
                            scores[i, 0], event_ids[i], group,
                        )
                        plot_names.append(filename)
                        writer.writerow([group, int(event_ids[i]), int(indices[i]),
                                         float(targets[indices[i]]), float(original[i])])
                np.savez_compressed(
                    tmp / "local_attributions.npz",
                    event_id=event_ids, dataset_index=indices,
                    group=np.asarray(groups), time_ps=time_ps,
                    waveform_normalized=waves, waveform_mv=physical,
                    attribution_ps_per_sample=maps,
                    raw_window_scores_ps=scores,
                    window_starts=frozen.starts, window_samples=window_samples,
                    target_correction_ps=targets[indices],
                    predicted_correction_ps=original,
                )
                atomic_json(tmp / "summary.json", {
                    "identity_sha256": fingerprint, "identity": identity,
                    "n_explained_events": len(waves), "plots": plot_names,
                    "imputation": "frozen_selected_benchmark_imputers",
                })
            else:
                positive, negative = base._signed_means(scores)
                mean_absolute = np.mean(np.abs(maps), axis=(0, 1))
                base._write_global_importance_csv(tmp, time_ps, mean_absolute)
                base._write_signed_csv(
                    tmp, time_ps, frozen.starts, positive, negative, window_samples,
                )
                base._plot_global(
                    tmp, time_ps, mean_absolute, physical.mean(axis=0),
                    model_name=spec.name, mode=config["mode"], window_name="wide",
                    board_name=board, n_events=len(waves), window_samples=window_samples,
                    window_starts=frozen.starts, signed_positive=positive,
                    signed_negative=negative, std_waveform_mv=physical.std(axis=0),
                )
                np.savez_compressed(
                    tmp / "attributions.npz",
                    event_id=event_ids, time_ps=time_ps,
                    window_attribution_ps=maps, raw_window_scores_ps=scores,
                    window_starts=frozen.starts, window_samples=window_samples,
                    mean_waveform_mv=physical.mean(axis=0),
                    std_waveform_mv=physical.std(axis=0),
                    original_prediction_ps=original,
                )
                atomic_json(tmp / "summary.json", {
                    "identity_sha256": fingerprint, "identity": identity,
                    "n_explained_events": len(waves), "plot_version": base.GLOBAL_PLOT_VERSION,
                    "imputation": "frozen_selected_benchmark_imputers",
                    "selected_imputers": frozen.selected, "board": board,
                    "window_name": "wide", "tshap_window_samples": window_samples,
                    "tshap_stride_samples": stride_samples,
                    "interpretation": "Paired two-coalition Shapley scores; independently benchmark-selected frozen imputers per task",
                })
            base._publish_directory(tmp, destination, overwrite=True)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
        outputs.append(destination)
    if selected == 0:
        raise ValueError("No saved model runs match the requested XAI selection")
    return outputs
