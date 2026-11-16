"""Fit, compare and freeze waveform imputers BEFORE running TSHAP.

One global hyperparameter choice per (board, mode, wide-window, coalition task).
Model fitting uses Development only. Tuning and method selection use separate,
disjoint Development subsets. Blind and saved predictive models are never touched.
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from waveform_analysis.core.io import atomic_json
from waveform_analysis.data.cache_store import get_or_build
from waveform_analysis.data.imputation import load_wide_development
from waveform_analysis.engine.conditional_knn_imputer import (
    PairedWindowKNN, WEIGHT_NAMES,
)
from waveform_analysis.models.ppca_imputer import ProbabilisticPCA
from waveform_analysis.engine.imputer_diagnostics import position_rows, save_diagnostics, diagnostics_complete
from waveform_analysis.scripts.evaluate_imputation_benchmark import conditional_mean
from waveform_analysis.scripts.train_ppca_interpolation_blend import linear_interpolation_batch

TASKS = ("remove_window", "keep_only")
VERSION = 1


def _starts(length, width, stride, retained):
    # Identical valid positions for the subsequent frozen-imputer TSHAP.
    from waveform_analysis.engine.tshap_explainer import _valid_window_starts
    result, _, _ = _valid_window_starts(length, width, stride, retained)
    starts = np.asarray(result, dtype=np.int64)
    if not len(starts):
        raise ValueError("No permitted temporal windows")
    return starts


def _masks(length, starts, width, retained):
    out = []
    for task in TASKS:
        for start in starts:
            window = np.zeros(length, dtype=bool)
            window[start:start + width] = True
            observed = retained & (~window if task == "remove_window" else window)
            hidden = retained & ~observed
            if not observed.any() or not hidden.any():
                raise ValueError("Empty observed or hidden coalition")
            out.append((task, int(start), observed, hidden))
    return out


def _scale(transform, length):
    minimum = np.broadcast_to(np.asarray(transform.minimum, np.float64), (2, length))
    maximum = np.broadcast_to(np.asarray(transform.maximum, np.float64), (2, length))
    if not np.isfinite(minimum).all() or not np.all(maximum > minimum):
        raise ValueError("Invalid frozen physical scale")
    return minimum, maximum - minimum


def _physical(waves, minimum, span):
    return np.asarray(waves, dtype=np.float64) * span + minimum


def _score_candidates(waves, masks, *, minimum, span, models, median,
                      control, knn, knn_k, logger=None, return_positions=False):
    """Physical-mV pooled MSE, with equal weight per hidden sample and event."""
    physical = _physical(waves, minimum, span)
    flat = physical.reshape(len(waves), -1)
    control_phys = _physical(control, minimum, span)
    n_windows = len(masks) // 2
    errors = {}
    counts = {task: 0 for task in TASKS}
    positional_errors = {}
    positional_counts = {}

    def add(task, method, setting, error, position):
        key = (task, method, setting)
        value = float(np.sum(error))
        errors[key] = errors.get(key, 0.0) + value
        if return_positions:
            position_key = (*key, position)
            positional_errors[position_key] = positional_errors.get(position_key, 0.0) + value

    # One nearest-neighbor search per event, reused for every task/position/k.
    neighbors = [knn.search(query) for query in waves] if knn is not None else None
    for mask_number, (task, start, observed, hidden) in enumerate(masks):
        window_index = mask_number % n_windows
        hidden_flat = np.tile(hidden, 2)
        observed_flat = np.tile(observed, 2)
        truth = flat[:, hidden_flat]
        counts[task] += int(truth.size)
        if return_positions:
            positional_counts[(task, window_index)] = int(truth.size)
        add(task, "median", "", (truth - median[hidden_flat]) ** 2, window_index)
        # Reference substitution is scored as E_b[(x_M-b_M)^2], not the
        # squared error of its average; this matches reference TSHAP draws.
        reference_truth = control_phys[:, :, hidden].reshape(len(control), -1)
        reference_error = np.mean(
            (truth[:, None, :] - reference_truth[None, :, :]) ** 2, axis=1
        )
        add(task, "reference", "", reference_error, window_index)
        if task == "remove_window":
            width = int(knn.ends[mask_number] - start)
            interpolation = np.stack([
                linear_interpolation_batch(physical[:, d, :], start, width)
                for d in range(2)
            ], axis=1)
            add(task, "linear", "", (
                interpolation.reshape(len(waves), -1) - truth
            ) ** 2, window_index)
        for components, model in models.items():
            prediction = conditional_mean(model, flat, observed_flat, hidden_flat)
            add(task, "ppca_joint", str(components), (prediction - truth) ** 2, window_index)
        if neighbors is not None:
            scale = span[:, hidden]
            truth_normalized = waves[:, :, hidden]
            for event, (indices, distances) in enumerate(neighbors):
                knn_mask = (mask_number + n_windows) % (2 * n_windows)
                for k in knn_k:
                    for weighting in WEIGHT_NAMES:
                        prediction, _ = knn.reconstruct(
                            waves[event], knn_mask, indices[knn_mask],
                            distances[knn_mask], k, weighting
                        )
                        residual = (
                            prediction[:, hidden] - truth_normalized[event]
                        ) * scale
                        add(task, "conditional_knn", f"{k}:{weighting}", residual ** 2, window_index)
        if logger and (mask_number + 1) % max(1, n_windows // 3) == 0:
            logger.info("Imputer scoring | %d/%d masks", mask_number + 1, len(masks))
    pooled = {key: float(np.sqrt(sse / counts[key[0]]))
              for key, sse in errors.items()}
    if not return_positions:
        return pooled
    per_position = {
        key: float(np.sqrt(sse / positional_counts[(key[0], key[3])]))
        for key, sse in positional_errors.items()
    }
    return pooled, per_position


def _winner_per_method(scores, task):
    chosen = {}
    for (group, method, setting), rmse in scores.items():
        if group != task:
            continue
        if method not in chosen or rmse < chosen[method]["rmse_mv"]:
            chosen[method] = {"method": method, "setting": setting,
                              "rmse_mv": rmse}
    return chosen


def _save_csv(path, records):
    if not records:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def benchmark_imputers(
    results, *, cache_root, output_root, modes=("energy", "timing"),
    window_samples=20, stride_samples=20, fit_events=2000,
    tuning_events=200, evaluation_events=200, n_baselines=8,
    components=(8, 16, 32), knn_k=(1, 3, 5, 10, 20),
    reference_block=512, seed=42, overwrite=False, logger=None,
):
    logger = logger or logging.getLogger(__name__)
    results = Path(results).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if results.is_relative_to(output_root) or output_root.is_relative_to(results):
        raise ValueError("Imputer output must be separate from saved model results")
    if any(v < 1 for v in (window_samples, stride_samples, fit_events,
                          tuning_events, evaluation_events, n_baselines, reference_block)):
        raise ValueError("Imputer event counts and window settings must be positive")
    if not components or min(components) < 1 or not knn_k or min(knn_k) < 1:
        raise ValueError("Invalid PPCA or kNN candidate parameters")
    board = results.parent.name
    outputs = []
    for channel in modes:
        if channel not in ("energy", "timing"):
            raise ValueError(f"Unknown waveform channel: {channel}")
        loaded = load_wide_development(
            results, channel, cache_root, return_transform=True,
            include_control=True, include_sample_mask=True,
        )
        waves, time_ps, _, dataset_manifest, transform, controls, retained = loaded
        retained = np.asarray(retained, dtype=bool)
        length = len(time_ps)
        minimum, span = _scale(transform, length)
        starts = _starts(length, window_samples, stride_samples, retained)
        masks = _masks(length, starts, window_samples, retained)
        total = fit_events + tuning_events + evaluation_events
        if total > len(waves):
            raise ValueError(
                f"{board}/{channel}: requires {total} disjoint Development events, "
                f"but only {len(waves)} available"
            )
        order = np.random.default_rng(seed).permutation(len(waves))[:total]
        fit_ids = np.sort(order[:fit_events])
        tune_ids = np.sort(order[fit_events:fit_events + tuning_events])
        eval_ids = np.sort(order[fit_events + tuning_events:])
        fit = np.asarray(waves[fit_ids], np.float32)
        tuning = np.asarray(waves[tune_ids], np.float32)
        evaluation = np.asarray(waves[eval_ids], np.float32)
        control_ids = np.sort(np.random.default_rng(seed + 17).choice(
            len(controls), size=min(n_baselines, len(controls)), replace=False
        ))
        control = np.asarray(controls[control_ids], np.float32)
        if not len(control):
            raise ValueError("No Control reference waveforms")

        destination = (output_root / board / channel / "wide" /
                       f"w{window_samples}_s{stride_samples}")
        expected = {
            "version": VERSION, "board": board, "mode": channel,
            "window": "wide", "development_identity":
                dataset_manifest["analysis_protocol_identity"],
            "window_samples": int(window_samples),
            "stride_samples": int(stride_samples),
            "seed": int(seed),
            "fit_events": int(fit_events), "tuning_events": int(tuning_events),
            "evaluation_events": int(evaluation_events),
            "ppca_components": [int(c) for c in components],
            "knn_k": [int(k) for k in knn_k],
            "n_baselines": int(n_baselines),
        }

        def build(directory):
            physical_fit = _physical(fit, minimum, span)
            flat_fit = physical_fit.reshape(len(fit), -1)
            valid_components = sorted(set(int(c) for c in components
                                          if c < min(flat_fit.shape)))
            if not valid_components:
                raise ValueError("No valid PPCA component candidates")
            models = {}
            for comp in valid_components:
                logger.info("Imputer PPCA fit | %s %s | components=%d",
                            board, channel, comp)
                models[comp] = ProbabilisticPCA(comp).fit(flat_fit)
            maximum_k = max(int(k) for k in knn_k)
            if maximum_k > len(fit):
                raise ValueError("kNN candidate k exceeds fit-reference population")
            engine = PairedWindowKNN(
                fit, starts, window_samples, retained, knn_k,
                scale_mv=span, reference_block=reference_block,
            )
            med = np.median(flat_fit, axis=0)
            logger.info("Imputer tuning | %s/%s | n=%d", board, channel, len(tuning))
            tune_scores, tune_positions = _score_candidates(
                tuning, masks, minimum=minimum, span=span, models=models,
                median=med, control=control, knn=engine,
                knn_k=sorted(set(knn_k)), logger=logger, return_positions=True,
            )
            best_variants = {task: _winner_per_method(tune_scores, task)
                             for task in TASKS}
            logger.info("Imputer held-out evaluation | %s/%s | n=%d",
                        board, channel, len(evaluation))
            eval_all, eval_positions = _score_candidates(
                evaluation, masks, minimum=minimum, span=span, models=models,
                median=med, control=control, knn=engine,
                knn_k=sorted(set(knn_k)), logger=logger, return_positions=True,
            )
            rows, selected = [], {}
            for task in TASKS:
                for method, selected_tuning in sorted(best_variants[task].items()):
                    setting = selected_tuning["setting"]
                    score = eval_all[(task, method, setting)]
                    rows.append(dict(
                        task=task, method=method, setting=setting,
                        tuning_rmse_mv=selected_tuning["rmse_mv"],
                        evaluation_rmse_mv=score,
                    ))
                allowed = [r for r in rows if r["task"] == task]
                selected[task] = min(allowed, key=lambda x: (
                    x["tuning_rmse_mv"], x["method"]
                )).copy()
                logger.info("Imputer SELECTED | %s/%s | %s: %s %s | "
                            "held-out RMSE=%.5g mV", board, channel, task,
                            selected[task]["method"], selected[task]["setting"],
                            selected[task]["evaluation_rmse_mv"])

            arrays = {
                "fit_reference": fit,
                "control_baselines": control,
                "median_physical": med,
                "minimum": minimum,
                "span": span,
                "retained": retained,
                "starts": starts,
                "fit_event_index": np.asarray(fit_ids, dtype=np.int64),
                "tuning_event_index": np.asarray(tune_ids, dtype=np.int64),
                "evaluation_event_index": np.asarray(eval_ids, dtype=np.int64),
            }
            for c, model in models.items():
                arrays[f"ppca_{c}_mean"] = model.mean
                arrays[f"ppca_{c}_W"] = model.W
                arrays[f"ppca_{c}_sigma2"] = np.asarray(model.sigma2)
            np.savez(directory / "fitted_imputers.npz", **arrays)
            _save_csv(directory / "comparison.csv", rows)
            plots = position_rows(
                best_variants, tune_positions, eval_positions,
                starts, time_ps, window_samples,
            )
            save_diagnostics(directory, plots, rows, selected, board, channel)
            atomic_json(directory / "selection.json", {
                **expected, "selected": selected, "methods": rows,
                "selection_rule": "min tuning pooled hidden-sample RMSE (mV), separately per task; independent evaluation reported without selection",
                "tuning_rule": "one global PPCA components or kNN (k, weighting) per task, over all positions",
                "no_blind_used": True, "frozen": True,
            })
        artifact = get_or_build(output_root, destination, build, rebuild=overwrite)
        metadata = json.loads((artifact.path / "selection.json").read_text(encoding="utf-8"))
        for name, value in expected.items():
            if metadata.get(name) != value:
                raise ValueError(f"Frozen imputer cache configuration mismatch ({name}). "
                                 "Choose a new output-root or --overwrite after reviewing inputs.")
        if not diagnostics_complete(artifact.path):
            logger.info("Existing benchmark has no position plots; regenerate them with plot-imputers")
        outputs.append(artifact.path)
        logger.info("Imputer benchmark saved | %s", artifact.path)
    return outputs


class FrozenImputers:
    """Prediction-time pure imputation adapter. No fit or tuning methods."""

    def __init__(self, folder, *, development_identity, window_samples, stride_samples):
        folder = Path(folder).expanduser().resolve()
        if not (folder / ".complete").is_file():
            raise FileNotFoundError(f"Frozen imputer benchmark incomplete: {folder}")
        self.metadata = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
        for key, expected in (
            ("development_identity", development_identity),
            ("window_samples", window_samples), ("stride_samples", stride_samples),
        ):
            if self.metadata.get(key) != expected:
                raise ValueError(f"Frozen imputer {key} mismatch")
        with np.load(folder / "fitted_imputers.npz", allow_pickle=False) as data:
            self.arrays = {key: np.asarray(data[key]) for key in data.files}
        self.starts = self.arrays["starts"]
        self.retained = self.arrays["retained"]
        self.minimum = self.arrays["minimum"]
        self.span = self.arrays["span"]
        self.selected = self.metadata["selected"]
        self.knn = None
        if any(x["method"] == "conditional_knn" for x in self.selected.values()):
            ks = sorted(set(int(x["setting"].split(":")[0]) for x in
                            self.selected.values() if x["method"] == "conditional_knn"))
            self.knn = PairedWindowKNN(
                self.arrays["fit_reference"], self.starts, window_samples,
                self.retained, ks, scale_mv=self.span,
            )

    def _ppca(self, sample, mask, setting):
        prefix = f"ppca_{setting}_"
        mean = self.arrays[prefix + "mean"]
        W = self.arrays[prefix + "W"]
        sigma2 = float(self.arrays[prefix + "sigma2"])
        observed = np.tile(mask, 2)
        missing = np.tile(self.retained & ~mask, 2)
        W_o, W_m = W[observed], W[missing]
        coefficient = np.linalg.solve(
            np.eye(W.shape[1]) + W_o.T @ W_o / sigma2,
            W_o.T / sigma2,
        ).T @ W_m.T
        physical = _physical(sample, self.minimum, self.span).reshape(-1)
        prediction = mean[missing] + (physical[observed] - mean[observed]) @ coefficient
        out = physical.copy()
        out[missing] = prediction
        return ((out.reshape(sample.shape) - self.minimum) / self.span).astype(np.float32)

    def reconstruct(self, sample, task, position, *, nearest=None):
        """Return [draws,2,time] normalized hybrid signals."""
        sample = np.asarray(sample, dtype=np.float32)
        if sample.shape != self.arrays["fit_reference"].shape[1:]:
            raise ValueError("Frozen imputer waveform grid mismatch")
        start = int(self.starts[position])
        width = int(self.metadata["window_samples"])
        window = np.zeros(sample.shape[-1], dtype=bool)
        window[start:start + width] = True
        observed = self.retained & (~window if task == "remove_window" else window)
        missing = self.retained & ~observed
        choice = self.selected[task]
        method, setting = choice["method"], choice["setting"]
        if method == "conditional_knn":
            if self.knn is None:
                raise RuntimeError("Frozen KNN reference missing")
            if nearest is None:
                nearest = self.knn.search(sample)
            side = 0 if task == "keep_only" else 1
            mask = side * len(self.starts) + position
            k_str, weighting = setting.split(":")
            hybrid, _ = self.knn.reconstruct(
                sample, mask, nearest[0][mask], nearest[1][mask],
                int(k_str), weighting,
            )
            return hybrid[None]
        if method == "ppca_joint":
            return self._ppca(sample, observed, setting)[None]
        if method == "reference":
            baselines = self.arrays["control_baselines"]
            batch = np.broadcast_to(sample, baselines.shape).copy()
            batch[:, :, missing] = baselines[:, :, missing]
            return batch
        hybrid = sample.copy()
        if method == "median":
            med = self.arrays["median_physical"].reshape(sample.shape)
            normalized_med = (med - self.minimum) / self.span
            hybrid[:, missing] = normalized_med[:, missing]
        elif method == "linear":
            if task != "remove_window":
                raise ValueError("Linear interpolation valid only for remove_window")
            values = np.stack([
                linear_interpolation_batch(sample[d:d+1], start, width)[0]
                for d in range(2)
            ])
            hybrid[:, start:start+width] = values
        else:
            raise ValueError(f"Unknown frozen imputer: {method}")
        return hybrid[None]

    def empty_references(self):
        # v(empty) is the empirical Development reference expectation.
        return self.arrays["fit_reference"]
