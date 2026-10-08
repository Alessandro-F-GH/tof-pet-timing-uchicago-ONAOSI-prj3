from __future__ import annotations

import numpy as np

from .sample_mask import apply_sample_mask
from .splits import semantic_seed
from .view import mode_family, waveform_view

# Increment when the meaning or shape of persisted XAI scores changes.
XAI_METHOD_VERSION = 2


def _predict_array(spec, fitted, pair):
    values = apply_sample_mask(np.asarray(pair, dtype=np.float32), fitted.sample_mask)
    if fitted.feature_transform is not None:
        values = fitted.feature_transform.apply(values)
    prediction = np.asarray(spec.predict(fitted.artifact, values), dtype=np.float64).reshape(-1)
    if fitted.output_max_abs_ps is not None:
        prediction = np.clip(prediction, -fitted.output_max_abs_ps, fitted.output_max_abs_ps)
    return prediction


def _interpolate_group(pair, start, stop):
    """Replace a temporal interval using the neighboring sample values."""
    n_samples = pair.shape[-1]
    left = max(0, start - 1)
    right = min(n_samples - 1, stop)
    if left == right:
        return pair[:, :, left:left + 1]
    alpha = np.linspace(0.0, 1.0, stop - start + 2, dtype=np.float32)[1:-1]
    return (
        pair[:, :, left:left + 1] * (1.0 - alpha)[None, None, :]
        + pair[:, :, right:right + 1] * alpha[None, None, :]
    )


def temporal_occlusion_importance(spec, fitted, dataset, mode, *, group_size_samples, max_events, seed):
    group_size_samples = int(group_size_samples)
    max_events = int(max_events)
    if group_size_samples < 1 or max_events < 1:
        raise ValueError("XAI group size and max events must be positive")
    n_events = int(dataset.n_events)
    if n_events < 1:
        raise ValueError("XAI requires at least one blind event")

    formulation = spec.estimator_formulation
    if formulation not in {"shared", "direct"}:
        raise ValueError(f"Unsupported XAI estimator formulation: {formulation}")
    n_selected = min(max_events, n_events)
    rng = np.random.default_rng(semantic_seed(seed, "xai_sample"))
    indices = np.arange(n_events, dtype=np.int64)
    if n_selected < n_events:
        indices = np.sort(rng.choice(indices, size=n_selected, replace=False))

    view = waveform_view(dataset, mode, indices)
    pair = np.asarray(view.materialize(), dtype=np.float32)
    baseline = _predict_array(spec, fitted, pair)
    n_samples = pair.shape[-1]
    n_channels = 2 if formulation == "direct" else 1
    importance = np.zeros((n_channels, n_samples), dtype=np.float64)
    starts, stops, group_scores = [], [], []

    for start in range(0, n_samples, group_size_samples):
        stop = min(start + group_size_samples, n_samples)
        replacement = _interpolate_group(pair, start, stop)
        scores = []
        for channel in range(n_channels):
            perturbed = pair.copy()
            if formulation == "shared":
                # Shared scorer: retain one joint temporal-importance estimate.
                perturbed[:, :, start:stop] = replacement
            else:
                # Direct model: intervene on only one physical input channel.
                perturbed[:, channel, start:stop] = replacement[:, channel, :]
            changed = _predict_array(spec, fitted, perturbed)
            score = float(np.mean(np.abs(changed - baseline)))
            importance[channel, start:stop] = score
            scores.append(score)
        starts.append(start)
        stops.append(stop)
        group_scores.append(scores)

    family = mode_family(mode)
    transform = dataset.energy_transform if family == "energy" else dataset.timing_transform
    if transform is None:
        raise ValueError(f"Missing {family} waveform input transform for XAI example")
    example = np.asarray(transform.inverse(pair[0]), dtype=np.float32)

    # Shared scores stay one-dimensional. Direct scores are [detector, sample]
    # and [detector, group], allowing unambiguous channel-specific plotting.
    if formulation == "shared":
        importance = importance[0]
        scores_array = np.asarray(group_scores, dtype=np.float64)[:, 0]
    else:
        scores_array = np.asarray(group_scores, dtype=np.float64).T

    return {
        "estimator_formulation": np.asarray(formulation),
        "method_version": np.asarray(XAI_METHOD_VERSION, dtype=np.int32),
        "example_waveforms_mV": example,
        "example_event_index": np.asarray(dataset.event_index[indices[0]], dtype=np.int64),
        "time_ps": np.asarray(view.time_ps, dtype=np.float64),
        "importance_ps": importance,
        "group_start": np.asarray(starts, dtype=np.int32),
        "group_stop": np.asarray(stops, dtype=np.int32),
        "group_importance_ps": scores_array,
        "event_index": np.asarray(dataset.event_index[indices], dtype=np.int64),
        "n_events": np.asarray(n_selected, dtype=np.int64),
        "group_size_samples": np.asarray(group_size_samples, dtype=np.int64),
    }
