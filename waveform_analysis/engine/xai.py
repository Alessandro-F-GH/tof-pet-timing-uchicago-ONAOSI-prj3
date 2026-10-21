from __future__ import annotations

from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from waveform_analysis.core.io import canonical_hash
from waveform_analysis.data.storage import _atomic_npz
from waveform_analysis.data.dataset import PreparedDataset

from waveform_analysis.signal.sample_mask import apply_sample_mask
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import mode_family, waveform_view

# Increment when the meaning or shape of persisted XAI scores changes.
XAI_METHOD_VERSION = 3


def _predict_array(spec, fitted, pair):
    # Match train._model_input exactly: feature transforms such as MultiRocket
    # may rely on contiguous [event, detector, sample] arrays after masking.
    values = np.ascontiguousarray(
        apply_sample_mask(np.asarray(pair, dtype=np.float32), fitted.sample_mask),
        dtype=np.float32,
    )
    if fitted.feature_transform is not None:
        values = fitted.feature_transform.apply(values)
    prediction = np.asarray(
        spec.predict(fitted.artifact, values), dtype=np.float64
    ).reshape(-1)
    if fitted.output_max_abs_ps is not None:
        prediction = np.clip(
            prediction, -fitted.output_max_abs_ps, fitted.output_max_abs_ps
        )
    return prediction


def calibration_mean_waveform(
    control_dataset: PreparedDataset, mode: str, *, cache_root: str | Path | None = None,
) -> NDArray[np.float32]:
    """Return detector-specific sample means from normalized control waveforms.

    The returned shape is ``(2, n_samples)`` on the full prepared input grid,
    before model masking or feature transformation. No blind statistics are used.
    Accumulation uses float64 and the replacement is stored as float32.
    """
    manifest = control_dataset.manifest
    if manifest.get("dataset_role") != "control" or control_dataset.n_events < 1:
        raise ValueError("XAI replacement requires a nonempty prepared control dataset")
    protocol = manifest.get("analysis_protocol_identity") or manifest["fingerprint"]
    identity = canonical_hash({"method_version": XAI_METHOD_VERSION, "mode": mode,
                               "control_protocol": protocol})
    path = (Path(cache_root) / "xai_control_means" / identity[:16] / "mean.npz"
            if cache_root is not None else None)
    family = mode_family(mode)
    time = np.asarray(getattr(control_dataset, f"{family}_time_ps"), dtype=np.float64)
    if path is not None and path.is_file():
        try:
            with np.load(path) as cached:
                mean = np.asarray(cached["mean"], dtype=np.float32)
                if (str(cached["identity"].item()) == identity
                        and mean.shape == (2, time.size) and np.all(np.isfinite(mean))):
                    return mean
        except (OSError, KeyError, ValueError):
            pass
    waveforms = np.asarray(getattr(control_dataset, f"{family}_windows"), dtype=np.float32)
    if waveforms.shape != (control_dataset.n_events, 2, time.size):
        raise ValueError("Control waveforms do not match the XAI input grid")
    mean = np.mean(waveforms, axis=0, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(mean)):
        raise ValueError("Control sample means must be finite")
    if path is not None:
        _atomic_npz(path, mean=mean, identity=np.asarray(identity))
    return mean


def temporal_occlusion_importance(
    spec, fitted, dataset, mode, *, control_dataset, group_size_samples, max_events, seed,
    cache_root=None
):
    group_size_samples = int(group_size_samples)
    max_events = int(max_events)
    if group_size_samples < 1 or max_events < 1:
        raise ValueError("XAI group size and max events must be positive")
    n_events = int(dataset.n_events)
    if n_events < 1:
        raise ValueError("XAI requires at least one blind event")

    family = mode_family(mode)
    if not np.array_equal(getattr(control_dataset, f"{family}_time_ps"),
                          getattr(dataset, f"{family}_time_ps")):
        raise ValueError("Control and explained waveform time grids differ")
    control_transform = getattr(control_dataset, f"{family}_transform")
    target_transform = getattr(dataset, f"{family}_transform")
    if (control_transform is None or target_transform is None
            or not np.array_equal(control_transform.minimum, target_transform.minimum)
            or not np.array_equal(control_transform.maximum, target_transform.maximum)):
        raise ValueError("Control and explained waveform normalization limits differ")
    control_mean = calibration_mean_waveform(control_dataset, mode, cache_root=cache_root)
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
        replacement = control_mean[None, :, start:stop]
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
    transform = (
        dataset.energy_transform if family == "energy" else dataset.timing_transform
    )
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
        "replacement_method": np.asarray("control_sample_mean"),
        "control_mean_normalized": control_mean,
        "control_n_events": np.asarray(control_dataset.n_events, dtype=np.int64),
        "control_analysis_protocol_identity": np.asarray(
            control_dataset.manifest.get("analysis_protocol_identity") or control_dataset.manifest["fingerprint"]
        ),
        "example_waveforms_mV": example,
        "example_event_index": np.asarray(
            dataset.event_index[indices[0]], dtype=np.int64
        ),
        "time_ps": np.asarray(view.time_ps, dtype=np.float64),
        "importance_ps": importance,
        "group_start": np.asarray(starts, dtype=np.int32),
        "group_stop": np.asarray(stops, dtype=np.int32),
        "group_importance_ps": scores_array,
        "event_index": np.asarray(dataset.event_index[indices], dtype=np.int64),
        "n_events": np.asarray(n_selected, dtype=np.int64),
        "group_size_samples": np.asarray(group_size_samples, dtype=np.int64),
    }

def model_xai_method(spec):
    name = getattr(spec, "name", spec)
    if name in {"direct_multirocket_rdst", "shared_multirocket_rdst"}:
        return None
    from waveform_analysis.models.kernel.rdst_common import RDST_MODELS
    return "shapelet_coefficients" if name in RDST_MODELS else "grouped_temporal_occlusion"


def model_xai_version(spec):
    if model_xai_method(spec) is None:
        return None
    from waveform_analysis.engine.shapelet_xai import SHAPELET_XAI_VERSION
    return SHAPELET_XAI_VERSION if model_xai_method(spec) == "shapelet_coefficients" else XAI_METHOD_VERSION


def model_importance(spec, fitted, dataset, mode, **kwargs):
    if model_xai_method(spec) is None:
        raise ValueError(f"XAI is unavailable for {spec.name}")
    if model_xai_method(spec) == "shapelet_coefficients":
        from waveform_analysis.engine.shapelet_xai import coefficient_importance
        return coefficient_importance(fitted)
    return temporal_occlusion_importance(spec, fitted, dataset, mode, **kwargs)
