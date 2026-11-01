"""Reuse identical frozen MultiRocket/RDST feature blocks during CPA.

Only direct energy models participate. Reuse is *never* inferred from the
algorithm family: separately fitted random transforms can differ.
"""
from __future__ import annotations

import numpy as np

from waveform_analysis.signal.sample_mask import apply_sample_mask


KERNEL_BLOCKS = {
    "direct_multirocket": "multirocket",
    "direct_rdst": "rdst",
}


def _same_arrays(lhs, rhs):
    if isinstance(lhs, (tuple, list)) and isinstance(rhs, (tuple, list)):
        return len(lhs) == len(rhs) and all(
            _same_arrays(a, b) for a, b in zip(lhs, rhs))
    if lhs is None or rhs is None:
        return lhs is rhs
    a, b = np.asarray(lhs), np.asarray(rhs)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def _same_scaler(a, b):
    return all(_same_arrays(getattr(a, key, None), getattr(b, key, None))
               for key in ("scale_", "mean_", "var_", "n_features_in_"))


def _same_transformer_settings(a, b, ignored=()):
    """Compare transform-time parameters, ignoring only fit RNG and worker count."""
    if type(a) is not type(b):
        return False
    if not hasattr(a, "get_params") or not hasattr(b, "get_params"):
        return False
    pa = a.get_params(deep=False)
    pb = b.get_params(deep=False)
    for key in ignored:
        pa.pop(key, None)
        pb.pop(key, None)
    if pa.keys() != pb.keys():
        return False
    return all(_same_arrays(pa[k], pb[k]) for k in pa)


def component_compatibility(combined_fitted, separate_fitted, block):
    """Return (can_reuse, explanation) for exact fitted-feature equivalence.

    Model-family names or matching hyperparameters are not enough:
    transform kernels/shapelets AND scaler state must coincide. These
    diagnostics are reported once per model, not per input batch.
    """
    if block not in KERNEL_BLOCKS.values():
        return False, f"unsupported block {block!r}"
    a = getattr(getattr(combined_fitted, "feature_transform", None), "artifact", None)
    b = getattr(getattr(separate_fitted, "feature_transform", None), "artifact", None)
    if a is None or b is None:
        return False, "saved fitted transform missing"
    if not hasattr(a, block):
        return False, f"combined transform missing {block} block"
    part = getattr(a, block)
    if type(part) is not type(b):
        return False, "fitted transform artifact types differ"
    if part.shared or b.shared:
        return False, "shared-detector transform cannot replace direct block"

    for key in ("scale_", "mean_", "var_", "n_features_in_"):
        if not _same_arrays(getattr(part.scaler, key, None),
                            getattr(b.scaler, key, None)):
            return False, f"fitted scaler differs ({key})"

    if type(part.transformer) is not type(b.transformer):
        return False, "transformer types differ"
    if not hasattr(part.transformer, "get_params") or not hasattr(b.transformer, "get_params"):
        return False, "transformer parameter inspection unavailable"
    pa = part.transformer.get_params(deep=False).copy()
    pb = b.transformer.get_params(deep=False).copy()
    # Worker configuration and initial RNG state do not change a previously
    # learned transform. Compare learned arrays below, not random_state.
    for key in ("n_jobs", "random_state"):
        pa.pop(key, None)
        pb.pop(key, None)
    if pa.keys() != pb.keys():
        return False, "transformer parameter names differ"
    for key in sorted(pa):
        if not _same_arrays(pa[key], pb[key]):
            return False, f"transformer hyperparameter differs ({key})"

    if block == "multirocket":
        for key in ("input_samples", "normalise", "feature_count"):
            if part.metadata.get(key) != b.metadata.get(key):
                return False, f"MultiROCKET metadata differs ({key})"
        if not _same_arrays(part.transformer.parameter, b.transformer.parameter):
            return False, "learned MultiROCKET raw kernels/biases differ"
        if not _same_arrays(part.transformer.parameter1, b.transformer.parameter1):
            return False, "learned MultiROCKET difference kernels/biases differ"
    elif block == "rdst":
        if part.transformer.n_shapelets_ != b.transformer.n_shapelets_:
            return False, "RDST fitted shapelet counts differ"
        if not _same_arrays(part.transformer.shapelets_, b.transformer.shapelets_):
            return False, "learned RDST shapelets differ"
    return True, f"fitted {block} features and scaler match exactly"


def compatible_component(combined_fitted, separate_fitted, block):
    """Backward-compatible boolean check for batch shared-feature inference."""
    return component_compatibility(combined_fitted, separate_fitted, block)[0]


def predict_shared_energy_group(models, values, *, chunk_size=64):
    """Predict combined and compatible standalone models in one feature pass.

    models maps model name to (spec, fitted, feature_block_or_None).
    The combined entry is required and owns the frozen feature transformer.
    Intermediate features are freed at each batch; no feature cache on disk.
    """
    if "direct_multirocket_rdst" not in models:
        raise ValueError("Combined MultiRocket-RDST model is required")
    if chunk_size < 1:
        raise ValueError("Prediction chunk size must be positive")
    pair = np.asarray(values, np.float32)
    if pair.ndim != 3 or pair.shape[1] != 2:
        raise ValueError("Expected [events, 2 detectors, samples]")
    _, reference, _ = models["direct_multirocket_rdst"]
    mask = np.asarray(reference.sample_mask, bool)
    combined = reference.feature_transform
    if combined is None:
        raise ValueError("Combined fitted feature transform missing")
    limits = [chunk_size, int(combined.artifact.metadata["batch_size"])]
    chunk = min(limits)
    outputs = {key: np.empty(len(pair), np.float64) for key in models}
    blocks = combined.artifact.metadata["feature_blocks"]
    for name, (spec, fitted, part) in models.items():
        if name == "direct_multirocket_rdst":
            if part is not None:
                raise ValueError("Combined model must receive the full feature matrix")
        elif part != KERNEL_BLOCKS.get(name):
            raise ValueError(f"Incorrect feature block for {name}")
        elif not compatible_component(reference, fitted, part):
            raise ValueError(f"Incompatible saved feature transform for {name}")
        if not np.array_equal(np.asarray(fitted.sample_mask, bool), mask):
            raise ValueError("Model sample masks differ")
    for start in range(0, len(pair), chunk):
        stop = min(len(pair), start+chunk)
        waveforms = np.ascontiguousarray(
            apply_sample_mask(pair[start:stop], mask), dtype=np.float32)
        features = combined.apply(waveforms)
        for name, (spec, fitted, block) in models.items():
            low, high = (0, features.shape[1]) if block is None else blocks[block]
            result = np.asarray(spec.predict(fitted.artifact, features[:, low:high]),
                                np.float64).reshape(-1)
            if result.shape != (stop-start,) or not np.all(np.isfinite(result)):
                raise ValueError(f"Invalid grouped prediction from {name}")
            if fitted.output_max_abs_ps is not None:
                result = np.clip(result, -fitted.output_max_abs_ps,
                                 fitted.output_max_abs_ps)
            outputs[name][start:stop] = result
    return outputs
