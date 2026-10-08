from __future__ import annotations

import numpy as np

from waveform_analysis.models.linear.linear_ridge_common import (
    LinearTransformArtifact,
    candidates,
    explain,
    fit_ridge,
    predict,
    save,
    transform_parameters,
    validate_pair,
)
from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec


def _concatenate(pair: np.ndarray) -> np.ndarray:
    values = validate_pair(pair, "direct_linear_ridge")
    contiguous = np.ascontiguousarray(values)
    return contiguous.reshape(contiguous.shape[0], -1)


def fit_transform(parameters, train_x, *, seed, config):
    del parameters, seed, config
    concatenated = _concatenate(train_x)
    artifact = LinearTransformArtifact(
        metadata={
            "definition": "detector-wise concatenation [s1,s2]",
            "feature_count": int(concatenated.shape[1]),
            "training_events": int(concatenated.shape[0]),
        }
    )
    return artifact, concatenated


def transform(artifact: LinearTransformArtifact, pair: np.ndarray) -> np.ndarray:
    del artifact
    return _concatenate(pair)


def fit(params, train_x, train_target, *, seed, config):
    del seed
    return fit_ridge(
        params,
        train_x,
        train_target,
        model_name="direct_linear_ridge",
        fit_intercept=True,
        config=config,
        metadata={
            "input_definition": "concatenated normalized detector waveforms [s1,s2]",
            "prediction_definition": "w1^T s1 + w2^T s2 + b [ps]",
            "detector_swap_antisymmetry_enforced": False,
            "equivalent_formulation": "joint direct linear correction on concatenated detector waveforms",
        },
    )


FEATURE_TRANSFORM = FeatureTransformSpec(
    name="paired_waveform_concatenation",
    parameters=transform_parameters,
    fit_transform=fit_transform,
    transform=transform,
    log_fit=False,
)


MODEL_SPEC = ModelSpec(
    name="direct_linear_ridge",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    estimator_formulation="direct",
    feature_transform=FEATURE_TRANSFORM,
    selection_method="ridge_cv",
)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models.direct_linear_ridge"
)
