from __future__ import annotations

import numpy as np

from ._linear_ridge_common import (
    LinearTransformArtifact,
    candidates,
    explain,
    fit_ridge,
    predict,
    save,
    transform_parameters,
    validate_pair,
)
from .spec import FeatureTransformSpec, ModelSpec


def _difference(pair: np.ndarray) -> np.ndarray:
    values = validate_pair(pair, "shared_linear_ridge")
    return np.ascontiguousarray(values[:, 0, :] - values[:, 1, :])


def fit_transform(parameters, train_x, *, seed, config):
    del parameters, seed, config
    difference = _difference(train_x)
    artifact = LinearTransformArtifact(
        metadata={
            "definition": "deterministic sample-wise detector difference s1-s2",
            "feature_count": int(difference.shape[1]),
            "training_events": int(difference.shape[0]),
        }
    )
    return artifact, difference


def transform(artifact: LinearTransformArtifact, pair: np.ndarray) -> np.ndarray:
    del artifact
    return _difference(pair)


def fit(params, train_x, train_target, *, seed, config):
    del seed, config
    return fit_ridge(
        params,
        train_x,
        train_target,
        model_name="shared_linear_ridge",
        fit_intercept=False,
        metadata={
            "input_definition": "sample-wise difference of normalized detector waveforms: s1-s2",
            "prediction_definition": "w^T(s1-s2) = g(s1)-g(s2) [ps]",
            "detector_swap_antisymmetry_enforced": True,
            "equivalent_formulation": "shared linear scorer g(s1)-g(s2), with g(s)=w^T s",
        },
    )


FEATURE_TRANSFORM = FeatureTransformSpec(
    name="paired_waveform_difference",
    parameters=transform_parameters,
    fit_transform=fit_transform,
    transform=transform,
)


MODEL_SPEC = ModelSpec(
    name="shared_linear_ridge",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    preserve_temporal_grid=True,
    estimator_formulation="shared",
    feature_transform=FEATURE_TRANSFORM,
)
