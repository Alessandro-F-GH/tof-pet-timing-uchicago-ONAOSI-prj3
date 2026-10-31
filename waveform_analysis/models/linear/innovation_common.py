"""Control-fitted, temporally ordered linear innovations for paired waveforms.

The Cholesky factor of the (regularized) control covariance defines a fixed,
invertible linear change of coordinates. An innovation at time t is the
conditional *linear* residual after the earlier positions in the chosen order.
TShap must always occlude the ORIGINAL waveform and then apply this transform.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_triangular

from waveform_analysis.models.linear.linear_ridge_common import (
    candidates, fit_ridge, predict, save, validate_pair,
)
from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec


@dataclass
class InnovationTransformArtifact:
    mean: np.ndarray
    lower: np.ndarray
    order: np.ndarray
    metadata: dict[str, Any]


def _positive_int(value, key):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"linear innovations {key} must be a positive integer")
    return int(value)


def transform_parameters(params, config):
    del params
    settings = config.get("transform", {})
    if not isinstance(settings, dict):
        raise ValueError("linear innovations transform must be an object")
    allowed = {"max_fit_events", "batch_size", "shrinkage", "relative_jitter", "order", "max_samples"}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown linear innovations settings: {sorted(unknown)}")
    fraction = float(settings.get("shrinkage", 0.01))
    jitter = float(settings.get("relative_jitter", 1e-6))
    if not (np.isfinite(fraction) and 0 <= fraction < 1):
        raise ValueError("linear innovations shrinkage must be in [0,1)")
    if not (np.isfinite(jitter) and jitter > 0):
        raise ValueError("linear innovations relative_jitter must be finite and positive")
    order = str(settings.get("order", "forward"))
    if order not in ("forward", "reverse"):
        raise ValueError("linear innovations order must be forward or reverse")
    return {
        "max_fit_events": _positive_int(settings.get("max_fit_events", 4096), "max_fit_events"),
        "batch_size": _positive_int(settings.get("batch_size", 256), "batch_size"),
        "max_samples": _positive_int(settings.get("max_samples", 2048), "max_samples"),
        "shrinkage": fraction, "relative_jitter": jitter, "order": order,
        "version": 1,
    }


def fit_transform(parameters, train_x, *, seed, config):
    """Fit on control signals only; never inspect timing targets or blind events."""
    pair = validate_pair(train_x, "linear innovations")
    n_events, _, n_samples = pair.shape
    if n_samples > parameters["max_samples"]:
        raise ValueError(
            f"linear innovations needs {n_samples} samples; max_samples="
            f"{parameters['max_samples']} to bound O(p^3) covariance factorization"
        )
    if n_events < 2:
        raise ValueError("linear innovations needs at least two control events")
    n_fit = min(n_events, parameters["max_fit_events"])
    if n_fit < n_events:
        chosen = np.sort(np.random.default_rng(int(seed)).choice(
            n_events, size=n_fit, replace=False,
        ))
        subset = pair[chosen]
    else:
        subset = pair

    # Pool the two detector channels. The same factor is subsequently applied
    # separately to each channel, preserving the shared model's antisymmetry.
    order = np.arange(n_samples, dtype=np.intp)
    if parameters["order"] == "reverse":
        order = order[::-1].copy()
    pooled = np.concatenate((subset[:, 0, :], subset[:, 1, :]), axis=0)
    pooled = np.asarray(pooled[:, order], dtype=np.float64)
    mean = pooled.mean(axis=0)
    centered = pooled - mean
    covariance = (centered.T @ centered) / (len(centered) - 1)
    variance = np.diag(covariance).copy()
    scale = float(variance.max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("linear innovations control signals have zero variance")
    shrinkage = parameters["shrinkage"]
    # Positive diagonal jitter ensures an invertible factor even if samples are
    # perfectly correlated or some active samples are constant.
    covariance *= (1.0 - shrinkage)
    diagonal = np.diag_indices_from(covariance)
    covariance[diagonal] += shrinkage * variance + parameters["relative_jitter"] * scale
    lower = np.linalg.cholesky(covariance)

    artifact = InnovationTransformArtifact(
        mean=mean, lower=lower, order=order,
        metadata={
            "input_samples": n_samples, "feature_count": 2 * n_samples,
            "fit_role": "control", "training_events": n_fit,
            "pooled_waveforms": 2 * n_fit,
            "ordering": parameters["order"], "shrinkage": shrinkage,
            "relative_jitter": parameters["relative_jitter"],
            "batch_size": parameters["batch_size"],
            "feature_definition": "standardized sequential linear innovations, independently per detector",
            "transform_version": 1,
        },
    )
    if config.get("_frozen_transform_only", False):
        return artifact, np.empty((0, 2 * n_samples), dtype=np.float32)
    return artifact, transform(artifact, pair)


def transform(artifact, pair):
    values = validate_pair(pair, "linear innovations")
    count, _, samples = values.shape
    if samples != artifact.metadata["input_samples"]:
        raise ValueError("linear innovations input sample count differs from frozen control grid")
    result = np.empty((count, 2, samples), dtype=np.float32)
    # Each solve is triangular; avoid constructing a dense inverse. Keep the
    # output coordinates in original chronological order, even for reverse fit.
    for start in range(0, count, artifact.metadata["batch_size"]):
        stop = min(start + artifact.metadata["batch_size"], count)
        for detector in range(2):
            ordered = np.asarray(values[start:stop, detector, artifact.order], dtype=np.float64)
            rhs = np.asfortranarray((ordered - artifact.mean).T)
            innovations = solve_triangular(
                artifact.lower, rhs, lower=True, check_finite=False,
                overwrite_b=True,
            ).T
            result[start:stop, detector, artifact.order] = innovations
    return result.reshape(count, -1)


def fit_model(params, train_x, train_target, *, seed, config, shared):
    del seed
    return fit_ridge(
        params, train_x, train_target,
        model_name="shared_linear_innovations" if shared else "direct_linear_innovations",
        fit_intercept=not shared, config=config,
        metadata={
            "input_definition": "frozen control-fitted linear innovations of the original waveforms",
            "prediction_definition": "w^T(z1-z2)" if shared else "w^T[z1,z2]+b",
            "detector_swap_antisymmetry_enforced": shared,
            "regularization_note": "Ridge refitting after whitening is not prediction-invariant",
        },
    )


def make_spec(name, shared):
    def fit(params, train_x, train_target, *, seed, config):
        return fit_model(params, train_x, train_target, seed=seed, config=config, shared=shared)

    def features(parameters, train_x, *, seed, config):
        artifact, matrix = fit_transform(parameters, train_x, seed=seed, config=config)
        if shared and matrix.shape[0]:
            # Frozen training uses _frozen_transform_only=True, so this is
            # mainly for direct standalone transform tests.
            pair = matrix.reshape(len(train_x), 2, -1)
            matrix = pair[:, 0, :] - pair[:, 1, :]
        return artifact, matrix

    def apply(artifact, pair):
        matrix = transform(artifact, pair)
        if shared:
            values = matrix.reshape(len(pair), 2, -1)
            return values[:, 0, :] - values[:, 1, :]
        return matrix

    return ModelSpec(
        name=name, candidates=candidates, fit=fit, predict=predict, save=save,
        estimator_formulation="shared" if shared else "direct",
        feature_transform=FeatureTransformSpec(
            name="linear_innovations_v1",
            parameters=transform_parameters, fit_transform=features, transform=apply,
            fit_role="control",
        ),
        selection_method="development_cv", artifact_format="pickle",
        bounded_transform_batches=True,
    )
