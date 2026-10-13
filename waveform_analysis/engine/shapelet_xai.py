"""Coefficient explanations for the frozen RDST features; no data refitting."""
from __future__ import annotations

import numpy as np

from waveform_analysis.models.kernel.rdst_common import export_shapelets

SHAPELET_XAI_VERSION = 1


def coefficient_importance(fitted):
    transform = fitted.feature_transform
    if transform is None:
        raise ValueError("Shapelet XAI requires the saved frozen transform")
    artifact = transform.artifact
    arrays = export_shapelets(artifact.transformer)
    n = len(arrays["shapelet_id"])
    coefficients = np.asarray(fitted.artifact.regressor.coef_, dtype=np.float64)
    scales = np.asarray(artifact.scaler.scale_, dtype=np.float64)
    if coefficients.shape != (3 * n,) or scales.shape != coefficients.shape:
        raise ValueError("Shapelet coefficient/scaler feature mapping mismatch")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("Shapelet coefficients and control scales must be finite")
    # Ridge is fitted on control-standardized features. These coefficients
    # already describe a change of one control SD, unlike raw-feature weights.
    scaled = coefficients.reshape(n, 3)
    importance = np.sum(np.abs(scaled), axis=1)
    order = np.lexsort((arrays["shapelet_id"], -importance))
    ranks = np.empty(n, dtype=np.int32)
    ranks[order] = np.arange(1, n + 1)
    arrays.update(
        method=np.asarray("shapelet_coefficients"),
        method_version=np.asarray(SHAPELET_XAI_VERSION, dtype=np.int32),
        estimator_formulation=np.asarray("shared" if artifact.shared else "direct"),
        coefficient_normalization=np.asarray("per_feature_control_standard_deviation"),
        aggregation=np.asarray("sum_absolute_scaled_coefficients"),
        coefficients_scaled_ps=scaled,
        coefficients_raw= (coefficients / scales).reshape(n, 3),
        control_feature_scale=scales.reshape(n, 3),
        shapelet_importance_ps=importance, rank=ranks, ranking=order,
        aeon_version=np.asarray(artifact.metadata["aeon_version"]),
        transform_identity=np.asarray(transform.identity),
        control_analysis_protocol_identity=np.asarray(artifact.metadata.get("control_analysis_protocol_identity", "")),
        intercept_ps=np.asarray(fitted.artifact.regressor.intercept_),
    )
    if artifact.time_ps is None or artifact.sample_indices is None:
        raise ValueError("Shapelet XAI requires the saved retained time grid")
    arrays["time_ps"] = np.asarray(artifact.time_ps)
    arrays["sample_indices"] = np.asarray(artifact.sample_indices)
    return arrays
