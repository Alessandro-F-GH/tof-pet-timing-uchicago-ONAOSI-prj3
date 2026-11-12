"""Frozen MultiRocket/RDST features with independent block Ridge penalties."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from waveform_analysis.models.kernel import multirocket_common as multirocket
from waveform_analysis.models.kernel import rdst_common as rdst
from waveform_analysis.models.spec import FeatureTransformSpec, ModelSpec

TRANSFORM_VERSION = 1


@dataclass
class CombinedTransformArtifact:
    multirocket: multirocket.MultiRocketTransformArtifact
    rdst: rdst.RDSTTransformArtifact
    shared: bool
    metadata: dict
    time_ps: np.ndarray | None = None
    sample_indices: np.ndarray | None = None


def transform_parameters(params, config):
    # Penalties are deliberately absent from the frozen-transform identity.
    return {
        "multirocket": multirocket.transform_parameters(params, config),
        "rdst": rdst.transform_parameters(params, config),
        "transform_version": TRANSFORM_VERSION,
    }


def fit_transform(parameters, train_x, *, seed, config, shared):
    mr, mr_features = multirocket.fit_transform(
        parameters["multirocket"], train_x, seed=seed, config=config, shared=shared,
    )
    rd, rd_features = rdst.fit_transform(
        parameters["rdst"], train_x, seed=seed, config=config, shared=shared,
    )
    split = mr.metadata["feature_count"]
    total = split + rd.metadata["feature_count"]
    artifact = CombinedTransformArtifact(mr, rd, shared, {
        "feature_blocks": {"multirocket": [0, split], "rdst": [split, total]},
        "feature_count": total, "feature_dtype": "float32",
        "input_samples": mr.metadata["input_samples"],
        "batch_size": mr.metadata["batch_size"],
        "transform_version": TRANSFORM_VERSION,
        "feature_definition": "phi(s1)-phi(s2)" if shared else "phi(s1,s2)",
        "scaling": "per_family_control_standard_deviation",
    })
    if config.get("_frozen_transform_only", False):
        return artifact, np.empty((0, total), dtype=np.float32)
    return artifact, np.concatenate([mr_features, rd_features], axis=1)


def transform(artifact, values):
    if np.shape(values)[-1] != artifact.metadata["input_samples"]:
        raise ValueError("Combined input length differs from its frozen control grid")
    return np.concatenate([
        multirocket.transform(artifact.multirocket, values),
        rdst.transform(artifact.rdst, values),
    ], axis=1)


def transform_block(artifact, values, block_name):
    """Apply one feature family without computing or concatenating the other."""
    if np.shape(values)[-1] != artifact.metadata["input_samples"]:
        raise ValueError("Combined input length differs from its frozen control grid")
    if block_name == "multirocket":
        return multirocket.transform(artifact.multirocket, values)
    if block_name == "rdst":
        return rdst.transform(artifact.rdst, values)
    raise ValueError(f"Unknown combined feature block: {block_name}")


def save_transform(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for name, common in (("multirocket", multirocket), ("rdst", rdst)):
        child = getattr(artifact, name)
        child.time_ps, child.sample_indices = artifact.time_ps, artifact.sample_indices
        for key in ("transform_identity", "control_analysis_protocol_identity"):
            if key in artifact.metadata:
                child.metadata[key] = artifact.metadata[key]
        common.save_transform(child, path / name)
    with (path / "transform.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def fit(params, train_x, train_target, *, seed, config, shared):
    alphas = {}
    for family in ("multirocket", "rdst"):
        value = float(params[f"{family}_alpha"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{family}_alpha must be finite and positive")
        alphas[family] = value
    metadata = config.get("_feature_transform_metadata", {})
    blocks = metadata.get("feature_blocks")
    if not blocks:
        raise ValueError("Combined Ridge requires frozen feature-block metadata")
    split, total = blocks["rdst"]
    x = np.asarray(train_x, dtype=np.float32)
    if (x.ndim != 2 or x.shape[1] != total or split < 1 or split >= total
            or blocks["multirocket"] != [0, split]):
        raise ValueError("Combined Ridge feature-block layout mismatch")
    # Only the engine's disposable row buffer may be scaled/centered in place.
    if not config.get("_fit_input_owned", False):
        x = x.copy()
    scale = np.sqrt(alphas["multirocket"]) / np.sqrt(alphas["rdst"])
    if not np.isfinite(scale) or scale < np.finfo(np.float32).tiny or scale > np.finfo(np.float32).max:
        raise ValueError("Regularization ratio cannot be represented in float32")
    x[:, split:] *= scale
    if not np.all(np.isfinite(x)):
        raise ValueError("Regularization scaling produced non-finite features")
    fitted = multirocket.fit(
        {"ridge_alpha": alphas["multirocket"]}, x, train_target,
        seed=seed, config={**config, "_fit_input_owned": True}, shared=shared,
    )
    # Fold the scaling into coefficients: all inference uses the original,
    # control-standardized cached features, including after model reload.
    fitted.regressor.coef_[split:] *= scale
    fitted.metadata.update(
        multirocket_alpha=alphas["multirocket"], rdst_alpha=alphas["rdst"],
        rdst_fit_scale=float(scale), feature_blocks=blocks,
        coefficient_coordinates="control_standardized_features",
    )
    return fitted


def make_spec(name, fit_transform_callback, fit_callback):
    return ModelSpec(
        name=name, candidates=multirocket.candidates, fit=fit_callback,
        predict=multirocket.predict, save=multirocket.save,
        artifact_format="pickle", requires_frozen_features=True,
        bounded_transform_batches=True, inplace_fit_input=True,
        estimator_formulation="shared" if name.startswith("shared_") else "direct",
        feature_transform=FeatureTransformSpec(
            name=f"{name}_aeon_{multirocket.AEON_VERSION}_v{TRANSFORM_VERSION}",
            parameters=transform_parameters, fit_transform=fit_transform_callback,
            transform=transform, save=save_transform,
            block_names=("multirocket", "rdst"),
            transform_block=transform_block,
        ),
    )
