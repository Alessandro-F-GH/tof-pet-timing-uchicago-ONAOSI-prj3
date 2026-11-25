"""Direct LightGBM regression on concatenated detector waveform samples.

The pipeline owns candidate selection through development cross-validation;
LightGBM is fitted only on the training rows supplied by each pipeline fold.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from waveform_analysis.engine.search import fixed_parameters, grid_candidates, optimization_config
from waveform_analysis.models.spec import ModelSpec


@dataclass
class LightGBMArtifact:
    regressor: object
    metadata: dict


def candidates(space):
    strategy = optimization_config(space).strategy
    return grid_candidates(space) if strategy == "grid" else [fixed_parameters(space)]


def _features(pair):
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("Direct LightGBM expects finite [event, detector=2, time] input")
    return np.ascontiguousarray(x.reshape(x.shape[0], -1))


def fit(params, train_x, train_target, *, seed, config):
    del config
    # Lazy import keeps reporting, registry and unrelated model families usable
    # even when the optional LightGBM dependency is not installed.
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError(
            "direct_lightgbm_regressor requires lightgbm; install waveform_analysis/requirements.txt"
        ) from exc

    x = _features(train_x)
    y = np.asarray(train_target, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or not len(y) or len(y) != len(x) or not np.all(np.isfinite(y)):
        raise ValueError("LightGBM requires aligned, finite one-dimensional targets")

    model = LGBMRegressor(
        objective="regression",
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=int(seed),
        n_jobs=1,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(x, y)
    return LightGBMArtifact(model, {
        "input_definition": "concatenated detector waveforms [s1,s2]",
        "prediction_definition": "f_lightgbm([s1,s2]) [ps]",
        "detector_swap_antisymmetry_enforced": False,
        "training_events": int(len(y)),
        "feature_count": int(x.shape[1]),
        "n_estimators": int(model.n_estimators_),
        "tree_count": int(model.booster_.num_trees()),
        "validation_definition": "external pipeline cross-validation",
    })


def predict(artifact, pair):
    return np.asarray(artifact.regressor.predict(_features(pair)), dtype=np.float64).reshape(-1)


def explain(artifact, pair):
    _features(pair)
    # Normalized per-sample split frequency, following the direct tree
    # explain interface. This is not SHAP and does not encode effect direction.
    importance = np.asarray(artifact.regressor.feature_importances_, dtype=np.float64)
    total = float(importance.sum())
    return importance / total if total > 0 else importance


def save(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


MODEL_SPEC = ModelSpec(
    name="direct_lightgbm_regressor",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    estimator_formulation="direct",
    artifact_format="pickle",
    swap_antisymmetry_enforced=False,
)
