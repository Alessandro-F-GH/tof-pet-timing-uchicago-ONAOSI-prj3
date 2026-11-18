"""Shared implementation for direct and waveform-difference decision trees."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from waveform_analysis.engine.search import fixed_parameters, grid_candidates, optimization_config
from waveform_analysis.models.spec import ModelSpec


@dataclass
class DecisionTreeArtifact:
    regressor: DecisionTreeRegressor
    metadata: dict


def candidates(space):
    strategy = optimization_config(space).strategy
    return grid_candidates(space) if strategy == "grid" else [fixed_parameters(space)]


def _features(pair, *, shared):
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("Decision tree expects finite [event, detector=2, time] input")
    if shared:
        return np.ascontiguousarray(x[:, 0, :] - x[:, 1, :])
    return np.ascontiguousarray(x.reshape(x.shape[0], -1))


def fit(params, train_x, train_target, *, seed, config, shared):
    del config
    x = _features(train_x, shared=shared)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    if not len(y) or len(y) != len(x) or not np.all(np.isfinite(y)):
        raise ValueError("Decision tree requires aligned, finite training targets")
    regressor = DecisionTreeRegressor(
        criterion=params["criterion"],
        max_depth=int(params["max_depth"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        ccp_alpha=float(params["ccp_alpha"]),
        random_state=int(seed),
    ).fit(x, y)
    return DecisionTreeArtifact(regressor, {
        "input_definition": "sample-wise detector difference s1-s2" if shared
                            else "concatenated detector waveforms [s1,s2]",
        "prediction_definition": "f_tree(s1-s2) [ps]" if shared
                                 else "f_tree([s1,s2]) [ps]",
        "detector_swap_antisymmetry_enforced": False,
        "training_events": int(len(y)),
        "feature_count": int(x.shape[1]),
        "tree_depth": int(regressor.get_depth()),
        "tree_leaves": int(regressor.get_n_leaves()),
    })


def predict(artifact, pair):
    shared = artifact.metadata["input_definition"].startswith("sample-wise")
    x = _features(pair, shared=shared)
    return np.asarray(artifact.regressor.predict(x), dtype=np.float64).reshape(-1)


def explain(artifact, pair):
    shared = artifact.metadata["input_definition"].startswith("sample-wise")
    _features(pair, shared=shared)
    return np.asarray(artifact.regressor.feature_importances_, dtype=np.float64)


def save(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def make_spec(name, *, shared):
    def fit_model(params, train_x, train_target, *, seed, config):
        return fit(params, train_x, train_target, seed=seed, config=config, shared=shared)

    return ModelSpec(
        name=name,
        candidates=candidates,
        fit=fit_model,
        predict=predict,
        save=save,
        explain=explain,
        estimator_formulation="shared" if shared else "direct",
        artifact_format="pickle",
        swap_antisymmetry_enforced=False,
    )
