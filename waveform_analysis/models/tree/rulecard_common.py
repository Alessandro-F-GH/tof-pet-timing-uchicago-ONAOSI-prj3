"""RuleCard-style additive regression with shallow univariate/pairwise trees.

Adapted from the user-provided RuleCardGAM2 regression implementation.
Uses sklearn trees and feature ranking to avoid optional RuleCard/RuleTree,
interpret/FAST and external feature-importance packages. Pair candidates
are drawn from the highest-ranked features, *not* FAST interactions.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from waveform_analysis.engine.search import fixed_parameters, grid_candidates, optimization_config
from waveform_analysis.models.spec import ModelSpec


@dataclass
class RuleCardArtifact:
    regressor: "RuleCardRegressor"
    metadata: dict


class RuleCardRegressor:
    """Greedy additive regression of one/two-feature decision-tree rules.

    Each round fits trees on current residuals, selects the smallest
    training squared error, and uses an internal validation split *only*
    for early stopping. Outer model selection remains pipeline CV.
    """

    def __init__(self, *, learning_rate=0.1, max_n_iter=20, patience=4,
                 val_size=0.15, max_depth=3, min_samples_leaf=20,
                 top_k_features=8, include_pairs=True, random_state=42):
        self.learning_rate = float(learning_rate)
        self.max_n_iter = int(max_n_iter)
        self.patience = int(patience)
        self.val_size = float(val_size)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.top_k_features = int(top_k_features)
        self.include_pairs = bool(include_pairs)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or not len(X) or X.shape[0] != len(y):
            raise ValueError("RuleCard requires aligned 2D features and 1D targets")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ValueError("RuleCard requires finite features and targets")
        if not 0 < self.val_size < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid validation fraction or learning rate")
        if min(self.max_n_iter, self.patience, self.max_depth,
               self.min_samples_leaf, self.top_k_features) < 1:
            raise ValueError("Iterations, patience, depth, leaf size and top-k must be positive")

        train_x, val_x, train_y, val_y = train_test_split(
            X, y, test_size=self.val_size, random_state=self.random_state
        )
        self.n_features_in_ = X.shape[1]
        self.base_prediction_ = float(np.mean(train_y))
        prediction_train = np.full(len(train_y), self.base_prediction_)
        prediction_val = np.full(len(val_y), self.base_prediction_)
        self.estimators_ = []
        best_loss = float(np.mean((val_y - prediction_val) ** 2))
        best_iteration = 0
        wait = 0

        # Feature ranking is learned strictly on this fit's training subset.
        selector = RandomForestRegressor(
            n_estimators=32, max_depth=3, n_jobs=1,
            random_state=self.random_state
        )
        selector.fit(train_x, train_y)
        self.feature_importances_ = np.asarray(selector.feature_importances_)
        ranking = np.argsort(-self.feature_importances_, kind="stable")
        selected = ranking[:min(self.top_k_features, X.shape[1])].tolist()
        self.selected_features_ = tuple(int(i) for i in selected)
        candidates = [(int(i),) for i in selected]
        if self.include_pairs:
            candidates.extend(tuple(int(k) for k in pair)
                              for pair in combinations(selected, 2))

        for iteration in range(self.max_n_iter):
            residuals = train_y - prediction_train
            chosen = None
            for indices in candidates:
                tree = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    random_state=self.random_state
                )
                train_slice = train_x[:, indices].reshape(len(train_x), -1)
                tree.fit(train_slice, residuals)
                # Leaf correction is the mean residual in that leaf.
                delta_train = self.learning_rate * tree.predict(train_slice)
                train_loss = float(np.mean(
                    (train_y - prediction_train - delta_train) ** 2
                ))
                if chosen is None or train_loss < chosen[0]:
                    val_slice = val_x[:, indices].reshape(len(val_x), -1)
                    delta_val = self.learning_rate * tree.predict(val_slice)
                    chosen = (train_loss, indices, tree, delta_train, delta_val)

            if chosen is None:
                break
            _, indices, tree, delta_train, delta_val = chosen
            self.estimators_.append((indices, tree))
            prediction_train += delta_train
            prediction_val += delta_val
            val_loss = float(np.mean((val_y - prediction_val) ** 2))
            if val_loss < best_loss - 1e-12:
                best_loss = val_loss
                best_iteration = len(self.estimators_)
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        self.estimators_ = self.estimators_[:best_iteration]
        self.best_iteration_ = best_iteration
        self.best_validation_loss_ = best_loss
        return self

    def predict(self, X):
        if not hasattr(self, "estimators_"):
            raise ValueError("RuleCard is not fitted")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.all(np.isfinite(X)):
            raise ValueError("RuleCard expects finite 2D inputs with fitted feature count")
        prediction = np.full(len(X), self.base_prediction_)
        for indices, tree in self.estimators_:
            prediction += self.learning_rate * tree.predict(
                X[:, indices].reshape(len(X), -1)
            )
        return prediction

    def additive_importances(self):
        importance = np.zeros(self.n_features_in_, dtype=np.float64)
        for indices, tree in self.estimators_:
            # Allocate the tree's split-based importance to original time samples.
            for local, weight in enumerate(tree.feature_importances_):
                importance[indices[local]] += float(weight)
        total = importance.sum()
        return importance / total if total > 0 else importance


def _features(pair, *, shared):
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2 or not np.all(np.isfinite(x)):
        raise ValueError("RuleCard expects finite [event, detector=2, time] input")
    if shared:
        return np.ascontiguousarray(x[:, 0, :] - x[:, 1, :])
    return np.ascontiguousarray(x.reshape(len(x), -1))


def candidates(space):
    strategy = optimization_config(space).strategy
    return grid_candidates(space) if strategy == "grid" else [fixed_parameters(space)]


def fit(params, train_x, train_target, *, seed, config, shared):
    del config
    x = _features(train_x, shared=shared)
    y = np.asarray(train_target, dtype=np.float64).reshape(-1)
    model = RuleCardRegressor(
        learning_rate=params["learning_rate"],
        max_n_iter=params["max_n_iter"],
        patience=params["patience"],
        val_size=params["val_size"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        top_k_features=params["top_k_features"],
        include_pairs=params["include_pairs"],
        random_state=seed,
    ).fit(x, y)
    return RuleCardArtifact(model, {
        "shared_input_difference": bool(shared),
        "input_definition": "sample-wise detector difference s1-s2" if shared
                            else "concatenated detector waveforms [s1,s2]",
        "prediction_definition": "additive_rulecard(s1-s2) [ps]" if shared
                                 else "additive_rulecard([s1,s2]) [ps]",
        "detector_swap_antisymmetry_enforced": False,
        "training_events": int(len(y)),
        "feature_count": int(x.shape[1]),
        "best_iteration": int(model.best_iteration_),
        "best_validation_mse": float(model.best_validation_loss_),
        "selected_features": list(model.selected_features_),
        "interaction_selection": "pairs from top-ranked random-forest features; not FAST",
        "rulecard_variant": "squared-loss additive shallow-tree regression",
    })


def predict(artifact, pair):
    return np.asarray(
        artifact.regressor.predict(_features(pair, shared=artifact.metadata["shared_input_difference"])),
        dtype=np.float64
    ).reshape(-1)


def explain(artifact, pair):
    _features(pair, shared=artifact.metadata["shared_input_difference"])
    return artifact.regressor.additive_importances()


def save(artifact, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "model.pkl").open("wb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)


def make_spec(name, *, shared):
    def fit_model(params, train_x, train_target, *, seed, config):
        return fit(params, train_x, train_target, seed=seed, config=config, shared=shared)

    return ModelSpec(
        name=name, candidates=candidates, fit=fit_model,
        predict=predict, save=save, explain=explain,
        estimator_formulation="shared" if shared else "direct",
        artifact_format="pickle", swap_antisymmetry_enforced=False,
    )
