from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import itertools
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from .spec import ModelSpec


@dataclass
class PCARidgeArtifact:
    pca: PCA
    model: Ridge
    n_components: int
    metadata: dict[str, Any]


_PCA_CACHE: OrderedDict[tuple[Any, ...], tuple[PCA, np.ndarray]] = OrderedDict()
_SCORE_CACHE: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
_CACHE_ENTRIES = 3


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {}) or {}
    components = [int(v) for v in parameters.get("n_components", [8, 16, 32, 64, 128])]
    alphas = [float(v) for v in parameters.get("alpha", [0.01, 0.1, 1.0, 10.0, 100.0])]
    return [{"n_components": n, "alpha": alpha} for n, alpha in itertools.product(components, alphas)]


def _array_key(values: np.ndarray) -> tuple[Any, ...]:
    x = np.asarray(values)
    return (int(x.__array_interface__["data"][0]), tuple(x.shape), tuple(x.strides), str(x.dtype))


def _pca_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("pca", {}) or {}


def _max_components(config: dict[str, Any], x: np.ndarray) -> int:
    requested = max(int(v) for v in (config.get("parameters", {}) or {}).get("n_components", [128]))
    return max(1, min(requested, int(x.shape[-1]), int(x.shape[0] * x.shape[1] - 1)))


def _signature(config: dict[str, Any], x: np.ndarray) -> tuple[Any, ...]:
    pca = _pca_config(config)
    return (
        _array_key(x),
        _max_components(config, x),
        bool(pca.get("whiten", True)),
        str(pca.get("svd_solver", "randomized")),
        int(pca.get("seed", 20260815)),
    )


def _pair_scores(pca: PCA, pair: np.ndarray) -> np.ndarray:
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("pca_ridge expects [event, detector, sample]")
    # The same PCA encoder is applied to each detector. Its centering therefore
    # cancels exactly in the difference, preserving detector-swap antisymmetry.
    z0 = pca.transform(x[:, 0, :])
    z1 = pca.transform(x[:, 1, :])
    return np.asarray(z0 - z1, dtype=np.float32)


def _fit_or_get_pca(config: dict[str, Any], train_x: np.ndarray) -> tuple[PCA, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32); key = _signature(config, x)
    cached = _PCA_CACHE.get(key)
    if cached is not None:
        _PCA_CACHE.move_to_end(key)
        return cached
    settings = _pca_config(config); stacked = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float32)
    pca = PCA(
        n_components=_max_components(config, x),
        whiten=bool(settings.get("whiten", True)),
        svd_solver=str(settings.get("svd_solver", "randomized")),
        random_state=int(settings.get("seed", 20260815)),
    )
    pca.fit(stacked); scores = _pair_scores(pca, x); value = (pca, scores); _PCA_CACHE[key] = value; _PCA_CACHE.move_to_end(key)
    while len(_PCA_CACHE) > _CACHE_ENTRIES:
        _PCA_CACHE.popitem(last=False)
    return value


def _cached_scores(artifact: PCARidgeArtifact, pair: np.ndarray) -> np.ndarray:
    x = np.asarray(pair, dtype=np.float32)
    key = (_array_key(x), id(artifact.pca), int(artifact.pca.n_components_))
    cached = _SCORE_CACHE.get(key)
    if cached is not None:
        _SCORE_CACHE.move_to_end(key)
        return cached
    scores = _pair_scores(artifact.pca, x); _SCORE_CACHE[key] = scores; _SCORE_CACHE.move_to_end(key)
    while len(_SCORE_CACHE) > _CACHE_ENTRIES:
        _SCORE_CACHE.popitem(last=False)
    return scores


def fit(params, train_x, train_target, *, seed, config, validation_x=None, validation_target=None, final_epochs=None):
    del seed, validation_x, validation_target, final_epochs
    x = np.asarray(train_x, dtype=np.float32); pca, scores = _fit_or_get_pca(config, x); n_components = int(params["n_components"])
    if n_components < 1 or n_components > int(pca.n_components_):
        raise ValueError(f"n_components={n_components} exceeds fitted PCA size {pca.n_components_}")
    ridge_config = config.get("ridge", {}) or {}
    model = Ridge(
        alpha=float(params["alpha"]),
        fit_intercept=False,
        solver=str(ridge_config.get("solver", "lsqr")),
        tol=float(ridge_config.get("tolerance", 1e-4)),
        max_iter=int(ridge_config.get("max_iterations", 2000)),
    )
    model.fit(scores[:, :n_components], np.asarray(train_target, dtype=np.float64))
    explained = float(np.sum(np.asarray(pca.explained_variance_ratio_[:n_components], dtype=np.float64)))
    metadata = {
        "n_components": n_components,
        "pca_fit_components": int(pca.n_components_),
        "explained_variance_fraction": explained,
        "whiten": bool(pca.whiten),
        "fit_population": "training_detectors_pooled",
    }
    return PCARidgeArtifact(pca, model, n_components, metadata)


def predict(artifact: PCARidgeArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    scores = _cached_scores(artifact, normalized_pair)
    return np.asarray(artifact.model.predict(scores[:, :artifact.n_components]), dtype=np.float64)


def save(artifact: PCARidgeArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path / "model.joblib")


def explain(artifact: PCARidgeArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    del normalized_pair
    components = np.asarray(artifact.pca.components_[:artifact.n_components], dtype=np.float64)
    weights = np.asarray(artifact.model.coef_, dtype=np.float64)
    if bool(artifact.pca.whiten):
        scale = np.sqrt(np.maximum(np.asarray(artifact.pca.explained_variance_[:artifact.n_components], dtype=np.float64), 1e-15))
        weights = weights / scale
    return np.abs(weights @ components)


MODEL_SPEC = ModelSpec(name="pca_ridge", candidates=candidates, fit=fit, predict=predict, save=save, explain=explain)
