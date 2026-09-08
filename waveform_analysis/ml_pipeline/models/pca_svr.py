from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import itertools
import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVR

from .spec import ModelSpec


@dataclass
class PCASVRArtifact:
    pca: PCA
    model: SVR
    n_components: int
    metadata: dict[str, Any]


_PCA_CACHE: OrderedDict[tuple[Any, ...], tuple[PCA, np.ndarray]] = OrderedDict()
_SCORE_CACHE: OrderedDict[tuple[Any, ...], np.ndarray] = OrderedDict()
_CACHE_ENTRIES = 3


def _kernel_variants(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    spaces = parameters.get("kernels", {"linear": {}, "rbf": {"gamma_scale": [1.0]}}) or {}
    variants: list[dict[str, Any]] = []
    for kernel, settings in spaces.items():
        kernel = str(kernel).lower(); settings = settings or {}
        if kernel == "linear":
            variants.append({"kernel": "linear"})
        elif kernel == "rbf":
            for gamma_scale in settings.get("gamma_scale", [1.0]):
                variants.append({"kernel": "rbf", "gamma_scale": float(gamma_scale)})
        elif kernel == "poly":
            for gamma_scale, degree, coef0 in itertools.product(
                settings.get("gamma_scale", [1.0]), settings.get("degree", [2]), settings.get("coef0", [1.0])
            ):
                variants.append({"kernel": "poly", "gamma_scale": float(gamma_scale), "degree": int(degree), "coef0": float(coef0)})
        else:
            raise ValueError(f"Unsupported PCA SVR kernel: {kernel}")
    if not variants:
        raise ValueError("PCA SVR requires at least one kernel candidate")
    return variants


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {}) or {}
    components = [int(v) for v in parameters.get("n_components", [16, 32, 64])]
    c_values = [float(v) for v in parameters.get("C", [0.1, 1.0, 10.0])]
    epsilons = [float(v) for v in parameters.get("epsilon_ps", [0.0, 10.0])]
    return [
        {"n_components": n, "C": c, "epsilon_ps": epsilon, **kernel}
        for n, c, epsilon, kernel in itertools.product(components, c_values, epsilons, _kernel_variants(parameters))
    ]


def _array_key(values: np.ndarray) -> tuple[Any, ...]:
    x = np.asarray(values)
    return (int(x.__array_interface__["data"][0]), tuple(x.shape), tuple(x.strides), str(x.dtype))


def _pca_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("pca", {}) or {}


def _max_components(config: dict[str, Any], x: np.ndarray) -> int:
    requested = max(int(v) for v in (config.get("parameters", {}) or {}).get("n_components", [64]))
    return max(1, min(requested, int(x.shape[-1]), int(x.shape[0] * x.shape[1] - 1)))


def _signature(config: dict[str, Any], x: np.ndarray) -> tuple[Any, ...]:
    pca = _pca_config(config)
    return (
        _array_key(x), _max_components(config, x), bool(pca.get("whiten", True)),
        str(pca.get("svd_solver", "randomized")), int(pca.get("seed", 20260815)),
    )


def _pair_scores(pca: PCA, pair: np.ndarray) -> np.ndarray:
    x = np.asarray(pair, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError("pca_svr expects [event, detector, sample]")
    return np.asarray(pca.transform(x[:, 0, :]) - pca.transform(x[:, 1, :]), dtype=np.float32)


def _fit_or_get_pca(config: dict[str, Any], train_x: np.ndarray) -> tuple[PCA, np.ndarray]:
    x = np.asarray(train_x, dtype=np.float32); key = _signature(config, x); cached = _PCA_CACHE.get(key)
    if cached is not None:
        _PCA_CACHE.move_to_end(key); return cached
    settings = _pca_config(config); stacked = np.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=np.float32)
    pca = PCA(
        n_components=_max_components(config, x), whiten=bool(settings.get("whiten", True)),
        svd_solver=str(settings.get("svd_solver", "randomized")), random_state=int(settings.get("seed", 20260815)),
    )
    pca.fit(stacked); value = (pca, _pair_scores(pca, x)); _PCA_CACHE[key] = value; _PCA_CACHE.move_to_end(key)
    while len(_PCA_CACHE) > _CACHE_ENTRIES: _PCA_CACHE.popitem(last=False)
    return value


def _cached_scores(artifact: PCASVRArtifact, pair: np.ndarray) -> np.ndarray:
    x = np.asarray(pair, dtype=np.float32); key = (_array_key(x), id(artifact.pca), int(artifact.pca.n_components_)); cached = _SCORE_CACHE.get(key)
    if cached is not None:
        _SCORE_CACHE.move_to_end(key); return cached
    scores = _pair_scores(artifact.pca, x); _SCORE_CACHE[key] = scores; _SCORE_CACHE.move_to_end(key)
    while len(_SCORE_CACHE) > _CACHE_ENTRIES: _SCORE_CACHE.popitem(last=False)
    return scores


def _gamma(params: dict[str, Any], n_components: int) -> str | float:
    if params["kernel"] == "linear": return "scale"
    return float(params.get("gamma_scale", 1.0)) / float(n_components)


def fit(params, train_x, train_target, *, seed, config, validation_x=None, validation_target=None, final_epochs=None):
    del seed, validation_x, validation_target, final_epochs
    x = np.asarray(train_x, dtype=np.float32); pca, scores = _fit_or_get_pca(config, x); n_components = int(params["n_components"])
    if n_components < 1 or n_components > int(pca.n_components_):
        raise ValueError(f"n_components={n_components} exceeds fitted PCA size {pca.n_components_}")
    svr_config = config.get("svr", {}) or {}; kernel = str(params["kernel"])
    model = SVR(
        kernel=kernel, C=float(params["C"]), epsilon=float(params["epsilon_ps"]), gamma=_gamma(params, n_components),
        degree=int(params.get("degree", 3)), coef0=float(params.get("coef0", 0.0)),
        tol=float(svr_config.get("tolerance", 1e-3)), max_iter=int(svr_config.get("max_iterations", 100000)),
        cache_size=float(svr_config.get("cache_size_mb", 2048.0)), shrinking=bool(svr_config.get("shrinking", True)),
    )
    model.fit(scores[:, :n_components], np.asarray(train_target, dtype=np.float64))
    explained = float(np.sum(np.asarray(pca.explained_variance_ratio_[:n_components], dtype=np.float64)))
    metadata = {
        "n_components": n_components, "pca_fit_components": int(pca.n_components_),
        "explained_variance_fraction": explained, "whiten": bool(pca.whiten),
        "fit_population": "training_detectors_pooled", "kernel": kernel, "gamma": float(model._gamma),
        "degree": int(model.degree), "coef0": float(model.coef0), "n_support_vectors": int(model.support_.size),
    }
    return PCASVRArtifact(pca, model, n_components, metadata)


def predict(artifact: PCASVRArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    scores = _cached_scores(artifact, normalized_pair)
    return np.asarray(artifact.model.predict(scores[:, :artifact.n_components]), dtype=np.float64)


def save(artifact: PCASVRArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True); joblib.dump(artifact, path / "model.joblib")


def _score_gradient(model: SVR, scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64); sv = np.asarray(model.support_vectors_, dtype=np.float64); alpha = np.asarray(model.dual_coef_[0], dtype=np.float64); kernel = str(model.kernel)
    if kernel == "linear":
        return np.broadcast_to(alpha @ sv, x.shape).copy()
    dot = x @ sv.T; gamma = float(model._gamma)
    if kernel == "rbf":
        distance2 = np.maximum(np.sum(x * x, axis=1)[:, None] + np.sum(sv * sv, axis=1)[None, :] - 2.0 * dot, 0.0)
        weighted = np.exp(-gamma * distance2) * alpha[None, :]
        return 2.0 * gamma * (weighted @ sv - np.sum(weighted, axis=1)[:, None] * x)
    if kernel == "poly":
        base = gamma * dot + float(model.coef0)
        weighted = alpha[None, :] * float(model.degree) * gamma * np.power(base, int(model.degree) - 1)
        return weighted @ sv
    raise ValueError(f"Unsupported PCA SVR kernel for XAI: {kernel}")


def explain(artifact: PCASVRArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    scores = _cached_scores(artifact, normalized_pair)[:, :artifact.n_components]
    gradient = _score_gradient(artifact.model, scores)
    basis = np.asarray(artifact.pca.components_[:artifact.n_components], dtype=np.float64)
    if bool(artifact.pca.whiten):
        scale = np.sqrt(np.maximum(np.asarray(artifact.pca.explained_variance_[:artifact.n_components], dtype=np.float64), 1e-15))
        basis = basis / scale[:, None]
    return np.mean(np.abs(gradient @ basis), axis=0)


MODEL_SPEC = ModelSpec(name="pca_svr", candidates=candidates, fit=fit, predict=predict, save=save, explain=explain)
