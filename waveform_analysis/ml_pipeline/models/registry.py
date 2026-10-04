from __future__ import annotations

from dataclasses import replace

from .antisymmetric_mlp import MODEL_SPEC as ANTISYMMETRIC_MLP_SPEC
from .direct_mlp import MODEL_SPEC as DIRECT_MLP_SPEC
from .independent_cnn1d import MODEL_SPEC as INDEPENDENT_CNN1D_SPEC
from .linear_ridge import MODEL_SPEC as LINEAR_RIDGE_SPEC
from .locally_connected_mlp import MODEL_SPEC as LOCALLY_CONNECTED_MLP_SPEC
from .minirocket import MODEL_SPEC as MINIROCKET_SPEC
from .onishi_cnn import MODEL_SPEC as _ONISHI_CNN_SPEC
from .shared_cnn1d import MODEL_SPEC as SHARED_CNN1D_SPEC
from .spec import ModelSpec

ONISHI_CNN_SPEC = replace(_ONISHI_CNN_SPEC, estimator_formulation="direct")

_REGISTRY: dict[str, ModelSpec] = {
    ANTISYMMETRIC_MLP_SPEC.name: ANTISYMMETRIC_MLP_SPEC,
    LOCALLY_CONNECTED_MLP_SPEC.name: LOCALLY_CONNECTED_MLP_SPEC,
    SHARED_CNN1D_SPEC.name: SHARED_CNN1D_SPEC,
    DIRECT_MLP_SPEC.name: DIRECT_MLP_SPEC,
    INDEPENDENT_CNN1D_SPEC.name: INDEPENDENT_CNN1D_SPEC,
    ONISHI_CNN_SPEC.name: ONISHI_CNN_SPEC,
    MINIROCKET_SPEC.name: MINIROCKET_SPEC,
    LINEAR_RIDGE_SPEC.name: LINEAR_RIDGE_SPEC,
}


def model_registry() -> dict[str, ModelSpec]:
    return dict(_REGISTRY)


def model_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model(name: str) -> ModelSpec:
    try:
        return _REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; available: {list(model_names())}") from exc
