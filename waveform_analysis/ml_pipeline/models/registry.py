from __future__ import annotations

from .locally_connected_mlp import MODEL_SPEC as LOCALLY_CONNECTED_MLP_SPEC
from .mlp import MODEL_SPEC as MLP_SPEC
from .onishi_cnn import MODEL_SPEC as ONISHI_CNN_SPEC
from .spec import ModelSpec

_REGISTRY: dict[str, ModelSpec] = {
    MLP_SPEC.name: MLP_SPEC,
    LOCALLY_CONNECTED_MLP_SPEC.name: LOCALLY_CONNECTED_MLP_SPEC,
    ONISHI_CNN_SPEC.name: ONISHI_CNN_SPEC,
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
