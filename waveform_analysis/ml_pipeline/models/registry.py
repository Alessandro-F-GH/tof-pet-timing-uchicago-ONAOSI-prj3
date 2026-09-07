from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from .spec import ModelSpec

_REGISTRY: dict[str, ModelSpec] = {}
_DISCOVERED = False
_EXCLUDED = {"__init__", "registry", "spec"}


def register_model(spec: ModelSpec, *, replace: bool = False) -> None:
    name = str(spec.name).strip()
    if not name:
        raise ValueError("ModelSpec.name must be non-empty")
    if name in _REGISTRY and not replace:
        raise ValueError(f"Duplicate model name: {name}")
    _REGISTRY[name] = spec


def unregister_model(name: str) -> None:
    _REGISTRY.pop(str(name), None)


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    package_path = Path(__file__).resolve().parent
    for info in pkgutil.iter_modules([str(package_path)]):
        if info.name in _EXCLUDED or info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__package__}.{info.name}")
        spec = getattr(module, "MODEL_SPEC", None)
        if spec is not None:
            if not isinstance(spec, ModelSpec):
                raise TypeError(f"{module.__name__}.MODEL_SPEC must be a ModelSpec")
            register_model(spec)
    _DISCOVERED = True


def model_registry() -> dict[str, ModelSpec]:
    _discover()
    return dict(_REGISTRY)


def model_names() -> tuple[str, ...]:
    _discover()
    return tuple(sorted(_REGISTRY))


def get_model(name: str) -> ModelSpec:
    _discover()
    try:
        return _REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name!r}; available: {list(model_names())}") from exc
