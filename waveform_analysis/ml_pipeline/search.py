from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from itertools import product
from typing import Any

from .common import canonical_json


SEARCH_STRATEGIES = ("fixed", "grid", "optuna")
PARAMETER_TYPES = ("fixed", "categorical", "int", "float")


@dataclass(frozen=True)
class OptimizationConfig:
    strategy: str
    sampler: str | None = None
    n_trials: int | None = None
    n_startup_trials: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class CandidateScore:
    candidate_id: str
    score: float


def candidate_id(candidate: dict[str, Any], length: int = 12) -> str:
    digest = hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()
    return digest[: int(length)]


def candidate_manifest(candidates):
    out = {}
    for raw in candidates:
        candidate = dict(raw or {})
        identifier = candidate_id(candidate)
        if identifier in out and canonical_json(out[identifier]) != canonical_json(candidate):
            raise RuntimeError(f"Candidate hash collision for {identifier}")
        out[identifier] = candidate
    if not out:
        raise ValueError("candidate list is empty")
    return dict(sorted(out.items()))


def choose_best(scores: list[CandidateScore]) -> CandidateScore:
    valid = [item for item in scores if math.isfinite(float(item.score))]
    if not valid:
        raise RuntimeError("No candidate produced a finite validation score")
    return min(valid, key=lambda item: (float(item.score), item.candidate_id))


def _parameter_specs(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parameters = space.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("model-space parameters must be an object")

    specs: dict[str, dict[str, Any]] = {}
    for name, raw in parameters.items():
        if not isinstance(raw, dict):
            raise ValueError(
                f"Parameter {name!r} must use the typed search-space form, e.g. "
                "{'type':'fixed','value':...} or {'type':'float','low':...,'high':...}"
            )
        spec = copy.deepcopy(raw)
        kind = str(spec.get("type", "")).strip().lower()
        if kind not in PARAMETER_TYPES:
            raise ValueError(
                f"Parameter {name!r} type must be one of {PARAMETER_TYPES}, got {kind!r}"
            )
        spec["type"] = kind

        if kind == "fixed":
            if set(spec) != {"type", "value"}:
                raise ValueError(f"Fixed parameter {name!r} requires exactly type and value")

        elif kind == "categorical":
            if set(spec) != {"type", "choices"}:
                raise ValueError(
                    f"Categorical parameter {name!r} requires exactly type and choices"
                )
            choices = spec["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Categorical parameter {name!r} choices must be non-empty")

        elif kind == "int":
            allowed = {"type", "low", "high", "step", "log"}
            if not set(spec) <= allowed or not {"low", "high"} <= set(spec):
                raise ValueError(
                    f"Integer parameter {name!r} requires low/high and optionally step/log"
                )
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            if high < low or step < 1:
                raise ValueError(f"Invalid integer range for parameter {name!r}")
            if log and step != 1:
                raise ValueError(
                    f"Integer parameter {name!r} cannot combine log=true with step != 1"
                )
            spec.update({"low": low, "high": high, "step": step, "log": log})

        elif kind == "float":
            allowed = {"type", "low", "high", "step", "log"}
            if not set(spec) <= allowed or not {"low", "high"} <= set(spec):
                raise ValueError(
                    f"Float parameter {name!r} requires low/high and optionally step/log"
                )
            low = float(spec["low"])
            high = float(spec["high"])
            step = spec.get("step")
            step = None if step is None else float(step)
            log = bool(spec.get("log", False))
            if not math.isfinite(low) or not math.isfinite(high) or high < low:
                raise ValueError(f"Invalid float range for parameter {name!r}")
            if step is not None and (not math.isfinite(step) or step <= 0):
                raise ValueError(f"Float parameter {name!r} step must be positive")
            if log and (low <= 0 or step is not None):
                raise ValueError(
                    f"Float parameter {name!r} with log=true requires low > 0 and no step"
                )
            spec.update({"low": low, "high": high, "step": step, "log": log})

        specs[str(name)] = spec
    return specs


def optimization_config(space: dict[str, Any]) -> OptimizationConfig:
    raw = space.get("optimization")
    if not isinstance(raw, dict):
        raise ValueError("Every model space must define an optimization object")

    strategy = str(raw.get("strategy", "")).strip().lower()
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(
            f"optimization.strategy must be one of {SEARCH_STRATEGIES}, got {strategy!r}"
        )

    specs = _parameter_specs(space)
    if strategy == "fixed":
        extra = set(raw) - {"strategy"}
        if extra:
            raise ValueError(f"Fixed optimization does not accept options: {sorted(extra)}")
        non_fixed = [name for name, spec in specs.items() if spec["type"] != "fixed"]
        if non_fixed:
            raise ValueError(
                "optimization.strategy='fixed' requires every parameter to be fixed; "
                f"non-fixed: {non_fixed}"
            )
        return OptimizationConfig(strategy="fixed")

    if strategy == "grid":
        extra = set(raw) - {"strategy"}
        if extra:
            raise ValueError(f"Grid optimization does not accept options: {sorted(extra)}")
        continuous = [
            name for name, spec in specs.items() if spec["type"] not in {"fixed", "categorical"}
        ]
        if continuous:
            raise ValueError(
                "Grid optimization requires finite fixed/categorical parameters; "
                f"continuous parameters: {continuous}"
            )
        return OptimizationConfig(strategy="grid")

    allowed = {"strategy", "sampler", "n_trials", "n_startup_trials", "seed"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Unsupported Optuna optimization options: {sorted(extra)}")
    sampler = str(raw.get("sampler", "tpe")).strip().lower()
    if sampler != "tpe":
        raise ValueError("Only Optuna TPE is currently supported")
    n_trials = int(raw.get("n_trials", 50))
    n_startup_trials = int(raw.get("n_startup_trials", 10))
    seed = raw.get("seed")
    seed = None if seed is None else int(seed)
    if n_trials < 1:
        raise ValueError("optimization.n_trials must be >= 1")
    if n_startup_trials < 0:
        raise ValueError("optimization.n_startup_trials must be >= 0")
    if n_startup_trials > n_trials:
        raise ValueError("optimization.n_startup_trials cannot exceed n_trials")
    return OptimizationConfig(
        strategy="optuna",
        sampler="tpe",
        n_trials=n_trials,
        n_startup_trials=n_startup_trials,
        seed=seed,
    )


def validate_search_space(space: dict[str, Any]) -> OptimizationConfig:
    return optimization_config(space)


def active_parameter_count(space: dict[str, Any]) -> int:
    return sum(spec["type"] != "fixed" for spec in _parameter_specs(space).values())


def fixed_parameters(space: dict[str, Any]) -> dict[str, Any]:
    optimization = optimization_config(space)
    if optimization.strategy != "fixed":
        raise ValueError("fixed_parameters requires optimization.strategy='fixed'")
    return {
        name: copy.deepcopy(spec["value"])
        for name, spec in _parameter_specs(space).items()
    }


def grid_candidates(space: dict[str, Any]) -> list[dict[str, Any]]:
    optimization = optimization_config(space)
    if optimization.strategy != "grid":
        raise ValueError("grid_candidates requires optimization.strategy='grid'")

    specs = _parameter_specs(space)
    names = list(specs)
    domains = []
    for name in names:
        spec = specs[name]
        if spec["type"] == "fixed":
            domains.append([copy.deepcopy(spec["value"])])
        else:
            domains.append([copy.deepcopy(value) for value in spec["choices"]])

    if not names:
        return [{}]
    return [dict(zip(names, values)) for values in product(*domains)]


def _is_optuna_categorical_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def suggest_parameters(trial, space: dict[str, Any]) -> dict[str, Any]:
    optimization = optimization_config(space)
    if optimization.strategy != "optuna":
        raise ValueError("suggest_parameters requires optimization.strategy='optuna'")

    resolved: dict[str, Any] = {}
    for name, spec in _parameter_specs(space).items():
        kind = spec["type"]
        if kind == "fixed":
            value = copy.deepcopy(spec["value"])
        elif kind == "categorical":
            choices = spec["choices"]
            if all(_is_optuna_categorical_scalar(choice) for choice in choices):
                value = trial.suggest_categorical(name, choices)
            else:
                index = trial.suggest_int(f"{name}__choice_index", 0, len(choices) - 1)
                value = copy.deepcopy(choices[index])
        elif kind == "int":
            value = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif kind == "float":
            value = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                step=spec.get("step"),
                log=bool(spec.get("log", False)),
            )
        else:  # pragma: no cover - validated above
            raise RuntimeError(f"Unhandled parameter type {kind!r}")
        resolved[name] = value
    return resolved
