from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from itertools import product
from typing import Any

from waveform_analysis.core.io import canonical_json

SEARCH_STRATEGIES = ("fixed", "grid", "optuna")
PARAMETER_TYPES = ("fixed", "categorical", "int", "float")


@dataclass(frozen=True)
class OptimizationConfig:
    strategy: str
    sampler: str | None = None
    n_trials: int | None = None
    n_startup_trials: int | None = None


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
        if identifier in out and canonical_json(out[identifier]) != canonical_json(
            candidate
        ):
            raise RuntimeError(f"Candidate hash collision for {identifier}")
        out[identifier] = candidate
    if not out:
        raise ValueError("candidate list is empty")
    return out


def choose_best(scores: list[CandidateScore]) -> CandidateScore:
    valid = [item for item in scores if math.isfinite(float(item.score))]
    if not valid:
        raise RuntimeError("No candidate produced a finite validation score")
    return min(valid, key=lambda item: (float(item.score), item.candidate_id))


def _parameter_specs(space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parameters = space.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ValueError("model-space parameters must be an object")
    specs = {}
    for name, raw in parameters.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Parameter {name!r} must use the typed search-space form")
        spec = copy.deepcopy(raw)
        kind = str(spec.get("type", "")).strip().lower()
        if kind not in PARAMETER_TYPES:
            raise ValueError(
                f"Parameter {name!r} type must be one of {PARAMETER_TYPES}"
            )
        spec["type"] = kind
        if kind == "fixed":
            if set(spec) != {"type", "value"}:
                raise ValueError(
                    f"Fixed parameter {name!r} requires exactly type and value"
                )
        elif kind == "categorical":
            if (
                set(spec) != {"type", "choices"}
                or not isinstance(spec["choices"], list)
                or not spec["choices"]
            ):
                raise ValueError(
                    f"Categorical parameter {name!r} requires non-empty choices"
                )
        elif kind == "int":
            allowed = {"type", "low", "high", "step", "log"}
            if not set(spec) <= allowed or not {"low", "high"} <= set(spec):
                raise ValueError(f"Integer parameter {name!r} requires low/high")
            low, high = int(spec["low"]), int(spec["high"])
            step, log = int(spec.get("step", 1)), bool(spec.get("log", False))
            if high < low or step < 1 or (log and step != 1):
                raise ValueError(f"Invalid integer range for parameter {name!r}")
            spec.update({"low": low, "high": high, "step": step, "log": log})
        else:
            allowed = {"type", "low", "high", "step", "log"}
            if not set(spec) <= allowed or not {"low", "high"} <= set(spec):
                raise ValueError(f"Float parameter {name!r} requires low/high")
            low, high = float(spec["low"]), float(spec["high"])
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
        raise ValueError("Outer-search model spaces must define an optimization object")
    strategy = str(raw.get("strategy", "")).strip().lower()
    if strategy not in SEARCH_STRATEGIES:
        raise ValueError(f"optimization.strategy must be one of {SEARCH_STRATEGIES}")
    specs = _parameter_specs(space)
    if strategy == "fixed":
        if set(raw) != {"strategy"}:
            raise ValueError("Fixed optimization accepts only strategy")
        non_fixed = [name for name, spec in specs.items() if spec["type"] != "fixed"]
        if non_fixed:
            raise ValueError(
                f"fixed optimization requires fixed parameters: {non_fixed}"
            )
        return OptimizationConfig("fixed")
    if strategy == "grid":
        if set(raw) != {"strategy"}:
            raise ValueError("Grid optimization accepts only strategy")
        continuous = [
            name
            for name, spec in specs.items()
            if spec["type"] not in {"fixed", "categorical"}
        ]
        if continuous:
            raise ValueError(
                f"Grid optimization requires finite parameters: {continuous}"
            )
        return OptimizationConfig("grid")
    allowed = {"strategy", "sampler", "n_trials", "n_startup_trials"}
    extra = set(raw) - allowed
    if extra:
        raise ValueError(f"Unsupported Optuna optimization options: {sorted(extra)}")
    sampler = str(raw.get("sampler", "tpe")).strip().lower()
    if sampler != "tpe":
        raise ValueError("Only Optuna TPE is supported")
    n_trials = int(raw.get("n_trials", 50))
    n_startup = int(raw.get("n_startup_trials", 10))
    if n_trials < 1 or n_startup < 0 or n_startup > n_trials:
        raise ValueError("Invalid Optuna trial counts")
    return OptimizationConfig("optuna", sampler, n_trials, n_startup)


def validate_search_space(space):
    return optimization_config(space)


def active_parameter_count(space):
    return sum(spec["type"] != "fixed" for spec in _parameter_specs(space).values())


def fixed_parameters(space):
    if optimization_config(space).strategy != "fixed":
        raise ValueError("fixed_parameters requires fixed strategy")
    return {
        name: copy.deepcopy(spec["value"])
        for name, spec in _parameter_specs(space).items()
    }


def grid_candidates(space):
    if optimization_config(space).strategy != "grid":
        raise ValueError("grid_candidates requires grid strategy")
    specs = _parameter_specs(space)
    names = list(specs)
    domains = [
        [copy.deepcopy(spec["value"])]
        if spec["type"] == "fixed"
        else [copy.deepcopy(v) for v in spec["choices"]]
        for spec in specs.values()
    ]
    if not names:
        return [{}]
    return [dict(zip(names, values)) for values in product(*domains)]


def _is_optuna_categorical_scalar(value):
    return value is None or isinstance(value, (bool, int, float, str))


def suggest_parameters(trial, space):
    if optimization_config(space).strategy != "optuna":
        raise ValueError("suggest_parameters requires optuna strategy")
    resolved = {}
    for name, spec in _parameter_specs(space).items():
        kind = spec["type"]
        if kind == "fixed":
            value = copy.deepcopy(spec["value"])
        elif kind == "categorical":
            choices = spec["choices"]
            if all(_is_optuna_categorical_scalar(choice) for choice in choices):
                value = trial.suggest_categorical(name, choices)
            else:
                token = trial.suggest_categorical(
                    f"{name}__choice", [str(i) for i in range(len(choices))]
                )
                value = copy.deepcopy(choices[int(token)])
        elif kind == "int":
            value = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step", 1),
                log=spec.get("log", False),
            )
        else:
            value = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                step=spec.get("step"),
                log=spec.get("log", False),
            )
        resolved[name] = value
    return resolved


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.search")
