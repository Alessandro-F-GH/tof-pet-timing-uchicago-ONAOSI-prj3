from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def semantic_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (int(base), *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def _split(indices: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size < 2:
        raise ValueError("Need at least two events to split")
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("Split fraction must be in (0, 0.5)")
    shuffled = np.random.default_rng(int(seed)).permutation(values)
    n_right = min(values.size - 1, max(1, int(round(values.size * float(fraction)))))
    return np.sort(shuffled[n_right:]), np.sort(shuffled[:n_right])


@dataclass(frozen=True)
class ExperimentSplit:
    development: np.ndarray
    blind: np.ndarray
    training: np.ndarray
    validation: np.ndarray

    def validate(self) -> None:
        development = set(map(int, self.development))
        blind = set(map(int, self.blind))
        training = set(map(int, self.training))
        validation = set(map(int, self.validation))
        if development & blind:
            raise AssertionError("development and blind overlap")
        if training & validation:
            raise AssertionError("training and validation overlap")
        if not training <= development or not validation <= development:
            raise AssertionError("training/validation must be subsets of development")
        if training | validation != development:
            raise AssertionError("training + validation must exactly partition development")

    def as_dict(self) -> dict[str, int]:
        return {
            "development": int(self.development.size),
            "blind": int(self.blind.size),
            "training": int(self.training.size),
            "validation": int(self.validation.size),
        }


def make_split(
    n_events: int,
    *,
    blind_fraction: float,
    validation_fraction: float,
    seed: int,
) -> ExperimentSplit:
    development, blind = _split(
        np.arange(int(n_events), dtype=np.int64),
        blind_fraction,
        semantic_seed(seed, "blind"),
    )
    training, validation = _split(
        development,
        validation_fraction,
        semantic_seed(seed, "validation"),
    )
    result = ExperimentSplit(development, blind, training, validation)
    result.validate()
    return result
