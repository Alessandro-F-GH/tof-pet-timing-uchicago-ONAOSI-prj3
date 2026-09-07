from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def semantic_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (int(base), *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def split_indices(indices: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(indices, dtype=np.int64).reshape(-1)
    if values.size < 2:
        raise ValueError("Need at least two events to split")
    if not 0.0 < float(fraction) < 0.5:
        raise ValueError("Split fraction must be in (0, 0.5)")
    shuffled = np.random.default_rng(int(seed)).permutation(values)
    n_right = min(values.size - 1, max(1, int(round(values.size * float(fraction)))))
    return np.sort(shuffled[n_right:]), np.sort(shuffled[:n_right])


@dataclass(frozen=True)
class DevelopmentTestSplit:
    development: np.ndarray
    test: np.ndarray

    def validate(self) -> None:
        if set(map(int, self.development)) & set(map(int, self.test)):
            raise AssertionError("development and test overlap")


@dataclass(frozen=True)
class TrainValidationSplit:
    training: np.ndarray
    validation: np.ndarray

    def validate(self, development: np.ndarray) -> None:
        development_set = set(map(int, development))
        training = set(map(int, self.training))
        validation = set(map(int, self.validation))
        if training & validation:
            raise AssertionError("training and validation overlap")
        if training | validation != development_set:
            raise AssertionError("training + validation must exactly partition development")


def split_development_test(indices: np.ndarray, *, test_fraction: float, seed: int) -> DevelopmentTestSplit:
    development, test = split_indices(indices, test_fraction, semantic_seed(seed, "test"))
    result = DevelopmentTestSplit(development, test)
    result.validate()
    return result


def split_training_validation(development: np.ndarray, *, validation_fraction: float, seed: int) -> TrainValidationSplit:
    training, validation = split_indices(development, validation_fraction, semantic_seed(seed, "validation"))
    result = TrainValidationSplit(training, validation)
    result.validate(development)
    return result
