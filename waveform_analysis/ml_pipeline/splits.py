from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def semantic_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (int(base), *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


@dataclass(frozen=True)
class FixedValidationSplit:
    tuning_train: np.ndarray
    validation: np.ndarray
    seed: int

    def validate(self, n_events: int) -> None:
        train = set(map(int, self.tuning_train))
        validation = set(map(int, self.validation))
        if train & validation:
            raise AssertionError("fixed tuning train and validation overlap")
        if train | validation != set(range(int(n_events))):
            raise AssertionError("fixed tuning split must cover the prepared population exactly")


@dataclass(frozen=True)
class ReplicaSplit:
    train: np.ndarray
    test: np.ndarray
    seed: int
    replica_index: int

    def validate(self, n_events: int) -> None:
        train = set(map(int, self.train))
        test = set(map(int, self.test))
        if train & test:
            raise AssertionError("replica train and blind test overlap")
        if train | test != set(range(int(n_events))):
            raise AssertionError("replica train/test must cover the prepared population exactly")


def _sample(values: np.ndarray, n_selected: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.int64)
    n_selected = int(n_selected)
    if not 0 < n_selected < values.size:
        raise ValueError("sample size must be between 1 and pool_size - 1")
    perm = rng.permutation(values)
    selected = np.sort(perm[:n_selected])
    remaining = np.sort(perm[n_selected:])
    return remaining, selected


def make_fixed_validation_split(
    n_events: int,
    *,
    analysis_identity: str,
    batch_seed: int,
    validation_fraction: float,
) -> FixedValidationSplit:
    n_events = int(n_events)
    validation_fraction = float(validation_fraction)
    if n_events < 3:
        raise ValueError("Need at least three prepared events")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie in (0, 1)")
    n_validation = int(round(n_events * validation_fraction))
    n_validation = min(n_events - 2, max(1, n_validation))
    seed = semantic_seed(int(batch_seed), "fixed_validation", analysis_identity)
    rng = np.random.default_rng(seed)
    tuning_train, validation = _sample(np.arange(n_events, dtype=np.int64), n_validation, rng)
    split = FixedValidationSplit(tuning_train=tuning_train, validation=validation, seed=seed)
    split.validate(n_events)
    return split


def make_replica_split(
    n_events: int,
    fixed_split: FixedValidationSplit,
    *,
    analysis_identity: str,
    batch_seed: int,
    replica_index: int,
    blind_fraction: float,
) -> ReplicaSplit:
    n_events = int(n_events)
    replica_index = int(replica_index)
    blind_fraction = float(blind_fraction)
    if replica_index < 1:
        raise ValueError("replica_index must be >= 1")
    if not 0.0 < blind_fraction < 1.0:
        raise ValueError("blind_fraction must lie in (0, 1)")
    fixed_split.validate(n_events)

    pool = np.asarray(fixed_split.tuning_train, dtype=np.int64)
    n_test = int(round(n_events * blind_fraction))
    if not 0 < n_test < pool.size:
        raise ValueError(
            "blind_fraction must leave a non-empty variable training subset outside the fixed validation set"
        )

    seed = semantic_seed(int(batch_seed), "bootstrap_replica", analysis_identity, replica_index)
    rng = np.random.default_rng(seed)
    variable_train, test = _sample(pool, n_test, rng)
    train = np.sort(np.concatenate([np.asarray(fixed_split.validation, dtype=np.int64), variable_train]))
    split = ReplicaSplit(train=train, test=test, seed=seed, replica_index=replica_index)
    split.validate(n_events)
    return split
