from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


def semantic_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (int(base), *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


@dataclass(frozen=True)
class CVFold:
    fold_id: int
    train: np.ndarray
    validation: np.ndarray


@dataclass(frozen=True)
class CVSplit:
    folds: tuple[CVFold, ...]
    seed: int
    shuffle: bool
    n_events: int

    def validate(self) -> None:
        universe = set(range(int(self.n_events)))
        seen_validation: set[int] = set()
        for expected_id, fold in enumerate(self.folds, 1):
            if int(fold.fold_id) != expected_id:
                raise AssertionError("CV fold IDs must be consecutive and one-based")
            train = set(map(int, np.asarray(fold.train, dtype=np.int64)))
            validation = set(map(int, np.asarray(fold.validation, dtype=np.int64)))
            if not validation:
                raise AssertionError("CV validation fold cannot be empty")
            if train & validation:
                raise AssertionError("CV training and validation subsets overlap")
            if train | validation != universe:
                raise AssertionError(
                    "Each CV fold must partition the development population"
                )
            if seen_validation & validation:
                raise AssertionError("CV validation folds overlap")
            seen_validation |= validation
        if seen_validation != universe:
            raise AssertionError(
                "CV validation folds must cover development exactly once"
            )


def make_cv_split(
    n_events: int,
    *,
    population_identity: str,
    batch_seed: int,
    n_folds: int,
    shuffle: bool,
) -> CVSplit:
    n_events = int(n_events)
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if n_events < n_folds:
        raise ValueError(
            "development population must contain at least one event per fold"
        )

    seed = semantic_seed(int(batch_seed), str(population_identity), "cv")
    order = np.arange(n_events, dtype=np.int64)
    if bool(shuffle):
        order = np.random.default_rng(seed).permutation(order)

    validation_parts = np.array_split(order, n_folds)
    all_indices = np.arange(n_events, dtype=np.int64)
    folds = []
    for fold_id, validation in enumerate(validation_parts, 1):
        validation = np.sort(np.asarray(validation, dtype=np.int64))
        mask = np.ones(n_events, dtype=bool)
        mask[validation] = False
        train = all_indices[mask]
        folds.append(CVFold(fold_id, train, validation))

    split = CVSplit(tuple(folds), seed, bool(shuffle), n_events)
    split.validate()
    return split


def fold_assignment(split: CVSplit) -> np.ndarray:
    split.validate()
    assignment = np.empty(int(split.n_events), dtype=np.int16)
    for fold in split.folds:
        assignment[np.asarray(fold.validation, dtype=np.int64)] = int(fold.fold_id)
    return assignment


def cv_split_from_assignment(
    assignment: np.ndarray, *, seed: int, shuffle: bool
) -> CVSplit:
    assignment = np.asarray(assignment, dtype=np.int64).reshape(-1)
    if assignment.size < 2:
        raise ValueError("fold assignment must contain at least two events")
    fold_ids = sorted(set(map(int, assignment)))
    if fold_ids != list(range(1, len(fold_ids) + 1)):
        raise ValueError("fold assignment must use consecutive one-based fold IDs")
    all_indices = np.arange(assignment.size, dtype=np.int64)
    folds = []
    for fold_id in fold_ids:
        validation = np.flatnonzero(assignment == fold_id).astype(np.int64, copy=False)
        train = np.flatnonzero(assignment != fold_id).astype(np.int64, copy=False)
        folds.append(CVFold(fold_id, train, validation))
    split = CVSplit(tuple(folds), int(seed), bool(shuffle), int(assignment.size))
    split.validate()
    return split


def bootstrap_draw_indices(n_events: int, rng: np.random.Generator) -> np.ndarray:
    n_events = int(n_events)
    if n_events < 1:
        raise ValueError("bootstrap requires at least one event")
    return rng.integers(0, n_events, size=n_events, dtype=np.int64)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.splits")
