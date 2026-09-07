from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass
class CandidateResult:
    candidate: Any
    score: float
    artifact: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "score": float(self.score),
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class SearchResult:
    best: CandidateResult
    candidates: list[CandidateResult]

    def as_dict(self) -> dict[str, Any]:
        return {
            "best": self.best.as_dict(),
            "candidates": [row.as_dict() for row in self.candidates],
        }


def select_candidate(
    candidates: Iterable[Any],
    train_data: Any,
    validation_data: Any,
    *,
    fit_candidate: Callable[[Any, Any, int], Any],
    predict_candidate: Callable[[Any, Any, Any], Any],
    score_candidate: Callable[[Any], float],
    seed: int,
) -> SearchResult:
    ordered = sorted(list(candidates), key=canonical_json)
    if not ordered:
        raise ValueError("Candidate list is empty")

    rows: list[CandidateResult] = []
    for index, candidate in enumerate(ordered):
        try:
            artifact = fit_candidate(candidate, train_data, int(seed) + index)
            prediction = predict_candidate(candidate, artifact, validation_data)
            score = float(score_candidate(prediction))
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            rows.append(CandidateResult(candidate, score, artifact, metadata))
        except Exception as exc:  # one bad candidate must not abort the search
            rows.append(
                CandidateResult(
                    candidate,
                    float("inf"),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    successful = [row for row in rows if row.error is None]
    if not successful:
        errors = "; ".join(row.error or "unknown failure" for row in rows)
        raise RuntimeError(f"Every candidate failed: {errors}")
    best = min(successful, key=lambda row: (row.score, canonical_json(row.candidate)))
    return SearchResult(best, rows)
