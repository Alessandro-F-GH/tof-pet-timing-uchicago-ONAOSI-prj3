from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .common import canonical_json


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
        return {"best": self.best.as_dict(), "candidates": [row.as_dict() for row in self.candidates]}


def select_candidate(
    candidates: Iterable[Any],
    *,
    fit_candidate: Callable[[Any, int], Any],
    predict_candidate: Callable[[Any, Any], Any],
    score_candidate: Callable[[Any], float],
    seed: int,
    on_candidate_start: Callable[[int, int, Any], None] | None = None,
    on_candidate_result: Callable[[int, int, CandidateResult], None] | None = None,
) -> SearchResult:
    ordered = sorted(candidates, key=canonical_json)
    if not ordered:
        raise ValueError("Candidate list is empty")

    rows: list[CandidateResult] = []
    best: CandidateResult | None = None
    for index, candidate in enumerate(ordered):
        number = index + 1
        if on_candidate_start:
            on_candidate_start(number, len(ordered), candidate)
        try:
            artifact = fit_candidate(candidate, int(seed) + index)
            score = float(score_candidate(predict_candidate(candidate, artifact)))
            result = CandidateResult(candidate, score, artifact, dict(getattr(artifact, "metadata", {}) or {}))
            if best is None or (score, canonical_json(candidate)) < (best.score, canonical_json(best.candidate)):
                if best is not None:
                    best.artifact = None
                best = result
            else:
                result.artifact = None
        except Exception as exc:
            result = CandidateResult(candidate, float("inf"), error=f"{type(exc).__name__}: {exc}")
        rows.append(result)
        if on_candidate_result:
            on_candidate_result(number, len(ordered), result)

    if best is None:
        raise RuntimeError("Every candidate failed: " + "; ".join(row.error or "unknown failure" for row in rows))
    return SearchResult(best, rows)
