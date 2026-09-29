from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

CandidateFactory = Callable[[dict[str, Any]], list[dict[str, Any]]]
ModelFit = Callable[..., Any]
ModelPredict = Callable[[Any, np.ndarray], np.ndarray]
ModelSave = Callable[[Any, Path], None]
ModelExplain = Callable[[Any, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    candidates: CandidateFactory
    fit: ModelFit
    predict: ModelPredict
    save: ModelSave
    explain: ModelExplain | None = None
    preserve_temporal_grid: bool = False
    estimator_formulation: str = "shared"

    def __post_init__(self) -> None:
        formulation = str(self.estimator_formulation).strip().lower()
        if formulation not in {"shared", "direct"}:
            raise ValueError(
                "estimator_formulation must be either 'shared' or 'direct'"
            )
        object.__setattr__(self, "estimator_formulation", formulation)
