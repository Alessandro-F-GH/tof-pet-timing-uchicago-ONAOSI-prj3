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
    normalization: str
    candidates: CandidateFactory
    fit: ModelFit
    predict: ModelPredict
    save: ModelSave
    explain: ModelExplain | None = None
