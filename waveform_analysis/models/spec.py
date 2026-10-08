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
TransformParameterSelector = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
TransformFit = Callable[..., tuple[Any, np.ndarray]]
TransformApply = Callable[[Any, np.ndarray], np.ndarray]
TransformSave = Callable[[Any, Path], None]


@dataclass(frozen=True)
class FeatureTransformSpec:
    """Reusable fit-time feature transformation for models such as MiniRocket.

    ``parameters`` extracts only the settings that define the transformation.
    Downstream estimator hyperparameters are deliberately excluded so one fitted
    transform can be reused across multiple candidates.
    """

    name: str
    parameters: TransformParameterSelector
    fit_transform: TransformFit
    transform: TransformApply
    save: TransformSave | None = None
    log_fit: bool = True

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("feature transform name cannot be empty")
        object.__setattr__(self, "name", name)


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
    feature_transform: FeatureTransformSpec | None = None

    def __post_init__(self) -> None:
        formulation = str(self.estimator_formulation).strip().lower()
        if formulation not in {"shared", "direct"}:
            raise ValueError(
                "estimator_formulation must be either 'shared' or 'direct'"
            )
        object.__setattr__(self, "estimator_formulation", formulation)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.models.spec")
