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
TransformBlockApply = Callable[[Any, np.ndarray, str], np.ndarray]
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
    fit_role: str = "control"
    block_names: tuple[str, ...] = ()
    transform_block: TransformBlockApply | None = None

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
    estimator_formulation: str = "shared"
    feature_transform: FeatureTransformSpec | None = None
    # Optional shared cache identity for models consuming the same frozen features.
    feature_cache_key: str | None = None
    selection_method: str = "development_cv"
    artifact_format: str = "torch"
    requires_frozen_features: bool = False
    bounded_transform_batches: bool = False
    inplace_fit_input: bool = False

    def __post_init__(self) -> None:
        if self.selection_method != "development_cv":
            raise ValueError("selection_method must be development_cv")
        formulation = str(self.estimator_formulation).strip().lower()
        if formulation not in {"shared", "direct"}:
            raise ValueError(
                "estimator_formulation must be either 'shared' or 'direct'"
            )
        object.__setattr__(self, "estimator_formulation", formulation)
        if self.feature_cache_key is not None:
            cache_key = str(self.feature_cache_key).strip()
            if not cache_key:
                raise ValueError("feature_cache_key cannot be empty")
            object.__setattr__(self, "feature_cache_key", cache_key)
