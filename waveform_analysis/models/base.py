"""Typed model interfaces that preserve existing architectures and artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

from .spec import ModelSpec

ArtifactT = TypeVar("ArtifactT")
Parameters = dict[str, Any]
FloatArray = NDArray[np.floating[Any]]


class BaseWaveformModel(ABC, Generic[ArtifactT]):
    """Estimator contract; fitted state stays in the existing artifact object.

    Inputs are [event, detector=2, sample] before feature transformation, or
    [event, feature] afterward. Targets and predictions are [event] in ps.
    Feature fitting, clipping, CV and seeding remain engine responsibilities.
    """

    @abstractmethod
    def fit(
        self,
        parameters: Parameters,
        inputs: FloatArray,
        target_ps: FloatArray,
        *,
        seed: int,
        config: Parameters,
    ) -> ArtifactT:
        """Fit using the existing estimator-specific training procedure."""

    @abstractmethod
    def predict(self, artifact: ArtifactT, inputs: FloatArray) -> NDArray[np.float64]:
        """Return the estimator's unmodified correction in ps."""

    @abstractmethod
    def save(self, artifact: ArtifactT, directory: Path) -> None:
        """Persist the existing artifact schema and filenames."""


class RegisteredWaveformModel(BaseWaveformModel[Any]):
    """Adapt a ModelSpec without introducing new fitted state or serialization."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec

    def fit(
        self,
        parameters: Parameters,
        inputs: FloatArray,
        target_ps: FloatArray,
        *,
        seed: int,
        config: Parameters,
    ) -> Any:
        return self.spec.fit(parameters, inputs, target_ps, seed=seed, config=config)

    def predict(self, artifact: Any, inputs: FloatArray) -> NDArray[np.float64]:
        return self.spec.predict(artifact, inputs)

    def save(self, artifact: Any, directory: Path) -> None:
        self.spec.save(artifact, directory)


class BaseTorchModel(nn.Module, ABC):
    """Parameter-free common interface for native PyTorch waveform modules.

    There is deliberately no constructor, buffer, layer, RNG draw or loss here.
    Inheriting this class preserves initialization order and state_dict keys.
    Training algorithms remain separate because Onishi MSE/full-split training
    and MLP RMSE/internal-holdout training have different scientific semantics.
    """

    @abstractmethod
    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """Map [event, detector=2, sample] to corrections [event] in ps."""

    def predict_array(
        self, pair: FloatArray, device: torch.device, batch_size: int = 512
    ) -> NDArray[np.float64]:
        """Use the established tensor conversion and ordered inference loader."""
        from .torch_runtime import predict_tensor

        return predict_tensor(self, pair, device, batch_size)
