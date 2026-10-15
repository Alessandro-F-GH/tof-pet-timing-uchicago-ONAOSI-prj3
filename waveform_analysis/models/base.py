"""Typed model interfaces that preserve existing architectures and artifacts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn


FloatArray = NDArray[np.floating[Any]]


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
