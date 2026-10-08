"""Torch dataset with the original contiguous float32 conversion policy."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.utils.data import TensorDataset


class WaveformDataset(TensorDataset):
    """Inputs [event, 2, sample] and targets [event], both converted to float32.

    Indexing, length and shape validation are inherited from TensorDataset.
    This introduces no sampling, normalization, augmentation or extra RNG draws.
    """

    def __init__(self, inputs: ArrayLike, target_ps: ArrayLike) -> None:
        super().__init__(
            torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
            torch.from_numpy(np.asarray(target_ps, dtype=np.float32)),
        )
