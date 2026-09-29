from __future__ import annotations

import torch
from torch import nn

from ._mlp_common import DenseStack, candidates, explain, fit_mlp, predict, save
from .spec import ModelSpec


class DirectPairMLP(nn.Module):
    """Unconstrained MLP acting jointly on both detector waveforms."""

    def __init__(self, input_samples: int, architecture, activation: str):
        super().__init__()
        self.regressor = DenseStack(2 * int(input_samples), architecture, activation)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"direct_mlp expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        return self.regressor(pair.flatten(start_dim=1))


def fit(params, train_x, train_target, *, seed, config):
    return fit_mlp(
        model_name="direct_mlp",
        model_factory=DirectPairMLP,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "paired normalized detector waveforms flattened jointly as [2*time]",
            "prediction_definition": "direct paired MLP correction f_theta(s1,s2) [ps]",
            "detector_swap_antisymmetry_enforced": False,
        },
    )


MODEL_SPEC = ModelSpec(
    name="direct_mlp",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    estimator_formulation="direct",
)
