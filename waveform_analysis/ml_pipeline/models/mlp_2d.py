from __future__ import annotations

import torch
from torch import nn

from ._mlp_common import DenseStack, MLPArtifact, candidates, explain, fit_mlp, predict, save
from .spec import ModelSpec


class JointPairMLP(nn.Module):
    """Single MLP over the complete ordered detector pair [s1, s2]."""

    def __init__(self, input_samples: int, architecture, activation: str):
        super().__init__()
        self.network = DenseStack(2 * input_samples, architecture, activation)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"mlp_2d expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        return self.network(pair.flatten(start_dim=1))


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
    validation_x=None,
    validation_target=None,
):
    return fit_mlp(
        model_name="mlp_2d",
        model_factory=JointPairMLP,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "ordered normalized detector pair flattened as [s1, s2]",
            "prediction_definition": "single joint MLP f_theta([s1;s2]) [ps]",
            "detector_swap_antisymmetry_enforced": False,
        },
    )


MODEL_SPEC = ModelSpec(
    name="mlp_2d",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
