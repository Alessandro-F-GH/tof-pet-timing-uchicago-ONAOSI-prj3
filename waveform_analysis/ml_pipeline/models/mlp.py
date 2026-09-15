from __future__ import annotations

import torch
from torch import nn

from ._mlp_common import DenseStack, MLPArtifact, candidates, explain, fit_mlp, predict, save
from .spec import ModelSpec


class SharedScorerMLP(nn.Module):
    """Shared detector scorer with exact swap antisymmetry g(s1) - g(s2)."""

    def __init__(self, input_samples: int, architecture, activation: str, batch_norm: bool = True):
        super().__init__()
        self.batch_norm = bool(batch_norm)
        self.scorer = DenseStack(input_samples, architecture, activation, self.batch_norm)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"mlp expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        return self.scorer(pair[:, 0, :]) - self.scorer(pair[:, 1, :])


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
        model_name="mlp",
        model_factory=SharedScorerMLP,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "two normalized detector waveforms scored independently by one shared MLP",
            "prediction_definition": "shared MLP correction g_theta(s1)-g_theta(s2) [ps]",
            "detector_swap_antisymmetry_enforced": True,
        },
    )


MODEL_SPEC = ModelSpec(
    name="mlp",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
