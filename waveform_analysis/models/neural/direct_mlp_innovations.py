from __future__ import annotations

import torch

from waveform_analysis.engine.neural_training import (
    DenseStack, candidates, explain, fit_mlp, predict, save,
)
from waveform_analysis.models.base import BaseTorchModel
from waveform_analysis.models.linear.direct_linear_innovations import MODEL_SPEC as _INNOVATIONS
from waveform_analysis.models.spec import ModelSpec


class DirectInnovationMLP(BaseTorchModel):
    """Direct fully connected MLP consuming paired, control-fitted innovations."""

    def __init__(self, input_samples: int, architecture, activation: str):
        super().__init__()
        self.regressor = DenseStack(2 * int(input_samples), architecture, activation)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] % 2:
            raise ValueError(
                "direct_mlp_innovations expects [event, 2*time] paired features"
            )
        return self.regressor(features)


def fit(params, train_x, train_target, *, seed, config):
    return fit_mlp(
        model_name="direct_mlp_innovations",
        model_factory=DirectInnovationMLP,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "paired control-fitted forward linear innovations [z1,z2], flattened",
            "prediction_definition": "direct MLP correction f_theta(z1,z2) [ps]",
            "feature_transform": "linear_innovations_v1",
            "detector_swap_antisymmetry_enforced": False,
        },
    )


MODEL_SPEC = ModelSpec(
    name="direct_mlp_innovations",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
    estimator_formulation="direct",
    feature_transform=_INNOVATIONS.feature_transform,
    feature_cache_key="direct_linear_innovations",
)
