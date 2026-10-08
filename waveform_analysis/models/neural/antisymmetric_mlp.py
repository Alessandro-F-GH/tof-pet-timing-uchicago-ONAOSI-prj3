from __future__ import annotations
from waveform_analysis.models.base import BaseTorchModel

import torch

from waveform_analysis.engine.neural_training import (
    DenseStack,
    candidates,
    explain,
    fit_mlp,
    predict,
    save,
)
from waveform_analysis.models.spec import ModelSpec


class AntisymmetricMLP(BaseTorchModel):
    """Shared detector scorer with exact swap antisymmetry g(s1) - g(s2)."""

    def __init__(self, input_samples: int, architecture, activation: str):
        super().__init__()
        self.scorer = DenseStack(input_samples, architecture, activation)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        if pair.ndim != 3 or pair.shape[1] != 2:
            raise ValueError(
                f"antisymmetric_mlp expects [event, detector=2, time], got {tuple(pair.shape)}"
            )
        return self.scorer(pair[:, 0, :]) - self.scorer(pair[:, 1, :])


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
):
    return fit_mlp(
        model_name="antisymmetric_mlp",
        model_factory=AntisymmetricMLP,
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
    name="antisymmetric_mlp",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(
    __name__, "waveform_analysis.ml_pipeline.models.antisymmetric_mlp"
)
