from __future__ import annotations

from ._cnn1d_common import SharedCNN1D, candidates, fit_cnn, predict, save
from .spec import ModelSpec


def fit(params, train_x, train_target, *, seed, config):
    return fit_cnn(
        model_name="shared_cnn1d",
        model_factory=SharedCNN1D,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "one normalized waveform at a time through one shared temporal CNN scorer",
            "prediction_definition": "shared CNN correction g_theta(s1)-g_theta(s2) [ps]",
            "detector_swap_antisymmetry_enforced": True,
        },
    )


MODEL_SPEC = ModelSpec(
    name="shared_cnn1d",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    preserve_temporal_grid=True,
    estimator_formulation="shared",
)
