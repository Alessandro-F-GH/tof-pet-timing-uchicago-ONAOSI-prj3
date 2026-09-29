from __future__ import annotations

from ._cnn1d_common import DirectCNN1D, candidates, fit_cnn, predict, save
from .spec import ModelSpec


def fit(params, train_x, train_target, *, seed, config):
    return fit_cnn(
        model_name="direct_cnn1d",
        model_factory=DirectCNN1D,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "paired normalized detector waveforms as two Conv1d input channels",
            "prediction_definition": "direct paired CNN correction f_theta(s1,s2) [ps]",
            "detector_swap_antisymmetry_enforced": False,
        },
    )


MODEL_SPEC = ModelSpec(
    name="direct_cnn1d",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    preserve_temporal_grid=True,
    estimator_formulation="direct",
)
