from __future__ import annotations

from ._cnn1d_common import IndependentCNN1D, candidates, fit_cnn, predict, save
from .spec import ModelSpec


def fit(params, train_x, train_target, *, seed, config):
    return fit_cnn(
        model_name="independent_cnn1d",
        model_factory=IndependentCNN1D,
        params=params,
        train_x=train_x,
        train_target=train_target,
        seed=seed,
        config=config,
        metadata_extra={
            "input_definition": "two normalized detector waveforms processed by separate channel-specific Conv1d scorers",
            "prediction_definition": "independent channel-specific CNN correction g1_theta(s1)-g2_phi(s2) [ps]",
            "detector_weight_sharing": False,
            "cross_detector_feature_mixing": False,
            "signal_concatenation": False,
            "detector_swap_antisymmetry_enforced": False,
        },
    )


MODEL_SPEC = ModelSpec(
    name="independent_cnn1d",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    preserve_temporal_grid=True,
    estimator_formulation="direct",
)
