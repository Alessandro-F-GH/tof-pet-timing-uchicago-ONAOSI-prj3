"""Direct Hydra + Ridge model entry point."""
from waveform_analysis.models.kernel import hydra_common as common


def fit_transform(parameters, train_x, *, seed, config):
    return common.fit_transform(parameters, train_x, seed=seed, config=config, shared=False)


def fit(params, train_x, train_target, *, seed, config):
    return common.fit(params, train_x, train_target, seed=seed, config=config, shared=False)


MODEL_SPEC = common.make_spec("direct_hydra", fit_transform, fit)
