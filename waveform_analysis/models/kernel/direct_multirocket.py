"""Direct Multirocket + Ridge model entry point."""
from waveform_analysis.models.kernel import multirocket_common as common


def fit_transform(parameters, train_x, *, seed, config):
    return common.fit_transform(parameters, train_x, seed=seed, config=config, shared=False)


def fit(params, train_x, train_target, *, seed, config):
    return common.fit(params, train_x, train_target, seed=seed, config=config, shared=False)


MODEL_SPEC = common.make_spec("direct_multirocket", fit_transform, fit)
