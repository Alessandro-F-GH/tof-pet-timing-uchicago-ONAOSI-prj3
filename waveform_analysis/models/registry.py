"""Lazy registry: metadata and reporting do not import training frameworks."""
from importlib import import_module
from waveform_analysis.models.spec import ModelSpec

_MODULES = {
    'antisymmetric_mlp': 'neural.antisymmetric_mlp',
    'direct_mlp': 'neural.direct_mlp',
    'locally_connected_mlp': 'neural.locally_connected_mlp',
    'independent_cnn1d': 'neural.independent_cnn1d',
    'shared_cnn1d': 'neural.shared_cnn1d',
    'onishi_cnn': 'neural.onishi_cnn',
    **{f'{form}_{family}': f'{kind}.{form}_{family}'
       for family, kind in [('linear_ridge', 'linear'), ('linear_innovations', 'linear'), ('minirocket', 'kernel'),
                            ('rdst', 'kernel'), ('hydra', 'kernel'), ('multirocket', 'kernel'),
                            ('multirocket_rdst', 'kernel')]
       for form in ('direct', 'shared')},
}


def model_names() -> tuple[str, ...]:
    return tuple(sorted(_MODULES))


def get_model(name: str) -> ModelSpec:
    try:
        module = _MODULES[str(name)]
    except KeyError as exc:
        raise ValueError(f'Unknown model {name!r}; available: {list(model_names())}') from exc
    return import_module(f'waveform_analysis.models.{module}').MODEL_SPEC


def model_registry() -> dict[str, ModelSpec]:
    return {name: get_model(name) for name in model_names()}
