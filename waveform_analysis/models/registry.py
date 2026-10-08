from __future__ import annotations

from dataclasses import replace

from waveform_analysis.models.neural.antisymmetric_mlp import (
    MODEL_SPEC as ANTISYMMETRIC_MLP_SPEC,
)
from waveform_analysis.models.linear.direct_linear_ridge import (
    MODEL_SPEC as DIRECT_LINEAR_RIDGE_SPEC,
)
from waveform_analysis.models.kernel.direct_minirocket import (
    MODEL_SPEC as DIRECT_MINIROCKET_SPEC,
)
from waveform_analysis.models.neural.direct_mlp import MODEL_SPEC as DIRECT_MLP_SPEC
from waveform_analysis.models.neural.independent_cnn1d import (
    MODEL_SPEC as INDEPENDENT_CNN1D_SPEC,
)
from waveform_analysis.models.neural.locally_connected_mlp import (
    MODEL_SPEC as LOCALLY_CONNECTED_MLP_SPEC,
)
from waveform_analysis.models.neural.onishi_cnn import MODEL_SPEC as _ONISHI_CNN_SPEC
from waveform_analysis.models.neural.shared_cnn1d import MODEL_SPEC as SHARED_CNN1D_SPEC
from waveform_analysis.models.linear.shared_linear_ridge import (
    MODEL_SPEC as SHARED_LINEAR_RIDGE_SPEC,
)
from waveform_analysis.models.kernel.shared_minirocket import (
    MODEL_SPEC as SHARED_MINIROCKET_SPEC,
)
from waveform_analysis.models.spec import ModelSpec

ONISHI_CNN_SPEC = replace(_ONISHI_CNN_SPEC, estimator_formulation="direct")

_REGISTRY: dict[str, ModelSpec] = {
    ANTISYMMETRIC_MLP_SPEC.name: ANTISYMMETRIC_MLP_SPEC,
    LOCALLY_CONNECTED_MLP_SPEC.name: LOCALLY_CONNECTED_MLP_SPEC,
    SHARED_CNN1D_SPEC.name: SHARED_CNN1D_SPEC,
    SHARED_LINEAR_RIDGE_SPEC.name: SHARED_LINEAR_RIDGE_SPEC,
    SHARED_MINIROCKET_SPEC.name: SHARED_MINIROCKET_SPEC,
    DIRECT_LINEAR_RIDGE_SPEC.name: DIRECT_LINEAR_RIDGE_SPEC,
    DIRECT_MLP_SPEC.name: DIRECT_MLP_SPEC,
    INDEPENDENT_CNN1D_SPEC.name: INDEPENDENT_CNN1D_SPEC,
    ONISHI_CNN_SPEC.name: ONISHI_CNN_SPEC,
    DIRECT_MINIROCKET_SPEC.name: DIRECT_MINIROCKET_SPEC,
}


def model_registry() -> dict[str, ModelSpec]:
    return dict(_REGISTRY)


def model_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model(name: str) -> ModelSpec:
    try:
        return _REGISTRY[str(name)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model {name!r}; available: {list(model_names())}"
        ) from exc


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.models.registry")
