from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from waveform_analysis.models.spec import FeatureTransformSpec
from waveform_analysis.engine.train import (
    FeatureTransformCache,
    FittedModel,
    predict_indices,
)


def test_feature_transform_cache_keeps_only_current_transform():
    fit_calls = []

    def parameters(params, config):
        del config
        return {"scale": int(params["scale"])}

    def fit_transform(params, values, *, seed, config):
        del seed, config
        fit_calls.append(int(params["scale"]))
        scale = float(params["scale"])
        return {"scale": scale}, np.asarray(values, float) * scale

    def transform(artifact, values):
        return np.asarray(values, float) * float(artifact["scale"])

    spec = FeatureTransformSpec(
        name="test_transform",
        parameters=parameters,
        fit_transform=fit_transform,
        transform=transform,
        save=None,
    )
    values = np.ones((4, 2, 3), dtype=np.float32)
    cache = FeatureTransformCache()

    first, first_values = cache.prepare(
        spec,
        {"scale": 2},
        values,
        seed_base=1,
        scope_key="same-split",
        config={},
    )
    repeated, repeated_values = cache.prepare(
        spec,
        {"scale": 2},
        values,
        seed_base=1,
        scope_key="same-split",
        config={},
    )
    second, second_values = cache.prepare(
        spec,
        {"scale": 3},
        values,
        seed_base=1,
        scope_key="same-split",
        config={},
    )

    assert repeated is first
    assert repeated_values is first_values
    assert second is not first
    assert len(cache._items) == 1
    assert fit_calls == [2, 3]
    np.testing.assert_allclose(second_values, 3.0)


def test_predict_indices_materializes_only_configured_chunks():
    calls = []

    class Spec:
        estimator_formulation = "shared"

        @staticmethod
        def predict(artifact, values):
            del artifact
            calls.append(int(values.shape[0]))
            return np.asarray(values[:, 0, 0], dtype=np.float64)

    n_events = 11
    waves = np.zeros((n_events, 2, 3), dtype=np.float32)
    waves[:, 0, 0] = np.arange(n_events, dtype=np.float32)
    dataset = SimpleNamespace(
        manifest={"analysis_protocol_identity": "protocol"},
        energy_windows=waves,
        timing_windows=None,
        energy_time_ps=np.arange(3, dtype=np.float64),
        timing_time_ps=None,
    )
    fitted = FittedModel(
        artifact=object(),
        metadata={"prediction_chunk_size": 4},
        output_max_abs_ps=None,
        sample_mask=np.ones(3, dtype=bool),
        feature_transform=None,
    )

    prediction = predict_indices(
        Spec(),
        fitted,
        dataset,
        "energy_to_energy",
        np.arange(n_events, dtype=np.int64),
    )

    np.testing.assert_array_equal(prediction, np.arange(n_events, dtype=np.float64))
    assert calls == [4, 4, 3]
