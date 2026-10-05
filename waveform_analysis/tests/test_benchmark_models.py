import inspect

import numpy as np
import torch

from waveform_analysis.ml_pipeline.models import get_model, model_names
from waveform_analysis.ml_pipeline.models._cnn1d_common import IndependentCNN1D, SharedCNN1D
from waveform_analysis.ml_pipeline.models.direct_linear_ridge import candidates as direct_linear_ridge_candidates
from waveform_analysis.ml_pipeline.models.direct_minirocket import candidates as direct_minirocket_candidates
from waveform_analysis.ml_pipeline.models.direct_mlp import DirectPairMLP
from waveform_analysis.ml_pipeline.models.shared_linear_ridge import candidates as shared_linear_ridge_candidates
from waveform_analysis.ml_pipeline.models.shared_minirocket import (
    SharedMiniRocketTransformArtifact,
    candidates as shared_minirocket_candidates,
    fit_transform as shared_minirocket_fit_transform,
    transform as shared_minirocket_transform,
)
from waveform_analysis.ml_pipeline.models.spec import FeatureTransformSpec
from waveform_analysis.ml_pipeline.train import FeatureTransformCache


def test_benchmark_registry_contains_all_model_families():
    expected = {
        "antisymmetric_mlp",
        "locally_connected_mlp",
        "shared_cnn1d",
        "shared_linear_ridge",
        "shared_minirocket",
        "direct_linear_ridge",
        "direct_mlp",
        "independent_cnn1d",
        "onishi_cnn",
        "direct_minirocket",
    }
    assert expected <= set(model_names())
    assert get_model("antisymmetric_mlp").estimator_formulation == "shared"
    assert get_model("locally_connected_mlp").estimator_formulation == "shared"
    assert get_model("shared_cnn1d").estimator_formulation == "shared"
    assert get_model("shared_linear_ridge").estimator_formulation == "shared"
    assert get_model("shared_minirocket").estimator_formulation == "shared"
    assert get_model("direct_linear_ridge").estimator_formulation == "direct"
    assert get_model("direct_mlp").estimator_formulation == "direct"
    assert get_model("independent_cnn1d").estimator_formulation == "direct"
    assert get_model("onishi_cnn").estimator_formulation == "direct"
    assert get_model("direct_minirocket").estimator_formulation == "direct"
    assert get_model("shared_linear_ridge").feature_transform is not None
    assert get_model("direct_linear_ridge").feature_transform is not None
    assert get_model("shared_minirocket").feature_transform is not None
    assert get_model("direct_minirocket").feature_transform is not None


def test_shared_cnn1d_is_exactly_antisymmetric():
    model = SharedCNN1D(32, [8], "silu", conv_channels=[4, 8], kernel_samples=[5, 3])
    pair = torch.randn(6, 2, 32)
    forward = model(pair)
    reverse = model(pair[:, [1, 0], :])
    torch.testing.assert_close(forward, -reverse)


def test_independent_cnn1d_uses_unshared_single_channel_branches():
    model = IndependentCNN1D(32, [8], "silu", conv_channels=[4, 8], kernel_samples=[5, 3])
    pair = torch.randn(6, 2, 32)
    assert model(pair).shape == (6,)
    assert model.backbone_1 is not model.backbone_2
    assert model.head_1 is not model.head_2
    first_conv_1 = next(layer for layer in model.backbone_1.network if isinstance(layer, torch.nn.Conv1d))
    first_conv_2 = next(layer for layer in model.backbone_2.network if isinstance(layer, torch.nn.Conv1d))
    assert first_conv_1.in_channels == 1
    assert first_conv_2.in_channels == 1
    assert next(model.backbone_1.parameters()).data_ptr() != next(model.backbone_2.parameters()).data_ptr()


def test_direct_mlp_preserves_pair_shape_contract():
    model = DirectPairMLP(24, [8, 4], "silu")
    pair = torch.randn(5, 2, 24)
    assert model(pair).shape == (5,)


def test_minirocket_variants_share_same_alpha_grid_contract():
    config = {
        "parameters": {"num_kernels": [10000], "ridge_alpha": [0.1, 1.0, 10.0]},
        "transform": {"n_jobs": -1},
    }
    expected = [
        {"num_kernels": 10000, "ridge_alpha": 0.1},
        {"num_kernels": 10000, "ridge_alpha": 1.0},
        {"num_kernels": 10000, "ridge_alpha": 10.0},
    ]
    assert direct_minirocket_candidates(config) == expected
    assert shared_minirocket_candidates(config) == expected
    for name in ("direct_minirocket", "shared_minirocket"):
        transform = get_model(name).feature_transform
        assert transform.parameters(expected[0], config) == {"num_kernels": 10000, "n_jobs": -1}
        assert transform.parameters(expected[2], config) == {"num_kernels": 10000, "n_jobs": -1}


def test_shared_minirocket_fits_one_transform_on_pooled_detector_waveforms():
    source = inspect.getsource(shared_minirocket_fit_transform)
    assert "np.concatenate([pair[:, 0, :], pair[:, 1, :]], axis=0)" in source
    assert "transformer.fit_transform" in source
    assert "z1 - z2" in source


def test_shared_minirocket_transform_is_exactly_antisymmetric():
    class IdentityTransformer:
        def transform(self, x):
            return np.asarray(x, dtype=np.float64)[:, 0, :]

    class IdentityScaler:
        def transform(self, x):
            return np.asarray(x, dtype=np.float64)

    artifact = SharedMiniRocketTransformArtifact(
        transformer=IdentityTransformer(),
        scaler=IdentityScaler(),
        metadata={},
    )
    pair = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
            [[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    forward = shared_minirocket_transform(artifact, pair)
    reverse = shared_minirocket_transform(artifact, pair[:, ::-1, :])
    np.testing.assert_allclose(forward, -reverse)


def test_linear_ridge_variants_share_same_alpha_grid_contract():
    config = {"parameters": {"ridge_alpha": [0.1, 1.0, 10.0]}}
    expected = [
        {"ridge_alpha": 0.1},
        {"ridge_alpha": 1.0},
        {"ridge_alpha": 10.0},
    ]
    assert shared_linear_ridge_candidates(config) == expected
    assert direct_linear_ridge_candidates(config) == expected
    for name in ("shared_linear_ridge", "direct_linear_ridge"):
        transform = get_model(name).feature_transform
        assert transform.parameters(expected[0], {}) == {}
        assert transform.parameters(expected[2], {}) == {}


def test_shared_linear_ridge_uses_detector_difference():
    transform = get_model("shared_linear_ridge").feature_transform
    pair = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
            [[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    artifact, difference = transform.fit_transform({}, pair, seed=7, config={})
    np.testing.assert_allclose(difference, pair[:, 0, :] - pair[:, 1, :])
    np.testing.assert_allclose(transform.transform(artifact, pair[:, ::-1, :]), -difference)


def test_direct_linear_ridge_uses_detector_concatenation():
    transform = get_model("direct_linear_ridge").feature_transform
    pair = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
            [[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]],
        ],
        dtype=np.float32,
    )
    artifact, features = transform.fit_transform({}, pair, seed=7, config={})
    expected = np.concatenate([pair[:, 0, :], pair[:, 1, :]], axis=1)
    np.testing.assert_allclose(features, expected)
    np.testing.assert_allclose(
        transform.transform(artifact, pair[:, ::-1, :]),
        np.concatenate([pair[:, 1, :], pair[:, 0, :]], axis=1),
    )


def test_shared_linear_ridge_is_exactly_antisymmetric_without_intercept():
    spec = get_model("shared_linear_ridge")
    pair = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
            [[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[2.0, 1.0, 4.0], [0.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([1.0, 2.0, -1.0, 0.5])
    transform = spec.feature_transform
    _, features = transform.fit_transform({}, pair, seed=1, config={})
    artifact = spec.fit(
        {"ridge_alpha": 1.0},
        features,
        target,
        seed=1,
        config={},
    )
    forward = spec.predict(artifact, features)
    reverse = spec.predict(artifact, transform.transform(None, pair[:, ::-1, :]))
    np.testing.assert_allclose(forward, -reverse, rtol=1e-10, atol=1e-10)
    assert artifact.regressor.fit_intercept is False


def test_direct_linear_ridge_is_unconstrained_and_uses_intercept():
    spec = get_model("direct_linear_ridge")
    pair = np.asarray(
        [
            [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]],
            [[4.0, 3.0, 2.0], [1.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            [[2.0, 1.0, 4.0], [0.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    target = np.asarray([1.0, 2.0, -1.0, 0.5])
    transform = spec.feature_transform
    _, features = transform.fit_transform({}, pair, seed=1, config={})
    artifact = spec.fit(
        {"ridge_alpha": 1.0},
        features,
        target,
        seed=1,
        config={},
    )
    assert features.shape[1] == 2 * pair.shape[-1]
    assert artifact.regressor.fit_intercept is True
    assert artifact.metadata["detector_swap_antisymmetry_enforced"] is False


def test_feature_transform_cache_reuses_same_transform_across_downstream_candidates():
    calls = {"fit": 0, "apply": 0}

    def parameters(candidate, config):
        return {"scale": candidate["transform_scale"]}

    def fit_transform(params, x, *, seed, config):
        calls["fit"] += 1
        artifact = {"scale": params["scale"]}
        return artifact, np.asarray(x) * params["scale"]

    def transform(artifact, x):
        calls["apply"] += 1
        return np.asarray(x) * artifact["scale"]

    spec = FeatureTransformSpec("fake", parameters, fit_transform, transform)
    cache = FeatureTransformCache()
    x = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    cfg = {}
    a, xa = cache.prepare(
        spec,
        {"transform_scale": 2, "alpha": 0.1},
        x,
        seed_base=7,
        scope_key="train",
        config=cfg,
    )
    b, xb = cache.prepare(
        spec,
        {"transform_scale": 2, "alpha": 10.0},
        x,
        seed_base=7,
        scope_key="train",
        config=cfg,
    )
    assert a is b and calls["fit"] == 1
    np.testing.assert_array_equal(xa, xb)
    val = np.ones((2, 2, 4), dtype=np.float32)
    first = a.apply(val)
    second = b.apply(val)
    assert calls["apply"] == 2
    np.testing.assert_array_equal(first, second)


def test_feature_transform_cache_refits_when_transform_parameters_change():
    calls = {"fit": 0}

    def parameters(candidate, config):
        return {"scale": candidate["transform_scale"]}

    def fit_transform(params, x, *, seed, config):
        calls["fit"] += 1
        return dict(params), np.asarray(x) * params["scale"]

    spec = FeatureTransformSpec(
        "fake",
        parameters,
        fit_transform,
        lambda artifact, x: np.asarray(x) * artifact["scale"],
    )
    cache = FeatureTransformCache()
    x = np.ones((3, 2, 4), dtype=np.float32)
    cache.prepare(spec, {"transform_scale": 2}, x, seed_base=7, scope_key="train", config={})
    cache.prepare(spec, {"transform_scale": 3}, x, seed_base=7, scope_key="train", config={})
    assert calls["fit"] == 2
