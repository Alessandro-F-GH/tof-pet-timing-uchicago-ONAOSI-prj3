import numpy as np
from sklearn.linear_model import LinearRegression

from waveform_analysis.models import get_model
from waveform_analysis.models.linear import innovation_common as module


def _settings():
    return module.transform_parameters({}, {"transform": {
        "shrinkage": 0.0,
        "relative_jitter": 1e-9,
        "max_fit_events": 500,
        "batch_size": 64,
        "max_samples": 128,
    }})


def test_invertible_predictions_and_uncorrelated_innovations():
    rng = np.random.default_rng(7)
    pair = rng.normal(size=(300, 2, 12))
    pair[:, :, 1:] += 0.7 * pair[:, :, :-1]
    artifact, z = module.fit_transform(_settings(), pair, seed=8, config={})
    z_pair = z.reshape(len(pair), 2, -1)

    reconstructed = z_pair @ artifact.lower.T + artifact.mean
    np.testing.assert_allclose(reconstructed, pair, rtol=2e-5, atol=2e-5)

    raw = pair.reshape(len(pair), -1)
    target = rng.normal(size=len(pair))
    raw_model = LinearRegression().fit(raw, target)
    transformed_model = LinearRegression().fit(z, target)
    # The feature-transform interface stores float32 matrices. OLS predictions
    # are mathematically invariant, with sub-micro-unit floating-point noise.
    np.testing.assert_allclose(
        raw_model.predict(raw), transformed_model.predict(z),
        atol=2e-6, rtol=2e-6,
    )

    covariance = np.cov(z_pair.reshape(-1, 12), rowvar=False)
    np.testing.assert_allclose(covariance, np.eye(12), atol=2e-5, rtol=2e-5)


def test_shared_antisymmetry_and_raw_waveform_transformation():
    rng = np.random.default_rng(12)
    pair = rng.normal(size=(40, 2, 10))
    spec = get_model("shared_linear_innovations")
    artifact, _ = spec.feature_transform.fit_transform(
        _settings(), pair, seed=1, config={"_frozen_transform_only": True},
    )
    prediction = spec.feature_transform.transform(artifact, pair)
    swapped_prediction = spec.feature_transform.transform(
        artifact, pair[:, ::-1],
    )
    np.testing.assert_allclose(prediction, -swapped_prediction, atol=2e-5)


def test_constant_samples_regularized_without_nan():
    rng = np.random.default_rng(1)
    pair = rng.normal(size=(60, 2, 10))
    pair[:, :, 4] = pair[:, :, 3]
    artifact, x = module.fit_transform(_settings(), pair, seed=4, config={})
    assert np.isfinite(x).all()
    assert np.isfinite(artifact.lower).all()


def test_frozen_transform_does_not_use_test_inputs_for_fit():
    rng = np.random.default_rng(4)
    control = rng.normal(size=(30, 2, 10))
    artifact, _ = module.fit_transform(_settings(), control, seed=4, config={})
    mean = artifact.mean.copy()
    module.transform(artifact, rng.normal(loc=50, size=(9, 2, 10)))
    np.testing.assert_array_equal(artifact.mean, mean)


def test_registry_and_config_parameters():
    for name in ("direct_linear_innovations", "shared_linear_innovations"):
        spec = get_model(name)
        assert spec.feature_transform.fit_role == "control"
        assert spec.feature_transform.name == "linear_innovations_v1"
