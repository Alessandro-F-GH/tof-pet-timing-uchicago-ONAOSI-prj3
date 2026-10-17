"""Frozen original features, exact contributions and independent tree comparison."""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.engine import regression_explainer as explainer
from waveform_analysis.tests.result_factory import persist_result


def test_fidelity_is_prediction_agreement_in_ps():
    result = explainer.fidelity([1., 2., 3.], [2., 2., 2.])
    assert result['mae_ps'] == pytest.approx(2 / 3)
    assert result['rmse_ps'] == pytest.approx(np.sqrt(2 / 3))
    assert result['r2'] == 0
    assert explainer.fidelity([2., 2.], [2., 2.])['r2'] is None
    with pytest.raises(ValueError, match='finite'):
        explainer.fidelity([1., np.nan], [1., 2.])


def test_tree_uses_original_features_and_disjoint_selection():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(50, 12)).astype('float32')
    predictions = features[:, 0] * 10 - features[:, 6] * 4
    before = features.copy()
    first, selection = explainer.fit_surrogate(features, predictions, transform_identity='original', seed=12,
                                              depths=(1, 2), min_samples_leaf=3)
    train, validation = selection['train_indices'], selection['validation_indices']
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(50))
    np.testing.assert_array_equal(features, before)
    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_array_equal(restored.predict(features), first.predict(features))
    assert restored.transform_identity == 'original'
    second, _ = explainer.fit_surrogate(features, predictions, transform_identity='original', seed=12,
                                       depths=(1, 2), min_samples_leaf=3)
    np.testing.assert_array_equal(first.tree.tree_.threshold, second.tree.tree_.threshold)


@pytest.mark.parametrize('kwargs', [{'depths': []}, {'max_events': 4}, {'min_samples_leaf': 0}])
def test_invalid_settings_rejected_before_loading_results(kwargs):
    with pytest.raises(ValueError):
        explainer.explain_regression('missing', **kwargs)


def test_grouped_contributions_reconstruct_with_cancellation_and_clipping():
    features = np.array([[5., -5., 0., 1., 2., 3.], [1., -1., 0., 2., 2., 2.]], dtype=np.float32)
    weights = np.ones(6)
    fitted = SimpleNamespace(artifact=SimpleNamespace(regressor=SimpleNamespace(coef_=weights, intercept_=2.)),
                             output_max_abs_ps=4.)
    result = explainer.contribution_summary(fitted, features, np.array([4., 4.]), group_size=3, chunk=1)
    np.testing.assert_array_equal(result['mean_absolute_contribution_ps'], [0., 6.])
    assert result['feature_mean_absolute_contribution_ps'][0] == 3.
    assert result['max_reconstruction_error_ps'] == 0
    assert result['clipped_events'] == 2
    np.testing.assert_array_equal(result['ranking'], [1, 0])
    with pytest.raises(ValueError, match='reconstruction'):
        explainer.contribution_summary(fitted, features, [0., 0.], group_size=3, chunk=1)


def test_verified_cache_skips_transform_and_preserves_feature_order(tmp_path):
    from waveform_analysis.data.feature_cache import _CACHE_VERSION, _transform_root
    from waveform_analysis.core.io import canonical_hash
    config = {'mode': 'energy_to_energy', 'window_name': 'onishi'}
    spec = SimpleNamespace(name='direct_rdst')
    dataset = SimpleNamespace(n_events=10, manifest={'analysis_protocol_identity': 'original-cohort'})
    def forbidden(*args):
        pytest.fail('Verified cached features must not run the transform')
    fitted = SimpleNamespace(feature_transform=SimpleNamespace(identity='frozen', apply=forbidden))
    identity = canonical_hash({'schema_version': _CACHE_VERSION, 'transform_identity': 'frozen',
                              'dataset_identity': 'original-cohort', 'role': 'development'})
    root = _transform_root(tmp_path, config['mode'], config['window_name'], spec.name, 'frozen')
    root.mkdir(parents=True)
    matrix = np.arange(30, dtype=np.float32).reshape(10, 3)
    np.save(root / f'development_{identity[:12]}.npy', matrix)
    (root / f'development_{identity[:12]}.json').write_text(json.dumps({
        'identity': identity, 'transform_identity': 'frozen', 'dataset_identity': 'original-cohort',
        'role': 'development', 'feature_count': 3}))
    features, source = explainer.frozen_features(spec, fitted, dataset, config, np.arange(10), tmp_path, 'development', tmp_path)
    assert isinstance(features, np.memmap)
    np.testing.assert_array_equal(features, matrix)
    features._mmap.close()
    subset, _ = explainer.frozen_features(spec, fitted, dataset, config, np.array([2, 5]), tmp_path, 'development', tmp_path)
    np.testing.assert_array_equal(subset, matrix[[2, 5]])
    assert source['source'] == 'verified_existing_cache'


@pytest.mark.parametrize('shared', [False, True])
def test_saved_rdst_end_to_end_never_refits_and_uses_original_shapelets(tmp_path, monkeypatch, shared):
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
    from waveform_analysis.engine.train import FittedModel, FittedFeatureTransform, save_model, predict_indices
    from waveform_analysis.models import get_model
    from waveform_analysis.models.kernel import rdst_common
    from waveform_analysis.signal.sample_mask import apply_sample_mask
    name = 'shared_rdst' if shared else 'direct_rdst'
    directory = tmp_path / name
    spec = get_model(name)
    rng = np.random.default_rng(4)
    datasets = {}
    for role, count in [('development', 50), ('blind', 25)]:
        waves = rng.normal(size=(count, 2, 15)).astype('float32')
        datasets[role] = SimpleNamespace(n_events=count, event_index=np.arange(count), energy_windows=waves,
            energy_time_ps=np.arange(15) * 100., energy_transform=SimpleNamespace(minimum=0., maximum=1.),
            energy_target_ps=rng.normal(0, 20, count), manifest={'analysis_protocol_identity': role})
    mask = np.ones(15, dtype=bool)
    mask[[1, 4]] = False
    params = {'max_shapelets': 8, 'shapelet_lengths': [3], 'proba_normalization': .8,
              'threshold_percentiles': [5, 10], 'alpha_similarity': .5, 'use_prime_dilations': False, 'n_jobs': 1}
    original_transform, features = rdst_common.fit_transform(params,
        apply_sample_mask(datasets['development'].energy_windows, mask), seed=6, config={}, shared=shared)
    original_transform.metadata['transform_identity'] = 'original-frozen-bank'
    original_transform.time_ps = np.arange(15)[mask] * 100.
    original_transform.sample_indices = np.flatnonzero(mask)
    head = rdst_common.fit({'ridge_alpha': 1.}, features, datasets['development'].energy_target_ps,
                           seed=6, config={}, shared=shared)
    wrapper = FittedFeatureTransform(spec.feature_transform, original_transform, 'original-frozen-bank', {}, 6, {})
    fitted = FittedModel(head, {}, 2000., mask, wrapper)
    expected = predict_indices(spec, fitted, datasets['blind'], 'energy_to_energy', np.arange(25), chunk_size=9)
    manifest = {'status': 'complete', 'model': name, 'stage_fingerprints': {'final_fit': 'original-fit'}}
    persist_result(directory, config={'mode': 'energy_to_energy', 'runtime': {'prediction_chunk_size': 9},
                                     'preprocessing': {'cache_dir': str(tmp_path / 'cache')}}, manifest=manifest)
    (directory / 'metadata/best.json').write_text(json.dumps({'parameters': {'ridge_alpha': 1.}}))
    save_model(spec, fitted, directory / 'model', {})
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    monkeypatch.setattr(explainer.PreparedCacheLookup, 'find', lambda self, root, role, config, meta: datasets[role])
    def forbidden(*args, **kwargs):
        pytest.fail('The original transform or original predictor must NEVER be fitted')
    monkeypatch.setattr(RandomDilatedShapeletTransform, 'fit', forbidden)
    monkeypatch.setattr(rdst_common, 'fit_transform', forbidden)
    from waveform_analysis.reporting import stats
    metric_calls = []
    def metric(values, config, seed):
        metric_calls.append(values.copy())
        return {'ctr_ps': float(2.355 * np.std(values)), 'rmse_ps': float(np.sqrt(np.mean(values ** 2)))}
    monkeypatch.setattr(stats, 'metric_values', metric)
    paths = explainer.explain_regression(directory, depths=[1, 2], min_samples_leaf=3)
    output = paths[0]
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['transform_identity'] == 'original-frozen-bank'
    assert summary['exact_contributions']
    assert summary['fidelity_blind']['n_events'] == 25
    assert summary['selection']['transform_fit_role'] == 'original_saved_model; never refitted'
    with np.load(output / 'predictions.npz') as data:
        np.testing.assert_array_equal(data['original_ps'], expected)
        np.testing.assert_array_equal(metric_calls[0], data['led_residual_ps'])
        np.testing.assert_array_equal(metric_calls[1], data['led_residual_ps'] - data['original_ps'])
        np.testing.assert_array_equal(metric_calls[2], data['led_residual_ps'] - data['surrogate_ps'])
        with (output / 'surrogate.pkl').open('rb') as stream:
            recovered = pickle.load(stream)
        blind_features = wrapper.apply(apply_sample_mask(datasets['blind'].energy_windows, mask))
        np.testing.assert_array_equal(recovered.predict(blind_features), data['surrogate_ps'])
    with np.load(output / 'shapelets.npz') as exported:
        np.testing.assert_array_equal(exported['values'], original_transform.transformer.shapelets_[0])
    with np.load(output / 'contributions.npz') as contributions:
        assert contributions['max_reconstruction_error_ps'] < 1e-3
        assert len(contributions['ranking']) == original_transform.transformer.n_shapelets_
    for file in ['summary.json', 'tree.json', 'tree.png', 'rules.txt', 'performance.csv', 'performance.png',
                 'contributions.png', 'contributions.csv', 'used_shapelets.png']:
        assert (output / file).is_file()
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert not list(output.parent.glob('.frozen-features-*'))
    assert explainer.explain_regression(directory, depths=[1, 2], min_samples_leaf=3) == paths


def test_neural_model_rejected_without_inventing_transform(tmp_path):
    directory = tmp_path / 'cnn'
    persist_result(directory, manifest={'status': 'complete', 'model': 'onishi_cnn'})
    with pytest.raises(RuntimeError, match='no explicit frozen feature transform'):
        explainer.explain_regression(directory)
