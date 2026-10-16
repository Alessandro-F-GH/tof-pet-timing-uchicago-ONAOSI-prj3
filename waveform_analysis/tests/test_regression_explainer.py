"""Global surrogate fitting, held-out evaluation and saved-result isolation."""
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.engine import regression_explainer as explainer
from waveform_analysis.signal.sample_mask import apply_sample_mask
from waveform_analysis.tests.result_factory import persist_result


def test_fidelity_is_prediction_agreement_in_ps():
    result = explainer.fidelity([1., 2., 3.], [2., 2., 2.])
    assert result['mae_ps'] == pytest.approx(2 / 3)
    assert result['rmse_ps'] == pytest.approx(np.sqrt(2 / 3))
    assert result['r2'] == 0
    assert explainer.fidelity([2., 2.], [2., 2.])['r2'] is None
    with pytest.raises(ValueError, match='finite'):
        explainer.fidelity([1., np.nan], [1., 2.])


def test_real_transform_masked_roundtrip_and_disjoint_selection():
    rng = np.random.default_rng(7)
    pair = rng.normal(size=(50, 2, 16)).astype('float32')
    mask = np.ones(16, dtype=bool)
    mask[[0, 3]] = False
    predictions = pair[:, 0, 5] * 10 - pair[:, 1, 6] * 4
    surrogate, selection = explainer.fit_surrogate(pair, predictions, sample_mask=mask, seed=12,
        max_shapelets=8, shapelet_lengths=(3,), depths=(1, 2), min_samples_leaf=3)
    train, validation = selection['train_indices'], selection['validation_indices']
    assert not set(train) & set(validation)
    assert set(train) | set(validation) == set(range(50))
    # The bank is univariate: matching never joins the two detector streams.
    assert surrogate.transformer.shapelets_[0].shape[1] == 1
    np.testing.assert_array_equal(surrogate.features(pair), surrogate.features(pair[..., mask]))
    restored = pickle.loads(pickle.dumps(surrogate))
    np.testing.assert_array_equal(restored.predict(pair), surrogate.predict(pair))
    changed = pair.copy()
    changed[..., ~mask] = 999
    np.testing.assert_array_equal(surrogate.predict(changed), surrogate.predict(pair))


@pytest.mark.parametrize('kwargs', [{'max_shapelets': 0}, {'depths': []}, {'max_events': 4}, {'shapelet_lengths': [1]}])
def test_invalid_settings_rejected_before_loading_results(kwargs):
    with pytest.raises(ValueError):
        explainer.explain_regression('missing', **kwargs)


def test_saved_run_end_to_end_is_paired_reproducible_and_isolated(tmp_path, monkeypatch):
    directory = tmp_path / 'run'
    manifest = {'status': 'complete', 'model': 'direct_linear_ridge',
                'stage_fingerprints': {'final_fit': 'original-fit'}}
    persist_result(directory, config={'mode': 'energy_to_energy', 'runtime': {'prediction_chunk_size': 9},
                                     'preprocessing': {'cache_dir': str(tmp_path / 'cache')}}, manifest=manifest)
    (directory / 'metadata/best.json').write_text(json.dumps({'parameters': {}}))
    (directory / 'model').mkdir()
    (directory / 'model/model.pkl').write_bytes(b'original model')
    before = {p: p.read_bytes() for p in directory.rglob('*') if p.is_file()}
    rng = np.random.default_rng(4)
    datasets = {}
    for role, count in [('development', 60), ('blind', 30)]:
        waves = rng.normal(size=(count, 2, 15)).astype('float32')
        datasets[role] = SimpleNamespace(n_events=count, event_index=np.arange(count), energy_windows=waves,
            energy_time_ps=np.arange(15) * 100., energy_transform=SimpleNamespace(minimum=0., maximum=1.),
            energy_target_ps=rng.normal(0, 20, count), manifest={'analysis_protocol_identity': role})
    mask = np.ones(15, dtype=bool)
    mask[[1, 4]] = False
    calls = []
    monkeypatch.setattr(explainer.PreparedCacheLookup, 'find', lambda self, root, role, config, meta: datasets[role])
    monkeypatch.setattr(explainer, 'saved_model_complete', lambda *a: True)
    monkeypatch.setattr(explainer, 'load_fitted_model', lambda *a: SimpleNamespace(sample_mask=mask))
    def predict(spec, fitted, dataset, mode, indices, **kwargs):
        calls.append(dataset.manifest['analysis_protocol_identity'])
        pair = apply_sample_mask(dataset.energy_windows[indices], fitted.sample_mask)
        return pair[:, 0].mean(axis=1) * 30 - pair[:, 1].mean(axis=1) * 20
    monkeypatch.setattr(explainer, 'predict_indices', predict)
    from waveform_analysis.reporting import stats
    metric_calls = []
    def metric(values, config, seed):
        metric_calls.append(values.copy())
        return {'ctr_ps': float(2.355 * np.std(values)), 'rmse_ps': float(np.sqrt(np.mean(values ** 2)))}
    monkeypatch.setattr(stats, 'metric_values', metric)
    paths = explainer.explain_regression(directory, max_shapelets=8, shapelet_lengths=[3], depths=[1, 2], min_samples_leaf=3)
    assert calls == ['development', 'blind']
    output = paths[0]
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['fidelity_blind']['n_events'] == 30
    assert [row['model'] for row in summary['timing_blind']] == ['LED', 'Original', 'Surrogate']
    with np.load(output / 'predictions.npz') as data:
        np.testing.assert_array_equal(metric_calls[0], data['led_residual_ps'])
        np.testing.assert_array_equal(metric_calls[1], data['led_residual_ps'] - data['original_ps'])
        np.testing.assert_array_equal(metric_calls[2], data['led_residual_ps'] - data['surrogate_ps'])
        with (output / 'surrogate.pkl').open('rb') as stream:
            recovered = pickle.load(stream)
        np.testing.assert_array_equal(recovered.predict(datasets['blind'].energy_windows), data['surrogate_ps'])
    for name in ['summary.json', 'tree.json', 'tree.png', 'rules.txt', 'performance.csv', 'performance.png', 'shapelets.npz']:
        assert (output / name).is_file()
    assert all(p.read_bytes() == contents for p, contents in before.items())
    assert not set(p for p in directory.rglob('*') if p.is_file()) - set(before) - set(output.rglob('*'))
    assert explainer.explain_regression(directory, max_shapelets=8, shapelet_lengths=[3], depths=[1, 2], min_samples_leaf=3) == paths
    assert calls == ['development', 'blind']  # Existing identical explanation is reused.


def test_shapelet_and_tree_fit_reproducible():
    # The public fitting API accepts development inputs only, and fitting twice
    # produces identical trees and transform features for a fixed seed.
    rng = np.random.default_rng(2)
    pair = rng.normal(size=(40, 2, 12)).astype('float32')
    teacher = pair.mean(axis=(1, 2))
    first, _ = explainer.fit_surrogate(pair, teacher, sample_mask=None, seed=9,
        max_shapelets=4, shapelet_lengths=(3,), depths=(2,), min_samples_leaf=3)
    second, _ = explainer.fit_surrogate(pair, teacher, sample_mask=None, seed=9,
        max_shapelets=4, shapelet_lengths=(3,), depths=(2,), min_samples_leaf=3)
    np.testing.assert_array_equal(first.features(pair), second.features(pair))
    np.testing.assert_array_equal(first.tree.tree_.threshold, second.tree.tree_.threshold)
