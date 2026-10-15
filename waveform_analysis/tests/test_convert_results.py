"""One-time conversion preserves source data, metrics and fitted predictions."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import types

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from waveform_analysis.core.config import load_batch_config
from waveform_analysis.data.dataset import load_prepared_dataset
from waveform_analysis.data.results import discover_runs, read_run, resolve_run_config
from waveform_analysis.models.linear.linear_ridge_common import LinearRidgeArtifact
converter = pytest.importorskip('waveform_analysis.scripts.convert_results', reason='Disposable conversion tool was removed')
convert_results = converter.convert_results
HistoricalUnpickler = converter.HistoricalUnpickler


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def old_run(tmp_path, *, model='direct_linear_ridge', legacy=False, mask='control',
            subsampling=1, window='onishi', window_ns=None):
    batch = load_batch_config(Path(__file__).parents[1] / 'config/batches/test_ridge.json')
    cfg = deepcopy(batch.runs[0])
    root = tmp_path / 'old/FBK/batch'
    directory = root / 'timing' / window / model
    directory.mkdir(parents=True, exist_ok=True)
    metadata = directory if legacy else directory / 'metadata'
    metadata.mkdir(exist_ok=True)
    cfg.update(output_dir=str(directory), batch_output_dir=str(root), window_name=window)
    cfg['model'] = {'name': model, 'space': {'model': model}}
    cfg['ml_input']['subsampling'] = subsampling
    if window_ns:
        cfg['window_ns'] = window_ns
    cfg['plot_config']['output']['dpi'] = 50
    (root / 'config.json').write_text(json.dumps({'name': 'old_batch', 'datasets': batch.datasets,
                'protocol': batch.protocol, 'results': {'folder': 'FBK/batch'}}))
    manifest = {'schema_version': 52, 'status': 'complete', 'model': model,
                'mode': cfg['mode'], 'window_name': window, 'window_ns': cfg['window_ns'],
                'sample_mask_fit_role': mask, 'sample_mask_identity': f'mask-{mask}',
                'led_control_mean_ps': 4., 'led_centering_fit_role': 'control',
                'estimator_formulation': 'shared' if model.startswith('shared') else 'direct',
                'dataset_populations': {r: {'n_selected': 32, 'population_identity': r} for r in ('control','development','blind')},
                **{f'{r}_dataset': cfg[r] for r in ('control','development','blind')}}
    for name, value in {
        'config': cfg, 'manifest': manifest,
        'best': {'candidate_id': 'selected', 'parameters': {'ridge_alpha': 1.}, 'validation_ctr_mean_ps': 99.},
        'blind': {'ctr_ps': 100., 'rmse_ps': 50., 'led_ctr_ps': 140., 'led_rmse_ps': 80.},
        'bootstrap': {'ctr_bootstrap_std_ps': 2., 'rmse_bootstrap_std_ps': 1.},
        'state': {'schema_version': 52, 'stages': {'final_fit': {'status': 'complete', 'fingerprint': 'original'}}},
    }.items():
        (metadata / f'{name}.json').write_text(json.dumps(value))
    arrays = directory if legacy else directory / 'artifacts'
    arrays.mkdir(exist_ok=True)
    np.savez(arrays / 'pred.npz', event_id=np.arange(32), prediction_ps=np.arange(32.) / 2,
             corrected_ps=np.arange(32.), led_residual_ps=np.arange(32.) * 1.5)
    (directory / 'plots').mkdir(exist_ok=True)
    (directory / 'plots/blind.png').write_bytes(b'original figure')
    return directory, cfg


def legacy_pickle(path, artifact):
    # Manufacture a real pickle with the historical import path, without leaving
    # those aliases installed for the converter or the canonical model loader.
    cls = type(artifact)
    old = 'waveform_analysis.ml_pipeline.models._linear_ridge_common'
    original = cls.__module__
    installed = {}
    for name in ('waveform_analysis.ml_pipeline', 'waveform_analysis.ml_pipeline.models', old):
        installed[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    setattr(sys.modules[old], cls.__name__, cls)
    try:
        cls.__module__ = old
        path.write_bytes(pickle.dumps(artifact))
    finally:
        cls.__module__ = original
        for name, previous in installed.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_conversion_full_model_arrays_masks_and_source_immutability(tmp_path, monkeypatch):
    directory, cfg = old_run(tmp_path, legacy=True)
    model = directory / 'model'
    model.mkdir()
    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 6)).astype(np.float32)
    regressor = Ridge(alpha=1.).fit(x, rng.normal(size=32))
    artifact = LinearRidgeArtifact(regressor, {'original': True})
    legacy_pickle(model / 'model.pkl', artifact)
    np.save(model / 'sample_mask.npy', np.array([True, False, True, True]))
    expected = regressor.predict(x)
    from waveform_analysis.models import get_model
    expected_pipeline = get_model('direct_linear_ridge').predict(artifact, x)
    before = snapshot(tmp_path / 'old')
    # A conversion must never run a model or recompute statistics.
    from waveform_analysis.engine import train
    monkeypatch.setattr(train, 'fit_on_indices', lambda *a, **k: pytest.fail('no fitting'))
    monkeypatch.setattr(train, 'predict_indices', lambda *a, **k: pytest.fail('no inference'))
    output = tmp_path / 'converted'
    result = convert_results([tmp_path / 'old'], output)
    assert len(result['runs']) == 1 and not result['skipped']
    target = output / result['runs'][0]['destination']
    restored = read_run(target)
    assert restored['blind']['ctr_ps'] == 100.
    assert restored['manifest']['conversion']['original_schema'] == 52
    assert json.loads((target / 'metadata/config.json').read_text()) == {'model': 'direct_linear_ridge'}
    assert snapshot(tmp_path / 'old') == before
    assert (target / 'artifacts/pred.npz').read_bytes() == (directory / 'pred.npz').read_bytes()
    assert (target / 'model/sample_mask.npy').read_bytes() == (model / 'sample_mask.npy').read_bytes()
    with (target / 'model/model.pkl').open('rb') as stream:
        loaded = pickle.load(stream)  # Regular loader; no aliases or converter needed.
    np.testing.assert_array_equal(loaded.regressor.predict(x), expected)
    from waveform_analysis.models import get_model
    from waveform_analysis.engine.train import load_fitted_model
    spec = get_model('direct_linear_ridge')
    fitted = load_fitted_model(spec, target / 'model', {'ridge_alpha': 1.}, resolve_run_config(target))
    np.testing.assert_array_equal(spec.predict(fitted.artifact, x), expected_pipeline)
    np.testing.assert_array_equal(fitted.sample_mask, np.array([True, False, True, True]))
    assert type(loaded).__module__ == 'waveform_analysis.models.linear.linear_ridge_common'
    assert not any(name.startswith('waveform_analysis.ml_pipeline') for name in sys.modules)
    assert len(discover_runs(output)) == 1


def test_check_metrics_only_and_explicit_skips(tmp_path):
    first, _ = old_run(tmp_path)
    model = first / 'model'
    model.mkdir()
    (model / 'model.pkl').write_bytes(b'unsupported old artifact')
    second, _ = old_run(tmp_path, model='shared_linear_ridge')
    cfg = json.loads((second / 'metadata/config.json').read_text())
    cfg.pop('ml_input')
    # Remove the batch fallback too, to ensure no subsampling is guessed.
    (second / 'metadata/config.json').write_text(json.dumps(cfg))
    parent = json.loads((second.parents[2] / 'config.json').read_text())
    parent['protocol'].pop('ml_input', None)
    (second.parents[2] / 'config.json').write_text(json.dumps(parent))
    output = tmp_path / 'converted'
    plan = convert_results([tmp_path / 'old'], output, check=True)
    assert len(plan['runs']) == 1 and 'subsampling' in plan['skipped'][0]['reason']
    assert not output.exists()
    result = convert_results([tmp_path / 'old'], output, metrics_only=True)
    target = output / result['runs'][0]['destination']
    assert not (target / 'model').exists()
    assert (target / 'artifacts/pred.npz').is_file()
    assert not read_run(target)['manifest']['conversion']['models_recovered']
    with pytest.raises(ValueError, match='must be new'):
        convert_results([tmp_path / 'old'], output, metrics_only=True)


def test_model_specific_policies_and_conflicting_settings_are_preserved(tmp_path):
    old_run(tmp_path, model='direct_linear_ridge', mask='control')
    old_run(tmp_path, model='shared_linear_ridge', mask='training')
    # Different subsampling belongs to a separate migrated experiment variant.
    old_run(tmp_path, model='direct_hydra', subsampling=4)
    result = convert_results([tmp_path / 'old'], tmp_path / 'converted', metrics_only=True)
    targets = [tmp_path / 'converted' / r['destination'] for r in result['runs']]
    assert len({p.parents[2] for p in targets}) == 2
    restored = {read_run(t)['manifest']['model']: read_run(t) for t in targets}
    assert restored['shared_linear_ridge']['manifest']['sample_mask_fit_role'] == 'training'
    assert restored['direct_linear_ridge']['manifest']['sample_mask_fit_role'] == 'control'
    for model in ('shared_linear_ridge', 'direct_linear_ridge'):
        assert 'sample_mask_fit_role' in restored[model]['manifest']['input_policy']
    hydra = next(t for t in targets if t.name == 'direct_hydra')
    assert resolve_run_config(hydra)['ml_input']['subsampling'] == 4


def test_failed_model_conversion_is_atomic(tmp_path):
    directory, _ = old_run(tmp_path)
    (directory / 'model').mkdir()
    (directory / 'model/model.pkl').write_bytes(b'bad pickle')
    before = snapshot(tmp_path / 'old')
    output = tmp_path / 'converted'
    with pytest.raises(ValueError, match='model.pkl'):
        convert_results([tmp_path / 'old'], output)
    assert not output.exists()
    assert snapshot(tmp_path / 'old') == before
    assert not list(tmp_path.glob('.converted-conversion-*'))


def test_explicit_cache_conversion_keeps_array_bytes(tmp_path):
    old_run(tmp_path)
    cache = tmp_path / 'old_cache'
    prepared = cache / 'control_ml/prepared/cohort'
    prepared.mkdir(parents=True)
    for name, values in {'event_index': np.arange(3), 'bias_voltage_V': np.full(3, 48.),
                         'energy_windows': np.ones((3, 2, 4), np.float32)}.items():
        np.save(prepared / f'{name}.npy', values)
    (prepared / 'manifest.json').write_text(json.dumps({'format_version': 34}))
    before = snapshot(cache)
    output = tmp_path / 'converted'
    result = convert_results([tmp_path / 'old'], output, metrics_only=True, cache_source=cache)
    target = output / result['runs'][0]['destination']
    assert resolve_run_config(target)['preprocessing']['cache_dir'] == str(output / '_caches')
    migrated = output / '_caches/control_ml/prepared/cohort'
    assert (migrated / 'energy_windows.npy').read_bytes() == (prepared / 'energy_windows.npy').read_bytes()
    loaded = load_prepared_dataset(migrated)
    np.testing.assert_array_equal(loaded.event_index, np.arange(3))
    assert loaded.manifest['conversion_original_format'] == 34
    assert snapshot(cache) == before


def test_pytorch_checkpoint_and_shared_folds_are_recovered(tmp_path):
    import torch
    directory, _ = old_run(tmp_path, model='direct_mlp')
    model = directory / 'model'
    model.mkdir()
    payload = {'state_dict': {'weight': torch.arange(12, dtype=torch.float32).reshape(3, 4)},
               'metadata': {'epoch': 7, 'selection_metric': 'ctr'}}
    torch.save(payload, model / 'model.pt')
    shared = directory.parents[2] / 'artifacts/timing/onishi/original'
    shared.mkdir(parents=True)
    np.savez(shared / 'folds.npz', assignment=np.array([0, 1, 0, 1]), seed=np.array(123))
    before = (shared / 'folds.npz').read_bytes()
    result = convert_results([tmp_path / 'old'], tmp_path / 'converted')
    target = tmp_path / 'converted' / result['runs'][0]['destination']
    restored = torch.load(target / 'model/model.pt', weights_only=False)
    assert torch.equal(restored['state_dict']['weight'], payload['state_dict']['weight'])
    assert restored['metadata'] == payload['metadata']
    assert (target.parents[2] / 'artifacts/timing/onishi/original/folds.npz').read_bytes() == before
    assert (target.parents[2] / 'tables/runs.csv').is_file()


def test_expanded_batch_snapshot_recovers_missing_model_config(tmp_path):
    directory, cfg = old_run(tmp_path)
    (directory / 'metadata/config.json').unlink()
    batch = directory.parents[2]
    config = json.loads((batch / 'config.json').read_text())
    config['runs'] = [cfg]
    (batch / 'config.json').write_text(json.dumps(config))
    result = convert_results([tmp_path / 'old'], tmp_path / 'converted', metrics_only=True)
    target = tmp_path / 'converted' / result['runs'][0]['destination']
    recovered = resolve_run_config(target)
    assert recovered['preprocessing'] == cfg['preprocessing']
    assert recovered['model'] == cfg['model']


def test_converted_results_can_be_grouped_without_old_modules(tmp_path):
    from waveform_analysis.scripts.group_results import group_results
    old_run(tmp_path, model='direct_linear_ridge', legacy=True)
    old_run(tmp_path, model='shared_linear_ridge')
    converted = tmp_path / 'converted'
    convert_results([tmp_path / 'old'], converted, metrics_only=True)
    output = tmp_path / 'grouped'
    result = group_results([converted], output, summary_plots=False)
    assert len(result['groups']) == 1
    leaf = output / result['groups'][0]['group'] / result['groups'][0]['leaf']
    table = (leaf / 'metrics.tex').read_text()
    assert 'D-LIN' in table and 'S-LIN' in table and 'LED &' in table
    assert not list(output.rglob('metadata/config.json'))
