"""The timing input mask is shared by all models and fitted on control only."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.data.dataset import DATASET_FORMAT_VERSION, InputTransform, PreparedDataset, load_prepared_dataset
from waveform_analysis.data import sample_mask as policy
from waveform_analysis.data.feature_cache import prepare_frozen_features, prepare_frozen_transform
from waveform_analysis.engine.train import FitInputCache, FittedModel, _model_input
from waveform_analysis.engine.xai import temporal_occlusion_importance
from waveform_analysis.models import get_model, model_names
from waveform_analysis.models.spec import FeatureTransformSpec
from waveform_analysis.signal.sample_mask import control_clipping_sample_mask


@pytest.mark.parametrize('chunk_size', [1, 17, 1000])
def test_only_consistent_clipping_on_both_detectors_is_removed(chunk_size):
    x = np.full((100, 2, 7), .4, dtype=np.float32)
    x[:, :, 0] = 0.
    x[:99, :, 1] = 1.
    x[:, 0, 2] = 1.  # One detector still has an informative sample.
    x[:50, :, 4] = 0.
    x[50:, :, 4] = 1.  # Alternating limits do not mean constant clipping.
    x[:, 0, 5], x[:, 1, 5] = 1., 0.
    x[:98, :, 6] = 1.  # Below the 99% threshold.
    np.testing.assert_array_equal(control_clipping_sample_mask(x, chunk_size=chunk_size),
                                  [False, False, True, True, True, False, True])
    with pytest.raises(ValueError, match='remove every'):
        control_clipping_sample_mask(np.ones((10, 2, 3)))


def dataset(tmp_path, role, *, mode='timing_to_timing', seed=1):
    family = 'timing' if mode == 'timing_to_timing' else 'energy'
    directory = tmp_path/role
    directory.mkdir()
    x = np.random.default_rng(seed).uniform(.1, .9, (100, 2, 24)).astype(np.float32)
    x[:, :, 1] = .5  # Ordinary constant samples are retained.
    if role == 'control':
        x[:, :, -8:] = 1.
    elif role == 'development':
        x[:, :, :4] = 0.  # Training-specific clipping must not determine the mask.
    time = np.arange(24, dtype=float)*50.
    manifest = {'format_version': DATASET_FORMAT_VERSION, 'mode': mode, 'family': family,
                'fingerprint': role, 'dataset_role': role, 'event_population_identity': role,
                'analysis_protocol_identity': role, 'analysis_population_identity': role}
    values = {name: None for name in ('energy_windows','timing_windows','energy_time_ps',
              'timing_time_ps','energy_transform','timing_transform','energy_target_ps',
              'timing_target_ps','energy_led_time_ps','timing_led_time_ps')}
    values.update({f'{family}_windows': x, f'{family}_time_ps': time,
                   f'{family}_transform': InputTransform(np.zeros((2,1)),np.ones((2,1))),
                   f'{family}_target_ps': np.arange(100.), f'{family}_led_time_ps': np.zeros((100,2))})
    d = PreparedDataset(directory, manifest, np.arange(100), np.full(100,48.), **values)
    for name in ('event_index','bias_voltage_V',f'{family}_windows',f'{family}_time_ps',
                 f'{family}_target_ps',f'{family}_led_time_ps'):
        np.save(directory/f'{name}.npy',getattr(d,name))
    return d


def calibrated(tmp_path, monkeypatch=None):
    control = dataset(tmp_path,'control')
    dev = dataset(tmp_path,'development')
    blind = dataset(tmp_path,'blind',seed=2)
    calls = []
    if monkeypatch is not None:
        original = policy.control_clipping_sample_mask
        monkeypatch.setattr(policy,'control_clipping_sample_mask',
                            lambda x: calls.append(x.shape) or original(x))
    control = policy.attach_control_sample_mask(control,control,cache_root=tmp_path)
    dev = policy.attach_control_sample_mask(dev,control,cache_root=tmp_path)
    blind = policy.attach_control_sample_mask(blind,control,cache_root=tmp_path)
    return control, dev, blind, calls


def test_control_fit_once_cached_persisted_and_reloaded(tmp_path, monkeypatch):
    control, dev, blind, calls = calibrated(tmp_path,monkeypatch)
    assert len(calls) == 1
    assert dev.sample_mask.sum() == 16
    assert dev.sample_mask[:4].all()  # No train-derived filtering.
    for d in (control,dev,blind):
        np.testing.assert_array_equal(d.sample_mask,control.sample_mask)
        assert d.manifest['sample_mask_control_events'] == 100
        loaded = load_prepared_dataset(d.directory)
        np.testing.assert_array_equal(loaded.sample_mask,control.sample_mask)
        np.testing.assert_array_equal(loaded.timing_windows,d.timing_windows)
        assert loaded.manifest['sample_mask_identity'] == control.manifest['sample_mask_identity']
        policy.attach_control_sample_mask(loaded,control,cache_root=tmp_path)
    assert len(calls) == 1


@pytest.mark.parametrize('name', model_names())
def test_every_model_uses_identical_control_mask_before_transform(tmp_path,name):
    control, dev, _, _ = calibrated(tmp_path)
    fitted = FitInputCache().prepare(get_model(name),dev,'timing_to_timing',np.arange(20))
    np.testing.assert_array_equal(fitted.x,dev.timing_windows[:20,:,:16])
    np.testing.assert_array_equal(fitted.time_ps,dev.timing_time_ps[:16])
    np.testing.assert_array_equal(fitted.sample_mask,control.sample_mask)


@pytest.mark.parametrize('name', ['direct_linear_ridge','shared_linear_ridge','direct_minirocket','shared_minirocket'])
def test_frozen_transform_and_feature_cache_receive_masked_control_or_development(tmp_path,name):
    control, dev, blind, _ = calibrated(tmp_path)
    received = []
    def apply(artifact,x):
        assert x.shape[-1] == 16
        received.append(x.copy())
        return x.reshape(len(x),-1)
    def fit(params,x,**kw):
        return SimpleNamespace(metadata={}),apply(None,x)
    transform = FeatureTransformSpec('test_transform',lambda *a: {},fit,apply)
    spec = replace(get_model(name),feature_transform=transform)
    cfg = {'mode':'timing_to_timing','window_name':'wide','seed':1,
           'runtime':{'prediction_chunk_size':25}}
    frozen = prepare_frozen_transform(spec,{},cfg,control_dataset=control,
                development_dataset=dev,cache_root=tmp_path)
    fit_calls = len(received)
    reused = prepare_frozen_transform(spec,{},cfg,control_dataset=control,
                development_dataset=dev,cache_root=tmp_path)
    assert len(received) == fit_calls
    np.testing.assert_array_equal(reused.sample_mask,control.sample_mask)
    expected = control if name.endswith('minirocket') else dev
    np.testing.assert_array_equal(received[0],expected.timing_windows[:len(received[0]),:,:16])
    features = prepare_frozen_features(spec,frozen,cfg,blind,role='blind',cache_root=tmp_path)
    np.testing.assert_array_equal(features.features,blind.timing_windows[:,:,:16].reshape(100,-1))
    np.testing.assert_array_equal(features.sample_mask,control.sample_mask)
    before = len(received)
    prepare_frozen_features(spec,frozen,cfg,blind,role='blind',cache_root=tmp_path)
    assert len(received) == before


def test_energy_never_filters_even_identical_or_clipped_values(tmp_path):
    d = dataset(tmp_path,'control',mode='energy_to_energy')
    d = policy.attach_control_sample_mask(d,None,cache_root=tmp_path)
    assert d.sample_mask.all()
    for name in model_names():
        prepared = FitInputCache().prepare(get_model(name),d,'energy_to_energy',np.arange(10))
        np.testing.assert_array_equal(prepared.x,d.energy_windows[:10])
        assert prepared.sample_mask.all()


def test_missing_timing_control_mask_and_grid_mismatch_fail(tmp_path):
    d = dataset(tmp_path,'development')
    with pytest.raises(ValueError,match='control-derived'):
        FitInputCache().prepare(get_model('direct_mlp'),d,'timing_to_timing',[0,1])
    control = dataset(tmp_path,'control')
    with pytest.raises(ValueError,match='grids differ'):
        policy.attach_control_sample_mask(replace(d,timing_time_ps=d.timing_time_ps+1),control,cache_root=tmp_path)


def test_prediction_and_xai_exclude_clipped_columns_but_keep_physical_axis(tmp_path):
    _, _, blind, _ = calibrated(tmp_path)
    fitted = FittedModel(None,{},sample_mask=blind.sample_mask)
    actual = _model_input(fitted,blind,'timing_to_timing',np.arange(5))
    np.testing.assert_array_equal(actual,blind.timing_windows[:5,:,:16])
    spec = SimpleNamespace(estimator_formulation='direct',
                           predict=lambda _,x: x.sum(axis=(1,2)))
    values = temporal_occlusion_importance(spec,fitted,blind,'timing_to_timing',
                        group_size_samples=4,max_events=5,seed=1)
    np.testing.assert_array_equal(values['time_ps'],blind.timing_time_ps)
    np.testing.assert_array_equal(values['importance_ps'][:,-8:],0.)


def test_mask_refits_when_control_identity_changes(tmp_path,monkeypatch):
    control,dev,_,calls = calibrated(tmp_path,monkeypatch)
    changed = replace(control,manifest={**control.manifest,'fingerprint':'new-control'})
    updated = policy.attach_control_sample_mask(dev,changed,cache_root=tmp_path)
    assert len(calls)==2
    assert updated.manifest['sample_mask_identity']!=dev.manifest['sample_mask_identity']
    with pytest.raises(ValueError,match='requires the control'):
        policy.attach_control_sample_mask(dev,dev,cache_root=tmp_path)
