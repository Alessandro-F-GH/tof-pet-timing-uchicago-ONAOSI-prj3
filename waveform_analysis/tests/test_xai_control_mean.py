"""Control-mean replacement and saved-model-only XAI replay contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.data.dataset import InputTransform, load_prepared_dataset
from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.results import read_run_manifest
from waveform_analysis.tests.result_factory import persist_result
from waveform_analysis.engine import xai, xai_replay
from waveform_analysis.tests.test_control_sample_mask import dataset as prepared_fixture


def cohort(values, role):
    x = np.asarray(values, dtype=np.float32)
    return SimpleNamespace(n_events=len(x),event_index=np.arange(len(x)),energy_windows=x,
        energy_time_ps=np.arange(x.shape[-1])*100.,
        energy_transform=InputTransform(np.zeros((2,1)),np.ones((2,1))),
        manifest={'dataset_role':role,'analysis_protocol_identity':role})


@pytest.mark.parametrize('formulation',['direct','shared'])
@pytest.mark.parametrize('group_size',[1,3])
def test_replacement_is_per_detector_control_sample_mean(formulation,group_size):
    control = cohort([[[.1,.3,.9,.7],[.7,.1,.5,.3]],
                      [[.3,.5,.7,.5],[.9,.3,.7,.1]]], 'control')
    blind = cohort([[[.8,.9,.1,.2],[.2,.5,.8,.6]],
                    [[.2,.4,.3,.8],[.6,.7,.1,.4]],
                    [[.5,.1,.7,.9],[.3,.4,.2,.8]]], 'blind')
    seen = []
    weights = np.array([1.,2.]) if formulation=='direct' else np.array([1.,-1.])
    def predict(_,pair):
        seen.append(pair.copy())
        return (pair.sum(axis=-1)*weights).sum(axis=1)
    fitted = SimpleNamespace(sample_mask=None,feature_transform=None,artifact=None,output_max_abs_ps=None)
    result = xai.temporal_occlusion_importance(
        SimpleNamespace(estimator_formulation=formulation,predict=predict), fitted,blind,
        'energy_to_energy', control_dataset=control,group_size_samples=group_size,max_events=3,seed=1)
    mean = control.energy_windows.mean(axis=0,dtype=np.float64).astype(np.float32)
    np.testing.assert_array_equal(result['control_mean_normalized'],mean)
    assert str(result['replacement_method'])=='control_sample_mean'
    assert int(result['control_n_events'])==2
    assert not np.allclose(mean,blind.energy_windows.mean(axis=0))
    expected = []
    channels = range(2) if formulation=='direct' else [None]
    for start in range(0,4,group_size):
        stop=min(start+group_size,4)
        for channel in channels:
            pair=blind.energy_windows.copy()
            if channel is None:
                pair[:,:,start:stop]=mean[None,:,start:stop]
            else:
                pair[:,channel,start:stop]=mean[None,channel,start:stop]
            expected.append(pair)
    for actual,wanted in zip(seen[1:],expected):
        np.testing.assert_array_equal(actual,wanted)


def test_mean_cached_and_never_learned_from_blind(tmp_path,monkeypatch):
    control=cohort(np.full((4,2,5),.25),'control')
    expected=xai.calibration_mean_waveform(control,'energy_to_energy',cache_root=tmp_path)
    monkeypatch.setattr(xai.np,'mean',lambda *a,**k: pytest.fail('cached control mean must not be recalculated'))
    np.testing.assert_array_equal(xai.calibration_mean_waveform(control,'energy_to_energy',cache_root=tmp_path),expected)
    control.manifest['dataset_role']='blind'
    with pytest.raises(ValueError,match='control dataset'):
        xai.calibration_mean_waveform(control,'energy_to_energy',cache_root=tmp_path)


@pytest.mark.parametrize('version',[33,34])
def test_legacy_prepared_arrays_read_only_for_diagnostics(tmp_path,version):
    original=prepared_fixture(tmp_path,'control',mode='energy_to_energy')
    path=original.directory/'manifest.json'
    path.write_text(json.dumps({**original.manifest,'format_version':version}))
    before=path.read_bytes()
    with pytest.raises(ValueError,match='format mismatch'):
        load_prepared_dataset(original.directory)
    assert path.read_bytes()==before  # Only the explicit converter can upgrade cache formats.


def test_cache_lookup_requires_original_cohort_and_unique_match(tmp_path,monkeypatch):
    directory=tmp_path/'control_ml/prepared/one'
    directory.mkdir(parents=True)
    cfg={'mode':'energy_to_energy','window_ns':{'start':-1,'end':2},'ml_input':{'subsampling':2},
         'control':{'root_file':'original.root','true_tof_ps':0}}
    metadata={'dataset_role':'control','dataset_source':'original.root','mode':cfg['mode'],
              'window_ns':cfg['window_ns'],'subsampling':2,'true_tof_ps':0,
              'analysis_protocol_identity':'exact-control','event_population_identity':'events'}
    (directory/'manifest.json').write_text(json.dumps(metadata))
    manifest={'dataset_populations':{'control':{'analysis_protocol_identity':'exact-control','population_identity':'events'}}}
    calls=[]
    monkeypatch.setattr(xai_replay,'load_prepared_dataset',lambda p,**kw:calls.append(p) or 'loaded')
    lookup=xai_replay.PreparedCacheLookup()
    assert lookup.find(tmp_path,'control',cfg,manifest)=='loaded'
    assert lookup.find(tmp_path,'control',cfg,manifest)=='loaded' and len(calls)==1
    with pytest.raises(ValueError,match='found 0'):
        lookup.find(tmp_path,'control',cfg,{'dataset_populations':{'control':{'analysis_protocol_identity':'wrong'}}})
    other=directory.parent/'two';other.mkdir()
    (other/'manifest.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='found 2'):
        xai_replay.PreparedCacheLookup().find(tmp_path,'control',cfg,manifest)


def test_replay_updates_only_xai_and_keeps_other_stages(tmp_path,monkeypatch):
    store=RunStore(tmp_path/'run')
    manifest={'schema_version':52,'status':'complete','model':'direct_linear_ridge',
              'stage_fingerprints':{'final_fit':'weights','blind':'evaluation','xai':'old'}}
    persist_result(store.root, manifest=manifest)
    config={'mode':'energy_to_energy','window_name':'short','seed':1,
            'preprocessing':{'cache_dir':str(tmp_path)},'xai':{'enabled':False,'group_size_samples':2,'max_events':3},
            'plot_config':{}}
    persist_result(store.root, config=config)
    store.write_best({'parameters':{}})
    store.write_blind({'ctr_ps':100.,'rmse_ps':60.})
    store.save_predictions(event_id=[0],prediction_ps=[1],corrected_ps=[2],led_residual_ps=[3])
    store.model_dir.mkdir()
    (store.model_dir/'model.pkl').write_bytes(b'unchanged weights')
    initial={'stages':{'final_fit':{'status':'complete','fingerprint':'weights'},
                                         'blind':{'status':'complete','fingerprint':'evaluation'}}}
    store.state_path.write_text(json.dumps(initial))
    snapshots={p:p.read_bytes() for p in [store.predictions_path,store.blind_path,store.metadata_dir/'config.json',
                                         store.model_dir/'model.pkl']}
    monkeypatch.setattr(xai_replay,'_runs',lambda _: [store])
    monkeypatch.setattr(xai_replay,'_cache_root',lambda *a:tmp_path)
    control=cohort(np.full((3,2,4),.2),'control');blind=cohort(np.full((3,2,4),.8),'blind')
    monkeypatch.setattr(xai_replay.PreparedCacheLookup,'find',lambda self,root,role,*a:control if role=='control' else blind)
    monkeypatch.setattr(xai_replay,'get_model',lambda name:SimpleNamespace(name=name,estimator_formulation='direct'))
    monkeypatch.setattr(xai_replay,'saved_model_complete',lambda *a:True)
    fitted=SimpleNamespace(sample_mask=None,feature_transform=None,artifact=None,output_max_abs_ps=None)
    spec=SimpleNamespace(name='direct_linear_ridge',estimator_formulation='direct',predict=lambda _,pair:pair.sum(axis=(1,2)))
    monkeypatch.setattr(xai_replay,'get_model',lambda name:spec)
    monkeypatch.setattr(xai_replay,'load_fitted_model',lambda *a:fitted)
    monkeypatch.setattr(xai_replay,'plot_run_xai',lambda *a:None)
    monkeypatch.setattr(xai_replay,'release_training_memory',lambda:None)
    assert xai_replay.rerun_xai(tmp_path,max_events=2)==[store.xai_path]
    for p,before in snapshots.items():assert p.read_bytes()==before
    state=json.loads(store.state_path.read_text())
    assert 'schema_version' not in state
    assert state['stages']['final_fit']==initial['stages']['final_fit']
    assert state['stages']['blind']==initial['stages']['blind']
    assert state['stages']['xai']['metadata']['replacement_method']=='control_sample_mean'
    assert read_run_manifest(store.root)['stage_fingerprints']['final_fit']=='weights'
    with np.load(store.xai_path) as scores:assert int(scores['control_n_events'])==3
