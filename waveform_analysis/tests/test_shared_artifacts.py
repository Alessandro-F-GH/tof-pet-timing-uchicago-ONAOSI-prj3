from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from waveform_analysis.ml_pipeline import batch, control_preprocessing, shared_artifacts, train
from waveform_analysis.ml_pipeline.shared_artifacts import ExperimentArtifactStore


class DummyDataset:
    def __init__(self,directory:Path,protocol="protocol-A",population="population-A",n=40):
        self.directory=directory
        self.manifest={"analysis_protocol_identity":protocol,"analysis_population_identity":protocol,"event_population_identity":population,"analysis_source":"dummy.root"}
        self.event_index=np.arange(n,dtype=np.int64)+1000
        self.n_events=n


def _config():
    return {"mode":"energy_to_energy","window_name":"onishi","window_ns":{"start":-1.5,"end":2.0},"resampling":{"policy":"repeated_holdout","validation_fraction":0.2,"test_fraction":0.2,"minimum_events_per_split":3},"fit":{"histogram_bin_width_ps":20.0}}


def test_shared_replica_is_written_once_and_split_contains_only_positions(tmp_path,monkeypatch):
    monkeypatch.setattr(shared_artifacts,"ctr_estimate",lambda *a,**k:SimpleNamespace(ctr_ps=75.0))
    dataset=DummyDataset(tmp_path/"prepared");target=np.linspace(-100.0,100.0,dataset.n_events);store=ExperimentArtifactStore(tmp_path/"artifacts")
    first=store.prepare_replica(dataset,_config(),123,target);second=store.prepare_replica(dataset,_config(),123,target)
    assert first.directory==second.directory;np.testing.assert_array_equal(first.split.test,second.split.test)
    split_files=list((tmp_path/"artifacts").rglob("split.npz"));assert len(split_files)==1
    with np.load(split_files[0]) as data:assert set(data.files)=={"train","validation","test"}
    references=list((tmp_path/"artifacts").rglob("blind_reference.npz"));assert len(references)==1
    with np.load(references[0]) as data:
        np.testing.assert_array_equal(data["event_index"],dataset.event_index[first.split.test]);np.testing.assert_allclose(data["led_ps"],target[first.split.test])


def test_analysis_protocol_separates_shared_population_artifacts(tmp_path,monkeypatch):
    monkeypatch.setattr(shared_artifacts,"ctr_estimate",lambda *a,**k:SimpleNamespace(ctr_ps=75.0));target=np.linspace(-100.0,100.0,40);store=ExperimentArtifactStore(tmp_path/"artifacts")
    a=store.prepare_replica(DummyDataset(tmp_path/"a",protocol="protocol-A"),_config(),123,target);b=store.prepare_replica(DummyDataset(tmp_path/"b",protocol="protocol-B"),_config(),123,target)
    assert a.directory!=b.directory


def test_control_cache_identity_includes_mode_specific_ctr_fit(tmp_path):
    reference=tmp_path/"control.root";reference.write_bytes(b"x");dataset={"root_file":str(reference),"true_tof_ps":0.0,"channels":{"energy":[1,2],"timing":[3,4]}};preprocessing={"selection":{},"led_selection":{}};modes=("energy_to_energy","timing_to_timing")
    a={"energy_to_energy":{"histogram_bin_width_ps":20.0},"timing_to_timing":{"histogram_bin_width_ps":10.0}};b={"energy_to_energy":{"histogram_bin_width_ps":20.0},"timing_to_timing":{"histogram_bin_width_ps":20.0}}
    assert control_preprocessing._artifact_fingerprint(reference,dataset,preprocessing,a,modes)!=control_preprocessing._artifact_fingerprint(reference,dataset,preprocessing,b,modes)
    assert control_preprocessing._mode_fingerprint(reference,dataset,preprocessing,"energy_to_energy",a["energy_to_energy"])==control_preprocessing._mode_fingerprint(reference,dataset,preprocessing,"energy_to_energy",b["energy_to_energy"])


def test_batch_control_protocol_rejects_mode_dependent_mismatch():
    configs=[{"mode":"energy_to_energy","fit":{"histogram_bin_width_ps":20.0}},{"mode":"energy_to_energy","fit":{"histogram_bin_width_ps":10.0}}]
    with pytest.raises(ValueError,match="one control CTR fit definition per mode"):batch._control_protocol(configs)


def test_fit_input_cache_materializes_one_training_view_per_scope(monkeypatch):
    calls={"materialize":0,"mask":0}
    class View:
        time_ps=np.arange(4,dtype=float)
        def materialize(self):calls["materialize"]+=1;return np.ones((3,2,4),dtype=np.float32)
    dataset=SimpleNamespace(manifest={"analysis_protocol_identity":"protocol"});monkeypatch.setattr(train,"waveform_view",lambda dataset,mode,idx:View());monkeypatch.setattr(train,"model_target",lambda dataset,mode:np.arange(10,dtype=float))
    def mask(x):calls["mask"]+=1;return np.asarray([True,True,False,True])
    monkeypatch.setattr(train,"training_sample_mask",mask);spec=SimpleNamespace(preserve_temporal_grid=False);cache=train.FitInputCache();idx=np.asarray([1,2,3]);first=cache.prepare(spec,dataset,"energy_to_energy",idx);second=cache.prepare(spec,dataset,"energy_to_energy",idx)
    assert first is second and calls=={"materialize":1,"mask":1}
