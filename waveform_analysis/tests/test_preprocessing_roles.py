from types import SimpleNamespace
from waveform_analysis.engine import preprocessing as pipeline
def cfg(tmp):
    return {"mode":"timing_to_timing","control":{"root_file":"control.root"},"development":{"root_file":"development.root"},"blind":{"root_file":"blind.root"},"preprocessing":{"cache_dir":str(tmp),"artifact_root":str(tmp/"artifact_store")},"plot_config":{},"fit":{},"_control_modes":("timing_to_timing",),"_control_fit_by_mode":{"timing_to_timing":{}}}
def test_control_fit_and_frozen_application(monkeypatch,tmp_path):
    c=cfg(tmp_path);monkeypatch.setattr(pipeline,"configure_plotting",lambda v:None);calls=[];monkeypatch.setattr(pipeline,"fit_control_artifact",lambda root,dataset,*a,**k:(calls.append((root,dataset)) or ({},tmp_path)));pipeline.fit_control(c);assert calls==[("control.root",c["control"])]
    seen=[];rules={"x":1};monkeypatch.setattr(pipeline,"apply_selection_rules",lambda root,dataset,pre,r,mode,**kw:(seen.append((root,r)) or SimpleNamespace(manifest={"fingerprint":root,"family":"timing"},directory=tmp_path)));monkeypatch.setattr(pipeline,"preprocess_selected",lambda *a,**k:SimpleNamespace());pipeline.apply_frozen_preprocessing(c,"development",{"selection_rules":rules});pipeline.apply_frozen_preprocessing(c,"blind",{"selection_rules":rules});assert seen==[("development.root",rules),("blind.root",rules)]


def test_prepare_role_dataset_reuses_prepared_cache(monkeypatch, tmp_path):
    c = cfg(tmp_path)
    c["reuse_prepared_inputs"] = True
    dataset = SimpleNamespace(n_events=12, manifest={"dataset_source": "development.root"})
    calls = []

    class Lookup:
        def find(self, role, config, manifest, *, control_artifact=None):
            calls.append((role, control_artifact))
            return dataset

    monkeypatch.setattr(pipeline, "PreparedCacheLookup", Lookup)
    monkeypatch.setattr(
        pipeline, "apply_frozen_preprocessing",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("raw preprocessing called")),
    )
    selection, native, prepared = pipeline.prepare_role_dataset(
        c, "development", {"fingerprint": "control"}
    )
    assert (selection, native, prepared) == (None, None, dataset)
    assert calls == [("development", {"fingerprint": "control"})]
