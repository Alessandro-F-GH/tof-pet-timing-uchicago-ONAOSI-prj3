import json
from pathlib import Path
from waveform_analysis.tests.result_factory import persist_result
from waveform_analysis.engine.batch import _planner_fingerprints,_run_state
def config(tmp):
    return {"run_id":"timing__onishi__m","output_dir":str(tmp/"m"),"control":{"root_file":"c"},"development":{"root_file":"d"},"blind":{"root_file":"b"},"preprocessing":{"p":1},"mode":"timing_to_timing","fit":{"histogram_bin_width_ps":5},"window_ns":{"start":-1,"end":2},"ml_input":{"subsampling":1},"cross_validation":{"folds":5},"model":{"name":"m","space":{"x":1}},"seed":1,"bootstrap":{"n_resamples":100},"xai":{"enabled":True},"plot_config":{"dpi":100}}
def complete(c):
    p=Path(c["output_dir"]);p.mkdir();persist_result(p,c,{"status":"complete"})
def test_scoped_planner(tmp_path):
    c=config(tmp_path);complete(c);old=_planner_fingerprints(c);assert _run_state(c,{c["run_id"]:old})[:2]==("keep","complete");c["bootstrap"]={"n_resamples":200};assert _run_state(c,{c["run_id"]:old})[:2]==("resume","bootstrap")
def test_blind_vs_development_invalidation(tmp_path):
    c=config(tmp_path);complete(c);old=_planner_fingerprints(c);c["blind"]={"root_file":"new"};assert _run_state(c,{c["run_id"]:old})[:2]==("resume","blind");c=config(tmp_path/"x");Path(c["output_dir"]).parent.mkdir(parents=True,exist_ok=True);complete(c);old=_planner_fingerprints(c);c["model"]={"name":"m","space":{"x":2}};assert _run_state(c,{c["run_id"]:old})[:2]==("resume","cv")

def test_runtime_configs_deep_copy_nested_settings(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace
    from waveform_analysis.engine.batch import _runtime_configs

    first = config(tmp_path / "first")
    second = deepcopy(first)
    first["run_id"] = "first"
    second["run_id"] = "second"
    batch = SimpleNamespace(
        runs=(first, second),
        results={"folder": "UC/energy/wide"},
    )
    runtime = _runtime_configs(batch)
    assert runtime[0]["cross_validation"] is not runtime[1]["cross_validation"]
    runtime[0]["cross_validation"]["folds"] = 2
    assert runtime[1]["cross_validation"]["folds"] == 5

def test_planner_fingerprints_cover_declared_graph_and_propagate_parents(tmp_path):
    from copy import deepcopy
    from waveform_analysis.core.dependencies import STAGE_DEPENDENCIES, STAGE_ORDER

    c = config(tmp_path)
    baseline = _planner_fingerprints(c)
    assert set(STAGE_ORDER).issubset(baseline)
    assert all(stage in baseline for stage in STAGE_DEPENDENCIES)

    changed = deepcopy(c)
    changed["bootstrap"]["n_resamples"] += 1
    after_bootstrap = _planner_fingerprints(changed)
    for stage in ("cv", "selection", "final_fit", "blind"):
        assert after_bootstrap[stage] == baseline[stage]
    for stage in ("bootstrap", "plots"):
        assert after_bootstrap[stage] != baseline[stage]
    assert after_bootstrap["xai"] == baseline["xai"]

    changed = deepcopy(c)
    changed["blind"]["root_file"] = "another-blind.root"
    after_blind = _planner_fingerprints(changed)
    for stage in ("cv", "selection", "final_fit"):
        assert after_blind[stage] == baseline[stage]
    for stage in ("blind", "bootstrap", "xai", "plots"):
        assert after_blind[stage] != baseline[stage]

    changed = deepcopy(c)
    changed["plot_config"]["dpi"] += 1
    after_plots = _planner_fingerprints(changed)
    for stage in STAGE_ORDER:
        if stage != "plots":
            assert after_plots[stage] == baseline[stage]
    assert after_plots["plots"] != baseline["plots"]


def test_planner_ignores_machine_local_cache_roots(tmp_path):
    from copy import deepcopy

    c = config(tmp_path)
    baseline = _planner_fingerprints(c)
    changed = deepcopy(c)
    changed["preprocessing"]["cache_dir"] = "/different/cache"
    changed["preprocessing"]["artifact_root"] = "/different/artifacts"
    assert _planner_fingerprints(changed) == baseline
