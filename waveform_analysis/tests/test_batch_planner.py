from copy import deepcopy
from types import SimpleNamespace

from waveform_analysis.engine.batch import _run_state, _runtime_configs
from waveform_analysis.data.storage import RunStore


def config(tmp):
    return {
        "run_id": "timing__onishi__m",
        "output_dir": str(tmp / "m"),
        "mode": "timing_to_timing",
        "window_name": "onishi",
        "model": {"name": "direct_linear_ridge"},
        "save_model": False,
        "cross_validation": {"folds": 3},
        "fit": {"histogram_bin_width_ps": 5},
    }


def test_run_state_uses_files_not_recipe_hashes(tmp_path):
    c = config(tmp_path)
    assert _run_state(c)[:2] == ("run", "all")
    store = RunStore(c["output_dir"])
    assert _run_state(c)[:2] == ("run", "all")
    store.write_manifest({"status": "running"})
    assert _run_state(c)[:2] == ("resume", "incomplete")
    store.write_manifest({"status": "complete"})
    assert _run_state(c)[:2] == ("resume", "cv")
    store.cv_path.parent.mkdir(parents=True, exist_ok=True)
    store.cv_path.write_text("candidate_id\n")
    assert _run_state(c)[:2] == ("resume", "selection")


def test_resume_ignores_changed_nonstructural_config(tmp_path):
    c = config(tmp_path)
    store = RunStore(c["output_dir"])
    store.write_manifest({"status": "complete"})
    for name in (store.cv_path, store.best_path, store.final_fit_path,
                 store.predictions_path, store.development_predictions_path,
                 store.bootstrap_path, store.bootstrap_draws_path):
        name.parent.mkdir(parents=True, exist_ok=True)
        name.touch()
    assert _run_state(c)[:2] == ("keep", "complete")
    changed = deepcopy(c)
    changed["seed"] = 999
    assert _run_state(changed)[:2] == ("keep", "complete")


def test_runtime_configs_deep_copy_nested_settings(tmp_path):
    first = config(tmp_path / "first")
    first["seed"] = 1
    first["preprocessing"] = {"cache_dir": "cache"}
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
    assert runtime[1]["cross_validation"]["folds"] == 3
