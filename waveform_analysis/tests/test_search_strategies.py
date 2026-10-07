import json
from pathlib import Path
import pytest
from waveform_analysis.ml_pipeline.search import fixed_parameters,grid_candidates,optimization_config,suggest_parameters
class Trial:
    def suggest_categorical(self,name,choices):return list(choices)[-1]
    def suggest_int(self,name,low,high,*,step=1,log=False):return high
    def suggest_float(self,name,low,high,*,step=None,log=False):return high
def test_strategies_and_seed_policy():
    assert fixed_parameters({"optimization":{"strategy":"fixed"},"parameters":{"a":{"type":"fixed","value":1}}})=={"a":1}
    assert grid_candidates({"optimization":{"strategy":"grid"},"parameters":{"a":{"type":"categorical","choices":[1,2]}}})==[{"a":1},{"a":2}]
    opt={"optimization":{"strategy":"optuna","n_trials":3,"n_startup_trials":1},"parameters":{"a":{"type":"float","low":1e-4,"high":1e-2,"log":True}}};assert suggest_parameters(Trial(),opt)["a"]==1e-2
    with pytest.raises(ValueError):optimization_config({"optimization":{"strategy":"optuna","seed":4},"parameters":{}})
def test_model_spaces_are_seedless():
    root=Path(__file__).resolve().parents[1]/"config"/"model_spaces"
    for path in root.glob("*.json"):
        space=json.loads(path.read_text());assert "seed" not in (space.get("optimization") or {});assert optimization_config(space).strategy in {"fixed","grid","optuna"}
