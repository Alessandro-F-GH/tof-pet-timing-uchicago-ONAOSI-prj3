import pytest
from waveform_analysis.core.config import ConfigError,_cross_validation
def value(t=5.0):return {"folds":5,"shuffle":True,"metric":"ctr","minimum_events_per_fold":50,"pruning":{"enabled":True,"startup_complete_candidates":3,"min_folds_before_prune":1,"max_degradation_ps":t,"prune_if_worse_than_led":True,"led_max_degradation_ps":0.0}}
def test_scalar_and_mapping_tolerance():
    assert _cross_validation(value())["pruning"]["max_degradation_ps"]==5.0;mapping={"1":10.0,"2":7.5,"3":5.0,"4":5.0};assert _cross_validation(value(mapping))["pruning"]["max_degradation_ps"]==mapping
def test_bad_tolerance_rejected():
    with pytest.raises(ConfigError):_cross_validation(value(-1))
    with pytest.raises(ConfigError):_cross_validation(value({"1":5.0}))
