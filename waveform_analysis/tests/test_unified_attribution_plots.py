import numpy as np
from waveform_analysis.reporting.attribution_plots import window_importance

def test_signs_do_not_cancel():
    v,_,_=window_importance(np.arange(4.),[0],4,[[5]], [[-5]])
    np.testing.assert_array_equal(v,[[10,10,10,10]])

def test_overlapping_windows():
    v,_,_=window_importance(np.arange(4.),[0,1],2,[[2,4]], [[0,0]])
    np.testing.assert_array_equal(v,[[2,3,4,0]])

def test_masked_gap():
    _,parts,_=window_importance([0,1,10,11],[0],4,[[2]],[[-1]])
    assert len(parts[0])==2

def test_detector_specific():
    v,_,_=window_importance(np.arange(2.),[0],2,[[2],[3]],[[0],[-1]])
    np.testing.assert_array_equal(v,[[2,2],[4,4]])
