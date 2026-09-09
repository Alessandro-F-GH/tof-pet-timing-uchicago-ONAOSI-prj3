import unittest
from pathlib import Path
import numpy as np
from waveform_analysis.ml_pipeline.data import PreprocessedData
from waveform_analysis.ml_pipeline.prepared_data import _materialize_family
from waveform_analysis.ml_pipeline.timing import anchor_grid, led_grid

class DatasetPreparationTests(unittest.TestCase):
    def _data(self):
        n,length=6,32; waves=np.zeros((n,2,length),dtype=np.float32)
        for event in range(n): waves[event,0]=np.arange(length)+event*0.1; waves[event,1]=2*np.arange(length)+event*0.2
        starts=np.zeros((n,2)); intervals=np.full((n,2),1e-9); rising_start=np.full((n,2),4,dtype=np.int32); rising_stop=np.full((n,2),20,dtype=np.int32)
        return PreprocessedData(Path('.'), {"source":"synthetic.root","fingerprint":"x"}, np.arange(n), np.array([0,0,0,0,1,1],dtype=np.int8), np.ones(n)*45, waves, None, starts, None, intervals, None, rising_start, None, rising_stop, None)
    def test_led_anchor_uses_native_sample_closest_to_threshold(self):
        data=self._data(); grid=led_grid(data,"energy",np.array([0]),np.array([10.5])); self.assertTrue(np.all(np.isfinite(grid)))
        anchor_index,anchor_time=anchor_grid(data,"energy",10.5); self.assertEqual(anchor_index[0,0],10); self.assertEqual(anchor_index[0,1],5); self.assertTrue(np.all(np.isfinite(anchor_time)))
    def test_normalization_uses_fixed_detector_limits(self):
        data=self._data(); anchors=np.full((data.n_events,2),10,dtype=np.int32); kept=np.arange(data.n_events,dtype=np.int64); config={"ml_input":{"window_ns":{"start":-2.0,"end":4.0},"subsampling":2},"preprocessing":{"energy":{"vertical_scale_limit_mV":[[-10.0,40.0],[-20.0,80.0]]}}}
        normalized,_time,minimum,maximum=_materialize_family(data,"energy",anchors,kept,config); self.assertEqual(minimum.shape,(2,1)); self.assertEqual(maximum.shape,(2,1)); np.testing.assert_allclose(minimum[:,0],[-10.0,-20.0]); np.testing.assert_allclose(maximum[:,0],[40.0,80.0]); reconstructed=normalized*(maximum-minimum)[None,:,:]+minimum[None,:,:]; offsets=np.array([-2,0,2,4]); expected=np.stack([data.energy_windows_mV[event][:,10+offsets] for event in kept]); np.testing.assert_allclose(reconstructed,expected,rtol=1e-6,atol=1e-6)
