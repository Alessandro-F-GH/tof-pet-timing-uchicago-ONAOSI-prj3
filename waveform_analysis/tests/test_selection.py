import unittest
import numpy as np
from waveform_analysis.ml_pipeline.event_selection import Hit, _choose_main_hits, pulse_hits, robust_center_scale

class SelectionTests(unittest.TestCase):
    def test_pulse_hits_preserve_multiple_hits_and_trace_end(self):
        signal = np.array([0,0,2,4,2,0,0,3,4,4,4], dtype=float)
        hits = pulse_hits(signal, 1.0, 1e-9)
        self.assertEqual(len(hits), 2); self.assertGreater(hits[1].duration_ns, hits[0].duration_ns); self.assertEqual(hits[1].stop_index, signal.size - 1)
    def test_longest_acceptable_hit_recovers_multi_hit_event(self):
        events = {"energy": [[[Hit(2,5,3.0), Hit(8,20,12.0)], [Hit(3,9,6.0)]]]}; limits = {"energy": np.array([[2.0,8.0],[2.0,8.0]])}
        accepted, chosen, trigger, stop = _choose_main_hits(events, limits, np.array([True]))
        self.assertTrue(accepted[0]); self.assertEqual(chosen["energy"][0,0], 0); self.assertEqual(trigger["energy"][0,0], 2); self.assertEqual(stop["energy"][0,0], 5)
    def test_mad_scale_is_robust_to_outlier(self):
        center, scale = robust_center_scale(np.array([10,10,11,9,10,1000], dtype=float)); self.assertAlmostEqual(center, 10.0); self.assertLess(scale, 2.0)
