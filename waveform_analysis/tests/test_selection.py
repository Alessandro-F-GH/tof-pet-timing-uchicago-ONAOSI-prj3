import unittest
import numpy as np
from waveform_analysis.ml_pipeline.event_selection import Hit, _choose_main_hits, pulse_hits, robust_center_scale
from waveform_analysis.ml_pipeline.selection_outputs import _outside_counts
from waveform_analysis.utils.peak import fit_histogram_peak

class SelectionTests(unittest.TestCase):
    def test_pulse_hits_preserve_multiple_hits_and_trace_end(self):
        signal = np.array([0,0,2,4,2,0,0,3,4,4,4], dtype=float)
        hits = pulse_hits(signal, 1.0, 1e-9)
        self.assertEqual(len(hits), 2); self.assertGreater(hits[1].duration_ns, hits[0].duration_ns); self.assertEqual(hits[1].stop_index, signal.size - 1)
    def test_energy_selection_keeps_first_hit(self):
        events = {"energy": [[[Hit(2,5,3.0), Hit(8,20,12.0)], [Hit(3,9,6.0)]]]}; limits = {"energy": np.array([[2.0,8.0],[2.0,8.0]])}
        accepted, chosen, trigger, stop = _choose_main_hits(events, limits, np.array([True]))
        self.assertTrue(accepted[0]); self.assertEqual(chosen["energy"][0,0], 0); self.assertEqual(trigger["energy"][0,0], 2); self.assertEqual(stop["energy"][0,0], 5)
    def test_timing_selection_chooses_hit_closest_to_peak_center(self):
        events={"timing":[[[Hit(2,8,6.30),Hit(10,20,6.48)],[Hit(3,9,6.39)]]]}; limits={"timing":np.array([[6.2,6.6],[6.2,6.6]])}
        accepted,chosen,trigger,_stop=_choose_main_hits(events,limits,np.array([True]),[6.45,6.40])
        self.assertTrue(accepted[0]); self.assertEqual(chosen["timing"][0,0],1); self.assertEqual(trigger["timing"][0,0],10)
    def test_shared_peak_fit_finds_dominant_population(self):
        rng=np.random.default_rng(4); values=np.concatenate([rng.normal(6.40,0.04,2000),rng.normal(7.2,0.05,200)])
        config={"histogram_bin_ns":0.02,"search_quantile_min":0.2,"smoothing_sigma_bins":2.0,"initial_half_width_ns":0.20,"iteration_sigma":2.5,"max_iterations":6,"convergence_tolerance_ns":0.001,"selection_sigma_low":-3.0,"selection_sigma_high":2.0}
        fit=fit_histogram_peak(values,config=config,unit_suffix="ns",fit_name="ToT detector 1",value_name="pulse durations")
        self.assertTrue(fit.success); self.assertAlmostEqual(fit.mean,6.40,places=2); self.assertLess(fit.selection_high,6.6)
    def test_mad_scale_is_robust_to_outlier(self):
        center, scale = robust_center_scale(np.array([10,10,11,9,10,1000], dtype=float)); self.assertAlmostEqual(center, 10.0); self.assertLess(scale, 2.0)

    def test_plot_outside_counts_distinguish_low_and_high(self):
        low, high = _outside_counts(np.array([0.5, 1.0, 1.5, 2.0, 3.0, np.nan]), (0.8, 2.5))
        self.assertEqual(low, 1)
        self.assertEqual(high, 1)
