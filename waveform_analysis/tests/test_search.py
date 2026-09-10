import unittest

import numpy as np

from waveform_analysis.ml_pipeline.models.spec import ModelSpec
from waveform_analysis.ml_pipeline.search import select_candidate
from waveform_analysis.ml_pipeline.train import _target_range_candidates, _target_range_mask


class SearchTests(unittest.TestCase):
    def test_target_range_candidates_cross_model_parameters_and_ranges(self):
        spec = ModelSpec(
            name="dummy",
            candidates=lambda _cfg: [{"alpha": 1}, {"alpha": 2}],
            fit=lambda *args, **kwargs: None,
            predict=lambda *args, **kwargs: None,
            save=lambda *args, **kwargs: None,
        )
        candidates = _target_range_candidates(
            spec,
            {},
            {"ml_training": {"target_abs_max_ps": [100.0, 250.0]}},
        )
        self.assertEqual(
            candidates,
            [
                {"alpha": 1, "target_abs_max_ps": 100.0},
                {"alpha": 1, "target_abs_max_ps": 250.0},
                {"alpha": 2, "target_abs_max_ps": 100.0},
                {"alpha": 2, "target_abs_max_ps": 250.0},
            ],
        )

    def test_target_range_mask_is_symmetric_and_inclusive(self):
        target = np.asarray([-201.0, -200.0, -3.0, 0.0, 200.0, 201.0, np.nan])
        np.testing.assert_array_equal(
            _target_range_mask(target, 200.0),
            [False, True, True, True, True, False, False],
        )


    def test_best_validation_candidate_is_selected_without_folds(self):
        result=select_candidate([3,1,2],fit_candidate=lambda candidate,_seed:candidate,predict_candidate=lambda _candidate,artifact:artifact,score_candidate=lambda value:abs(value-2),seed=4)
        self.assertEqual(result.best.candidate,2); self.assertEqual(len(result.candidates),3)
        self.assertIsNotNone(result.best.artifact)
        self.assertEqual(sum(row.artifact is not None for row in result.candidates),1)

    def test_failed_candidate_does_not_abort_search(self):
        def fit(candidate,_seed):
            if candidate==1: raise RuntimeError("bad")
            return candidate
        result=select_candidate([1,2],fit_candidate=fit,predict_candidate=lambda _candidate,artifact:artifact,score_candidate=float,seed=1)
        self.assertEqual(result.best.candidate,2); self.assertIsNotNone(result.candidates[0].error)
