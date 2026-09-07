import unittest

from waveform_analysis.ml_pipeline.search import select_candidate


class SearchTests(unittest.TestCase):
    def test_best_validation_candidate_is_selected_without_folds(self):
        result = select_candidate(
            [3, 1, 2], None, None,
            fit_candidate=lambda candidate, _data, _seed: candidate,
            predict_candidate=lambda _candidate, artifact, _data: artifact,
            score_candidate=lambda value: abs(value - 2),
            seed=4,
        )
        self.assertEqual(result.best.candidate, 2)
        self.assertEqual(len(result.candidates), 3)

    def test_failed_candidate_does_not_abort_search(self):
        def fit(candidate, _data, _seed):
            if candidate == 1:
                raise RuntimeError("bad")
            return candidate
        result = select_candidate(
            [1, 2], None, None,
            fit_candidate=fit,
            predict_candidate=lambda _candidate, artifact, _data: artifact,
            score_candidate=float,
            seed=1,
        )
        self.assertEqual(result.best.candidate, 2)
        self.assertIsNotNone(result.candidates[0].error)
