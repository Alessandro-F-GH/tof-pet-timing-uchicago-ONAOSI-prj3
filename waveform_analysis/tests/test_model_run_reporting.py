import tempfile
import unittest
from pathlib import Path

import numpy as np

from waveform_analysis.ml_pipeline.model_run_reporting import (
    _aligned_model_outputs,
)
from waveform_analysis.ml_pipeline.model_output_reporting import _pearson


class ModelRunReportingTests(unittest.TestCase):
    def _write_run(
        self,
        root: Path,
        dataset: str,
        model: str,
        event_ids: np.ndarray,
        outputs: np.ndarray,
    ) -> None:
        split_dir = root / "splits"
        artifact_dir = root / "artifacts" / dataset
        split_dir.mkdir(parents=True, exist_ok=True)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_dir / f"{dataset}.npz", test_event_index=event_ids)
        np.save(artifact_dir / f"{model}_test_model_output_ps.npy", outputs)

    def test_aligned_outputs_reorders_second_run_by_event_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a"
            run_b = root / "b"
            self._write_run(
                run_a,
                "46V",
                "mlp",
                np.array([10, 20, 30]),
                np.array([1.0, 2.0, 3.0]),
            )
            self._write_run(
                run_b,
                "46V",
                "onishi_cnn",
                np.array([30, 10, 20]),
                np.array([30.0, 10.0, 20.0]),
            )

            left, right = _aligned_model_outputs(
                run_a,
                "46V",
                "mlp",
                run_b,
                "46V",
                "onishi_cnn",
            )
            np.testing.assert_allclose(left, [1.0, 2.0, 3.0])
            np.testing.assert_allclose(right, [10.0, 20.0, 30.0])
            correlation, n = _pearson(left, right)
            self.assertAlmostEqual(correlation, 1.0, places=12)
            self.assertEqual(n, 3)

    def test_aligned_outputs_rejects_different_blind_populations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "a"
            run_b = root / "b"
            self._write_run(
                run_a,
                "46V",
                "mlp",
                np.array([10, 20, 30]),
                np.array([1.0, 2.0, 3.0]),
            )
            self._write_run(
                run_b,
                "46V",
                "onishi_cnn",
                np.array([10, 20, 40]),
                np.array([1.0, 2.0, 4.0]),
            )
            with self.assertRaisesRegex(ValueError, "Blind event identities differ"):
                _aligned_model_outputs(
                    run_a,
                    "46V",
                    "mlp",
                    run_b,
                    "46V",
                    "onishi_cnn",
                )


if __name__ == "__main__":
    unittest.main()
