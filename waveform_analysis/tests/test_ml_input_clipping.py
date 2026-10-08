import numpy as np

from waveform_analysis.ml_pipeline.dataset import InputTransform


def test_detector_specific_clipping_and_inverse():
    transform = InputTransform(
        minimum=np.array([[-100.0], [-40.0]], dtype=np.float32),
        maximum=np.array([[0.0], [60.0]], dtype=np.float32),
    )
    waveforms = np.array(
        [[[-140.0, -50.0, 25.0], [-80.0, 10.0, 100.0]]],
        dtype=np.float32,
    )
    normalized = transform.transform(waveforms)
    np.testing.assert_allclose(
        normalized,
        [[[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]],
    )
    np.testing.assert_allclose(
        transform.inverse(normalized),
        [[[-100.0, -50.0, 0.0], [-40.0, 10.0, 60.0]]],
    )
