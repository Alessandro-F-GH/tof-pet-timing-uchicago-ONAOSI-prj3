import numpy as np

from waveform_analysis.ml_pipeline.batch import _planner_fingerprints, _run_state
from waveform_analysis.ml_pipeline.dataset import DATASET_FORMAT_VERSION, InputTransform


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


def test_prepared_data_version_invalidates_cv(tmp_path):
    config = {
        "run_id": "test", "output_dir": str(tmp_path / "study"),
        "control": {"root_file": "control.root"},
        "development": {"root_file": "dev.root"},
        "blind": {"root_file": "blind.root"},
        "preprocessing": {"cache_dir": str(tmp_path)},
        "mode": "energy_to_energy", "fit": {},
        "window_ns": {"start": -1, "end": 3},
        "ml_input": {"subsampling": 1},
        "cross_validation": {"folds": 3},
        "model": {"name": "shared_linear_ridge"},
        "seed": 1, "bootstrap": {}, "xai": {}, "plot_config": {},
    }
    study = tmp_path / "study"
    study.mkdir()
    (study / "manifest.json").write_text(
        '{"schema_version": 51, "status": "complete"}', encoding="utf-8"
    )
    current = _planner_fingerprints(config)
    assert DATASET_FORMAT_VERSION == 33
    assert _run_state(config, {"test": current})[:2] == ("keep", "complete")
    legacy = dict(current)
    from waveform_analysis.ml_pipeline.common import canonical_hash
    legacy["development"] = canonical_hash({
        "preprocessing": current["preprocessing"],
        "development": config["development"],
        "window": config["window_ns"],
        "ml_input": config["ml_input"],
    })
    assert _run_state(config, {"test": legacy})[:2] == ("rebuild", "cv")
