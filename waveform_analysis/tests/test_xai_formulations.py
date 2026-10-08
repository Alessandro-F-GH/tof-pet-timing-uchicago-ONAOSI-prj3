from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from waveform_analysis.reporting.plotting import _xai_one_ns, plot_run_xai
from waveform_analysis.engine.xai import temporal_occlusion_importance


class DummyDataset:
    def __init__(self):
        from waveform_analysis.data.dataset import InputTransform
        self.n_events = 3
        self.event_index = np.arange(3, dtype=np.int64)
        self.energy_windows = np.array([
            [[0., .2, .8, .1], [.9, .3, .5, .7]],
            [[.1, .3, .9, .2], [.8, .4, .6, .9]],
            [[.2, .4, .7, .3], [.7, .5, .4, .8]],
        ], dtype=np.float32)
        self.energy_time_ps = np.arange(4, dtype=np.float64)*500
        self.energy_transform = InputTransform(
            np.array([[-100.], [-100.]], dtype=np.float32),
            np.array([[0.], [0.]], dtype=np.float32),
        )


def _spec(formulation):
    def predict(_, values):
        return np.sum(values[:, 0, :], axis=1) + 2*np.sum(values[:, 1, :], axis=1)
    return SimpleNamespace(estimator_formulation=formulation, predict=predict)


def test_xai_direct_is_channel_specific():
    dataset = DummyDataset()
    fitted = SimpleNamespace(sample_mask=None, feature_transform=None, artifact=None, output_max_abs_ps=None)
    values = temporal_occlusion_importance(
        _spec("direct"), fitted, dataset, "energy_to_energy",
        group_size_samples=2, max_events=3, seed=10,
    )
    assert values["importance_ps"].shape == (2, 4)
    assert values["group_importance_ps"].shape == (2, 2)
    assert not np.allclose(values["importance_ps"][0], values["importance_ps"][1])
    assert str(values["estimator_formulation"]) == "direct"


def test_xai_shared_retains_joint_importance():
    dataset = DummyDataset()
    fitted = SimpleNamespace(sample_mask=None, feature_transform=None, artifact=None, output_max_abs_ps=None)
    values = temporal_occlusion_importance(
        _spec("shared"), fitted, dataset, "energy_to_energy",
        group_size_samples=2, max_events=3, seed=10,
    )
    assert values["importance_ps"].shape == (4,)
    assert values["group_importance_ps"].shape == (2,)


@pytest.mark.parametrize("formulation", ["shared", "direct"])
def test_xai_plot_exports(formulation, tmp_path):
    run = tmp_path / "study"
    (run / "artifacts").mkdir(parents=True)
    importance = np.array([1., 2., 3., 1.]) if formulation == "shared" else np.array([[1., 2., 3., 1.], [2., 3., 1., 1.]])
    np.savez(
        run / "artifacts" / "xai.npz",
        time_ps=np.arange(4)*1000., importance_ps=importance,
        example_waveforms_mV=np.array([[-80., -60., -30., -10.], [-40., -20., -10., -5.]]),
        estimator_formulation=formulation,
    )
    config = {
        "font": {"family": "DejaVu Sans", "size": 10, "title_size": 11, "label_size": 10, "tick_size": 9, "legend_size": 9},
        "output": {"format": "png", "dpi": 60},
        "line": {"width": 1.2},
        "grid": {"enabled": True, "alpha": .22, "linestyle": ":"},
        "xai": {"figsize": [7., 5.]},
    }
    assert plot_run_xai(run, config).is_file()


def test_one_ns_unscaled_scores():
    _, _, raw = _xai_one_ns(np.array([0., .5, 1., 1.5]), np.array([2., 2., 4., 4.]), normalize=False)
    np.testing.assert_allclose(raw, [2., 4.])
