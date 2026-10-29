"""Canonical module identity, configuration and tensor policy contracts."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import importlib.util

import numpy as np

from waveform_analysis.core.config import (
    BaselineConfig,
    TimingConfig,
    MLPTrainingConfig,
    OnishiTrainingConfig,
)
from waveform_analysis.data.torch_dataset import WaveformDataset
from waveform_analysis.models.base import BaseTorchModel
from waveform_analysis.models import get_model

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_modules_and_serialized_dataclasses():
    from waveform_analysis.data.splits import make_cv_split
    split = make_cv_split(20, population_identity="pickle", batch_seed=1, n_folds=4, shuffle=True)
    assert type(split).__module__ == "waveform_analysis.data.splits"
    assert type(pickle.loads(pickle.dumps(split))) is type(split)
    assert importlib.util.find_spec('waveform_analysis.ml_pipeline') is None


def test_typed_settings_preserve_existing_json_values_and_defaults():
    config = json.loads(
        (ROOT / "waveform_analysis/config/preprocessing/UC.json").read_text()
    )
    baseline = BaselineConfig.from_preprocessing(config)
    assert baseline.window_ns == tuple(config["selection"]["baseline_window_ns"])
    assert (
        baseline.clipping_margin_mV
        == config["selection"]["baseline_clipping"]["margin_mV"]
    )
    timing = TimingConfig.from_preprocessing(config, fractions=(0.2, 0.5))
    assert timing.thresholds_mV == tuple(config["led_selection"]["thresholds_mV"])
    assert timing.fractions == (0.2, 0.5)
    assert MLPTrainingConfig.from_mapping({}) == MLPTrainingConfig()
    assert OnishiTrainingConfig.from_mapping({}) == OnishiTrainingConfig()
    assert (
        MLPTrainingConfig.from_mapping({"epochs": 3, "optimizer": " ADAM "}).optimizer
        == "adam"
    )


def test_tensor_dataset_and_model_adapter_use_original_operations(tmp_path):
    rng = np.random.default_rng(81)
    x = rng.normal(size=(24, 2, 12))[:, :, ::2]
    y = rng.normal(size=24)
    dataset = WaveformDataset(x, y)
    np.testing.assert_array_equal(
        dataset.tensors[0].numpy(), np.ascontiguousarray(x, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        dataset.tensors[1].numpy(), np.asarray(y, dtype=np.float32)
    )
    spec = get_model("direct_linear_ridge")
    _, features = spec.feature_transform.fit_transform({}, x, seed=1, config={})
    artifact = spec.fit({"ridge_alpha": 0.1}, features, y, seed=1, config={})
    assert spec.predict(artifact, features).shape == y.shape
    spec.save(artifact, tmp_path)
    assert (tmp_path / "model.pkl").is_file()
    from waveform_analysis.models.neural.direct_mlp import DirectPairMLP

    model = DirectPairMLP(6, [4], "silu")
    assert isinstance(model, BaseTorchModel)
    assert tuple(model.state_dict()) == (
        "regressor.network.0.weight",
        "regressor.network.0.bias",
        "regressor.network.2.weight",
        "regressor.network.2.bias",
    )


def test_signal_imports_do_not_initialize_torch_or_io_frameworks():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import waveform_analysis.signal.timing; import waveform_analysis.signal.baseline; import waveform_analysis.signal.pulses; import waveform_analysis.signal.metrics; assert not {'torch', 'uproot', 'awkward'} & sys.modules.keys()",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


