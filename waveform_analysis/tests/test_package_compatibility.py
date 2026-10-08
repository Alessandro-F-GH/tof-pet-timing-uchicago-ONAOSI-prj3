"""Migration contracts: module identity, serialization, config and tensor policy."""

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
from waveform_analysis.models.base import BaseTorchModel, RegisteredWaveformModel
from waveform_analysis.models import get_model

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_modules_and_serialized_dataclasses_keep_identity():
    mapping = json.loads(
        (ROOT / "waveform_analysis" / "module_migration.json").read_text()
    )
    for old, new in mapping.items():
        legacy = importlib.import_module(old)
        modern = importlib.import_module(new)
        if old.endswith(".models"):
            assert legacy.get_model is modern.get_model
        else:
            assert legacy is modern
    from waveform_analysis.data.splits import make_cv_split

    split = make_cv_split(
        20, population_identity="pickle", batch_seed=1, n_folds=4, shuffle=True
    )
    assert type(split).__module__ == "waveform_analysis.ml_pipeline.splits"
    assert type(pickle.loads(pickle.dumps(split))) is type(split)


def test_typed_settings_preserve_existing_json_values_and_defaults():
    config = json.loads(
        (ROOT / "waveform_analysis/config/preprocessing/default_ctr.json").read_text()
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
    adapter = RegisteredWaveformModel(spec)
    artifact = adapter.fit({"ridge_alpha": 0.1}, features, y, seed=1, config={})
    np.testing.assert_array_equal(
        adapter.predict(artifact, features), spec.predict(artifact, features)
    )
    adapter.save(artifact, tmp_path)
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


def test_normalized_configuration_and_fingerprints_are_unchanged(tmp_path):
    """Evaluate the baseline parser with the same paths and JSON as the new one."""
    from waveform_analysis.core.config import load_batch_config, public_batch_config
    from waveform_analysis.core.io import canonical_hash

    source = subprocess.run(
        [
            "git",
            "show",
            "59a095a4cbc8b699c5bce16f656d4156483ce86b:waveform_analysis/ml_pipeline/config.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    path = tmp_path / "original_config.py"
    path.write_text(source)
    name = "waveform_analysis.ml_pipeline._original_config"
    spec = importlib.util.spec_from_file_location(name, path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[name] = reference
    try:
        spec.loader.exec_module(reference)
        config_path = ROOT / "waveform_analysis/config/batches/benchmark_FBK.json"
        project = ROOT / "waveform_analysis"
        expected = reference.public_batch_config(
            reference.load_batch_config(config_path, project)
        )
        # The original parser preserves raw ranges; the new parser expands them.
        from waveform_analysis.core.ridge import normalize_ridge_space

        for run in expected["runs"]:
            model = run["model"]
            if model["name"] in {"direct_linear_ridge", "shared_linear_ridge"}:
                model["space"] = normalize_ridge_space(model["space"], model["name"])
        actual = public_batch_config(load_batch_config(config_path, project))
        assert actual == expected
        assert canonical_hash(actual) == canonical_hash(expected)
        assert public_batch_config(load_batch_config(config_path)) == actual
    finally:
        sys.modules.pop(name, None)
