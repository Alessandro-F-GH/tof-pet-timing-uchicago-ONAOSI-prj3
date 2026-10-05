import json
from pathlib import Path

import pytest

from waveform_analysis.ml_pipeline.search import (
    fixed_parameters,
    grid_candidates,
    optimization_config,
    suggest_parameters,
)


class FakeTrial:
    def __init__(self):
        self.calls = []

    def suggest_categorical(self, name, choices):
        self.calls.append(("categorical", name, list(choices)))
        return list(choices)[-1]

    def suggest_int(self, name, low, high, *, step=1, log=False):
        self.calls.append(("int", name, low, high, step, log))
        return high

    def suggest_float(self, name, low, high, *, step=None, log=False):
        self.calls.append(("float", name, low, high, step, log))
        return high


def test_fixed_strategy_returns_single_fixed_parameter_set():
    space = {
        "optimization": {"strategy": "fixed"},
        "parameters": {
            "learning_rate": {"type": "fixed", "value": 1e-3},
            "batch_size": {"type": "fixed", "value": 128},
        },
    }
    assert optimization_config(space).strategy == "fixed"
    assert fixed_parameters(space) == {"learning_rate": 1e-3, "batch_size": 128}


def test_grid_strategy_expands_only_finite_domains():
    space = {
        "optimization": {"strategy": "grid"},
        "parameters": {
            "num_kernels": {"type": "categorical", "choices": [1000, 3000]},
            "ridge_alpha": {"type": "categorical", "choices": [0.1, 1.0, 10.0]},
            "activation": {"type": "fixed", "value": "relu"},
        },
    }
    rows = grid_candidates(space)
    assert len(rows) == 6
    assert rows[0]["activation"] == "relu"
    assert {row["num_kernels"] for row in rows} == {1000, 3000}
    assert {row["ridge_alpha"] for row in rows} == {0.1, 1.0, 10.0}


def test_grid_rejects_continuous_parameter():
    space = {
        "optimization": {"strategy": "grid"},
        "parameters": {
            "learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 1e-2,
                "log": True,
            }
        },
    }
    with pytest.raises(ValueError, match="finite"):
        optimization_config(space)


def test_optuna_space_supports_ranges_fixed_values_and_structured_choices():
    space = {
        "optimization": {
            "strategy": "optuna",
            "sampler": "tpe",
            "n_trials": 20,
            "n_startup_trials": 5,
        },
        "parameters": {
            "architecture": {
                "type": "categorical",
                "choices": [[64], [128, 64]],
            },
            "activation": {"type": "fixed", "value": "relu"},
            "learning_rate": {
                "type": "float",
                "low": 3e-4,
                "high": 1e-2,
                "log": True,
            },
            "batch_size": {
                "type": "categorical",
                "choices": [16, 32, 128],
            },
        },
    }
    trial = FakeTrial()
    params = suggest_parameters(trial, space)
    assert params == {
        "architecture": [128, 64],
        "activation": "relu",
        "learning_rate": 1e-2,
        "batch_size": 128,
    }
    assert ("categorical", "architecture__choice", ["0", "1"]) in trial.calls
    assert ("float", "learning_rate", 3e-4, 1e-2, None, True) in trial.calls


def test_all_repository_model_spaces_declare_valid_strategy():
    root = Path(__file__).resolve().parents[1] / "config" / "model_spaces"
    strategies = {}
    for path in sorted(root.glob("*.json")):
        space = json.loads(path.read_text(encoding="utf-8"))
        strategies[path.stem] = optimization_config(space).strategy

    assert strategies["onishi_cnn"] == "fixed"
    grid_models = {
        "direct_minirocket",
        "shared_minirocket",
        "direct_linear_ridge",
        "shared_linear_ridge",
    }
    for name in grid_models:
        assert strategies[name] == "grid"
    assert all(
        strategy == "optuna"
        for name, strategy in strategies.items()
        if name not in {"onishi_cnn", *grid_models}
    )
