"""Typed views of existing JSON settings; no new scientific defaults.

These classes do not replace serialized dictionaries in fingerprints, logs or
artifacts. Acquisition sampling intervals remain per-event input metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RuntimeConfig:
    """Prediction chunking default; the JSON key is prediction_chunk_size."""

    prediction_chunk_size: int = 4096


@dataclass(frozen=True)
class XAIConfig:
    """Existing grouped-occlusion defaults, separate from gradient attribution."""

    enabled: bool = True
    group_size_samples: int = 8
    max_events: int = 512


@dataclass(frozen=True)
class BaselineConfig:
    """Trigger-relative baseline window in ns and clipping margin in mV.

    The window is required. Its documented repository example is (-2, -1) ns;
    silently inventing it for other datasets would change scientific behavior.
    """

    window_ns: tuple[float, float]
    clipping_margin_mV: float

    @classmethod
    def from_preprocessing(cls, config: Mapping[str, Any]) -> BaselineConfig:
        selection = config["selection"]
        low, high = selection["baseline_window_ns"]
        return cls(
            (float(low), float(high)),
            float(selection["baseline_clipping"]["margin_mV"]),
        )


@dataclass(frozen=True)
class TimingConfig:
    """Explicit LED offsets and optional CFD fractions; no fitted rules stored.

    LED thresholds are read from preprocessing. CFD fractions are supplied by
    the caller because the current batch schema does not configure a CFD scan.
    """

    thresholds_mV: tuple[float, ...]
    baseline: BaselineConfig
    fractions: tuple[float, ...] = ()

    @classmethod
    def from_preprocessing(
        cls, config: Mapping[str, Any], *, fractions: Sequence[float] = ()
    ) -> TimingConfig:
        return cls(
            tuple(map(float, config["led_selection"]["thresholds_mV"])),
            BaselineConfig.from_preprocessing(config),
            tuple(map(float, fractions)),
        )


@dataclass(frozen=True)
class MLPTrainingConfig:
    """Existing MLP/CNN defaults; all fields use the existing training JSON keys.

    min_delta is ps, gradient_clip_norm is the global L2 norm, and the internal
    holdout is separate from development CV. Validation remains in the trainer.
    """

    epochs: int = 300
    patience: int = 10
    min_delta: float = 0.01
    early_stopping_fraction: float = 0.20
    gradient_clip_norm: float = 10.0
    optimizer: str = "sgd_nesterov"
    momentum: float = 0.9
    gradient_early_stop: bool = False
    gradient_min_norm: float = 0.0
    gradient_patience: int = 4

    @classmethod
    def from_mapping(cls, training: Mapping[str, Any]) -> MLPTrainingConfig:
        # Keep every original coercion, including bool/string behavior.
        return cls(
            epochs=int(training.get("epochs", 300)),
            patience=int(training.get("patience", 10)),
            min_delta=float(training.get("min_delta", 0.01)),
            early_stopping_fraction=float(
                training.get("early_stopping_fraction", 0.20)
            ),
            gradient_clip_norm=float(training.get("gradient_clip_norm", 10.0)),
            optimizer=str(training.get("optimizer", "sgd_nesterov")).strip().lower(),
            momentum=float(training.get("momentum", 0.9)),
            gradient_early_stop=bool(training.get("gradient_early_stop", False)),
            gradient_min_norm=float(training.get("gradient_min_norm", 0.0)),
            gradient_patience=int(training.get("gradient_patience", 4)),
        )


@dataclass(frozen=True)
class OnishiTrainingConfig:
    """Existing Onishi full-split/MSE defaults, distinct from MLP settings."""

    epochs: int = 600
    lr_decay_epochs: tuple[int, ...] = (180, 360)
    lr_decay_factor: float = 0.1

    @classmethod
    def from_mapping(cls, training: Mapping[str, Any]) -> OnishiTrainingConfig:
        return cls(
            int(training.get("epochs", 600)),
            tuple(int(v) for v in training.get("lr_decay_epochs", [180, 360])),
            float(training.get("lr_decay_factor", 0.1)),
        )
