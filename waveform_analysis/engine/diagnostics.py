"""Post-fit diagnostics that do not participate in model selection."""

from __future__ import annotations

from contextlib import contextmanager
import random
from pathlib import Path
from typing import Any

import numpy as np

from waveform_analysis.data.storage import RunStore
from waveform_analysis.data.view import model_target
from waveform_analysis.engine.train import predict_indices

DEVELOPMENT_DISTRIBUTION_VERSION = 1


@contextmanager
def diagnostic_random_state():
    """Restore random generators after diagnostic model loading/inference."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        import torch
    except ImportError:
        torch = None
    try:
        if torch is None:
            yield
        else:
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def development_predictions_current(store: RunStore, final_fit_fingerprint: str) -> bool:
    """Check that the diagnostic artifact belongs to the current final fit."""
    if not store.development_predictions_path.is_file():
        return False
    with np.load(store.development_predictions_path) as data:
        return ("final_fit_fingerprint" in data
                and str(data["final_fit_fingerprint"].item()) == final_fit_fingerprint)


def save_development_diagnostic(
    store: RunStore, spec: Any, fitted: Any, dataset: Any, config: dict[str, Any],
    final_fit_fingerprint: str, *, frozen_features: Any = None,
) -> Path:
    """Persist final-model residuals on all prepared development events.

    This is a training-population diagnostic, not out-of-fold validation.
    Prediction arrays have shape ``(n_development_events,)`` and use the same
    output limits and native target definition as blind prediction.
    """
    with diagnostic_random_state():
        prediction = predict_indices(
            spec, fitted, dataset, config["mode"],
            np.arange(dataset.n_events, dtype=np.int64),
            chunk_size=int(config["runtime"]["prediction_chunk_size"]),
            frozen_features=frozen_features,
        )
    led = np.asarray(model_target(dataset, config["mode"]), dtype=np.float64)
    return store.save_predictions(
        event_id=dataset.event_index, prediction_ps=prediction,
        corrected_ps=led - prediction, led_residual_ps=led,
        dataset_role="development",
        metadata={
            "dataset_role": np.asarray("development"),
            "led_control_mean_ps": np.asarray(dataset.manifest["led_control_mean_ps"]),
            "final_fit_fingerprint": np.asarray(final_fit_fingerprint),
            "analysis_protocol_identity": np.asarray(dataset.manifest["analysis_protocol_identity"]),
        },
    )
