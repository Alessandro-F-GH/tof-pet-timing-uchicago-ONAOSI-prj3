"""Original Onishi full-split MSE trainer, separate from RMSE/holdout fitting."""

from __future__ import annotations

from typing import Any
import numpy as np
import torch
from waveform_analysis.core.config import OnishiTrainingConfig
from waveform_analysis.models.neural.onishi_cnn import (
    OnishiPairedCNN,
    OnishiCNNArtifact,
    _mse_loss,
)
from waveform_analysis.models.torch_runtime import (
    configure_reproducibility as _configure_reproducibility,
    device_from_config as _device,
    gradient_norm as _gradient_norm,
    make_loader as _loader,
    predict_tensor as _predict_tensor,
    rmse as _rmse,
)


def fit_onishi(
    params: dict[str, Any],
    train_x: np.ndarray,
    train_target: np.ndarray,
    *,
    seed: int,
    config: dict[str, Any],
) -> OnishiCNNArtifact:
    training_seed = _configure_reproducibility(seed)
    training = config.get("training", {})
    verbose = bool(config.get("verbose", False))
    logger = config.get("_logger")
    device = _device(config)

    batch = int(params.get("batch_size", training.get("batch_size", 128)))
    epochs = int(training.get("epochs", OnishiTrainingConfig.epochs))
    learning_rate = float(params.get("learning_rate", 1e-3))
    decay_epochs = [
        int(v)
        for v in training.get("lr_decay_epochs", OnishiTrainingConfig.lr_decay_epochs)
    ]
    decay_factor = float(
        training.get("lr_decay_factor", OnishiTrainingConfig.lr_decay_factor)
    )

    x = np.asarray(train_x, dtype=np.float32)
    target = np.asarray(train_target, dtype=np.float64)
    if x.shape[0] != target.shape[0]:
        raise ValueError(
            "onishi_cnn train_x and train_target must contain the same number of events"
        )
    if x.shape[0] < 1:
        raise ValueError("onishi_cnn training requires at least one event")

    model = OnishiPairedCNN(config.get("architecture", {})).to(device)
    with torch.no_grad():
        model(torch.from_numpy(np.ascontiguousarray(x[:1])).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=decay_epochs,
        gamma=decay_factor,
    )
    loader = _loader(x, target, batch, shuffle=True, seed=training_seed)
    output_limit = config.get("_prediction_max_abs_ps")

    if verbose and logger is not None:
        logger.info(
            "onishi_cnn training | configurable Onishi-style CNN | loss=MSE | optimizer=Adam | "
            "lr=%.6g | batch=%d | epochs=%d | lr_decay_epochs=%s | "
            "lr_decay_factor=%.6g | train=%d | device=%s",
            learning_rate,
            batch,
            epochs,
            decay_epochs,
            decay_factor,
            target.size,
            device,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_gradient_norms = []
        for pair, batch_target in loader:
            pair = pair.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _mse_loss(model(pair), batch_target)
            loss.backward()
            if verbose and logger is not None:
                epoch_gradient_norms.append(_gradient_norm(model))
            optimizer.step()
        scheduler.step()

        if verbose and logger is not None:
            prediction = _predict_tensor(model, x, device, batch)
            if output_limit is not None:
                prediction = np.clip(
                    prediction, -float(output_limit), float(output_limit)
                )
            train_rmse = _rmse(prediction - target)
            logger.info(
                "onishi_cnn epoch %d/%d | train RMSE=%.4f ps | grad norm=%.6g | "
                "lr=%.6g | pred mean=%.4f ps | pred std=%.4f ps | "
                "pred min=%.4f ps | pred max=%.4f ps",
                epoch,
                epochs,
                train_rmse,
                float(np.mean(epoch_gradient_norms))
                if epoch_gradient_norms
                else float("nan"),
                float(optimizer.param_groups[0]["lr"]),
                float(np.mean(prediction)),
                float(np.std(prediction)),
                float(np.min(prediction)),
                float(np.max(prediction)),
            )

    return OnishiCNNArtifact(
        model=model,
        device=str(device),
        metadata={
            "training_loss": "mse",
            "optimizer": "adam",
            "epochs": epochs,
            "training_events": int(target.size),
            "training_uses_full_split": True,
            "refit_on_full_training_split": False,
            "learning_rate": learning_rate,
            "lr_decay_epochs": decay_epochs,
            "lr_decay_factor": decay_factor,
            "batch_size": batch,
            "output_max_abs_ps": None if output_limit is None else float(output_limit),
            "training_seed": training_seed,
            "deterministic_algorithms": True,
            "architecture_reference": "Onishi-style paired waveform CNN; depth, widths, kernels, and training settings are configurable",
            "input_definition": "paired normalized detector waveforms stacked as [2,time]",
            "prediction_definition": "joint CNN correction f_theta([s1;s2]) [ps]",
            "detector_axis_policy": "first 2x5 convolution fuses the two detector rows immediately",
            "detector_swap_antisymmetry_enforced": False,
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        },
    )
