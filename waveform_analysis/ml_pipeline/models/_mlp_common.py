from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from .cnn import (
    _configure_reproducibility,
    _device,
    _gradient_norm,
    _internal_early_stopping_split,
    _loader,
    _predict_tensor,
    _rmse,
    _rmse_loss,
)


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
}


def _activation(name: str) -> nn.Module:
    key = str(name).strip().lower()
    try:
        return _ACTIVATIONS[key]()
    except KeyError as exc:
        raise ValueError(
            f"Unsupported MLP activation {name!r}; available: {sorted(_ACTIVATIONS)}"
        ) from exc


def _validate_architecture(architecture) -> list[int]:
    if not isinstance(architecture, (list, tuple)) or not architecture:
        raise ValueError("MLP architecture must be a non-empty list of hidden-layer widths")
    widths = [int(value) for value in architecture]
    if any(width <= 0 for width in widths):
        raise ValueError("MLP architecture widths must all be positive")
    return widths


class DenseStack(nn.Module):
    def __init__(self, input_dim: int, architecture, activation: str):
        super().__init__()
        widths = _validate_architecture(architecture)
        layers: list[nn.Module] = []
        incoming = int(input_dim)
        for width in widths:
            layers.extend([nn.Linear(incoming, width), _activation(activation)])
            incoming = width
        layers.append(nn.Linear(incoming, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


@dataclass
class MLPArtifact:
    model: nn.Module
    device: str
    metadata: dict[str, Any]


def candidates(config):
    parameters = config.get("parameters", {})
    training = config.get("training", {})
    architectures = parameters.get("architecture", [[128, 64]])
    activations = parameters.get("activation", ["silu"])
    learning_rates = parameters.get("learning_rate", [1e-3])
    batch_sizes = parameters.get("batch_size", [training.get("batch_size", 128)])
    weight_decays = parameters.get("weight_decay", [1e-5])
    return [
        {
            "architecture": _validate_architecture(architecture),
            "activation": str(activation).strip().lower(),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
        }
        for architecture, activation, learning_rate, batch_size, weight_decay in itertools.product(
            architectures, activations, learning_rates, batch_sizes, weight_decays
        )
    ]


def fit_mlp(
    *,
    model_name: str,
    model_factory,
    params,
    train_x,
    train_target,
    seed,
    config,
    metadata_extra: dict[str, Any],
):
    training_seed = _configure_reproducibility(seed)
    training = config.get("training", {})
    verbose = bool(config.get("verbose", False))
    logger = config.get("_logger")
    device = _device(config)

    batch = int(params["batch_size"])
    max_epochs = int(training.get("epochs", 300))
    patience = int(training.get("patience", 10))
    min_delta = float(training.get("min_delta", 0.01))
    early_fraction = float(training.get("early_stopping_fraction", 0.20))
    clip = float(training.get("gradient_clip_norm", 10.0))
    output_limit = config.get("_prediction_max_abs_ps")
    split_seed = int(config.get("_early_stopping_seed", seed))

    fit_x, fit_target, early_x, early_target = _internal_early_stopping_split(
        train_x, train_target, early_fraction, split_seed
    )

    architecture = _validate_architecture(params["architecture"])
    activation = str(params["activation"]).strip().lower()
    model = model_factory(
        int(train_x.shape[-1]), architecture, activation
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = _rmse_loss
    loader = _loader(
        fit_x, fit_target, batch, shuffle=True, seed=training_seed
    )

    if verbose and logger is not None:
        logger.info(
            "%s training | loss=RMSE | architecture=%s | activation=%s | "
            "lr=%.6g | weight_decay=%.6g | batch=%d | epochs=%d | patience=%d | min_delta=%.6g | "
            "early_stop_fraction=%.3f | fit=%d | early_stop=%d | device=%s",
            model_name,
            architecture,
            activation,
            float(params["learning_rate"]),
            float(params["weight_decay"]),
            batch,
            max_epochs,
            patience,
            min_delta,
            early_fraction,
            fit_target.size,
            early_target.size,
            device,
        )

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_gradient_norms = []
        for pair, target in loader:
            pair = pair.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(pair), target)
            loss.backward()
            if verbose and logger is not None:
                epoch_gradient_norms.append(_gradient_norm(model))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        prediction = _predict_tensor(model, early_x, device, batch)
        if output_limit is not None:
            prediction = np.clip(
                prediction, -float(output_limit), float(output_limit)
            )
        score = _rmse(prediction - early_target)

        if verbose and logger is not None:
            fit_prediction = _predict_tensor(model, fit_x, device, batch)
            if output_limit is not None:
                fit_prediction = np.clip(
                    fit_prediction, -float(output_limit), float(output_limit)
                )
            fit_score = _rmse(fit_prediction - fit_target)
            logger.info(
                "%s epoch %d/%d | fit RMSE=%.4f ps | early-stop RMSE=%.4f ps | "
                "grad norm=%.6g | pred mean=%.4f ps | pred std=%.4f ps | "
                "pred min=%.4f ps | pred max=%.4f ps",
                model_name,
                epoch,
                max_epochs,
                fit_score,
                score,
                float(np.mean(epoch_gradient_norms))
                if epoch_gradient_norms
                else float("nan"),
                float(np.mean(prediction)),
                float(np.std(prediction)),
                float(np.min(prediction)),
                float(np.max(prediction)),
            )

        if score < best_score - min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError(f"{model_name} early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    metadata = {
        "best_epoch": int(best_epoch),
        "training_loss": "rmse",
        "best_early_stopping_rmse_ps": float(best_score),
        "early_stopping_metric": "internal_train_holdout_rmse",
        "early_stopping_fraction": early_fraction,
        "early_stopping_events": int(early_target.size),
        "optimizer_training_events": int(fit_target.size),
        "training_events_available": int(len(train_target)),
        "training_uses_full_split": False,
        "refit_on_full_training_split": False,
        "external_validation_used_for_early_stopping": False,
        "early_stopping_split_seed": split_seed,
        "architecture": architecture,
        "activation": activation,
        "learning_rate": float(params["learning_rate"]),
        "weight_decay": float(params["weight_decay"]),
        "batch_size": batch,
        "output_max_abs_ps": None
        if output_limit is None
        else float(output_limit),
        "training_seed": training_seed,
        "deterministic_algorithms": True,
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
    }
    metadata.update(metadata_extra)
    return MLPArtifact(model, str(device), metadata)


def predict(artifact: MLPArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    return _predict_tensor(
        artifact.model,
        normalized_pair,
        torch.device(artifact.device),
        512,
    )


def save(artifact: MLPArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifact.model.state_dict(),
            "metadata": artifact.metadata,
        },
        path / "model.pt",
    )


def explain(artifact: MLPArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    device = torch.device(artifact.device)
    pair = torch.tensor(
        np.asarray(normalized_pair, dtype=np.float32),
        device=device,
        requires_grad=True,
    )
    artifact.model.eval()
    artifact.model.zero_grad(set_to_none=True)
    artifact.model(pair).sum().backward()
    if pair.grad is None:
        raise RuntimeError("MLP XAI gradient is unavailable")
    return (
        pair.grad.detach()
        .abs()
        .mean(dim=(0, 1))
        .cpu()
        .numpy()
        .astype(np.float64)
    )
