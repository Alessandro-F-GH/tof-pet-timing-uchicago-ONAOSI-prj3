from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ._torch_common import (
    configure_reproducibility as _configure_reproducibility,
    device_from_config as _device,
    gradient_norm as _gradient_norm,
    internal_early_stopping_split as _internal_early_stopping_split,
    make_loader as _loader,
    predict_tensor as _predict_tensor,
    rmse as _rmse,
    rmse_loss as _rmse_loss,
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
    weight_decays = parameters.get("weight_decay", [0.0])
    return [
        {
            "architecture": _validate_architecture(architecture),
            "activation": str(activation).strip().lower(),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
        }
        for architecture, activation, learning_rate, batch_size, weight_decay
        in itertools.product(
            architectures,
            activations,
            learning_rates,
            batch_sizes,
            weight_decays,
        )
    ]


def _restart_seeds(seed: int, count: int) -> list[int]:
    if count < 1:
        raise ValueError("training.random_restarts must be >= 1")
    if count == 1:
        return [int(seed)]
    rng = np.random.default_rng(int(seed))
    extra = rng.integers(
        0,
        np.iinfo(np.int32).max,
        size=count - 1,
        dtype=np.int64,
    )
    return [int(seed), *[int(value) for value in extra]]


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
    random_restarts = int(training.get("random_restarts", 1))

    gradient_early_stop = bool(training.get("gradient_early_stop", False))
    gradient_min_norm = float(training.get("gradient_min_norm", 0.0))
    gradient_patience = int(training.get("gradient_patience", 4))
    if gradient_min_norm < 0:
        raise ValueError("training.gradient_min_norm must be non-negative")
    if gradient_patience < 1:
        raise ValueError("training.gradient_patience must be >= 1")

    output_limit = config.get("_prediction_max_abs_ps")
    split_seed = int(config.get("_early_stopping_seed", seed))

    # This split is fixed across restarts. Restarts differ only in initialization
    # and minibatch order, so restart selection cannot exploit a different holdout.
    fit_x, fit_target, early_x, early_target = _internal_early_stopping_split(
        train_x,
        train_target,
        early_fraction,
        split_seed,
    )

    architecture = _validate_architecture(params["architecture"])
    activation = str(params["activation"]).strip().lower()

    def run_restart(restart_seed: int) -> MLPArtifact:
        training_seed = _configure_reproducibility(restart_seed)
        model = model_factory(
            int(train_x.shape[-1]),
            architecture,
            activation,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        loader = _loader(
            fit_x,
            fit_target,
            batch,
            shuffle=True,
            seed=training_seed,
        )

        if verbose and logger is not None:
            logger.info(
                "%s training | loss=RMSE | architecture=%s | activation=%s | "
                "lr=%.6g | weight_decay=%.6g | batch=%d | "
                "epochs=%d | rmse_patience=%d | min_delta=%.6g | early_stop_fraction=%.3f | "
                "gradient_stop=%s | gradient_min_norm=%.6g | gradient_patience=%d | "
                "fit=%d | early_stop=%d | device=%s | seed=%d",
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
                gradient_early_stop,
                gradient_min_norm,
                gradient_patience,
                fit_target.size,
                early_target.size,
                device,
                training_seed,
            )

        best_score = float("inf")
        best_epoch = 0
        best_state = None
        rmse_stale_epochs = 0
        low_gradient_epochs = 0
        stop_reason = "max_epochs"
        stop_epoch = max_epochs
        last_gradient_norm = float("nan")

        for epoch in range(1, max_epochs + 1):
            model.train()
            epoch_gradient_norms: list[float] = []

            for pair, target in loader:
                pair = pair.to(device)
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)

                prediction = model(pair)
                loss = _rmse_loss(prediction, target)
                loss.backward()

                # Always compute the pre-clipping gradient norm because it is a
                # training stop criterion, not merely a verbose diagnostic.
                epoch_gradient_norms.append(float(_gradient_norm(model)))

                if clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            if not epoch_gradient_norms:
                raise RuntimeError(f"{model_name} produced no optimizer batches")
            last_gradient_norm = float(np.mean(epoch_gradient_norms))
            if not np.isfinite(last_gradient_norm):
                raise RuntimeError(
                    f"{model_name} produced a non-finite gradient norm at epoch {epoch}"
                )

            prediction = _predict_tensor(model, early_x, device, batch)
            if output_limit is not None:
                prediction = np.clip(
                    prediction,
                    -float(output_limit),
                    float(output_limit),
                )
            score = _rmse(prediction - early_target)

            if not np.isfinite(score):
                raise RuntimeError(
                    f"{model_name} produced a non-finite early-stopping RMSE at epoch {epoch}"
                )

            if verbose and logger is not None:
                fit_prediction = _predict_tensor(model, fit_x, device, batch)
                if output_limit is not None:
                    fit_prediction = np.clip(
                        fit_prediction,
                        -float(output_limit),
                        float(output_limit),
                    )
                fit_score = _rmse(fit_prediction - fit_target)
                logger.info(
                    "%s epoch %d/%d | fit RMSE=%.4f ps | early-stop RMSE=%.4f ps | "
                    "grad norm=%.6g | pred mean=%.4f ps | "
                    "pred std=%.4f ps | pred min=%.4f ps | pred max=%.4f ps",
                    model_name,
                    epoch,
                    max_epochs,
                    fit_score,
                    score,
                    last_gradient_norm,
                    float(np.mean(prediction)),
                    float(np.std(prediction)),
                    float(np.min(prediction)),
                    float(np.max(prediction)),
                )

            if score < best_score - min_delta:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                rmse_stale_epochs = 0
            else:
                rmse_stale_epochs += 1

            if gradient_early_stop and last_gradient_norm < gradient_min_norm:
                low_gradient_epochs += 1
            else:
                low_gradient_epochs = 0

            if rmse_stale_epochs >= patience:
                stop_reason = "rmse_patience"
                stop_epoch = epoch
                break

            if gradient_early_stop and low_gradient_epochs >= gradient_patience:
                stop_reason = "gradient_norm"
                stop_epoch = epoch
                break

        if best_state is None:
            raise RuntimeError(
                f"{model_name} early stopping did not produce a valid checkpoint"
            )

        model.load_state_dict(best_state)

        metadata = {
            "best_epoch": int(best_epoch),
            "stop_epoch": int(stop_epoch),
            "stop_reason": stop_reason,
            "training_loss": "rmse",
            "best_early_stopping_rmse_ps": float(best_score),
            "early_stopping_metric": "internal_train_holdout_rmse",
            "early_stopping_fraction": early_fraction,
            "early_stopping_events": int(early_target.size),
            "rmse_early_stopping_patience": patience,
            "rmse_early_stopping_min_delta_ps": min_delta,
            "gradient_early_stop_enabled": gradient_early_stop,
            "gradient_min_norm": gradient_min_norm,
            "gradient_patience": gradient_patience,
            "gradient_low_epochs_at_stop": int(low_gradient_epochs),
            "last_epoch_mean_gradient_norm": float(last_gradient_norm),
            "gradient_norm_definition": "mean pre-clipping global parameter gradient L2 norm across optimizer batches in the epoch",
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
            "output_max_abs_ps": (
                None if output_limit is None else float(output_limit)
            ),
            "training_seed": training_seed,
            "deterministic_algorithms": True,
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }
        metadata.update(metadata_extra)
        return MLPArtifact(model, str(device), metadata)

    restart_seeds = _restart_seeds(int(seed), random_restarts)
    restart_artifacts: list[MLPArtifact] = []
    restart_records: list[dict[str, Any]] = []

    for restart_index, restart_seed in enumerate(restart_seeds, start=1):
        artifact = run_restart(restart_seed)
        restart_artifacts.append(artifact)
        record = {
            "restart": restart_index,
            "seed": int(restart_seed),
            "best_epoch": int(artifact.metadata["best_epoch"]),
            "stop_epoch": int(artifact.metadata["stop_epoch"]),
            "stop_reason": str(artifact.metadata["stop_reason"]),
            "early_stopping_rmse_ps": float(
                artifact.metadata["best_early_stopping_rmse_ps"]
            ),
            "last_epoch_mean_gradient_norm": float(
                artifact.metadata["last_epoch_mean_gradient_norm"]
            ),
        }
        restart_records.append(record)

        if logger is not None and random_restarts > 1:
            logger.info(
                "Restart | %s | %d/%d | seed=%d | best epoch=%d | stop epoch=%d | "
                "stop=%s | early-stop RMSE=%.4f ps | grad norm=%.6g",
                model_name,
                restart_index,
                random_restarts,
                restart_seed,
                record["best_epoch"],
                record["stop_epoch"],
                record["stop_reason"],
                record["early_stopping_rmse_ps"],
                record["last_epoch_mean_gradient_norm"],
            )

    selected_index = min(
        range(len(restart_artifacts)),
        key=lambda index: (
            float(
                restart_artifacts[index].metadata[
                    "best_early_stopping_rmse_ps"
                ]
            ),
            index,
        ),
    )
    selected = restart_artifacts[selected_index]
    selected.metadata.update(
        {
            "random_restarts": random_restarts,
            "restart_selection_metric": "internal_train_holdout_rmse",
            "selected_restart": selected_index + 1,
            "restart_results": restart_records,
        }
    )

    if logger is not None and random_restarts > 1:
        logger.info(
            "Selected restart | %s | %d/%d | seed=%d | early-stop RMSE=%.4f ps",
            model_name,
            selected_index + 1,
            random_restarts,
            restart_seeds[selected_index],
            float(selected.metadata["best_early_stopping_rmse_ps"]),
        )

    return selected


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
