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
    input_group_lassos = parameters.get("input_group_lasso", [0.0])
    return [
        {
            "architecture": _validate_architecture(architecture),
            "activation": str(activation).strip().lower(),
            "learning_rate": float(learning_rate),
            "batch_size": int(batch_size),
            "weight_decay": float(weight_decay),
            "input_group_lasso": float(input_group_lasso),
        }
        for architecture, activation, learning_rate, batch_size, weight_decay, input_group_lasso in itertools.product(
            architectures,
            activations,
            learning_rates,
            batch_sizes,
            weight_decays,
            input_group_lassos,
        )
    ]


def _first_layer_weight(model: nn.Module) -> torch.Tensor:
    scorer = getattr(model, "scorer", None)
    network = getattr(scorer, "network", None)
    if not isinstance(network, nn.Sequential) or not network:
        raise TypeError("Input group lasso requires an MLP scorer with a Sequential first layer")
    first = network[0]
    if not isinstance(first, nn.Linear):
        raise TypeError("Input group lasso requires the first MLP scorer layer to be Linear")
    return first.weight


def _input_group_lasso_penalty(model: nn.Module) -> torch.Tensor:
    """Group lasso over temporal inputs: one group is one first-layer weight column."""
    weight = _first_layer_weight(model)
    return torch.linalg.vector_norm(weight, ord=2, dim=0).sum()


def _restart_seeds(seed: int, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("training.random_restarts must be >= 1")
    if count == 1:
        return [int(seed)]
    rng = np.random.default_rng(int(seed))
    extra = rng.integers(0, np.iinfo(np.int32).max, size=count - 1, dtype=np.int64)
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
    group_lambda = float(params.get("input_group_lasso", 0.0))
    if group_lambda < 0:
        raise ValueError("input_group_lasso must be non-negative")
    output_limit = config.get("_prediction_max_abs_ps")
    split_seed = int(config.get("_early_stopping_seed", seed))

    # Keep the internal early-stopping population identical for every restart.
    fit_x, fit_target, early_x, early_target = _internal_early_stopping_split(
        train_x, train_target, early_fraction, split_seed
    )

    architecture = _validate_architecture(params["architecture"])
    activation = str(params["activation"]).strip().lower()

    def run_restart(restart_seed: int) -> MLPArtifact:
        training_seed = _configure_reproducibility(restart_seed)
        model = model_factory(
            int(train_x.shape[-1]), architecture, activation
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(params["learning_rate"]),
            weight_decay=float(params["weight_decay"]),
        )
        loader = _loader(
            fit_x, fit_target, batch, shuffle=True, seed=training_seed
        )

        if verbose and logger is not None:
            logger.info(
                "%s training | loss=RMSE+input_group_lasso | architecture=%s | activation=%s | "
                "lr=%.6g | weight_decay=%.6g | input_group_lasso=%.6g | batch=%d | "
                "epochs=%d | patience=%d | min_delta=%.6g | early_stop_fraction=%.3f | "
                "fit=%d | early_stop=%d | device=%s | seed=%d",
                model_name,
                architecture,
                activation,
                float(params["learning_rate"]),
                float(params["weight_decay"]),
                group_lambda,
                batch,
                max_epochs,
                patience,
                min_delta,
                early_fraction,
                fit_target.size,
                early_target.size,
                device,
                training_seed,
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
                prediction = model(pair)
                rmse_loss = _rmse_loss(prediction, target)
                if group_lambda > 0:
                    loss = rmse_loss + group_lambda * _input_group_lasso_penalty(model)
                else:
                    loss = rmse_loss
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
                with torch.no_grad():
                    group_penalty = float(_input_group_lasso_penalty(model).detach().cpu())
                logger.info(
                    "%s epoch %d/%d | fit RMSE=%.4f ps | early-stop RMSE=%.4f ps | "
                    "input-group norm sum=%.6g | grad norm=%.6g | pred mean=%.4f ps | "
                    "pred std=%.4f ps | pred min=%.4f ps | pred max=%.4f ps",
                    model_name,
                    epoch,
                    max_epochs,
                    fit_score,
                    score,
                    group_penalty,
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

        with torch.no_grad():
            group_norms = (
                torch.linalg.vector_norm(_first_layer_weight(model), ord=2, dim=0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

        metadata = {
            "best_epoch": int(best_epoch),
            "training_loss": "rmse_plus_input_group_lasso",
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
            "input_group_lasso": group_lambda,
            "input_group_lasso_definition": "lambda * sum_t ||W_first[:, t]||_2; one group per temporal input sample",
            "input_group_norm_sum": float(np.sum(group_norms)),
            "input_group_norm_mean": float(np.mean(group_norms)),
            "input_group_norm_median": float(np.median(group_norms)),
            "input_group_near_zero_fraction": float(np.mean(group_norms <= 1e-6)),
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
            "early_stopping_rmse_ps": float(
                artifact.metadata["best_early_stopping_rmse_ps"]
            ),
        }
        restart_records.append(record)
        if logger is not None and random_restarts > 1:
            logger.info(
                "Restart | %s | %d/%d | seed=%d | best epoch=%d | early-stop RMSE=%.4f ps",
                model_name,
                restart_index,
                random_restarts,
                restart_seed,
                record["best_epoch"],
                record["early_stopping_rmse_ps"],
            )

    selected_index = min(
        range(len(restart_artifacts)),
        key=lambda index: (
            float(restart_artifacts[index].metadata["best_early_stopping_rmse_ps"]),
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
