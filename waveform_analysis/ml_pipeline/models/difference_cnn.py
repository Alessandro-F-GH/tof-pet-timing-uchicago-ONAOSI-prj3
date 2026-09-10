from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .cnn import SharedScorerCNN, _device, _loader, _rmse, candidates
from .spec import ModelSpec


def _raw_difference(pair: np.ndarray) -> np.ndarray:
    values = np.asarray(pair, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"difference_cnn expects [event, detector=2, time], got {values.shape}")
    return np.ascontiguousarray(values[:, 0, :] - values[:, 1, :], dtype=np.float32)


def _centered_difference(pair: np.ndarray, mean_difference: np.ndarray) -> np.ndarray:
    difference = _raw_difference(pair)
    mean = np.asarray(mean_difference, dtype=np.float32).reshape(1, -1)
    if difference.shape[1] != mean.shape[1]:
        raise ValueError(
            f"difference_cnn input length changed: waveform has {difference.shape[1]} samples, "
            f"training mean has {mean.shape[1]}"
        )
    return np.ascontiguousarray(difference - mean, dtype=np.float32)


@dataclass
class DifferenceCNNArtifact:
    model: SharedScorerCNN
    mean_difference: np.ndarray
    device: str
    metadata: dict[str, Any]


def _predict_centered(model: SharedScorerCNN, x: np.ndarray, device: torch.device, batch: int) -> np.ndarray:
    loader = DataLoader(
        torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
        batch_size=int(batch),
        shuffle=False,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for difference in loader:
            values.append(model.score(difference.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def fit(
    params,
    train_x,
    train_target,
    *,
    seed,
    config,
    validation_x=None,
    validation_target=None,
):
    if validation_x is None or validation_target is None:
        raise ValueError("difference_cnn training requires a validation set for early stopping")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    raw_train = _raw_difference(train_x)
    mean_difference = np.mean(raw_train, axis=0, dtype=np.float64).astype(np.float32)
    centered_train = np.ascontiguousarray(raw_train - mean_difference[None, :], dtype=np.float32)
    centered_validation = _centered_difference(validation_x, mean_difference)

    training = config.get("training", {})
    device = _device(config)
    model = SharedScorerCNN(config.get("architecture", {})).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    batch = int(params.get("batch_size", training.get("batch_size", 64)))
    max_epochs = int(training.get("epochs", 350))
    patience = int(training.get("patience", 30))
    min_delta = float(training.get("min_delta", 0.05))
    loader = _loader(centered_train, train_target, batch, shuffle=True, seed=seed)
    output_limit = config.get("_prediction_max_abs_ps")

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    validation_target = np.asarray(validation_target, dtype=np.float64)

    for epoch in range(1, max_epochs + 1):
        model.train()
        for difference, target in loader:
            difference = difference.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model.score(difference), target)
            loss.backward()
            clip = float(training.get("gradient_clip_norm", 10.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        prediction = _predict_centered(model, centered_validation, device, batch)
        if output_limit is not None:
            prediction = np.clip(prediction, -float(output_limit), float(output_limit))
        score = _rmse(prediction - validation_target)
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
        raise RuntimeError("difference_cnn early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    return DifferenceCNNArtifact(
        model=model,
        mean_difference=mean_difference,
        device=str(device),
        metadata={
            "best_epoch": int(best_epoch),
            "best_validation_rmse_ps": float(best_score),
            "selection_metric": "validation_rmse",
            "batch_size": batch,
            "output_max_abs_ps": None if output_limit is None else float(output_limit),
            "input_definition": "normalized window difference d=s1-s2, centered by the training-set mean difference waveform mu_d(t)",
            "difference_centering": "mu_d(t)=mean_training[s1(t)-s2(t)]; input=d(t)-mu_d(t)",
            "mean_difference_shape": list(mean_difference.shape),
            "prediction_definition": "single 1-D CNN h_theta((s1-s2)-mean_training(s1-s2)) [ps]",
        },
    )


def predict(artifact: DifferenceCNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    centered = _centered_difference(normalized_pair, artifact.mean_difference)
    return _predict_centered(artifact.model, centered, torch.device(artifact.device), 512)


def save(artifact: DifferenceCNNArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": artifact.model.state_dict(),
            "mean_difference": np.asarray(artifact.mean_difference, dtype=np.float32),
            "metadata": artifact.metadata,
        },
        path / "model.pt",
    )
    np.save(path / "training_mean_difference.npy", np.asarray(artifact.mean_difference, dtype=np.float32))


def explain(artifact: DifferenceCNNArtifact, normalized_pair: np.ndarray) -> np.ndarray:
    device = torch.device(artifact.device)
    pair = torch.tensor(np.asarray(normalized_pair, dtype=np.float32), device=device, requires_grad=True)
    mean = torch.tensor(np.asarray(artifact.mean_difference, dtype=np.float32), device=device)[None, :]
    centered_difference = pair[:, 0, :] - pair[:, 1, :] - mean
    artifact.model.zero_grad(set_to_none=True)
    artifact.model.score(centered_difference).sum().backward()
    return pair.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)


MODEL_SPEC = ModelSpec(
    name="difference_cnn",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
