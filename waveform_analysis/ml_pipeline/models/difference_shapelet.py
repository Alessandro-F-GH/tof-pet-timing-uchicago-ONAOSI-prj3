from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .cnn import _device, _loader, _rmse
from .spec import ModelSpec


def _difference(pair: np.ndarray) -> np.ndarray:
    values = np.asarray(pair, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"difference_shapelet expects [event, detector=2, time], got {values.shape}")
    return np.ascontiguousarray(values[:, 0, :] - values[:, 1, :], dtype=np.float32)


def candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = config.get("parameters", {})
    training = config.get("training", {})
    return [
        {
            "learning_rate": float(lr),
            "weight_decay": float(wd),
            "batch_size": int(batch),
        }
        for lr, wd, batch in itertools.product(
            parameters.get("learning_rate", [1e-3]),
            parameters.get("weight_decay", [1e-5]),
            parameters.get("batch_size", [training.get("batch_size", 64)]),
        )
    ]


def _shapelet_layout(architecture: dict[str, Any], input_length: int):
    lengths = [int(v) for v in architecture.get("shapelet_lengths", [17, 33, 65])]
    raw = architecture.get("shapelets_per_length", [8] * len(lengths))
    counts = [int(raw)] * len(lengths) if isinstance(raw, (int, float)) else [int(v) for v in raw]
    if not lengths or len(lengths) != len(counts):
        raise ValueError("shapelet_lengths and shapelets_per_length must be non-empty and have equal length")
    if any(v < 2 or v > input_length for v in lengths):
        raise ValueError(f"Shapelet lengths must lie in [2, {input_length}], got {lengths}")
    if any(v < 1 for v in counts):
        raise ValueError("Every shapelet group must contain at least one shapelet")
    return lengths, counts


def _initialize_shapelets(x, lengths, counts, *, seed):
    rng = np.random.default_rng(int(seed))
    n_events, n_samples = x.shape
    groups = []
    for length, count in zip(lengths, counts):
        templates = np.empty((count, 1, length), dtype=np.float32)
        for i in range(count):
            event = int(rng.integers(0, n_events))
            start = int(rng.integers(0, n_samples - length + 1))
            templates[i, 0] = x[event, start:start + length]
        groups.append(templates)
    return groups


class LearnableShapeletTransform(nn.Module):
    """Learn local templates and return each template's minimum subsequence MSE."""

    def __init__(self, initial_shapelets, *, match_stride):
        super().__init__()
        if int(match_stride) < 1:
            raise ValueError("match_stride must be >= 1")
        self.match_stride = int(match_stride)
        self.shapelets = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(np.asarray(g, dtype=np.float32))) for g in initial_shapelets]
        )
        self.lengths = [int(g.shape[-1]) for g in initial_shapelets]
        self.counts = [int(g.shape[0]) for g in initial_shapelets]

    @property
    def n_features(self):
        return int(sum(self.counts))

    def forward(self, difference):
        if difference.ndim != 2:
            raise ValueError(f"Shapelet transform expects [batch, time], got {tuple(difference.shape)}")
        signal = difference[:, None, :]
        features = []
        for shapelets, length in zip(self.shapelets, self.lengths):
            # Mean squared Euclidean distance from every learned shapelet to every
            # sliding subsequence. The convolution identity avoids materializing
            # [batch, position, length] windows.
            cross = F.conv1d(signal, shapelets, stride=self.match_stride)
            signal_sq = F.avg_pool1d(signal.square(), kernel_size=length, stride=self.match_stride)
            shapelet_sq = shapelets.square().mean(dim=2).view(1, -1, 1)
            distance = signal_sq - (2.0 / float(length)) * cross + shapelet_sq
            features.append(distance.clamp_min(0.0).amin(dim=2))
        return torch.cat(features, dim=1)


class DifferenceShapeletRegressor(nn.Module):
    def __init__(self, initial_shapelets, *, match_stride, dense_units, dropout):
        super().__init__()
        self.transform = LearnableShapeletTransform(initial_shapelets, match_stride=match_stride)
        incoming = self.transform.n_features
        head = [nn.BatchNorm1d(incoming)]
        for width in dense_units:
            head.extend([nn.Linear(incoming, int(width)), nn.SiLU()])
            if dropout > 0:
                head.append(nn.Dropout(float(dropout)))
            incoming = int(width)
        head.append(nn.Linear(incoming, 1))
        self.head = nn.Sequential(*head)

    def forward(self, difference):
        return self.head(self.transform(difference)).squeeze(1)


@dataclass
class DifferenceShapeletArtifact:
    model: DifferenceShapeletRegressor
    device: str
    metadata: dict[str, Any]


def _predict_difference(model, difference, device, batch_size):
    loader = DataLoader(
        torch.from_numpy(np.ascontiguousarray(difference, dtype=np.float32)),
        batch_size=int(batch_size),
        shuffle=False,
    )
    values = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            values.append(model(batch.to(device)).detach().cpu().numpy())
    return np.concatenate(values).astype(np.float64, copy=False)


def fit(params, train_x, train_target, *, seed, config, validation_x=None, validation_target=None):
    if validation_x is None or validation_target is None:
        raise ValueError("difference_shapelet training requires a validation set for early stopping")

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    train_difference = _difference(train_x)
    validation_difference = _difference(validation_x)
    architecture = config.get("architecture", {})
    lengths, counts = _shapelet_layout(architecture, train_difference.shape[1])
    match_stride = int(architecture.get("match_stride", 4))
    dense_units = [int(v) for v in architecture.get("dense_units", [32, 16])]
    dropout = float(architecture.get("dropout", 0.05))
    if match_stride < 1:
        raise ValueError("architecture.match_stride must be >= 1")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("architecture.dropout must be in [0, 1)")

    initial_shapelets = _initialize_shapelets(train_difference, lengths, counts, seed=seed)
    device = _device(config)
    model = DifferenceShapeletRegressor(
        initial_shapelets,
        match_stride=match_stride,
        dense_units=dense_units,
        dropout=dropout,
    ).to(device)

    training = config.get("training", {})
    batch_size = int(params.get("batch_size", training.get("batch_size", 64)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    loader = _loader(train_difference, train_target, batch_size, shuffle=True, seed=seed)
    max_epochs = int(training.get("epochs", 200))
    patience = int(training.get("patience", 15))
    min_delta = float(training.get("min_delta", 0.01))
    output_limit = config.get("_prediction_max_abs_ps")
    validation_target = np.asarray(validation_target, dtype=np.float64)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for difference, target in loader:
            difference = difference.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(difference), target)
            loss.backward()
            clip = float(training.get("gradient_clip_norm", 10.0))
            if clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        prediction = _predict_difference(model, validation_difference, device, batch_size)
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
        raise RuntimeError("difference_shapelet early stopping did not produce a valid checkpoint")
    model.load_state_dict(best_state)

    input_time_ps = np.asarray(config.get("_input_time_ps", []), dtype=np.float64)
    if input_time_ps.size > 1:
        dt_ps = float(np.median(np.diff(input_time_ps)))
        durations_ps = [float((length - 1) * dt_ps) for length in lengths]
    else:
        dt_ps = float("nan")
        durations_ps = [float("nan")] * len(lengths)

    metadata = {
        "best_epoch": int(best_epoch),
        "best_validation_rmse_ps": float(best_score),
        "selection_metric": "validation_rmse",
        "batch_size": batch_size,
        "output_max_abs_ps": None if output_limit is None else float(output_limit),
        "input_definition": "normalized window difference d(t)=s1(t)-s2(t)",
        "representation": "learnable shapelet transform: minimum sliding mean-squared distance per shapelet",
        "prediction_definition": "nonlinear regression from learned shapelet-distance features of s1-s2 [ps]",
        "shapelet_lengths_samples": lengths,
        "shapelets_per_length": counts,
        "shapelet_durations_ps": durations_ps,
        "input_sample_interval_ps": dt_ps,
        "match_stride_samples": match_stride,
        "n_shapelet_features": int(sum(counts)),
        "shapelet_initialization": "random training subsequences",
    }
    return DifferenceShapeletArtifact(model=model, device=str(device), metadata=metadata)


def predict(artifact, normalized_pair):
    return _predict_difference(
        artifact.model,
        _difference(normalized_pair),
        torch.device(artifact.device),
        512,
    )


def save(artifact, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": artifact.model.state_dict(), "metadata": artifact.metadata},
        path / "model.pt",
    )
    shapelets = {
        f"length_{length}_group_{index}": parameter.detach().cpu().numpy()
        for index, (length, parameter) in enumerate(
            zip(artifact.model.transform.lengths, artifact.model.transform.shapelets)
        )
    }
    np.savez_compressed(path / "learned_shapelets.npz", **shapelets)


def explain(artifact, normalized_pair):
    device = torch.device(artifact.device)
    pair = torch.tensor(
        np.asarray(normalized_pair, dtype=np.float32),
        device=device,
        requires_grad=True,
    )
    difference = pair[:, 0, :] - pair[:, 1, :]
    artifact.model.eval()
    artifact.model.zero_grad(set_to_none=True)
    artifact.model(difference).sum().backward()
    return pair.grad.detach().abs().mean(dim=(0, 1)).cpu().numpy().astype(np.float64)


MODEL_SPEC = ModelSpec(
    name="difference_shapelet",
    candidates=candidates,
    fit=fit,
    predict=predict,
    save=save,
    explain=explain,
)
