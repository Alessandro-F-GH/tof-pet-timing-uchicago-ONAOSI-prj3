from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
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


def _fixed_starts(input_length: int, length: int, count: int) -> np.ndarray:
    maximum = int(input_length) - int(length)
    if maximum < 0:
        raise ValueError(f"Shapelet length {length} exceeds input length {input_length}")
    if count < 1:
        raise ValueError("Every shapelet group must contain at least one shapelet")
    if count > maximum + 1:
        raise ValueError(
            f"Cannot place {count} distinct fixed shapelets of length {length} in {input_length} samples"
        )
    if count == 1:
        return np.asarray([maximum // 2], dtype=np.int64)
    starts = np.rint(np.linspace(0, maximum, count)).astype(np.int64)
    if np.unique(starts).size != starts.size:
        raise ValueError(
            f"Fixed shapelet positions are not distinct for length={length}, count={count}, input={input_length}"
        )
    return starts


def _shapelet_layout(architecture: dict[str, Any], input_length: int):
    lengths = [int(v) for v in architecture.get("shapelet_lengths", [17, 33, 65])]
    raw = architecture.get("shapelets_per_length", [8] * len(lengths))
    counts = [int(raw)] * len(lengths) if isinstance(raw, (int, float)) else [int(v) for v in raw]
    if not lengths or len(lengths) != len(counts):
        raise ValueError("shapelet_lengths and shapelets_per_length must be non-empty and have equal length")
    if any(v < 2 or v > input_length for v in lengths):
        raise ValueError(f"Shapelet lengths must lie in [2, {input_length}], got {lengths}")
    starts = [_fixed_starts(input_length, length, count) for length, count in zip(lengths, counts)]
    return lengths, counts, starts


def _initialize_shapelets(x, lengths, starts, *, seed):
    rng = np.random.default_rng(int(seed))
    n_events = int(x.shape[0])
    groups = []
    for length, group_starts in zip(lengths, starts):
        templates = np.empty((len(group_starts), length), dtype=np.float32)
        for i, start in enumerate(group_starts):
            event = int(rng.integers(0, n_events))
            templates[i] = x[event, int(start) : int(start) + int(length)]
        groups.append(templates)
    return groups


class FixedShapeletTransform(nn.Module):
    """Compare learned templates only with their fixed synchronized-time supports."""

    def __init__(self, initial_shapelets, starts):
        super().__init__()
        if len(initial_shapelets) != len(starts):
            raise ValueError("Shapelet groups and fixed-position groups must have equal length")
        self.shapelets = nn.ParameterList(
            [nn.Parameter(torch.from_numpy(np.asarray(group, dtype=np.float32))) for group in initial_shapelets]
        )
        self.starts = [np.asarray(group, dtype=np.int64) for group in starts]
        self.lengths = [int(group.shape[-1]) for group in initial_shapelets]
        self.counts = [int(group.shape[0]) for group in initial_shapelets]

    @property
    def n_features(self):
        return int(sum(self.counts))

    def forward(self, difference):
        if difference.ndim != 2:
            raise ValueError(f"Shapelet transform expects [batch, time], got {tuple(difference.shape)}")
        features = []
        for shapelets, starts, length in zip(self.shapelets, self.starts, self.lengths):
            windows = torch.stack(
                [difference[:, int(start) : int(start) + int(length)] for start in starts],
                dim=1,
            )
            features.append((windows - shapelets[None, :, :]).square().mean(dim=2))
        return torch.cat(features, dim=1)


class DifferenceShapeletRegressor(nn.Module):
    def __init__(self, initial_shapelets, starts, *, dense_units, dropout):
        super().__init__()
        self.transform = FixedShapeletTransform(initial_shapelets, starts)
        incoming = self.transform.n_features
        head = [nn.LayerNorm(incoming)]
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
    input_time_ps: np.ndarray
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
    lengths, counts, starts = _shapelet_layout(architecture, train_difference.shape[1])
    dense_units = [int(v) for v in architecture.get("dense_units", [32, 16])]
    dropout = float(architecture.get("dropout", 0.05))
    if not 0.0 <= dropout < 1.0:
        raise ValueError("architecture.dropout must be in [0, 1)")

    initial_shapelets = _initialize_shapelets(train_difference, lengths, starts, seed=seed)
    device = _device(config)
    model = DifferenceShapeletRegressor(
        initial_shapelets,
        starts,
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
    if input_time_ps.size not in {0, train_difference.shape[1]}:
        raise ValueError(
            f"input_time_ps has {input_time_ps.size} samples but waveform has {train_difference.shape[1]}"
        )
    if input_time_ps.size:
        group_start_times_ps = [[float(input_time_ps[int(start)]) for start in group] for group in starts]
        group_end_times_ps = [
            [float(input_time_ps[int(start) + int(length) - 1]) for start in group]
            for length, group in zip(lengths, starts)
        ]
        group_center_times_ps = [
            [0.5 * (left + right) for left, right in zip(lefts, rights)]
            for lefts, rights in zip(group_start_times_ps, group_end_times_ps)
        ]
        dt_ps = float(np.median(np.diff(input_time_ps))) if input_time_ps.size > 1 else float("nan")
    else:
        group_start_times_ps = [[] for _ in lengths]
        group_end_times_ps = [[] for _ in lengths]
        group_center_times_ps = [[] for _ in lengths]
        dt_ps = float("nan")

    metadata = {
        "best_epoch": int(best_epoch),
        "best_validation_rmse_ps": float(best_score),
        "selection_metric": "validation_rmse",
        "batch_size": batch_size,
        "output_max_abs_ps": None if output_limit is None else float(output_limit),
        "input_definition": "normalized synchronized-window difference d(t)=s1(t)-s2(t)",
        "representation": "position-locked learnable shapelets: MSE between each learned template and its fixed time support",
        "prediction_definition": "nonlinear regression from fixed-position learned shapelet distances of s1-s2 [ps]",
        "shapelet_lengths_samples": lengths,
        "shapelets_per_length": counts,
        "shapelet_starts_samples": [group.tolist() for group in starts],
        "shapelet_start_times_ps": group_start_times_ps,
        "shapelet_end_times_ps": group_end_times_ps,
        "shapelet_center_times_ps": group_center_times_ps,
        "input_sample_interval_ps": dt_ps,
        "n_shapelet_features": int(sum(counts)),
        "shapelet_position_policy": "fixed supports evenly distributed across the synchronized input window; no sliding or minimum-over-position",
        "shapelet_initialization": "random training-event subsequence at the same fixed support",
    }
    return DifferenceShapeletArtifact(
        model=model,
        input_time_ps=input_time_ps,
        device=str(device),
        metadata=metadata,
    )


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
    arrays: dict[str, np.ndarray] = {
        "input_time_ps": np.asarray(artifact.input_time_ps, dtype=np.float64),
    }
    for group_index, (length, starts, parameter) in enumerate(
        zip(
            artifact.model.transform.lengths,
            artifact.model.transform.starts,
            artifact.model.transform.shapelets,
        )
    ):
        arrays[f"group_{group_index}_shapelets"] = parameter.detach().cpu().numpy().astype(np.float32)
        arrays[f"group_{group_index}_starts_samples"] = np.asarray(starts, dtype=np.int64)
        arrays[f"group_{group_index}_length_samples"] = np.asarray([length], dtype=np.int64)
    np.savez_compressed(path / "learned_shapelets.npz", **arrays)


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
