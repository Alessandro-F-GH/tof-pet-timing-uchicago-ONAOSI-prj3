from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def plot_fixed_shapelets(run: Path, output: Path, dataset: str, paths: list[Path]) -> None:
    """Plot learned fixed-position shapelets on the synchronized input time axis."""
    import matplotlib.pyplot as plt

    model_dir = Path(run) / "models" / dataset / "difference_shapelet"
    archive = model_dir / "learned_shapelets.npz"
    metadata_path = model_dir / "metadata.json"
    if not archive.is_file() or not metadata_path.is_file():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = dict(metadata.get("training") or {})
    with np.load(archive) as data:
        time_ps = np.asarray(data.get("input_time_ps", []), dtype=np.float64)
        groups = []
        group_index = 0
        while f"group_{group_index}_shapelets" in data:
            shapelets = np.asarray(data[f"group_{group_index}_shapelets"], dtype=np.float64)
            starts = np.asarray(data[f"group_{group_index}_starts_samples"], dtype=np.int64)
            length = int(np.asarray(data[f"group_{group_index}_length_samples"]).reshape(-1)[0])
            groups.append((length, starts, shapelets))
            group_index += 1

    if not groups:
        return
    if time_ps.size == 0:
        maximum = max(int(start) + int(length) for length, starts, _ in groups for start in starts)
        time_ps = np.arange(maximum, dtype=np.float64)
        xlabel = "Prepared input sample"
        time_scale = 1.0
    else:
        xlabel = "Time relative to interpolated LED crossing [ns]"
        time_scale = 1.0 / 1000.0

    fig, axes = plt.subplots(len(groups), 1, figsize=(9.0, 2.8 * len(groups)), sharex=True, squeeze=False)
    for group_index, (ax, (length, starts, shapelets)) in enumerate(zip(axes[:, 0], groups)):
        for shapelet_index, (start, shapelet) in enumerate(zip(starts, shapelets), 1):
            stop = int(start) + int(length)
            if stop > time_ps.size:
                continue
            x = time_ps[int(start):stop] * time_scale
            ax.plot(x, shapelet, lw=1.35, label=f"#{shapelet_index}")
            ax.axvspan(float(x[0]), float(x[-1]), alpha=0.035)
        ax.set_ylabel("Normalized\ndifference")
        ax.set_title(f"Fixed shapelets · length {length} samples")
        ax.grid(alpha=0.2)
        if len(starts) <= 10:
            ax.legend(ncol=min(4, len(starts)), fontsize=8, loc="best")

    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(
        f"{dataset} · difference_shapelet · learned position-locked templates\n"
        "Each template is compared only with its own synchronized-time support"
    )
    fig.tight_layout()
    target = output / f"shapelets_{dataset}_difference_shapelet.pdf"
    fig.savefig(target, bbox_inches="tight")
    plt.close(fig)
    paths.append(target)

    rows = []
    start_times = training.get("shapelet_start_times_ps") or []
    end_times = training.get("shapelet_end_times_ps") or []
    center_times = training.get("shapelet_center_times_ps") or []
    for group_index, (length, starts, shapelets) in enumerate(groups):
        for shapelet_index, start in enumerate(starts):
            row = {
                "group": group_index,
                "shapelet": shapelet_index,
                "length_samples": int(length),
                "start_sample": int(start),
                "end_sample": int(start) + int(length) - 1,
            }
            if group_index < len(start_times) and shapelet_index < len(start_times[group_index]):
                row["start_time_ns"] = float(start_times[group_index][shapelet_index]) / 1000.0
            if group_index < len(end_times) and shapelet_index < len(end_times[group_index]):
                row["end_time_ns"] = float(end_times[group_index][shapelet_index]) / 1000.0
            if group_index < len(center_times) and shapelet_index < len(center_times[group_index]):
                row["center_time_ns"] = float(center_times[group_index][shapelet_index]) / 1000.0
            row["template_rms"] = float(np.sqrt(np.mean(np.asarray(shapelets[shapelet_index]) ** 2)))
            rows.append(row)

    if rows:
        import csv

        csv_target = output / f"shapelets_{dataset}_difference_shapelet.csv"
        with csv_target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(csv_target)
