"""Held-out reconstruction RMSE figures for globally selected imputers."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

TASKS = ("remove_window", "keep_only")
METHODS = ("reference", "median", "linear", "ppca_joint", "conditional_knn")
NAMES = {
    "reference": "Control reference",
    "median": "Development median",
    "linear": "Linear interpolation",
    "ppca_joint": "Joint PPCA",
    "conditional_knn": "Conditional kNN",
}


def method_label(method, setting):
    if method == "ppca_joint":
        return f"Joint PPCA (components={setting})"
    if method == "conditional_knn":
        k, weight = setting.split(":")
        return f"Conditional kNN (k={k}, {weight})"
    return NAMES[method]


def position_rows(best_variants, tuning_positions, evaluation_positions,
                  starts, time_ps, width):
    """Select a SINGLE tuned setting per method/task for every plotted position."""
    time_ps = np.asarray(time_ps, dtype=float)
    starts = np.asarray(starts, dtype=int)
    if time_ps.ndim != 1 or len(time_ps) < 2:
        raise ValueError("Missing time grid")
    dt = float(np.median(np.diff(time_ps)))
    if dt <= 0 or not np.allclose(np.diff(time_ps), dt, rtol=1e-5, atol=1e-3):
        raise ValueError("Nonuniform waveform time grid")
    result = []
    for task in TASKS:
        for method in METHODS:
            if method not in best_variants[task]:
                continue
            setting = best_variants[task][method]["setting"]
            for j, start in enumerate(starts):
                key = (task, method, setting, j)
                result.append({
                    "task": task, "method": method, "setting": setting,
                    "window_index": j, "start_sample": int(start),
                    "start_ns": float(time_ps[start] / 1000.),
                    "end_ns": float((time_ps[start] + width * dt) / 1000.),
                    "tuning_rmse_mv": float(tuning_positions[key]),
                    "evaluation_rmse_mv": float(evaluation_positions[key]),
                })
    return result


def save_diagnostics(directory, positions, method_rows, selected, board, channel):
    """Write machine-readable position profiles and the requested PNG/PDF plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not positions or not method_rows:
        raise ValueError("No imputer diagnostics available")
    with (directory / "position_rmse.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(positions[0]))
        w.writeheader()
        w.writerows(positions)
    # Preserve the exact per-position numbers, without rounding to CSV/plots.
    labels = sorted({(r["task"], r["method"], r["setting"]) for r in positions})
    windows = sorted(set(r["window_index"] for r in positions))
    matrix = lambda field: np.asarray([
        [next(r[field] for r in positions
              if (r["task"], r["method"], r["setting"]) == key
              and r["window_index"] == j) for j in windows]
        for key in labels
    ], dtype=np.float64)
    np.savez_compressed(
        directory / "position_rmse.npz",
        tasks=np.array([x[0] for x in labels]),
        methods=np.array([x[1] for x in labels]),
        settings=np.array([x[2] for x in labels]),
        window_index=np.asarray(windows),
        start_ns=np.array([next(r["start_ns"] for r in positions
                                if r["window_index"] == j) for j in windows]),
        tuning_rmse_mv=matrix("tuning_rmse_mv"),
        evaluation_rmse_mv=matrix("evaluation_rmse_mv"),
    )
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for method in METHODS:
            subset = sorted((r for r in positions if r["task"] == task and
                             r["method"] == method), key=lambda r: r["window_index"])
            if not subset:
                continue
            label = method_label(method, subset[0]["setting"])
            if (selected[task]["method"], selected[task]["setting"]) == (
                method, subset[0]["setting"]
            ):
                label += " [selected]"
            ax.plot([r["start_ns"] for r in subset],
                    [r["evaluation_rmse_mv"] for r in subset], label=label)
        ax.set(xlabel="Window start (ns)", ylabel="Hidden-region RMSE (mV)",
               title=f"{board} {channel} — {task} (independent evaluation)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        for ext in ("png", "pdf"):
            fig.savefig(directory / f"rmse_vs_position__{task}.{ext}", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, task in zip(axes, TASKS):
        rows = sorted((r for r in method_rows if r["task"] == task),
                      key=lambda r: r["evaluation_rmse_mv"])
        y = np.arange(len(rows))
        values = np.asarray([r["evaluation_rmse_mv"] for r in rows])
        ax.barh(y, values)
        ax.set_yticks(y, labels=[method_label(r["method"], r["setting"]) for r in rows])
        ax.invert_yaxis()
        ax.set(xlabel="Pooled hidden-region RMSE (mV)", title=task)
        ax.set_xlim(0, 1.3 * max(float(values.max()), 1e-12))
        ax.grid(axis="x", alpha=.2)
        for j, row in enumerate(rows):
            chosen = (selected[task]["method"], selected[task]["setting"]) == (
                row["method"], row["setting"]
            )
            label = f'{values[j]:.3g}' + (" [selected]" if chosen else "")
            ax.text(values[j] + .015 * ax.get_xlim()[1], j, label,
                    va="center", fontsize=8)
    fig.suptitle(f"{board} {channel} — best globally tuned settings per method")
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"best_methods_summary.{ext}", dpi=180)
    plt.close(fig)


DIAGNOSTIC_FILES = (
    "position_rmse.csv", "position_rmse.npz",
    "rmse_vs_position__remove_window.png",
    "rmse_vs_position__keep_only.png",
    "rmse_vs_position__remove_window.pdf",
    "rmse_vs_position__keep_only.pdf",
    "best_methods_summary.png", "best_methods_summary.pdf",
)


def diagnostics_complete(folder):
    return all((Path(folder) / name).is_file() for name in DIAGNOSTIC_FILES)
