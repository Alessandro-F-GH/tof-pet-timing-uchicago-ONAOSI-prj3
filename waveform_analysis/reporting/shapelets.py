"""Portable coefficient tables and ranked shapelet figures."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_shapelets(run, arrays, cfg):
    from waveform_analysis.reporting.plotting import output_path, plot_context

    run = Path(run)
    table = run / "tables" / "shapelet_importance.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    fields = ["rank", "shapelet_id", "importance_ps", "length", "dilation",
              "threshold", "normalized"]
    names = [str(v) for v in arrays["feature_names"]]
    fields += [f"{name}_{suffix}" for suffix in ("coefficient_scaled_ps", "coefficient_raw", "control_scale") for name in names]
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in arrays["ranking"]:
            row = dict(rank=int(arrays["rank"][i]), shapelet_id=int(i),
                       importance_ps=float(arrays["shapelet_importance_ps"][i]),
                       length=int(arrays["lengths"][i]), dilation=int(arrays["dilations"][i]),
                       threshold=float(arrays["thresholds"][i]), normalized=bool(arrays["normalizes"][i]))
            for suffix, key in (("coefficient_scaled_ps", "coefficients_scaled_ps"),
                                ("coefficient_raw", "coefficients_raw"), ("control_scale", "control_feature_scale")):
                row.update({f"{name}_{suffix}": float(arrays[key][i, j]) for j, name in enumerate(names)})
            writer.writerow(row)
    # One ranked page per 12 shapelets: every saved shapelet remains viewable.
    directory = run / "plots" / "shapelets"
    directory.mkdir(parents=True, exist_ok=True)
    time = np.asarray(arrays["time_ps"], float)
    output = None
    generated = set()
    colors = cfg["xai"].get("detector_colors", cfg["palette"])
    with plot_context(cfg):
        for start in range(0, len(arrays["ranking"]), 12):
            selected = arrays["ranking"][start:start + 12]
            fig, axes = plt.subplots(len(selected), 1, figsize=(8, 2.1 * len(selected)), squeeze=False)
            for axis, i in zip(axes[:, 0], selected):
                length, dilation = int(arrays["lengths"][i]), int(arrays["dilations"][i])
                origin = int(arrays["startpoints"][i])
                positions = origin + dilation * np.arange(length)
                if positions[-1] >= len(time):
                    raise ValueError("Shapelet support exceeds saved input time grid")
                x = (time[positions] - time[origin]) / 1000.0
                values = arrays["values"][i, :, :length]
                shared = str(arrays["estimator_formulation"].item()) == "shared"
                for channel, wave in enumerate(values):
                    axis.plot(x, wave, marker=".", color=colors[channel % len(colors)], label="Shared detector template" if shared else f"Detector {channel + 1}")
                normalized = bool(arrays["normalizes"][i])
                axis.set_ylabel("z score" if normalized else "Normalized amplitude")
                beta = arrays["coefficients_scaled_ps"][i]
                axis.set_title(f"Rank {int(arrays['rank'][i])} · shapelet {i} · score {arrays['shapelet_importance_ps'][i]:.3g} ps\n"
                               f"length={length}, dilation={dilation}, threshold={arrays['thresholds'][i]:.3g}; β(distance, location, count)={np.array2string(beta, precision=2)}", fontsize=9)
                axis.legend(fontsize=8, frameon=False)
                axis.set_xlabel("Time from source start (ns)")
            fig.tight_layout()
            output = output_path(directory, f"ranked_{start // 12 + 1:03d}", cfg)
            fig.savefig(output, dpi=cfg["output"]["dpi"])
            plt.close(fig)
            generated.add(output)
    # Remove obsolete pages when fewer shapelets are exported on regeneration.
    for old in directory.glob(f"ranked_*.{str(cfg['output'].get('format', 'png')).lstrip('.')}"):
        if old not in generated:
            old.unlink()
    return output
