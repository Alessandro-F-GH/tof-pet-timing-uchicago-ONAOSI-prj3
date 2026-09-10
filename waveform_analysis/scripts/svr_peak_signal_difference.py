#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from waveform_analysis.ml_pipeline.dataset import PreparedDataset, load_prepared_dataset
from waveform_analysis.ml_pipeline.view import inverse_pair, waveform_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Group events by peaks of the final model timing-residual distribution and compare "
            "the mean physical waveform difference s1(t)-s2(t) between groups. Designed to "
            "diagnose the multi-peak Linear-SVR residual distribution."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory.")
    parser.add_argument("--method", default="linear_svr", help="Residual artifact method. Default: linear_svr.")
    parser.add_argument(
        "--stage", choices=("train", "test"), default="test", help="Residual population. Default: test."
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Optional study dataset name. May be repeated. Default: all datasets with the requested artifact.",
    )
    parser.add_argument(
        "--hist-bin-width-ps",
        type=float,
        default=None,
        help="Histogram bin width used only for peak finding. Default: study fit.bin_width_ps.",
    )
    parser.add_argument(
        "--peak-prominence-fraction",
        type=float,
        default=0.08,
        help="Minimum smoothed-histogram peak prominence as a fraction of the maximum count. Default: 0.08.",
    )
    parser.add_argument(
        "--min-peak-distance-ps",
        type=float,
        default=8.0,
        help="Minimum separation between automatically detected peaks. Default: 8 ps.",
    )
    parser.add_argument(
        "--smooth-sigma-bins",
        type=float,
        default=1.0,
        help="Gaussian histogram smoothing used only for automatic peak detection. Default: 1 bin.",
    )
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=8,
        help="Keep at most this many automatically detected peaks, ranked by prominence. Default: 8.",
    )
    parser.add_argument(
        "--peak-centers-ps",
        type=float,
        nargs="+",
        default=None,
        help="Optional manual peak centers. When supplied, automatic peak detection is skipped.",
    )
    parser.add_argument(
        "--detection-quantiles",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(0.005, 0.995),
        help="Residual quantiles used for peak-detection range. Default: 0.005 0.995.",
    )
    parser.add_argument(
        "--max-distance-from-peak-ps",
        type=float,
        default=None,
        help="Optionally exclude events farther than this from their assigned peak. Default: keep all finite events.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Waveform batch size for mean/SEM accumulation. Default: 512.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/<method>_peak_signal_difference/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _fit_config(manifest: dict[str, Any]) -> dict[str, Any]:
    config = manifest.get("config") or {}
    fit = config.get("fit") or {}
    return dict(fit)


def _artifact(run: Path, dataset: str, method: str, stage: str) -> Path:
    return run / "artifacts" / dataset / f"{method}_{stage}_residuals_ps.npy"


def _stage_indices(run: Path, dataset: str, stage: str) -> np.ndarray:
    path = run / "splits" / f"{dataset}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing study split artifact: {path}")
    key = "training" if stage == "train" else "test"
    with np.load(path) as split:
        return np.asarray(split[key], dtype=np.int64)


def _median_centered_edges(values: np.ndarray, bin_width_ps: float, low: float, high: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    width = float(bin_width_ps)
    if width <= 0:
        raise ValueError("Histogram bin width must be positive")
    median = float(np.median(values))
    anchor = median - 0.5 * width
    steps_left = int(np.ceil((anchor - float(low)) / width))
    start = anchor - max(0, steps_left) * width
    if start > low:
        start -= width
    count = max(3, int(np.ceil((float(high) - start) / width)))
    edges = start + np.arange(count + 1, dtype=np.float64) * width
    if edges[-1] < high:
        edges = np.append(edges, edges[-1] + width)
    return edges


def _peak_indices_for_centers(bin_centers: np.ndarray, centers_ps: np.ndarray) -> np.ndarray:
    return np.asarray([int(np.argmin(np.abs(bin_centers - center))) for center in centers_ps], dtype=np.int64)


def _detect_groups(
    residual: np.ndarray,
    *,
    bin_width_ps: float,
    prominence_fraction: float,
    min_peak_distance_ps: float,
    smooth_sigma_bins: float,
    max_peaks: int,
    quantiles: tuple[float, float],
    manual_centers: list[float] | None,
    max_distance_ps: float | None,
) -> dict[str, Any]:
    values = np.asarray(residual, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size < 10:
        raise ValueError("Too few finite residuals for peak analysis")

    q_low, q_high = map(float, quantiles)
    if not 0.0 <= q_low < q_high <= 1.0:
        raise ValueError("detection quantiles must satisfy 0 <= LOW < HIGH <= 1")
    low, high = np.quantile(finite, [q_low, q_high])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Unable to determine a finite peak-detection residual range")

    edges = _median_centered_edges(finite, bin_width_ps, float(low), float(high))
    counts, edges = np.histogram(finite, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts.astype(np.float64), sigma=max(0.0, float(smooth_sigma_bins)))

    if manual_centers is not None:
        peak_centers = np.sort(np.asarray(manual_centers, dtype=np.float64))
        if peak_centers.size < 2:
            raise ValueError("Provide at least two --peak-centers-ps values for grouped analysis")
        peak_indices = _peak_indices_for_centers(centers, peak_centers)
        prominences = np.full(peak_centers.size, np.nan, dtype=np.float64)
    else:
        if not 0.0 < float(prominence_fraction) < 1.0:
            raise ValueError("peak-prominence-fraction must be in (0, 1)")
        distance_bins = max(1, int(np.ceil(float(min_peak_distance_ps) / float(bin_width_ps))))
        detected, properties = find_peaks(
            smooth,
            prominence=float(prominence_fraction) * float(np.max(smooth)),
            distance=distance_bins,
        )
        if detected.size < 2:
            raise RuntimeError(
                "Fewer than two residual peaks were detected. Lower --peak-prominence-fraction, "
                "change --hist-bin-width-ps, or supply --peak-centers-ps manually."
            )
        prominences_all = np.asarray(properties.get("prominences", np.zeros(detected.size)), dtype=np.float64)
        if detected.size > int(max_peaks):
            keep = np.argsort(prominences_all)[-int(max_peaks) :]
            detected = detected[keep]
            prominences_all = prominences_all[keep]
        order = np.argsort(centers[detected])
        peak_indices = np.asarray(detected[order], dtype=np.int64)
        peak_centers = np.asarray(centers[peak_indices], dtype=np.float64)
        prominences = np.asarray(prominences_all[order], dtype=np.float64)

    # Natural group boundaries are histogram valleys between adjacent residual peaks.
    boundaries = []
    for left_index, right_index in zip(peak_indices[:-1], peak_indices[1:]):
        lo, hi = sorted((int(left_index), int(right_index)))
        if hi <= lo + 1:
            boundaries.append(float(0.5 * (centers[lo] + centers[hi])))
            continue
        valley_local = int(np.argmin(smooth[lo : hi + 1]))
        valley_index = lo + valley_local
        boundaries.append(float(centers[valley_index]))
    boundaries = np.asarray(boundaries, dtype=np.float64)

    groups = np.full(values.size, -1, dtype=np.int64)
    finite_mask = np.isfinite(values)
    groups[finite_mask] = np.searchsorted(boundaries, values[finite_mask], side="right")
    if max_distance_ps is not None:
        if float(max_distance_ps) <= 0:
            raise ValueError("max-distance-from-peak-ps must be positive")
        positions = np.flatnonzero(groups >= 0)
        nearest_distance = np.abs(values[positions] - peak_centers[groups[positions]])
        groups[positions[nearest_distance > float(max_distance_ps)]] = -1

    return {
        "edges": edges,
        "counts": counts,
        "smooth": smooth,
        "bin_centers": centers,
        "peak_indices": peak_indices,
        "peak_centers_ps": peak_centers,
        "peak_prominences": prominences,
        "boundaries_ps": boundaries,
        "groups": groups,
        "detection_range_ps": (float(low), float(high)),
    }


def _physical_difference_statistics(
    dataset: PreparedDataset,
    mode: str,
    prepared_indices: np.ndarray,
    groups: np.ndarray,
    n_groups: int,
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if prepared_indices.size != groups.size:
        raise ValueError("Split indices and residual peak assignments are not aligned")
    first = waveform_view(dataset, mode, prepared_indices[:1])
    time_ps = np.asarray(first.time_ps, dtype=np.float64)
    samples = int(time_ps.size)
    sums = np.zeros((n_groups, samples), dtype=np.float64)
    sums_sq = np.zeros((n_groups, samples), dtype=np.float64)
    counts = np.zeros(n_groups, dtype=np.int64)

    for start in range(0, prepared_indices.size, int(batch_size)):
        stop = min(prepared_indices.size, start + int(batch_size))
        batch_groups = groups[start:stop]
        if not np.any(batch_groups >= 0):
            continue
        view = waveform_view(dataset, mode, prepared_indices[start:stop])
        physical = inverse_pair(dataset, mode, view.materialize())
        difference = np.asarray(physical[:, 0, :] - physical[:, 1, :], dtype=np.float64)
        for group in np.unique(batch_groups[batch_groups >= 0]):
            mask = batch_groups == int(group)
            selected = difference[mask]
            sums[group] += np.sum(selected, axis=0)
            sums_sq[group] += np.sum(selected * selected, axis=0)
            counts[group] += int(selected.shape[0])

    mean = np.full_like(sums, np.nan)
    sem = np.full_like(sums, np.nan)
    for group in range(n_groups):
        n = int(counts[group])
        if n == 0:
            continue
        mean[group] = sums[group] / n
        if n > 1:
            variance = np.maximum((sums_sq[group] - n * mean[group] ** 2) / (n - 1), 0.0)
            sem[group] = np.sqrt(variance / n)
    return time_ps, mean, sem, counts


def _source_metadata(dataset: PreparedDataset, prepared_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_dataset_path = dataset.directory / "source_dataset.npy"
    source_event_path = dataset.directory / "source_event_index.npy"
    if source_dataset_path.is_file():
        source_dataset = np.asarray(np.load(source_dataset_path, mmap_mode="r")[prepared_indices]).astype(str)
    else:
        source = Path(str(dataset.manifest.get("source", dataset.directory.name))).stem
        source_dataset = np.full(prepared_indices.size, source, dtype="U128")
    if source_event_path.is_file():
        source_event = np.asarray(np.load(source_event_path, mmap_mode="r")[prepared_indices], dtype=np.int64)
    else:
        source_event = np.asarray(dataset.event_index[prepared_indices], dtype=np.int64)
    return source_dataset, source_event


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        return
    fields = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_peak_assignment(path: Path, residual: np.ndarray, analysis: dict[str, Any], method: str, dataset_name: str) -> None:
    values = np.asarray(residual, dtype=np.float64)
    values = values[np.isfinite(values)]
    edges = np.asarray(analysis["edges"], dtype=np.float64)
    centers = np.asarray(analysis["bin_centers"], dtype=np.float64)
    counts = np.asarray(analysis["counts"], dtype=np.float64)
    smooth = np.asarray(analysis["smooth"], dtype=np.float64)
    peak_centers = np.asarray(analysis["peak_centers_ps"], dtype=np.float64)
    boundaries = np.asarray(analysis["boundaries_ps"], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.hist(values, bins=edges, alpha=0.28, label="Final residual")
    ax.plot(centers, smooth, lw=1.8, label="Smoothed counts for peak finding")
    for index, center in enumerate(peak_centers):
        ax.axvline(center, lw=1.5, label=f"Peak {index + 1}: {center:+.1f} ps")
    for boundary in boundaries:
        ax.axvline(boundary, color="black", ls="--", lw=1.0, alpha=0.65)
    ax.set_xlabel("Final timing residual [ps]")
    ax.set_ylabel("Events / detection bin")
    ax.set_title(f"{dataset_name} · {method} · residual peak groups")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_average_difference(
    path: Path,
    time_ps: np.ndarray,
    mean: np.ndarray,
    sem: np.ndarray,
    counts: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    peak_centers: np.ndarray,
    method: str,
    dataset_name: str,
) -> None:
    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    fig, ax = plt.subplots(figsize=(9.4, 5.5))
    for group in range(int(peak_centers.size)):
        n = int(counts[group])
        if n == 0:
            continue
        values = np.asarray(residual[groups == group], dtype=np.float64)
        label = (
            f"Peak {group + 1} ({peak_centers[group]:+.1f} ps) · n={n} · "
            f"residual {np.mean(values):+.1f}±{np.std(values):.1f} ps"
        )
        line = ax.plot(time_ns, mean[group], lw=1.8, label=label)[0]
        finite_sem = np.where(np.isfinite(sem[group]), sem[group], 0.0)
        ax.fill_between(
            time_ns,
            mean[group] - finite_sem,
            mean[group] + finite_sem,
            color=line.get_color(),
            alpha=0.14,
            linewidth=0,
        )
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.6)
    ax.axvline(0.0, color="black", ls="--", lw=0.9, alpha=0.6)
    ax.set_xlabel("Time relative to LED/native anchor [ns]")
    ax.set_ylabel(r"Mean signal difference $s_1-s_2$ [mV]")
    ax.set_title(f"{dataset_name} · {method} · mean waveform difference by final residual peak")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_voltage_composition(
    path: Path,
    voltage: np.ndarray,
    groups: np.ndarray,
    peak_centers: np.ndarray,
    dataset_name: str,
) -> bool:
    voltage = np.asarray(voltage, dtype=np.float64)
    finite_voltage = np.isfinite(voltage) & (groups >= 0)
    unique = np.unique(voltage[finite_voltage])
    if unique.size < 2:
        return False
    x = np.arange(peak_centers.size, dtype=float)
    bottom = np.zeros(peak_centers.size, dtype=float)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for value in unique:
        fractions = np.zeros(peak_centers.size, dtype=float)
        for group in range(peak_centers.size):
            mask = groups == group
            total = int(np.count_nonzero(mask))
            fractions[group] = 100.0 * np.count_nonzero(mask & np.isclose(voltage, value)) / total if total else 0.0
        ax.bar(x, fractions, bottom=bottom, label=f"{value:g} V")
        bottom += fractions
    ax.set_xticks(x)
    ax.set_xticklabels([f"Peak {i + 1}\n{center:+.1f} ps" for i, center in enumerate(peak_centers)])
    ax.set_ylabel("Events from bias voltage [%]")
    ax.set_ylim(0.0, 100.0)
    ax.set_title(f"{dataset_name} · voltage composition of residual peaks")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def analyse_dataset(
    run: Path,
    manifest: dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    artifact = _artifact(run, dataset_name, args.method, args.stage)
    residual = np.asarray(np.load(artifact), dtype=np.float64).reshape(-1)
    prepared_dir = Path(manifest["datasets"][dataset_name]["prepared_dir"])
    dataset = load_prepared_dataset(prepared_dir)
    indices = _stage_indices(run, dataset_name, args.stage)
    if residual.size != indices.size:
        raise ValueError(
            f"{dataset_name}: residual artifact has {residual.size} values but {args.stage} split has {indices.size} indices"
        )

    fit = _fit_config(manifest)
    bin_width = float(args.hist_bin_width_ps if args.hist_bin_width_ps is not None else fit.get("bin_width_ps", 5.0))
    analysis = _detect_groups(
        residual,
        bin_width_ps=bin_width,
        prominence_fraction=float(args.peak_prominence_fraction),
        min_peak_distance_ps=float(args.min_peak_distance_ps),
        smooth_sigma_bins=float(args.smooth_sigma_bins),
        max_peaks=int(args.max_peaks),
        quantiles=tuple(map(float, args.detection_quantiles)),
        manual_centers=args.peak_centers_ps,
        max_distance_ps=args.max_distance_from_peak_ps,
    )
    groups = np.asarray(analysis["groups"], dtype=np.int64)
    peak_centers = np.asarray(analysis["peak_centers_ps"], dtype=np.float64)

    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    time_ps, mean, sem, counts = _physical_difference_statistics(
        dataset,
        mode,
        indices,
        groups,
        int(peak_centers.size),
        batch_size=int(args.batch_size),
    )

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _plot_peak_assignment(dataset_dir / "residual_peak_groups.pdf", residual, analysis, args.method, dataset_name)
    _plot_average_difference(
        dataset_dir / "average_signal_difference_by_peak.pdf",
        time_ps,
        mean,
        sem,
        counts,
        residual,
        groups,
        peak_centers,
        args.method,
        dataset_name,
    )

    voltage = np.asarray(dataset.bias_voltage_V[indices], dtype=np.float64)
    _plot_voltage_composition(
        dataset_dir / "peak_voltage_composition.pdf",
        voltage,
        groups,
        peak_centers,
        dataset_name,
    )

    source_dataset, source_event = _source_metadata(dataset, indices)
    event_rows = []
    for position in range(residual.size):
        group = int(groups[position])
        event_rows.append(
            {
                "position": position,
                "prepared_index": int(indices[position]),
                "source_dataset": str(source_dataset[position]),
                "source_event_index": int(source_event[position]),
                "bias_voltage_V": float(voltage[position]),
                "residual_ps": float(residual[position]),
                "peak_group": "excluded" if group < 0 else f"peak_{group + 1}",
                "peak_center_ps": float("nan") if group < 0 else float(peak_centers[group]),
            }
        )
    _write_csv(dataset_dir / "event_peak_assignment.csv", event_rows)

    peak_rows = []
    composition_rows = []
    for group in range(peak_centers.size):
        mask = groups == group
        values = residual[mask]
        peak_rows.append(
            {
                "peak_group": f"peak_{group + 1}",
                "peak_center_ps": float(peak_centers[group]),
                "left_boundary_ps": float("-inf") if group == 0 else float(analysis["boundaries_ps"][group - 1]),
                "right_boundary_ps": float("inf") if group == peak_centers.size - 1 else float(analysis["boundaries_ps"][group]),
                "n": int(np.count_nonzero(mask)),
                "fraction": float(np.mean(mask)),
                "residual_mean_ps": float(np.mean(values)) if values.size else float("nan"),
                "residual_std_ps": float(np.std(values)) if values.size else float("nan"),
                "residual_median_ps": float(np.median(values)) if values.size else float("nan"),
                "detection_prominence": float(analysis["peak_prominences"][group]),
            }
        )
        for source in sorted(set(source_dataset[mask])):
            source_mask = mask & (source_dataset == source)
            composition_rows.append(
                {
                    "peak_group": f"peak_{group + 1}",
                    "peak_center_ps": float(peak_centers[group]),
                    "source_dataset": str(source),
                    "n": int(np.count_nonzero(source_mask)),
                    "fraction_within_peak": float(np.count_nonzero(source_mask) / max(1, np.count_nonzero(mask))),
                    "mean_voltage_V": float(np.mean(voltage[source_mask])) if np.any(source_mask) else float("nan"),
                }
            )
    _write_csv(dataset_dir / "peak_summary.csv", peak_rows)
    _write_csv(dataset_dir / "peak_source_composition.csv", composition_rows)

    waveform_rows = []
    for group in range(peak_centers.size):
        for sample, time in enumerate(time_ps):
            waveform_rows.append(
                {
                    "peak_group": f"peak_{group + 1}",
                    "peak_center_ps": float(peak_centers[group]),
                    "n": int(counts[group]),
                    "time_ns": float(time / 1000.0),
                    "mean_signal_difference_mV": float(mean[group, sample]),
                    "sem_signal_difference_mV": float(sem[group, sample]),
                }
            )
    _write_csv(dataset_dir / "average_signal_difference.csv", waveform_rows)

    summary = {
        "dataset": dataset_name,
        "method": args.method,
        "stage": args.stage,
        "prepared_dir": str(prepared_dir),
        "n_residuals": int(residual.size),
        "n_grouped": int(np.count_nonzero(groups >= 0)),
        "n_excluded": int(np.count_nonzero(groups < 0)),
        "hist_bin_width_ps": bin_width,
        "detection_quantiles": list(map(float, args.detection_quantiles)),
        "detection_range_ps": list(map(float, analysis["detection_range_ps"])),
        "peak_centers_ps": peak_centers.tolist(),
        "boundaries_ps": np.asarray(analysis["boundaries_ps"], dtype=float).tolist(),
        "group_counts": counts.tolist(),
        "assignment": "histogram valleys between adjacent residual peaks",
        "signal_quantity": "physical_mV_detector1_minus_detector2",
        "uncertainty_band": "SEM across events within each residual-peak group",
    }
    with (dataset_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=True)
        stream.write("\n")

    print(f"{dataset_name}: detected {peak_centers.size} peaks at {np.array2string(peak_centers, precision=1)} ps")
    for group, center in enumerate(peak_centers):
        print(f"  peak {group + 1}: center={center:+.1f} ps | n={int(counts[group])}")
    print(f"  outputs: {dataset_dir}")


def main() -> None:
    args = parse_args()
    if int(args.batch_size) < 1:
        raise ValueError("batch-size must be positive")
    if int(args.max_peaks) < 2:
        raise ValueError("max-peaks must be >= 2")

    run = args.run_dir.resolve()
    manifest = _read_json(run / "manifest.json")
    datasets = []
    for name in manifest.get("datasets", {}):
        if _artifact(run, name, args.method, args.stage).is_file():
            datasets.append(name)
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [name for name in datasets if name in wanted]
        missing = wanted - set(datasets)
        if missing:
            raise FileNotFoundError(
                f"Requested dataset(s) do not have {args.method}_{args.stage}_residuals_ps.npy: {sorted(missing)}"
            )
    if not datasets:
        raise FileNotFoundError(f"No study dataset has a {args.method}_{args.stage}_residuals_ps.npy artifact")

    output_root = (args.output_dir or run / f"{args.method}_peak_signal_difference").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for dataset_name in datasets:
        analyse_dataset(run, manifest, dataset_name, args, output_root)


if __name__ == "__main__":
    main()
