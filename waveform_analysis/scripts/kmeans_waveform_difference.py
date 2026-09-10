#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from utils_fit import fit_ctr_ps
from waveform_analysis.ml_pipeline.dataset import load_prepared_dataset
from waveform_analysis.ml_pipeline.splits import semantic_seed
from waveform_analysis.ml_pipeline.view import calibrated_led, waveform_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster the exact normalized waveform-difference representation used by LinearSVR. "
            "K-means centroids and per-cluster LED corrections are learned from training events only, "
            "then frozen and evaluated on the blind/test split."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed study directory.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Dataset name from the study manifest. Repeat to analyse multiple datasets. Default: all.",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of K-means clusters. Default: 5.")
    parser.add_argument("--n-init", type=int, default=10, help="K-means initializations. Default: 10.")
    parser.add_argument("--max-iter", type=int, default=300, help="Maximum K-means iterations. Default: 300.")
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=None,
        help="Paired bootstrap repeats for CTR improvement. Default: study fit.bootstrap_samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <run-dir>/kmeans_waveform_difference/.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _difference(dataset, mode: str, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return exactly the feature matrix used by LinearSVR: normalized s1 - s2."""
    view = waveform_view(dataset, mode, np.asarray(indices, dtype=np.int64))
    pair = view.materialize(dtype=np.float32)
    if pair.ndim != 3 or pair.shape[1] != 2:
        raise ValueError(f"Expected [event, detector=2, sample], got {pair.shape}")
    return np.ascontiguousarray(pair[:, 0, :] - pair[:, 1, :], dtype=np.float32), np.asarray(view.time_ps, dtype=np.float64)


def _fit_ctr(values: np.ndarray, fit_config: dict[str, Any], *, seed: int, bootstrap: bool = True):
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    minimum = int(fit_config.get("min_events", 100))
    if finite.size < minimum:
        raise ValueError(f"Only {finite.size} finite events; need at least {minimum}")
    return fit_ctr_ps(finite, fit_config, seed=int(seed), bootstrap=bool(bootstrap))


def _paired_ctr_improvement(
    reference: np.ndarray,
    corrected: np.ndarray,
    fit_config: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    corrected = np.asarray(corrected, dtype=np.float64).reshape(-1)
    if reference.shape != corrected.shape:
        raise ValueError("Paired CTR comparison requires aligned arrays")
    finite = np.isfinite(reference) & np.isfinite(corrected)
    reference = reference[finite]
    corrected = corrected[finite]
    minimum = int(fit_config.get("min_events", 100))
    if reference.size < minimum:
        raise ValueError(f"Only {reference.size} common finite test events")

    raw_full = _fit_ctr(reference, fit_config, seed=seed, bootstrap=False).ctr_ps
    corrected_full = _fit_ctr(corrected, fit_config, seed=seed + 1, bootstrap=False).ctr_ps
    central_improvement = float(raw_full - corrected_full)
    central_relative = float(100.0 * central_improvement / raw_full)

    rng = np.random.default_rng(int(seed))
    improvement = []
    relative = []
    for _ in range(int(samples)):
        draw = rng.integers(0, reference.size, size=reference.size)
        try:
            raw_ctr = _fit_ctr(reference[draw], fit_config, seed=0, bootstrap=False).ctr_ps
            corrected_ctr = _fit_ctr(corrected[draw], fit_config, seed=0, bootstrap=False).ctr_ps
        except ValueError:
            continue
        if np.isfinite(raw_ctr) and raw_ctr > 0.0 and np.isfinite(corrected_ctr):
            delta = float(raw_ctr - corrected_ctr)
            improvement.append(delta)
            relative.append(float(100.0 * delta / raw_ctr))

    return {
        "raw_ctr_ps": float(raw_full),
        "corrected_ctr_ps": float(corrected_full),
        "ctr_improvement_ps": central_improvement,
        "relative_improvement_percent": central_relative,
        "paired_bootstrap_improvement_uncertainty_ps": (
            float(np.std(improvement, ddof=1)) if len(improvement) > 1 else float("nan")
        ),
        "paired_bootstrap_relative_uncertainty_percent": (
            float(np.std(relative, ddof=1)) if len(relative) > 1 else float("nan")
        ),
        "paired_bootstrap_successful": int(len(improvement)),
    }


def _cluster_ctr(values: np.ndarray, fit_config: dict[str, Any]) -> float:
    try:
        return float(_fit_ctr(values, fit_config, seed=0, bootstrap=False).ctr_ps)
    except ValueError:
        return float("nan")


def _cluster_rows(
    labels: np.ndarray,
    led: np.ndarray,
    corrections: np.ndarray,
    fit_config: dict[str, Any],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    rows = []
    for cluster in range(corrections.size):
        mask = labels == cluster
        values = np.asarray(led[mask], dtype=np.float64)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "cluster": int(cluster),
                "stage": stage,
                "n": int(values.size),
                "fraction": float(values.size / max(1, labels.size)),
                "train_cluster_correction_ps": float(corrections[cluster]),
                "led_mean_ps": float(np.mean(values)) if values.size else float("nan"),
                "led_median_ps": float(np.median(values)) if values.size else float("nan"),
                "led_std_ps": float(np.std(values, ddof=1)) if values.size > 1 else float("nan"),
                "led_ctr_ps": _cluster_ctr(values, fit_config),
                "mean_minus_train_correction_ps": (
                    float(np.mean(values) - corrections[cluster]) if values.size else float("nan")
                ),
            }
        )
    return rows


def _voltage_composition(labels: np.ndarray, voltage: np.ndarray, *, stage: str, k: int) -> list[dict[str, Any]]:
    voltage = np.asarray(voltage, dtype=np.float64)
    rows = []
    finite_values = np.unique(voltage[np.isfinite(voltage)])
    for cluster in range(int(k)):
        cluster_mask = labels == cluster
        cluster_n = int(np.count_nonzero(cluster_mask))
        for value in finite_values:
            n = int(np.count_nonzero(cluster_mask & np.isclose(voltage, value, rtol=0.0, atol=1e-9)))
            rows.append(
                {
                    "stage": stage,
                    "cluster": int(cluster),
                    "voltage_V": float(value),
                    "n": n,
                    "fraction_within_cluster": float(n / max(1, cluster_n)),
                }
            )
    return rows


def _plot_centroids(path: Path, time_ps: np.ndarray, centroids: np.ndarray, corrections: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    time_ns = np.asarray(time_ps, dtype=np.float64) / 1000.0
    for cluster, centroid in enumerate(np.asarray(centroids, dtype=np.float64)):
        ax.plot(time_ns, centroid, label=f"cluster {cluster} · correction {corrections[cluster]:+.1f} ps")
    ax.axvline(0.0, ls="--", lw=1.0, alpha=0.7)
    ax.set_xlabel("Time relative to LED/native anchor [ns]")
    ax.set_ylabel("Normalized waveform difference $s_1-s_2$")
    ax.set_title("Training K-means centroids in the exact LinearSVR input space")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_led_by_cluster(
    path: Path,
    labels: np.ndarray,
    led: np.ndarray,
    corrections: np.ndarray,
    *,
    stage: str,
) -> None:
    import matplotlib.pyplot as plt

    finite_all = np.asarray(led, dtype=np.float64)
    finite_all = finite_all[np.isfinite(finite_all)]
    if not finite_all.size:
        return
    low, high = np.quantile(finite_all, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(np.min(finite_all)), float(np.max(finite_all))
    margin = 0.06 * max(high - low, 1.0)
    edges = np.linspace(low - margin, high + margin, 31)

    k = int(corrections.size)
    columns = 2
    rows = int(np.ceil(k / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.5, 3.1 * rows), squeeze=False, sharex=True)
    for cluster, ax in enumerate(axes.flat):
        if cluster >= k:
            ax.axis("off")
            continue
        values = np.asarray(led[labels == cluster], dtype=np.float64)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=edges, histtype="stepfilled", alpha=0.5)
        ax.axvline(float(corrections[cluster]), ls="--", lw=1.4, label="train mean correction")
        test_mean = float(np.mean(values)) if values.size else float("nan")
        if np.isfinite(test_mean):
            ax.axvline(test_mean, ls=":", lw=1.4, label=f"{stage} mean")
        ax.set_title(f"cluster {cluster} · n={values.size}")
        ax.set_ylabel("Events / bin")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Calibrated LED residual [ps]")
    fig.suptitle(f"LED residual distributions by waveform-difference cluster · {stage}")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_before_after(path: Path, raw, corrected, raw_fit, corrected_fit) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    raw = np.asarray(raw, dtype=np.float64)
    corrected = np.asarray(corrected, dtype=np.float64)
    raw = raw[np.isfinite(raw)]
    corrected = corrected[np.isfinite(corrected)]
    pooled = np.concatenate([raw, corrected])
    low, high = np.quantile(pooled, [0.005, 0.995])
    margin = 0.06 * max(high - low, 1.0)
    edges = np.linspace(low - margin, high + margin, 31)
    ax.hist(raw, bins=edges, histtype="step", lw=1.5, label=f"LED · CTR {raw_fit.ctr_ps:.2f} ± {raw_fit.ctr_error_ps:.2f} ps")
    ax.hist(corrected, bins=edges, histtype="step", lw=1.5, label=f"cluster-corrected · CTR {corrected_fit.ctr_ps:.2f} ± {corrected_fit.ctr_error_ps:.2f} ps")
    ax.axvline(float(raw_fit.left_half_ps), ls="--", lw=1.0, alpha=0.7)
    ax.axvline(float(raw_fit.right_half_ps), ls="--", lw=1.0, alpha=0.7)
    ax.axvline(float(corrected_fit.left_half_ps), ls=":", lw=1.0, alpha=0.8)
    ax.axvline(float(corrected_fit.right_half_ps), ls=":", lw=1.0, alpha=0.8)
    ax.set_xlabel("Blind/test timing residual [ps]")
    ax.set_ylabel("Events / display bin")
    ax.set_title("Blind LED CTR before and after train-cluster mean correction")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_voltage_composition(path: Path, rows: list[dict[str, Any]], *, k: int) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    voltages = sorted({float(row["voltage_V"]) for row in rows})
    if not voltages:
        return
    x = np.arange(int(k), dtype=float)
    bottom = np.zeros(int(k), dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for voltage in voltages:
        values = np.asarray(
            [
                next(
                    (
                        float(row["fraction_within_cluster"])
                        for row in rows
                        if int(row["cluster"]) == cluster
                        and np.isclose(float(row["voltage_V"]), voltage, rtol=0.0, atol=1e-9)
                    ),
                    0.0,
                )
                for cluster in range(int(k))
            ],
            dtype=np.float64,
        )
        ax.bar(x, values, bottom=bottom, label=f"{voltage:g} V")
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([f"cluster {i}" for i in range(int(k))])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction within cluster")
    ax.set_title("Blind/test bias-voltage composition of waveform-difference clusters")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def analyse_dataset(
    run: Path,
    manifest: dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    entry = manifest["datasets"][dataset_name]
    dataset = load_prepared_dataset(entry["prepared_dir"])
    mode = str(manifest.get("mode") or manifest["config"]["mode"])
    config = manifest.get("config") or {}
    fit_config = dict(config.get("fit") or {})
    base_seed = int((config.get("validation") or {}).get("seed", 0))
    seed = semantic_seed(base_seed, dataset_name, mode, "kmeans_waveform_difference")

    training = np.asarray(dataset.training, dtype=np.int64)
    test = np.asarray(dataset.test, dtype=np.int64)
    if training.size < int(args.k):
        raise ValueError(f"{dataset_name}: training set has {training.size} events, fewer than k={args.k}")
    if test.size == 0:
        raise ValueError(f"{dataset_name}: blind/test split is empty")

    train_difference, time_ps = _difference(dataset, mode, training)
    test_difference, test_time_ps = _difference(dataset, mode, test)
    if not np.array_equal(time_ps, test_time_ps):
        raise ValueError("Training and test waveform grids differ")

    finite_train_waveform = np.all(np.isfinite(train_difference), axis=1)
    finite_test_waveform = np.all(np.isfinite(test_difference), axis=1)
    if np.count_nonzero(finite_train_waveform) < int(args.k):
        raise ValueError(f"{dataset_name}: fewer than k finite training waveforms")

    kmeans = KMeans(
        n_clusters=int(args.k),
        init="k-means++",
        n_init=int(args.n_init),
        max_iter=int(args.max_iter),
        random_state=int(seed),
        algorithm="lloyd",
    )
    kmeans.fit(train_difference[finite_train_waveform])

    train_labels = np.full(training.size, -1, dtype=np.int64)
    train_labels[finite_train_waveform] = kmeans.labels_.astype(np.int64, copy=False)
    test_labels = np.full(test.size, -1, dtype=np.int64)
    test_labels[finite_test_waveform] = kmeans.predict(test_difference[finite_test_waveform]).astype(np.int64, copy=False)

    led_all = calibrated_led(dataset, mode)
    train_led = np.asarray(led_all[training], dtype=np.float64)
    test_led = np.asarray(led_all[test], dtype=np.float64)

    corrections = np.full(int(args.k), np.nan, dtype=np.float64)
    for cluster in range(int(args.k)):
        mask = (train_labels == cluster) & np.isfinite(train_led)
        if not np.any(mask):
            raise RuntimeError(f"{dataset_name}: cluster {cluster} has no finite training LED residuals")
        corrections[cluster] = float(np.mean(train_led[mask]))

    common_test = (test_labels >= 0) & np.isfinite(test_led)
    raw_test = test_led[common_test]
    assigned_test = test_labels[common_test]
    corrected_test = raw_test - corrections[assigned_test]

    raw_fit = _fit_ctr(raw_test, fit_config, seed=semantic_seed(seed, "raw_test"), bootstrap=True)
    corrected_fit = _fit_ctr(
        corrected_test,
        fit_config,
        seed=semantic_seed(seed, "corrected_test"),
        bootstrap=True,
    )
    bootstrap_samples = int(
        args.bootstrap_samples
        if args.bootstrap_samples is not None
        else fit_config.get("bootstrap_samples", 100)
    )
    paired = _paired_ctr_improvement(
        raw_test,
        corrected_test,
        fit_config,
        samples=bootstrap_samples,
        seed=semantic_seed(seed, "paired_test"),
    )

    output = output_root / dataset_name
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "cluster_centroids.npy", np.asarray(kmeans.cluster_centers_, dtype=np.float32))
    np.save(output / "cluster_correction_ps.npy", corrections)
    np.save(output / "time_ps.npy", time_ps)

    train_rows = _cluster_rows(train_labels[train_labels >= 0], train_led[train_labels >= 0], corrections, fit_config, stage="train")
    test_rows = _cluster_rows(assigned_test, raw_test, corrections, fit_config, stage="test")
    _write_csv(output / "train_cluster_summary.csv", train_rows)
    _write_csv(output / "test_cluster_summary.csv", test_rows)

    train_event_rows = []
    train_valid_rows = np.flatnonzero(train_labels >= 0)
    train_distances = np.full(training.size, np.nan, dtype=np.float64)
    if train_valid_rows.size:
        centers = kmeans.cluster_centers_[train_labels[train_valid_rows]]
        train_distances[train_valid_rows] = np.linalg.norm(
            train_difference[train_valid_rows] - centers,
            axis=1,
        )
    for position in range(training.size):
        cluster = int(train_labels[position])
        train_event_rows.append(
            {
                "position": int(position),
                "prepared_index": int(training[position]),
                "event_index": int(dataset.event_index[training[position]]),
                "bias_voltage_V": float(dataset.bias_voltage_V[training[position]]),
                "cluster": cluster,
                "euclidean_distance_to_centroid": float(train_distances[position]),
                "led_residual_ps": float(train_led[position]),
                "cluster_correction_ps": float(corrections[cluster]) if cluster >= 0 else float("nan"),
                "corrected_residual_ps": (
                    float(train_led[position] - corrections[cluster])
                    if cluster >= 0 and np.isfinite(train_led[position])
                    else float("nan")
                ),
            }
        )
    _write_csv(output / "train_events.csv", train_event_rows)

    test_event_rows = []
    test_distances = np.full(test.size, np.nan, dtype=np.float64)
    test_valid_rows = np.flatnonzero(test_labels >= 0)
    if test_valid_rows.size:
        centers = kmeans.cluster_centers_[test_labels[test_valid_rows]]
        test_distances[test_valid_rows] = np.linalg.norm(
            test_difference[test_valid_rows] - centers,
            axis=1,
        )
    for position in range(test.size):
        cluster = int(test_labels[position])
        test_event_rows.append(
            {
                "position": int(position),
                "prepared_index": int(test[position]),
                "event_index": int(dataset.event_index[test[position]]),
                "bias_voltage_V": float(dataset.bias_voltage_V[test[position]]),
                "cluster": cluster,
                "euclidean_distance_to_centroid": float(test_distances[position]),
                "led_residual_ps": float(test_led[position]),
                "cluster_correction_ps": float(corrections[cluster]) if cluster >= 0 else float("nan"),
                "corrected_residual_ps": (
                    float(test_led[position] - corrections[cluster])
                    if cluster >= 0 and np.isfinite(test_led[position])
                    else float("nan")
                ),
            }
        )
    _write_csv(output / "test_events.csv", test_event_rows)

    train_voltage_rows = _voltage_composition(
        train_labels[train_labels >= 0],
        np.asarray(dataset.bias_voltage_V[training], dtype=np.float64)[train_labels >= 0],
        stage="train",
        k=int(args.k),
    )
    test_voltage_rows = _voltage_composition(
        assigned_test,
        np.asarray(dataset.bias_voltage_V[test], dtype=np.float64)[common_test],
        stage="test",
        k=int(args.k),
    )
    _write_csv(output / "cluster_voltage_composition.csv", train_voltage_rows + test_voltage_rows)

    _plot_centroids(output / "cluster_centroids.pdf", time_ps, kmeans.cluster_centers_, corrections)
    _plot_led_by_cluster(output / "train_led_by_cluster.pdf", train_labels, train_led, corrections, stage="train")
    _plot_led_by_cluster(output / "test_led_by_cluster.pdf", test_labels, test_led, corrections, stage="test")
    _plot_before_after(output / "test_ctr_before_after_cluster_correction.pdf", raw_test, corrected_test, raw_fit, corrected_fit)
    _plot_voltage_composition(output / "test_cluster_voltage_composition.pdf", test_voltage_rows, k=int(args.k))

    summary = {
        "dataset": dataset_name,
        "mode": mode,
        "prepared_dir": str(dataset.directory),
        "k": int(args.k),
        "clustering": {
            "library": "sklearn.cluster.KMeans",
            "metric": "Euclidean distance (standard K-means / Lloyd algorithm)",
            "init": "k-means++",
            "n_init": int(args.n_init),
            "max_iter": int(args.max_iter),
            "random_seed": int(seed),
            "fit_population": "training split only",
            "test_used_to_fit_centroids": False,
        },
        "input": {
            "definition": "normalized prepared waveform difference s1-s2",
            "equivalence": "exact feature representation passed to repository LinearSVR before its linear regression",
            "n_samples": int(train_difference.shape[1]),
            "time_start_ns": float(time_ps[0] / 1000.0),
            "time_end_ns": float(time_ps[-1] / 1000.0),
        },
        "correction": {
            "definition": "for each training K-means cluster c, correction_c = mean calibrated LED residual of training events assigned to c",
            "cluster_correction_ps": corrections.tolist(),
            "test_application": "assign blind event to nearest frozen training centroid, then LED_corrected = LED_residual - correction_cluster",
            "test_used_to_fit_corrections": False,
            "anchor_correction_used": False,
        },
        "population": {
            "training_total": int(training.size),
            "training_finite_waveforms": int(np.count_nonzero(finite_train_waveform)),
            "test_total": int(test.size),
            "test_finite_waveforms": int(np.count_nonzero(finite_test_waveform)),
            "test_common_finite_for_ctr": int(raw_test.size),
        },
        "kmeans": {
            "inertia": float(kmeans.inertia_),
            "n_iter": int(kmeans.n_iter_),
            "train_cluster_counts": [int(np.count_nonzero(train_labels == c)) for c in range(int(args.k))],
            "test_cluster_counts": [int(np.count_nonzero(test_labels == c)) for c in range(int(args.k))],
        },
        "blind_test_ctr": {
            "raw_led_ctr_ps": float(raw_fit.ctr_ps),
            "raw_led_ctr_uncertainty_ps": float(raw_fit.ctr_error_ps),
            "cluster_corrected_ctr_ps": float(corrected_fit.ctr_ps),
            "cluster_corrected_ctr_uncertainty_ps": float(corrected_fit.ctr_error_ps),
            **paired,
        },
        "train_cluster_summary": train_rows,
        "test_cluster_summary": test_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"{dataset_name} | k={args.k} | train-only KMeans on normalized s1-s2 | "
        f"train={training.size} | blind/test={test.size} | common finite test={raw_test.size}"
    )
    for row in train_rows:
        print(
            f"  cluster {row['cluster']} | train n={row['n']} | "
            f"mean LED correction={row['train_cluster_correction_ps']:+.3f} ps | "
            f"train within-cluster CTR={row['led_ctr_ps']:.3f} ps"
        )
    print(
        f"  blind LED CTR: {raw_fit.ctr_ps:.3f} ± {raw_fit.ctr_error_ps:.3f} ps\n"
        f"  blind cluster-corrected CTR: {corrected_fit.ctr_ps:.3f} ± {corrected_fit.ctr_error_ps:.3f} ps\n"
        f"  paired improvement: {paired['ctr_improvement_ps']:+.3f} ± "
        f"{paired['paired_bootstrap_improvement_uncertainty_ps']:.3f} ps "
        f"({paired['relative_improvement_percent']:+.2f} ± "
        f"{paired['paired_bootstrap_relative_uncertainty_percent']:.2f}%)"
    )
    print(f"  outputs: {output}")


def main() -> None:
    args = parse_args()
    if int(args.k) < 2:
        raise ValueError("--k must be >= 2")
    if int(args.n_init) < 1:
        raise ValueError("--n-init must be >= 1")
    if int(args.max_iter) < 1:
        raise ValueError("--max-iter must be >= 1")
    if args.bootstrap_samples is not None and int(args.bootstrap_samples) < 2:
        raise ValueError("--bootstrap-samples must be >= 2")

    run = args.run_dir.resolve()
    manifest = _read_json(run / "manifest.json")
    available = list((manifest.get("datasets") or {}).keys())
    if args.dataset:
        requested = set(args.dataset)
        datasets = [name for name in available if name in requested]
        missing = requested - set(datasets)
        if missing:
            raise FileNotFoundError(f"Requested study dataset(s) not found: {sorted(missing)}")
    else:
        datasets = available
    if not datasets:
        raise FileNotFoundError("Study manifest contains no datasets")

    output_root = (args.output_dir or run / "kmeans_waveform_difference").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print("Blind/test split is used only after K-means centroids and cluster corrections are frozen from training.")
    for dataset_name in datasets:
        analyse_dataset(run, manifest, dataset_name, args, output_root)


if __name__ == "__main__":
    main()
