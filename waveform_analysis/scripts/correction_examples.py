"""Select real waveform examples at quantiles of saved model timing corrections.

This is a read-only, post-training analysis of the original model outputs.
No new feature transforms, model inference, fitting, clustering or synthetic data.

Run from the repository root:
    python -m waveform_analysis.scripts.correction_examples --results PATH
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.view import model_sample_mask, waveform_view
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time

DEFAULT_QUANTILES = (0.10, 0.30, 0.50, 0.70, 0.90)


def validate_quantiles(quantiles):
    """Keep order explicit and reject extreme/duplicate/invalid requests."""
    q = np.asarray(quantiles, dtype=np.float64)
    if q.ndim != 1 or not q.size or not np.all(np.isfinite(q)):
        raise ValueError("Quantiles must be a nonempty sequence of finite fractions")
    if np.any((q <= 0) | (q >= 1)) or np.any(np.diff(q) <= 0):
        raise ValueError("Quantiles must be strictly increasing and between 0 and 1")
    return q


def align_prediction_events(prepared_ids, prediction_ids, predictions):
    """Map each saved prediction to a prepared row, with strict ID verification.

    This deliberately does not assume the two artifacts have the same row order.
    """
    prepared_ids = np.asarray(prepared_ids, dtype=np.int64)
    prediction_ids = np.asarray(prediction_ids, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if (prepared_ids.ndim != 1 or prediction_ids.ndim != 1
            or predictions.shape != prediction_ids.shape):
        raise ValueError("Expected aligned one-dimensional event IDs and predictions")
    if not prediction_ids.size or len(prepared_ids) != len(prediction_ids):
        raise ValueError("Saved predictions do not cover the entire prepared event population")
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Saved predictions contain non-finite corrections")
    order = np.argsort(prepared_ids)
    if (np.any(np.diff(prepared_ids[order]) == 0)
            or len(np.unique(prediction_ids)) != len(prediction_ids)):
        raise ValueError("Event IDs must be unique in both saved artifacts")
    positions = np.searchsorted(prepared_ids[order], prediction_ids)
    if (np.any(positions >= len(order))
            or not np.array_equal(prepared_ids[order[np.minimum(positions, len(order)-1)]],
                                  prediction_ids)):
        raise ValueError("Saved prediction event IDs differ from prepared dataset IDs")
    return order[positions]


def select_examples(predictions, event_ids, quantiles=DEFAULT_QUANTILES):
    """Choose one distinct observed event nearest each prediction quantile.

    Ties are broken by the smallest event ID, independent of file row order.
    Never averages or generates waveforms.
    """
    q = validate_quantiles(quantiles)
    predictions = np.asarray(predictions, dtype=np.float64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if (predictions.ndim != 1 or predictions.shape != event_ids.shape
            or not predictions.size or not np.all(np.isfinite(predictions))):
        raise ValueError("Predictions and event IDs must be aligned finite 1-D arrays")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("Event IDs must be unique")
    target_values = np.quantile(predictions, q)
    chosen, rows = set(), []
    for fraction, target in zip(q, target_values):
        candidates = np.lexsort((event_ids, np.abs(predictions - target)))
        index = int(candidates[0])
        if index in chosen:
            raise ValueError(
                "Multiple quantiles select the same event; use fewer quantiles "
                "or a population with a wider prediction distribution"
            )
        chosen.add(index)
        rows.append({
            "quantile": float(fraction), "quantile_percent": float(100 * fraction),
            "quantile_correction_ps": float(target),
            "event_id": int(event_ids[index]),
            "prediction_row": index,
            "prediction_ps": float(predictions[index]),
            "quantile_difference_ps": float(predictions[index] - target),
        })
    return rows


def read_predictions(path, *, role, protocol_identity, final_fit_fingerprint=None):
    if not path.is_file():
        message = (f"Missing saved {role} predictions: {path}."
                   + (" Generate them with 'python -m waveform_analysis.cli diagnostics --results PATH'."
                      if role == "development" else " Complete the original model run first."))
        raise FileNotFoundError(message)
    with np.load(path, allow_pickle=False) as data:
        for key in ("event_id", "prediction_ps"):
            if key not in data:
                raise ValueError(f"{path} lacks required '{key}'")
        for key, expected in (("dataset_role", role),
                              ("analysis_protocol_identity", protocol_identity),
                              ("final_fit_fingerprint", final_fit_fingerprint)):
            if key in data and expected is not None and str(data[key].item()) != str(expected):
                raise ValueError(f"Stale or incompatible prediction file: {key} mismatch in {path}")
        return (np.asarray(data["event_id"], dtype=np.int64),
                np.asarray(data["prediction_ps"], dtype=np.float64))



def select_quantile_groups(predictions, event_ids, quantiles=DEFAULT_QUANTILES,
                           half_width=0.05):
    """Select all real events in equal-rank quantile bands.

    Deterministic ordering by (prediction, event ID) handles tied predictions
    without placing large tied populations unpredictably into one group.
    Percentile ranks (rank + .5)/N are independent of row order.
    """
    q = validate_quantiles(quantiles)
    predictions = np.asarray(predictions, dtype=np.float64)
    event_ids = np.asarray(event_ids, dtype=np.int64)
    if (predictions.ndim != 1 or predictions.shape != event_ids.shape
            or not len(predictions) or not np.all(np.isfinite(predictions))):
        raise ValueError("Predictions and event IDs must be aligned finite 1-D arrays")
    if len(np.unique(event_ids)) != len(event_ids):
        raise ValueError("Event IDs must be unique")
    if not np.isfinite(half_width) or not 0 < half_width < .5:
        raise ValueError("half_width must be finite and between 0 and 0.5")
    if np.any(q - half_width < 0) or np.any(q + half_width > 1):
        raise ValueError("Quantile bands extend beyond [0, 1]; reduce --half-width")

    ordered = np.lexsort((event_ids, predictions))
    ranks = (np.arange(len(predictions)) + .5) / len(predictions)
    groups = []
    for target in q:
        lower, upper = float(target - half_width), float(target + half_width)
        # Left inclusive, right exclusive; use the midrank of each observation.
        start = int(np.searchsorted(ranks, lower, side="left"))
        stop = int(np.searchsorted(ranks, upper, side="left"))
        selected = ordered[start:stop]
        if len(selected) < 2:
            raise ValueError(
                f"Q{target * 100:g}: only {len(selected)} events in percentile band "
                f"[{100*lower:.1f}, {100*upper:.1f}); "
                "increase --half-width or use a larger population"
            )
        values = predictions[selected]
        groups.append({
            "quantile": float(target),
            "quantile_percent": float(target * 100),
            "lower_percentile": float(lower * 100),
            "upper_percentile": float(upper * 100),
            "n_events": int(len(selected)),
            "prediction_mean_ps": float(np.mean(values)),
            "prediction_std_ps": float(np.std(values, ddof=1)),
            "prediction_min_ps": float(np.min(values)),
            "prediction_max_ps": float(np.max(values)),
            "prediction_indices": selected,
        })
    return groups


def summarize_group_waveforms(dataset, mode, prepared_rows, sample_mask, *,
                              chunk_size=1024):
    """Stable streaming sample mean and std, including paired difference.

    Std of detector1 - detector2 is computed from event-wise differences,
    retaining their covariance. The full event population is never materialized.
    """
    indices = np.asarray(prepared_rows, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 2 or chunk_size < 1:
        raise ValueError("At least two events and positive chunk_size are required")
    count = 0
    means = {}
    m2s = {}
    for start in range(0, len(indices), int(chunk_size)):
        view = waveform_view(dataset, mode, indices[start:start + chunk_size])
        waves = np.asarray(apply_sample_mask(view.materialize(), sample_mask),
                           dtype=np.float64)
        if waves.ndim != 3 or waves.shape[1] != 2 or not np.all(np.isfinite(waves)):
            raise ValueError("Prepared detector waveforms must be finite pairs")
        difference = waves[:, 0, :] - waves[:, 1, :]
        batches = {"pair": waves, "difference": difference}
        size = len(waves)
        for name, values in batches.items():
            batch_mean = np.mean(values, axis=0)
            batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
            if count == 0:
                means[name] = batch_mean
                m2s[name] = batch_m2
            else:
                delta = batch_mean - means[name]
                combined = count + size
                means[name] += delta * (size / combined)
                m2s[name] += batch_m2 + delta**2 * (count * size / combined)
        count += size
    return {
        "pair_mean": means["pair"],
        "pair_std": np.sqrt(np.maximum(m2s["pair"] / (count - 1), 0)),
        "difference_mean": means["difference"],
        "difference_std": np.sqrt(np.maximum(m2s["difference"] / (count - 1), 0)),
        "n_events": count,
    }



def compute_temporal_association(dataset, mode, prepared_rows, predictions_ps,
                                 sample_mask, *, chunk_size=1024):
    """Population Pearson r of event-wise D1-D2 and saved output at each time.

    Two-pass centering avoids cancellation in covariance/variance on small
    waveform differences. No new prediction or transformation is performed.
    Constant channels/time samples get NaN rather than misleading zero r.
    """
    indices = np.asarray(prepared_rows, dtype=np.int64)
    y = np.asarray(predictions_ps, dtype=np.float64)
    if indices.ndim != 1 or y.shape != indices.shape or len(y) < 2:
        raise ValueError("At least two aligned predictions and event indices required")
    if not np.all(np.isfinite(y)) or int(chunk_size) < 1:
        raise ValueError("Predictions must be finite and chunk_size positive")
    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    y_ss = float(np.dot(y_centered, y_centered))
    if y_ss <= 0:
        raise ValueError("Cannot correlate a constant model output")
    reference = summarize_group_waveforms(
        dataset, mode, indices, sample_mask, chunk_size=chunk_size
    )["difference_mean"]
    x_ss = np.zeros_like(reference)
    xy = np.zeros_like(reference)
    for start in range(0, len(indices), chunk_size):
        batch = waveform_view(dataset, mode, indices[start:start + chunk_size])
        waves = np.asarray(apply_sample_mask(batch.materialize(), sample_mask), np.float64)
        difference = waves[:, 0] - waves[:, 1] - reference
        x_ss += np.sum(difference * difference, axis=0)
        xy += difference.T @ y_centered[start:start + len(waves)]
    correlation = np.full(reference.shape, np.nan, dtype=np.float64)
    valid = x_ss > 0
    correlation[valid] = np.clip(xy[valid] / np.sqrt(x_ss[valid] * y_ss), -1, 1)
    return {"difference_correlation": correlation,
            "difference_mean": reference,
            "difference_std": np.sqrt(x_ss / (len(y) - 1)),
            "prediction_mean_ps": y_mean,
            "prediction_std_ps": np.sqrt(y_ss / (len(y) - 1)),
            "n_events": len(y)}


def plot_temporal_association(time_ps, association, output, *, zoom_max_ps=None):
    """Global Pearson correlation of channel difference with saved model output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.asarray(time_ps, np.float64)
    values = np.asarray(association["difference_correlation"], np.float64)
    if t.ndim != 1 or values.shape != t.shape:
        raise ValueError("Time grid and association must be aligned")
    keep = np.ones(t.shape, bool) if zoom_max_ps is None else t <= zoom_max_ps
    if np.count_nonzero(keep) < 2:
        raise ValueError("Zoom range contains fewer than two time samples")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(t[keep], values[keep], label="Pearson r(D1 - D2, prediction)", linewidth=1.7)
    ax.axhline(0, color="0.5", linestyle=":", linewidth=1)
    ax.set(xlabel="Time relative to waveform arrival [ps]",
           ylabel="Pearson correlation r",
           ylim=(-1, 1),
           title="Temporal association with original model correction")
    ax.grid(alpha=.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_group_statistics(summaries, time_ps, groups, output, *,
                          zoom_max_ps=None, global_difference_mean=None):
    """Original channel pair and population-centered D1-D2, mean ± 1 SD."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_ps = np.asarray(time_ps, dtype=np.float64)
    keep = np.ones(len(time_ps), dtype=bool) if zoom_max_ps is None else (
        time_ps <= float(zoom_max_ps)
    )
    if time_ps.ndim != 1 or np.count_nonzero(keep) < 2:
        raise ValueError("Zoom range contains fewer than two time samples")
    t = time_ps[keep]
    if global_difference_mean is None:
        baseline = np.zeros(len(time_ps), dtype=np.float64)
    else:
        baseline = np.asarray(global_difference_mean, dtype=np.float64)
        if baseline.shape != time_ps.shape or not np.all(np.isfinite(baseline)):
            raise ValueError("Population difference reference must match time grid")
    fig, axes = plt.subplots(
        len(groups), 2, figsize=(15, max(4, 2.65 * len(groups))),
        sharex="col", sharey="col", squeeze=False
    )
    for i, (row, summary) in enumerate(zip(groups, summaries)):
        left, right = axes[i]
        for detector in range(2):
            mean = summary["pair_mean"][detector, keep]
            std = summary["pair_std"][detector, keep]
            line, = left.plot(t, mean, linewidth=1.5,
                              label=f"Detector {detector + 1}")
            left.fill_between(t, mean - std, mean + std,
                              color=line.get_color(), alpha=.2, linewidth=0)
        diff_mean = summary["difference_mean"][keep] - baseline[keep]
        diff_std = summary["difference_std"][keep]
        line, = right.plot(t, diff_mean, linewidth=1.5, label="D1 - D2 minus population mean")
        right.fill_between(t, diff_mean - diff_std, diff_mean + diff_std,
                           color=line.get_color(), alpha=.23, linewidth=0)
        right.axhline(0, linewidth=.8, color="0.5", linestyle=":")
        label = (
            f"Q{row['quantile_percent']:g} "
            f"({row['lower_percentile']:g}–{row['upper_percentile']:g}%)"
            f" | N={summary['n_events']:,}"
            f" | correction {row['prediction_mean_ps']:+.2f} ± "
            f"{row['prediction_std_ps']:.2f} ps"
        )
        left.set_title(label, fontsize=10)
        right.set_title(label, fontsize=10)
        left.set_ylabel("Normalized amplitude")
        right.set_ylabel("Centered amplitude difference")
        left.grid(alpha=.2)
        right.grid(alpha=.2)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Time relative to waveform arrival [ps]")
    axes[-1, 1].set_xlabel("Time relative to waveform arrival [ps]")
    fig.suptitle(
        "Correction-conditioned mean ± 1 SD | channel pair and centered difference"
        + (f" | t ≤ {zoom_max_ps:g} ps" if zoom_max_ps is not None else ""),
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, .975))
    fig.savefig(output, dpi=180)
    plt.close(fig)

def plot_prediction_distribution(predictions, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(predictions, bins="auto", color="0.65", edgecolor="white",
            linewidth=.4, alpha=.8)
    for row in rows:
        q = row["quantile_percent"]
        x = row["prediction_ps"]
        ax.axvline(x, linestyle="--", linewidth=1.5, label=f"Q{q:g}: {x:.2f} ps")
    ax.set(xlabel="Original model correction [ps]",
           ylabel="Number of real events",
           title="Saved prediction distribution and selected examples")
    ax.legend(fontsize=9, ncol=min(3, len(rows)))
    ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_example_waveforms(waves, time_ps, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    waves, time_ps = np.asarray(waves), np.asarray(time_ps)
    if (waves.ndim != 3 or waves.shape[:2] != (len(rows), 2)
            or waves.shape[2] != len(time_ps)):
        raise ValueError("Expected one two-detector waveform pair per quantile")
    figure, axes = plt.subplots(len(rows), 1, figsize=(10, max(3, 2.25 * len(rows))),
                                sharex=True, sharey=True, squeeze=False)
    for i, row in enumerate(rows):
        ax = axes[i, 0]
        ax.plot(time_ps, waves[i, 0], linewidth=1.4, label="Detector 1")
        ax.plot(time_ps, waves[i, 1], linewidth=1.4, label="Detector 2")
        ax.set(ylabel="Norm. amplitude",
               title=(f"Q{row['quantile_percent']:g} | "
                      f"correction = {row['prediction_ps']:+.2f} ps | "
                      f"event {row['event_id']}"))
        ax.grid(alpha=.2)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("Time relative to waveform arrival [ps]")
    figure.suptitle("Real paired waveforms selected by model-output quantile")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def analyze_run(run, *, role="blind", cache_override=None,
                quantiles=DEFAULT_QUANTILES, half_width=0.05,
                zoom_max_ps=10000., logger=None):
    directory, manifest = Path(run["directory"]), run["manifest"]
    config = resolve_run_config(directory)
    if role not in ("blind", "development"):
        raise ValueError("role must be 'blind' or 'development'")
    q = validate_quantiles(quantiles)
    fingerprint = (manifest.get("stage_fingerprints") or {}).get("final_fit")
    cache_root = _cache_root(config, directory, cache_override, roles=(role,))
    prepared = PreparedCacheLookup().find(cache_root, role, config, manifest)
    predictions_path = directory / "artifacts" / (
        "pred.npz" if role == "blind" else "development_pred.npz"
    )
    prediction_ids, prediction_values = read_predictions(
        predictions_path, role=role,
        protocol_identity=prepared.manifest["analysis_protocol_identity"],
        final_fit_fingerprint=fingerprint,
    )
    prepared_rows = align_prediction_events(
        prepared.event_index, prediction_ids, prediction_values
    )
    if not np.isfinite(zoom_max_ps):
        raise ValueError("zoom_max_ps must be finite")
    selections = select_examples(prediction_values, prediction_ids, q)
    groups = select_quantile_groups(prediction_values, prediction_ids, q,
                                    half_width=half_width)
    positions = [int(prepared_rows[row["prediction_row"]]) for row in selections]
    view = waveform_view(prepared, config["mode"], positions)
    mask = model_sample_mask(prepared, config["mode"])
    waves = apply_sample_mask(view.materialize(), mask)
    time_ps = apply_sample_mask_to_time(view.time_ps, mask)
    for row, prepared_index in zip(selections, positions):
        row["prepared_row"] = int(prepared_index)
    if np.count_nonzero(time_ps <= zoom_max_ps) < 2:
        raise ValueError(
            f"--zoom-max-ps={zoom_max_ps:g} leaves fewer than two time samples"
        )

    # Reference uses the entire saved-prediction population, not only the
    # selected quantile bands. Streaming aggregation avoids a large allocation.
    population = summarize_group_waveforms(
        prepared, config["mode"], prepared_rows, mask
    )
    global_difference_mean = population["difference_mean"]

    association = compute_temporal_association(
        prepared, config["mode"], prepared_rows, prediction_values, mask
    )

    summaries = []
    for group in groups:
        members = prepared_rows[group["prediction_indices"]]
        summaries.append(
            summarize_group_waveforms(prepared, config["mode"], members, mask)
        )
        if logger:
            logger.info(
                "Correction examples | %s | Q%g | %d events | correction %.2f ± %.2f ps",
                manifest["model"], group["quantile_percent"], group["n_events"],
                group["prediction_mean_ps"], group["prediction_std_ps"],
            )

    identity = {
        "method": "saved_prediction_quantile_examples_v1",
        "model": manifest["model"], "role": role,
        "quantiles": q.tolist(), "half_width": float(half_width),
        "zoom_max_ps": float(zoom_max_ps),
        "analysis_version": 4,
        "difference_centering": "full_population_mean",
        "final_fit_fingerprint": fingerprint,
        "protocol_identity": prepared.manifest["analysis_protocol_identity"],
        "prediction_source": predictions_path.name,
    }
    destination = directory / "explanations" / "correction_examples" / canonical_hash(identity)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "examples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selections[0]))
        writer.writeheader()
        writer.writerows(selections)
    np.savez_compressed(destination / "examples.npz", quantiles=q,
                        event_id=np.asarray([row["event_id"] for row in selections], dtype=np.int64),
                        prediction_ps=np.asarray([row["prediction_ps"] for row in selections]),
                        quantile_correction_ps=np.asarray([row["quantile_correction_ps"] for row in selections]),
                        waveforms=np.asarray(waves, dtype=np.float32), time_ps=np.asarray(time_ps))
    group_rows = [
        {key: value for key, value in group.items() if key != "prediction_indices"}
        for group in groups
    ]
    with (destination / "groups.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(group_rows[0]))
        writer.writeheader()
        writer.writerows(group_rows)
    offsets = np.concatenate([[0], np.cumsum([g["n_events"] for g in groups])])
    np.savez_compressed(
        destination / "groups.npz",
        quantiles=q, lower_percentile=[g["lower_percentile"] for g in groups],
        upper_percentile=[g["upper_percentile"] for g in groups],
        n_events=[g["n_events"] for g in groups],
        prediction_mean_ps=[g["prediction_mean_ps"] for g in groups],
        prediction_std_ps=[g["prediction_std_ps"] for g in groups],
        member_offsets=offsets,
        member_event_ids=np.concatenate(
            [prediction_ids[g["prediction_indices"]] for g in groups]
        ),
        pair_mean=np.stack([s["pair_mean"] for s in summaries]),
        pair_std=np.stack([s["pair_std"] for s in summaries]),
        difference_mean=np.stack([s["difference_mean"] for s in summaries]),
        difference_centered_mean=np.stack([
            s["difference_mean"] - global_difference_mean for s in summaries
        ]),
        global_difference_mean=global_difference_mean,
        difference_std=np.stack([s["difference_std"] for s in summaries]),
        temporal_difference_correlation=association["difference_correlation"],
        temporal_difference_mean=association["difference_mean"],
        temporal_difference_std=association["difference_std"],
        temporal_prediction_mean_ps=association["prediction_mean_ps"],
        temporal_prediction_std_ps=association["prediction_std_ps"],
        time_ps=np.asarray(time_ps),
    )
    plot_prediction_distribution(prediction_values, selections, destination / "correction_distribution.png")
    plot_example_waveforms(waves, time_ps, selections, destination / "waveforms.png")
    plot_group_statistics(
        summaries, time_ps, groups, destination / "groups_full.png",
        global_difference_mean=global_difference_mean,
    )
    plot_group_statistics(
        summaries, time_ps, groups, destination / "groups_zoom.png",
        zoom_max_ps=zoom_max_ps,
        global_difference_mean=global_difference_mean,
    )
    plot_temporal_association(time_ps, association,
                              destination / "temporal_association.png")
    plot_temporal_association(time_ps, association,
                              destination / "temporal_association_zoom.png",
                              zoom_max_ps=zoom_max_ps)
    atomic_json(destination / "summary.json", {
        **identity, "run": str(directory), "n_events": int(len(prediction_ids)),
        "cache_root": str(cache_root), "preprocessed_waveforms": True,
        "model_inference": False, "feature_transform_applied": False,
        "selection": "closest observed prediction to each quantile; tie -> smallest event ID",
        "prototypes_represent": "model output quantiles, not guaranteed waveform morphology",
        "examples": selections, "groups": group_rows,
        "dispersion": "sample standard deviation (ddof=1) at each time sample",
        "channel_difference": "detector1 - detector2, with event-level covariance retained",
        "difference_centering": "Group mean D1-D2 minus whole-population mean D1-D2; within-group SD unchanged",
        "population_n_events": population["n_events"],
        "temporal_association": "Pearson r of event-wise D1-D2 with original model prediction, computed over entire selected role",
        "temporal_association_n_events": association["n_events"],
        "group_definition": "empirical prediction midrank percentile bands; tied predictions resolved by event ID",
    })
    if logger:
        logger.info("Correction examples | %s | %s | %s", manifest["model"], role, destination)
    return destination


def run(results, *, models=None, role="blind", cache_root=None,
        quantiles=DEFAULT_QUANTILES, half_width=0.05,
        zoom_max_ps=10000., logger=None):
    validate_quantiles(quantiles)
    if not np.isfinite(zoom_max_ps):
        raise ValueError("zoom_max_ps must be finite")
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No completed model runs found in {results}")
    present = {item["manifest"]["model"] for item in runs}
    if models is not None and set(models) - present:
        raise ValueError(f"Models not found in results: {sorted(set(models) - present)}")
    outputs, failures = [], []
    for item in runs:
        if models is not None and item["manifest"]["model"] not in models:
            continue
        try:
            outputs.append(analyze_run(
                item, role=role, cache_override=cache_root,
                quantiles=quantiles, half_width=half_width,
                zoom_max_ps=zoom_max_ps, logger=logger
            ))
        except Exception as exc:
            failures.append(f"{item['directory']}: {exc}")
            if logger:
                logger.exception("Correction examples failed: %s", item["directory"])
    if failures:
        raise RuntimeError("Some correction-example analyses failed:\n" + "\n".join(failures))
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Complete canonical batch root or model run")
    parser.add_argument("--models", nargs="+", metavar="MODEL", default=None)
    parser.add_argument("--role", choices=("blind", "development"), default="blind",
                        help="Prediction population used for examples (default: blind)")
    parser.add_argument("--cache-root", type=Path,
                        help="Original prepared cache root if it moved")
    parser.add_argument("--quantiles", nargs="+", type=float, default=DEFAULT_QUANTILES,
                        help="Strictly increasing fractions within (0,1); default .1 .3 .5 .7 .9")
    parser.add_argument("--half-width", type=float, default=0.05,
                        help="Half-width of correction quantile bands, default .05")
    parser.add_argument("--zoom-max-ps", type=float, default=10000.,
                        help="Show samples up to this time in grouped zoom plot (default 10000 ps)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    for output in run(args.results, models=args.models, role=args.role,
                      cache_root=args.cache_root, quantiles=args.quantiles,
                      half_width=args.half_width, zoom_max_ps=args.zoom_max_ps,
                      logger=logging.getLogger(__name__)):
        print(output)


if __name__ == "__main__":
    main()
