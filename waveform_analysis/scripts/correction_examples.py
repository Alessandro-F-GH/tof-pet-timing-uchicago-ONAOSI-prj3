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
                quantiles=DEFAULT_QUANTILES, logger=None):
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
    selections = select_examples(prediction_values, prediction_ids, q)
    positions = [int(prepared_rows[row["prediction_row"]]) for row in selections]
    view = waveform_view(prepared, config["mode"], positions)
    mask = model_sample_mask(prepared, config["mode"])
    waves = apply_sample_mask(view.materialize(), mask)
    time_ps = apply_sample_mask_to_time(view.time_ps, mask)
    for row, prepared_index in zip(selections, positions):
        row["prepared_row"] = int(prepared_index)

    identity = {
        "method": "saved_prediction_quantile_examples_v1",
        "model": manifest["model"], "role": role,
        "quantiles": q.tolist(),
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
    plot_prediction_distribution(prediction_values, selections, destination / "correction_distribution.png")
    plot_example_waveforms(waves, time_ps, selections, destination / "waveforms.png")
    atomic_json(destination / "summary.json", {
        **identity, "run": str(directory), "n_events": int(len(prediction_ids)),
        "cache_root": str(cache_root), "preprocessed_waveforms": True,
        "model_inference": False, "feature_transform_applied": False,
        "selection": "closest observed prediction to each quantile; tie -> smallest event ID",
        "prototypes_represent": "model output quantiles, not guaranteed waveform morphology",
        "examples": selections,
    })
    if logger:
        logger.info("Correction examples | %s | %s | %s", manifest["model"], role, destination)
    return destination


def run(results, *, models=None, role="blind", cache_root=None,
        quantiles=DEFAULT_QUANTILES, logger=None):
    validate_quantiles(quantiles)
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
                quantiles=quantiles, logger=logger
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
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    for output in run(args.results, models=args.models, role=args.role,
                      cache_root=args.cache_root, quantiles=args.quantiles,
                      logger=logging.getLogger(__name__)):
        print(output)


if __name__ == "__main__":
    main()
