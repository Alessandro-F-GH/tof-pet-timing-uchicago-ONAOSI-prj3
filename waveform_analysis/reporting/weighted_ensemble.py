"""Select pair weights on saved in-sample development predictions only."""
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.reporting.stats import align_by_event_id, metric_values

WEIGHT_GRID = tuple(index / 20 for index in range(21))
WEIGHTED_ENSEMBLE_VERSION = 1


def aligned_predictions(first, second):
    """A fixed finite intersection for every weight, including endpoints."""
    ids, a, b = align_by_event_id(
        first["event_id"], first["prediction_ps"],
        second["event_id"], second["prediction_ps"],
    )
    led_ids, led_a, led_b = align_by_event_id(
        first["event_id"], first["led_residual_ps"],
        second["event_id"], second["led_residual_ps"],
    )
    matched, prediction_idx, led_idx = np.intersect1d(ids, led_ids, return_indices=True)
    if len(matched) < 5:
        raise ValueError("fewer than five finite matched events")
    if not np.allclose(led_a[led_idx], led_b[led_idx], rtol=1e-10, atol=1e-8):
        raise ValueError("saved LED residuals differ on matched events")
    return matched, a[prediction_idx], b[prediction_idx], led_a[led_idx]


def read_development(run):
    """Read artifacts without inference; reject stale or unverified outputs."""
    manifest = run["manifest"]
    expected = manifest.get("stage_fingerprints", {}).get("final_fit")
    if not expected:
        raise ValueError("missing final-fit fingerprint")
    path = Path(run["directory"]) / "artifacts/development_pred.npz"
    with np.load(path, allow_pickle=False) as saved:
        data = {key: np.asarray(saved[key]) for key in saved.files}
    if str(data["final_fit_fingerprint"].item()) != expected:
        raise ValueError("stale development predictions: final-fit fingerprint differs")
    if str(data["dataset_role"].item()) != "development":
        raise ValueError("prediction dataset role is not development")
    actual_protocol = str(data["analysis_protocol_identity"].item())
    expected_protocol = manifest.get("dataset_populations", {}).get("development", {}).get("analysis_protocol_identity")
    if not actual_protocol or (expected_protocol is not None and actual_protocol != expected_protocol):
        raise ValueError("development prediction protocol identity differs")
    ids = np.asarray(data["event_id"])
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError("development event IDs must be unique, integer and one-dimensional")
    for key in ("prediction_ps", "led_residual_ps"):
        if data[key].shape != ids.shape:
            raise ValueError("development predictions and event IDs must have matching shapes")
    return data


def weighted_ensemble(group, blind_payloads, fit, *, seed):
    labels = [run["manifest"]["model"] for run in group]
    size = len(group)
    weights = np.full((size, size), np.nan)
    development_ctr = np.full_like(weights, np.nan)
    blind_ctr = np.full_like(weights, np.nan)
    development_n = np.zeros((size, size), dtype=int)
    blind_n = np.zeros_like(development_n)
    scan, unavailable, sources = [], [], []
    development = {}
    errors = {}
    for i, run in enumerate(group):
        weights[i, i] = 1.0  # A diagonal entry is the single model itself.
        blind_ctr[i, i] = float(run["blind"]["ctr_ps"])
        blind_n[i, i] = int(run["blind"]["n_events"])
        source = {"model": labels[i], "final_fit_fingerprint": run["manifest"].get("stage_fingerprints", {}).get("final_fit")}
        try:
            data = read_development(run)
            development[i] = data
            _, a, _, led = aligned_predictions(data, data)
            development_n[i, i] = len(a)
            source.update(status="available", analysis_protocol_identity=str(data["analysis_protocol_identity"].item()))
            try:
                development_ctr[i, i] = metric_values(led - a, fit, seed=semantic_seed(seed, labels[i], "development_single"))["ctr_ps"]
            except ValueError as exc:
                source["single_model_ctr_unavailable"] = str(exc)
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            errors[i] = str(exc)
            development.pop(i, None)
            source.update(status="unavailable", reason=str(exc))
        sources.append(source)

    # Complete every development selection before evaluating any blind pair.
    selected = []
    for i in range(size):
        for j in range(i + 1, size):
            a_idx, b_idx = sorted((i, j), key=lambda index: labels[index])
            pair = {"row": labels[a_idx], "column": labels[b_idx]}
            try:
                if a_idx not in development or b_idx not in development:
                    reason = "; ".join(f"{labels[k]}: {errors[k]}" for k in (a_idx, b_idx) if k in errors)
                    raise ValueError(reason)
                ids, a, b, led = aligned_predictions(development[a_idx], development[b_idx])
                development_n[i, j] = development_n[j, i] = len(ids)
                candidates = []
                pair_seed = semantic_seed(seed, group[i]["manifest"]["mode"],
                                          group[i]["manifest"].get("window_name"),
                                          labels[a_idx], labels[b_idx], "weighted_development")
                for index, weight in enumerate(WEIGHT_GRID):
                    row = {"model_a": labels[a_idx], "model_b": labels[b_idx],
                           "weight_a": weight, "weight_b": 1.0 - weight,
                           "development_ctr_ps": float("nan"), "n_events": len(ids),
                           "selected": False, "status": "failed", "reason": ""}
                    try:
                        ctr = metric_values(led - (weight * a + (1.0 - weight) * b), fit, seed=pair_seed)["ctr_ps"]
                        if not np.isfinite(ctr):
                            raise ValueError("non-finite development CTR")
                        row.update(development_ctr_ps=ctr, status="complete")
                        candidates.append((ctr, abs(index - 10), index, row))
                    except ValueError as exc:
                        row["reason"] = str(exc)
                    scan.append(row)
                if not candidates:
                    raise ValueError("no weight has a finite development CTR")
                ctr, _, index, row = min(candidates, key=lambda candidate: candidate[:3])
                row["selected"] = True
                weight = WEIGHT_GRID[index]
                weights[a_idx, b_idx], weights[b_idx, a_idx] = weight, 1.0 - weight
                development_ctr[i, j] = development_ctr[j, i] = ctr
                selected.append((a_idx, b_idx, weight))
            except (ValueError, KeyError) as exc:
                unavailable.append({**pair, "stage": "development_selection", "reason": str(exc)})

    for a_idx, b_idx, weight in selected:
        try:
            ids, a, b, led = aligned_predictions(blind_payloads[a_idx], blind_payloads[b_idx])
            blind_n[a_idx, b_idx] = blind_n[b_idx, a_idx] = len(ids)
            value = metric_values(led - (weight * a + (1.0 - weight) * b), fit,
                                  seed=semantic_seed(seed, labels[a_idx], labels[b_idx], "weighted_blind"))["ctr_ps"]
            blind_ctr[a_idx, b_idx] = blind_ctr[b_idx, a_idx] = value
        except (ValueError, KeyError) as exc:
            unavailable.append({"row": labels[a_idx], "column": labels[b_idx],
                                "stage": "blind_evaluation", "reason": str(exc)})
    return {"labels": labels, "weights": weights, "development_ctr": development_ctr,
            "blind_ctr": blind_ctr, "development_n": development_n, "blind_n": blind_n,
            "scan": scan, "unavailable_pairs": unavailable, "development_sources": sources}
