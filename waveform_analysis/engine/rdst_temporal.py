"""Temporal localization of RDST net Ridge contributions on the frozen time grid.

Each shapelet contributes at the START of its best-match subsequence. The
minimum-distance, argmin-position, and occurrence-count regression terms are
summed *before* absolute value. Positions are never interpreted as individual
sample-level causal attributions. No shapelets or model parameters are refitted.
"""
from __future__ import annotations

import numpy as np

from waveform_analysis.data.view import waveform_view
from waveform_analysis.signal.sample_mask import apply_sample_mask

GROUPS = ("all", "low", "central", "high")


def recover_argmin(scaled_features, scales, lengths, dilations, n_samples):
    """Invert RDST's StandardScaler(with_mean=False) for the argmin columns.

    aeon 1.6.0 emits [minimum L1 distance, argmin START, occurrence count]
    for each shapelet. The argmin is an integer 0-based retained-sample index.
    """
    matrix = np.asarray(scaled_features)
    scale = np.asarray(scales, dtype=np.float64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    dilations = np.asarray(dilations, dtype=np.int64).reshape(-1)
    count = len(lengths)
    if (matrix.ndim != 2 or matrix.shape[1] != 3*count
            or scale.shape != (3*count,) or dilations.shape != (count,)
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(scale))):
        raise ValueError("Invalid scaled RDST feature/shapelet layout")
    maximum = n_samples - 1 - (lengths - 1) * dilations
    if np.any(maximum < 0) or np.any(scale <= 0):
        raise ValueError("Saved shapelets do not fit the retained time grid")
    raw = np.asarray(matrix[:, 1::3], dtype=np.float64) * scale[None, 1::3]
    index = np.rint(raw).astype(np.int32)
    if (np.max(np.abs(raw - index)) > 0.02 or np.any(index < 0)
            or np.any(index > maximum[None, :])):
        raise ValueError("Recovered RDST argmin does not match valid sample starts")
    return index


def _histogram(positions, contributions, group_indices, hist_signed, hist_abs, hist_count, *, all_selected=None):
    """Add event-shapelet weights at their best-match locations, chunkwise."""
    n_samples = hist_signed.shape[-1]
    for detector, position in enumerate(positions):
        for row, selected in enumerate((np.ones(len(contributions), dtype=bool) if all_selected is None else all_selected,
                                        group_indices == 0, group_indices == 1,
                                        group_indices == 2)):
            if not np.any(selected):
                continue
            index = position[selected].ravel()
            weight = contributions[selected].ravel()
            hist_signed[row, detector] += np.bincount(index, weights=weight, minlength=n_samples)
            hist_abs[row, detector] += np.bincount(index, weights=np.abs(weight), minlength=n_samples)
            hist_count[row, detector] += np.bincount(index, minlength=n_samples)


def temporal_match_maps(*, spec, fitted, blind, mode, features, reference_mean,
                        coefficients, group_labels, time_ps, chunk_size, logger=None,
                        event_indices=None, all_membership=None):
    """Compute descriptive match-start maps without modifying model inputs.

    Direct: the multichannel shapelet has one joint match position.
    Shared: transform each detector separately with the ORIGINAL frozen bank.
    Each detector view receives the full net shapelet contribution; the reported
    combined map is their arithmetic mean, not their sum (prevents double count).
    """
    artifact = fitted.feature_transform.artifact
    shapelets = artifact.transformer.shapelets_
    lengths = np.asarray(shapelets[2], dtype=np.int64)
    dilations = np.asarray(shapelets[3], dtype=np.int64)
    scales = np.asarray(artifact.scaler.scale_, dtype=np.float64)
    time = np.asarray(time_ps, dtype=np.float64)
    ref = np.asarray(reference_mean, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    labels = np.asarray(group_labels, dtype=np.int8)
    n_total, p = features.shape
    indices = np.arange(n_total, dtype=np.int64) if event_indices is None else np.asarray(event_indices, dtype=np.int64)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= n_total) or np.any(np.diff(indices) <= 0):
        raise ValueError("Temporal event indices must be nonempty, sorted and unique")
    membership = np.ones(len(indices), dtype=bool) if all_membership is None else np.asarray(all_membership, dtype=bool)
    if membership.shape != indices.shape:
        raise ValueError("All-population membership must align with selected events")
    if (time.ndim != 1 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0) or ref.shape != (p,) or weights.shape != (p,)
            or p != 3 * len(lengths) or labels.shape != (n_total,) or chunk_size < 1):
        raise ValueError("Temporal XAI expects aligned original frozen features and grid")
    shared = bool(artifact.shared)
    detector_names = np.asarray(["detector_1", "detector_2"] if shared else ["joint_pair"])
    n_det = 2 if shared else 1
    hist_signed = np.zeros((4, n_det, len(time)), dtype=np.float64)
    hist_abs = np.zeros_like(hist_signed)
    hist_count = np.zeros_like(hist_signed, dtype=np.int64)
    sample_count = np.zeros(4, dtype=np.int64)
    net_abs_total = np.zeros(4, dtype=np.float64)
    net_signed_total = np.zeros(4, dtype=np.float64)
    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        selected_ids = indices[start:stop]
        data = np.asarray(features[selected_ids], dtype=np.float32)
        net = ((np.asarray(data, dtype=np.float64) - ref) * weights).reshape(-1, len(lengths), 3).sum(axis=2)
        positions = []
        if shared:
            # Differenced argmin columns are not individual detector positions.
            # Re-evaluate the SAVED shapelet bank on the two original detectors.
            pair = apply_sample_mask(waveform_view(blind, mode, selected_ids).materialize(),
                                     fitted.sample_mask)
            first = np.asarray(artifact.transformer.transform(pair[:, :1]), dtype=np.float32)
            second = np.asarray(artifact.transformer.transform(pair[:, 1:]), dtype=np.float32)
            # Reproduce the same float32 scaling and differencing as rdst_common.
            z1 = np.asarray(artifact.scaler.transform(first), dtype=np.float32)
            z2 = np.asarray(artifact.scaler.transform(second), dtype=np.float32)
            if not np.allclose(np.asarray(z1 - z2, np.float32), data, atol=2e-4, rtol=2e-4):
                raise ValueError("Separate-detector RDST features do not reconstruct saved shared features")
            positions = [
                recover_argmin(z1, scales, lengths, dilations, len(time)),
                recover_argmin(z2, scales, lengths, dilations, len(time)),
            ]
        else:
            positions = [recover_argmin(data, scales, lengths, dilations, len(time))]
        chosen = labels[selected_ids]
        all_selected = membership[start:stop]
        for g, mask in enumerate((all_selected,
                                  chosen == 0, chosen == 1, chosen == 2)):
            sample_count[g] += int(mask.sum())
            net_abs_total[g] += np.abs(net[mask]).sum()
            net_signed_total[g] += net[mask].sum()
        _histogram(positions, net, chosen, hist_signed, hist_abs, hist_count,
                   all_selected=all_selected)
        if logger and (stop == len(indices) or start == 0):
            logger.info("RDST temporal XAI | %d/%d blind events | shared=%s", stop, len(indices), shared)
    if np.any(sample_count <= 0):
        raise ValueError("Temporal explanation groups must be nonempty")
    signed = hist_signed / sample_count[:, None, None]
    absolute = hist_abs / sample_count[:, None, None]
    frequency = hist_count / sample_count[:, None, None]
    # Every shapelet maps once per event, so the map conserves its total
    # net-contribution magnitude independently in each shared-detector view.
    np.testing.assert_allclose(absolute.sum(axis=-1), net_abs_total[:, None] / sample_count[:, None],
                               rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(signed.sum(axis=-1), net_signed_total[:, None] / sample_count[:, None],
                               rtol=2e-12, atol=1e-8)
    combined_absolute = absolute.mean(axis=1)
    combined_signed = signed.mean(axis=1)
    combined_frequency = frequency.mean(axis=1)
    total = combined_absolute.sum(axis=1)
    normalized = np.divide(combined_absolute, total[:, None],
                           out=np.zeros_like(combined_absolute), where=total[:, None] > 0)
    # Semantics: a shapelet match is located at its starting sample, not
    # over all samples within its dilation-dependent receptive field.
    in_onishi = (time >= -1500.0) & (time <= 2000.0)
    fraction_onishi = np.sum(normalized[:, in_onishi], axis=1)
    return {
        "time_ps": time,
        "group_names": np.asarray(GROUPS),
        "detector_names": detector_names,
        "n_events": sample_count,
        "sampled_population_indices": indices,
        "all_population_membership": membership,
        "absolute_ps": absolute,
        "signed_ps": signed,
        "match_count_per_event": frequency,
        "combined_absolute_ps": combined_absolute,
        "combined_signed_ps": combined_signed,
        "combined_match_count_per_event": combined_frequency,
        "combined_fraction": normalized,
        "onishi_fraction": fraction_onishi,
        "net_shapelet_abs_ps": net_abs_total / sample_count,
        "net_shapelet_signed_ps": net_signed_total / sample_count,
        "method": np.asarray("contribution_weighted_best_match_start"),
        "method_version": np.asarray(1, dtype=np.int32),
        "shared_aggregation": np.asarray("mean_of_two_detector_maps" if shared else "joint_multichannel_match"),
    }
