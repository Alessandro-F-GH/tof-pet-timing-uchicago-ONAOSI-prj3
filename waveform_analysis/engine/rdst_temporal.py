"""Global temporal localization of frozen RDST Ridge contributions.

The three RDST feature contributions for each shapelet are combined BEFORE
taking their absolute value, and this net contribution is deposited at the
shapelet's best-match START position and separately over its dilated
sampling points. These are descriptive localization maps, not causality.

Only one reproducible blind sample is used, independently of the standard
prediction-quantile feature-space explanations.
"""
from __future__ import annotations

import numpy as np


def recover_argmin(scaled_features, scales, lengths, dilations, n_samples):
    """Invert RDST's StandardScaler(with_mean=False) for argmin columns."""
    matrix = np.asarray(scaled_features)
    scale = np.asarray(scales, dtype=np.float64).reshape(-1)
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    dilations = np.asarray(dilations, dtype=np.int64).reshape(-1)
    count = len(lengths)
    if (matrix.ndim != 2 or matrix.shape[1] != 3 * count
            or scale.shape != (3 * count,) or dilations.shape != (count,)
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



def distribute_dilated_matches(positions, contributions, lengths, dilations, n_samples):
    """Credit only the dilated sample points, never intervening samples."""
    p = np.asarray(positions, dtype=np.int64)
    c = np.asarray(contributions, dtype=np.float64)
    l = np.asarray(lengths, dtype=np.int64)
    d = np.asarray(dilations, dtype=np.int64)
    if (p.ndim != 2 or c.shape != p.shape or l.shape != (p.shape[1],)
            or d.shape != l.shape or np.any(l < 1) or np.any(d < 1)
            or not np.all(np.isfinite(c)) or np.any(p < 0)
            or np.any(p + (l[None, :] - 1) * d[None, :] >= n_samples)):
        raise ValueError("Invalid dilated shapelet match positions or contributions")
    signed = np.zeros(n_samples, dtype=np.float64)
    absolute = np.zeros(n_samples, dtype=np.float64)
    frequency = np.zeros(n_samples, dtype=np.float64)
    for tap in range(int(l.max())):
        active = np.flatnonzero(l > tap)
        sampled = (p[:, active] + tap * d[active][None, :]).reshape(-1)
        portions = (c[:, active] / l[active][None, :]).reshape(-1)
        signed += np.bincount(sampled, weights=portions, minlength=n_samples)
        absolute += np.bincount(sampled, weights=np.abs(portions), minlength=n_samples)
        frequency += np.bincount(
            sampled,
            weights=np.broadcast_to(
                1.0 / l[active][None, :], (len(p), len(active))).reshape(-1),
            minlength=n_samples)
    return signed, absolute, frequency


def temporal_match_maps(*, fitted, blind, mode, features, reference_mean,
                        coefficients, time_ps, chunk_size, logger=None,
                        event_indices=None, position_cache_root=None):
    """Compute one global match-position importance profile.

    Direct RDST: locate each multichannel shapelet's single joint match.
    Shared RDST: use each detector's original frozen argmin position from the
    persistent cache. Both maps carry the *paired* net contribution; their
    arithmetic mean provides a combined profile without double counting.
    """
    artifact = fitted.feature_transform.artifact
    lengths = np.asarray(artifact.transformer.shapelets_[2], dtype=np.int64)
    dilations = np.asarray(artifact.transformer.shapelets_[3], dtype=np.int64)
    scales = np.asarray(artifact.scaler.scale_, dtype=np.float64)
    time = np.asarray(time_ps, dtype=np.float64)
    reference = np.asarray(reference_mean, dtype=np.float64)
    weights = np.asarray(coefficients, dtype=np.float64)
    n_total, n_features = features.shape
    indices = (np.arange(n_total, dtype=np.int64) if event_indices is None
               else np.asarray(event_indices, dtype=np.int64))
    if (indices.ndim != 1 or len(indices) == 0 or np.any(indices < 0)
            or np.any(indices >= n_total) or np.any(np.diff(indices) <= 0)):
        raise ValueError("Temporal event indices must be nonempty, sorted and unique")
    if (time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time))
            or np.any(np.diff(time) <= 0) or n_features != 3 * len(lengths)
            or reference.shape != (n_features,) or weights.shape != (n_features,)
            or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(weights))
            or chunk_size < 1):
        raise ValueError("Temporal XAI expects aligned original frozen features and grid")

    shared = bool(artifact.shared)
    detector_names = np.asarray(
        ["detector_1", "detector_2"] if shared else ["joint_pair"]
    )
    n_detectors = 2 if shared else 1
    absolute_sum = np.zeros((n_detectors, len(time)), dtype=np.float64)
    signed_sum = np.zeros_like(absolute_sum)
    count_sum = np.zeros_like(absolute_sum, dtype=np.int64)
    sample_absolute_sum = np.zeros_like(absolute_sum)
    sample_signed_sum = np.zeros_like(absolute_sum)
    sample_count_sum = np.zeros_like(absolute_sum)
    net_absolute_sum = 0.0
    net_signed_sum = 0.0

    for start in range(0, len(indices), chunk_size):
        stop = min(start + chunk_size, len(indices))
        selected = indices[start:stop]
        data = np.asarray(features[selected], dtype=np.float32)
        net = ((data.astype(np.float64) - reference) * weights).reshape(
            -1, len(lengths), 3).sum(axis=2)
        net_absolute_sum += float(np.abs(net).sum())
        net_signed_sum += float(net.sum())

        if shared:
            if position_cache_root is None:
                raise ValueError("Shared RDST temporal XAI requires position_cache_root")
            from waveform_analysis.engine.rdst_position_cache import shared_positions_for_events
            recovered = shared_positions_for_events(
                cache_root=position_cache_root, fitted=fitted, dataset=blind,
                mode=mode, event_indices=selected, frozen_features=features,
                time_ps=time, logger=logger if start == 0 else None,
            )
            positions = (recovered[:, 0, :], recovered[:, 1, :])
        else:
            positions = (recover_argmin(data, scales, lengths, dilations, len(time)),)

        for detector, position in enumerate(positions):
            locs = position.ravel()
            contribution = net.ravel()
            absolute_sum[detector] += np.bincount(
                locs, weights=np.abs(contribution), minlength=len(time))
            signed_sum[detector] += np.bincount(
                locs, weights=contribution, minlength=len(time))
            count_sum[detector] += np.bincount(locs, minlength=len(time))
            ds, da, dc = distribute_dilated_matches(
                position, net, lengths, dilations, len(time))
            sample_signed_sum[detector] += ds
            sample_absolute_sum[detector] += da
            sample_count_sum[detector] += dc

        if logger and (stop == len(indices) or start == 0):
            logger.info("RDST temporal XAI | %d/%d blind events | shared=%s",
                        stop, len(indices), shared)

    n_events = len(indices)
    absolute = absolute_sum / n_events
    signed = signed_sum / n_events
    frequency = count_sum / n_events
    sample_absolute = sample_absolute_sum / n_events
    sample_signed = sample_signed_sum / n_events
    sample_frequency = sample_count_sum / n_events
    # Each shapelet deposits its net contribution exactly once in EACH
    # shared-detector view. The combined profile averages the two.
    np.testing.assert_allclose(
        absolute.sum(axis=1), net_absolute_sum / n_events, rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(
        signed.sum(axis=1), net_signed_sum / n_events, rtol=2e-12, atol=1e-8)

    np.testing.assert_allclose(
        sample_absolute.sum(axis=1), net_absolute_sum / n_events,
        rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(
        sample_signed.sum(axis=1), net_signed_sum / n_events,
        rtol=2e-12, atol=1e-8)
    combined_sample_absolute = sample_absolute.mean(axis=0)
    combined_sample_signed = sample_signed.mean(axis=0)
    combined_sample_frequency = sample_frequency.mean(axis=0)
    sample_total = combined_sample_absolute.sum()
    sample_normalized = (combined_sample_absolute / sample_total if sample_total > 0
                         else np.zeros_like(combined_sample_absolute))
    combined_absolute = absolute.mean(axis=0)
    combined_signed = signed.mean(axis=0)
    combined_frequency = frequency.mean(axis=0)
    total = combined_absolute.sum()
    normalized = (combined_absolute / total if total > 0
                  else np.zeros_like(combined_absolute))
    within_onishi = (time >= -1500.0) & (time <= 2000.0)

    return {
        "time_ps": time,
        "detector_names": detector_names,
        "n_events": np.asarray(n_events, dtype=np.int64),
        "sampled_population_indices": indices,
        "absolute_ps": absolute,
        "signed_ps": signed,
        "match_count_per_event": frequency,
        "combined_absolute_ps": combined_absolute,
        "combined_signed_ps": combined_signed,
        "combined_match_count_per_event": combined_frequency,
        "combined_fraction": normalized,
        "sample_absolute_ps": sample_absolute,
        "sample_signed_ps": sample_signed,
        "sample_count_per_event": sample_frequency,
        "combined_sample_absolute_ps": combined_sample_absolute,
        "combined_sample_signed_ps": combined_sample_signed,
        "combined_sample_count_per_event": combined_sample_frequency,
        "combined_sample_fraction": sample_normalized,
        "sample_onishi_fraction": np.asarray(sample_normalized[within_onishi].sum()),
        "sample_distribution_method": np.asarray(
            "uniform_over_exact_dilated_shapelet_sample_positions"),
        "onishi_fraction": np.asarray(normalized[within_onishi].sum()),
        "net_shapelet_abs_ps": np.asarray(net_absolute_sum / n_events),
        "net_shapelet_signed_ps": np.asarray(net_signed_sum / n_events),
        "method": np.asarray("global_contribution_weighted_best_match_start"),
        "method_version": np.asarray(3, dtype=np.int32),
        "shared_aggregation": np.asarray(
            "mean_of_two_detector_maps" if shared else "joint_multichannel_match"),
    }
