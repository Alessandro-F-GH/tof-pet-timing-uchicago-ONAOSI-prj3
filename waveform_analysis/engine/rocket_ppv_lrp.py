"""ROCKET-LRP's PPV pooling redistribution on frozen aeon 1.6 MultiRocket.

This implements the PPV *pooling* rule of ROCKET-LRP: distribute a PPV
feature's signed Ridge contribution uniformly over positions where the
corresponding frozen convolution response exceeds its fitted bias.
It is NOT full convolution-to-input LRP and does not attribute MPV/MIPV/LSPV.

The convolution and PPV are reconstructed independently of aeon's tabular
transform and verified against the original model's frozen feature values.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from waveform_analysis.data.view import waveform_view
from waveform_analysis.signal.sample_mask import apply_sample_mask


@njit(cache=True)
def _trace_rep(x, n_channels_by_kernel, channel_ids, offsets, dilations, per_dilation,
               biases, kernel_indices, feature_relevance, scale, trim, selected_mask=None, input_attribution=None):
    """Trace ONE waveform representation using aeon 1.6 convolution geometry.

    The output maps use convolution *response positions*, not input LRP.
    Even dilations may contribute distinct PPV feature thresholds per kernel.
    """
    channels, length = x.shape
    n_kernels = len(kernel_indices)
    features = n_kernels * np.sum(per_dilation)
    reconstructed = np.full(features, np.nan, dtype=np.float64)
    signed = np.zeros(length + trim, dtype=np.float64)
    absolute = np.zeros(length + trim, dtype=np.float64)
    hits = np.zeros(length + trim, dtype=np.int64)
    unlocalized_abs = 0.0
    unlocalized_signed = 0.0
    index = 0
    input_signed = np.zeros(length + trim, dtype=np.float64)
    input_absolute = np.zeros(length + trim, dtype=np.float64)
    input_residual = 0.0
    for di in range(len(dilations)):
        n_bias = int(per_dilation[di])
        first = index
        last = first + n_kernels * n_bias
        if selected_mask is not None:
            if not np.any(selected_mask[first:last]):
                index = last
                continue
        dilation = int(dilations[di])
        padding = 4 * dilation
        # Reproduce the C_alpha/C_gamma arithmetic in aeon 1.6.0.
        a = -x.copy()
        g = np.zeros((9, channels, length), dtype=np.float32)
        g[4] = x + x + x
        for tap in range(9):
            if tap == 4:
                continue
            # aeon 1.6.0 difference branch initializes its left-shift
            # loop with n_timepoints - padding, where n_timepoints is the
            # ORIGINAL series length. The differenced series is one shorter,
            # shifting the four negative-index taps right by one sample.
            # Preserve this exact (asymmetric) reference behavior.
            offset = (tap - 4) * dilation + (1 if trim and tap < 4 else 0)
            for ch in range(channels):
                for t in range(length):
                    at = t + offset
                    if 0 <= at < length:
                        a[ch, t] += -x[ch, at]
                        g[tap, ch, t] = x[ch, at] + x[ch, at] + x[ch, at]
        for ki in range(n_kernels):
            if selected_mask is not None:
                if not np.any(selected_mask[index:index + n_bias]):
                    index += n_bias
                    continue
            chosen = np.empty(length, dtype=np.float32)
            left = int(offsets[di * n_kernels + ki])
            right = int(offsets[di * n_kernels + ki + 1])
            a0, a1, a2 = kernel_indices[ki]
            for t in range(length):
                accumulator = np.float32(0.)
                for k in range(left, right):
                    ch = channel_ids[k]
                    v = a[ch, t] + g[a0, ch, t]
                    v = v + g[a1, ch, t]
                    v = v + g[a2, ch, t]
                    accumulator += v
                chosen[t] = accumulator
            use_trim = (di % 2 + ki) % 2 == 1
            lo = padding if use_trim else 0
            hi = length - padding if use_trim else length
            if hi <= lo:
                raise ValueError("Invalid MultiRocket dilation relative to samples")
            for bj in range(n_bias):
                if selected_mask is not None and not selected_mask[index]:
                    index += 1
                    continue
                bias = biases[index]
                positives = 0
                for t in range(lo, hi):
                    if chosen[t] > bias:
                        positives += 1
                ppv = positives / (hi - lo)
                reconstructed[index] = ppv / scale[index]
                relevance = float(feature_relevance[index])
                if positives:
                    portion = relevance / positives
                    for t in range(lo, hi):
                        if chosen[t] > bias:
                            at = t + (1 if trim else 0)
                            signed[at] += portion
                            absolute[at] += abs(portion)
                            hits[at] += 1
                            if input_attribution is not None:
                                denom = 0.0
                                for k in range(left, right):
                                    ch = channel_ids[k]
                                    for tap in range(9):
                                        source = t + (tap - 4) * dilation + (1 if trim and tap < 4 else 0)
                                        if 0 <= source < length:
                                            weight = 2.0 if (tap == a0 or tap == a1 or tap == a2) else -1.0
                                            denom += abs(float(x[ch, source]) * weight)
                                if denom <= 1e-30:
                                    input_residual += abs(portion)
                                else:
                                    for k in range(left, right):
                                        ch = channel_ids[k]
                                        for tap in range(9):
                                            source = t + (tap - 4) * dilation + (1 if trim and tap < 4 else 0)
                                            if 0 <= source < length:
                                                weight = 2.0 if (tap == a0 or tap == a1 or tap == a2) else -1.0
                                                part = portion * abs(float(x[ch, source]) * weight) / denom
                                                if trim:
                                                    input_signed[source] += 0.5 * part
                                                    input_signed[source + 1] += 0.5 * part
                                                    input_absolute[source] += 0.5 * abs(part)
                                                    input_absolute[source + 1] += 0.5 * abs(part)
                                                else:
                                                    input_signed[source] += part
                                                    input_absolute[source] += abs(part)
                else:
                    unlocalized_abs += abs(relevance)
                    unlocalized_signed += relevance
                index += 1
    if input_attribution is not None:
        # Arrays live inside a tuple. Mutate their elements, never the
        # immutable tuple slots (Numba rejects tuple-slot augmented assignment).
        for t in range(length + trim):
            input_attribution[0][t] += input_signed[t]
            input_attribution[1][t] += input_absolute[t]
        input_attribution[2][0] += input_residual
    return reconstructed, signed, absolute, hits, unlocalized_abs, unlocalized_signed


def _params(artifact, which):
    t = artifact.transformer
    params = t.parameter if which == "raw" else t.parameter1
    indices = np.asarray(t._indices, dtype=np.int32)
    if len(params) == 3:
        dilations, counts, biases = params
        n = len(indices) * len(dilations)
        channels = np.zeros(n, dtype=np.int32)
        nch = np.ones(n, dtype=np.int32)
        offsets = np.arange(n + 1, dtype=np.int32)
    elif len(params) == 5:
        nch, channels, dilations, counts, biases = params
        if which == "difference":
            # aeon 1.6 _transform_multi explicitly reuses channels from
            # self.parameter (raw), even though parameter1 carries its
            # own channel arrays. Match the actual fitted transformation.
            nch, channels = t.parameter[0], t.parameter[1]
        nch = np.asarray(nch, dtype=np.int32)
        channels = np.asarray(channels, dtype=np.int32)
        offsets = np.concatenate(([0], np.cumsum(nch, dtype=np.int32)))
    else:
        raise ValueError("Unsupported frozen aeon MultiRocket parameter schema")
    dilations = np.asarray(dilations, dtype=np.int32)
    counts = np.asarray(counts, dtype=np.int32)
    biases = np.asarray(biases, dtype=np.float32)
    f = len(indices) * int(counts.sum())
    if (len(nch) != len(indices) * len(dilations) or len(biases) != f
            or offsets[-1] != len(channels) or np.any(counts < 1)
            or np.any(dilations < 1)):
        raise ValueError("Invalid fitted MultiRocket kernel layout")
    return nch, channels, offsets, dilations, counts, biases, indices


def _ppv_columns(artifact):
    raw = _params(artifact, "raw")
    diff = _params(artifact, "difference")
    n_raw = len(raw[5])
    n_diff = len(diff[5])
    total = 4 * (n_raw + n_diff)
    if int(artifact.metadata["feature_count"]) != total:
        raise ValueError("MultiRocket fitted feature ordering mismatch")
    return (np.arange(n_raw), np.arange(4 * n_raw, 4 * n_raw + n_diff),
            raw, diff)


def ppv_temporal_map(*, fitted, blind, mode, features, reference_mean, coefficients,
                     time_ps, event_indices, logger=None, n_ppv_features=None):
    """Explain the PPV-only feature contribution with response-position maps.

    The RDST+MultiRocket combined estimator is supported using its frozen
    MultiRocket feature block; RDST contributions are excluded from PPV coverage.
    """
    from waveform_analysis.models.kernel.multirocket_rdst_common import CombinedTransformArtifact
    transform = fitted.feature_transform.artifact
    if isinstance(transform, CombinedTransformArtifact):
        block = transform.multirocket
        start, end = transform.metadata["feature_blocks"]["multirocket"]
        if start != 0:
            raise ValueError("Combined MultiRocket block must come first")
    else:
        block = transform
        start, end = 0, int(block.metadata["feature_count"])
    if not hasattr(block.transformer, "parameter1"):
        raise ValueError("PPV-LRP requires an aeon 1.6 MultiRocket frozen transformer")
    time = np.asarray(time_ps, dtype=np.float64)
    indices = np.asarray(event_indices, dtype=np.int64)
    if (indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0)
            or np.any(indices < 0) or np.any(indices >= features.shape[0])
            or len(time) < 10 or np.any(np.diff(time) <= 0)):
        raise ValueError("Invalid PPV temporal selection or retained time grid")
    if (len(reference_mean) != features.shape[1] or len(coefficients) != features.shape[1]
            or end > features.shape[1]):
        raise ValueError("Frozen Ridge/transform feature mismatch")
    raw_cols, diff_cols, raw_params, diff_params = _ppv_columns(block)
    ppv_cols = np.concatenate([raw_cols, diff_cols])
    if n_ppv_features is not None and (
            isinstance(n_ppv_features, bool) or not isinstance(n_ppv_features, int)
            or n_ppv_features < 1):
        raise ValueError("n_ppv_features must be a positive integer")
    # Rank PPV columns using ALL blind events, not the sampled XAI subset.
    # Use original development-centered linear contributions and stable ties.
    weighted = np.asarray(coefficients[start:end], dtype=np.float64)
    centered_ref = np.asarray(reference_mean[start:end], dtype=np.float64)
    ppv_scores = np.zeros(len(ppv_cols), dtype=np.float64)
    if logger:
        logger.info("ROCKET-LRP PPV ranking | full-blind events=%d | candidate PPV features=%d | requested top=%s",
                    features.shape[0], len(ppv_cols), n_ppv_features if n_ppv_features is not None else "all")
    for lo in range(0, features.shape[0], 256):
        hi = min(lo + 256, features.shape[0])
        block_features = np.asarray(features[lo:hi, start:end], dtype=np.float64)
        ppv_scores += np.abs(
            (block_features[:, ppv_cols] - centered_ref[ppv_cols])
            * weighted[ppv_cols]).sum(axis=0)
        if logger and (lo == 0 or hi == features.shape[0] or (lo // 256 + 1) % 20 == 0):
            logger.info("ROCKET-LRP PPV ranking | %d/%d blind events", hi, features.shape[0])
    ppv_scores /= features.shape[0]
    ranking = np.argsort(-ppv_scores, kind="stable")
    selected = np.sort(ranking[:min(n_ppv_features or len(ppv_cols), len(ppv_cols))])
    selected_ppv_columns = ppv_cols[selected]
    selected_mask = np.zeros(end - start, dtype=np.bool_)
    selected_mask[selected_ppv_columns] = True
    importance_coverage = (
        ppv_scores[selected].sum() / ppv_scores.sum()
        if ppv_scores.sum() else 0.0)
    if logger:
        logger.info("ROCKET-LRP PPV selection | selected=%d/%d | global importance covered=%.2f%%",
                    len(selected), len(ppv_cols), 100 * importance_coverage)
    scales = np.asarray(block.scaler.scale_, dtype=np.float64)
    if len(scales) != end - start or np.any(scales <= 0):
        raise ValueError("Missing MultiRocket control scaling")
    weights = np.asarray(coefficients[start:end], dtype=np.float64)
    reference = np.asarray(reference_mean[start:end], dtype=np.float64)
    shared = bool(block.shared)
    n_det = 2 if shared else 1
    # Keep representation-specific histograms, detector-specific for shared.
    signed = np.zeros((2, n_det, len(time)), dtype=np.float64)
    input_signed = np.zeros((2, n_det, len(time)), dtype=np.float64)
    input_absolute = np.zeros_like(input_signed)
    input_residual = np.zeros((2, n_det), dtype=np.float64)
    absolute = np.zeros_like(signed)
    counts = np.zeros_like(signed, dtype=np.int64)
    unlocalized_abs = np.zeros((2, n_det), dtype=np.float64)
    unlocalized_signed = np.zeros((2, n_det), dtype=np.float64)
    ppv_total_abs = 0.0
    selected_total_abs = 0.0
    all_features_abs = 0.0
    valid_tol = 2e-4
    if logger:
        logger.info("ROCKET-LRP PPV tracing | starting %d events | selected PPV features=%d",
                    len(indices), len(selected_ppv_columns))
    for pos, event_index in enumerate(indices):
        pair = apply_sample_mask(
            waveform_view(blind, mode, [int(event_index)]).materialize(),
            fitted.sample_mask)[0]
        event_features = np.asarray(features[event_index], dtype=np.float64)
        centered = (event_features[start:end] - reference) * weights
        ppv_total_abs += np.abs(centered[ppv_cols]).sum()
        selected_total_abs += np.abs(centered[selected_ppv_columns]).sum()
        all_features_abs += np.abs((event_features - reference_mean) * coefficients).sum()
        # Shared feature differences are indivisible at the feature level:
        # use the SAME net contribution in each detector view and average
        # the views, exactly as for the shared RDST best-match-position map.
        detector_views = [pair] if not shared else [pair[0:1], pair[1:2]]
        for detector, view in enumerate(detector_views):
            x = view
            if block.transformer.normalise:
                x = (x - x.mean(axis=-1, keepdims=True)) / (
                    x.std(axis=-1, keepdims=True) + 1e-8)
                x = np.asarray(x, dtype=np.float32)
            xdiff = np.diff(x, axis=1)
            for rep, (values, columns, params, trim) in enumerate((
                (x, raw_cols, raw_params, 0),
                (xdiff, diff_cols, diff_params, 1),
            )):
                if trim:
                    # Differenced responses are indexed by their right sample;
                    # this convention deliberately avoids half-grid bins.
                    pass
                attribution = np.zeros((2, len(time)), dtype=np.float64)
                residual = np.zeros(1, dtype=np.float64)
                r, s, a, n_hits, ua, us = _trace_rep(
                    np.ascontiguousarray(values, dtype=np.float32),
                    *params, np.asarray(centered[columns], dtype=np.float64),
                    np.asarray(scales[columns], dtype=np.float64), trim,
                    selected_mask=np.asarray(selected_mask[columns], dtype=np.bool_),
                    input_attribution=(attribution[0], attribution[1], residual))
                input_signed[rep, detector] += attribution[0]
                input_absolute[rep, detector] += attribution[1]
                input_residual[rep, detector] += residual[0]
                # Direct: compare to saved features. Shared: collect the two
                # detector feature vectors, then check their difference.
                if not shared:
                    expected = event_features[start:end][columns]
                    active = selected_mask[columns]
                    if not np.allclose(r[active], expected[active], atol=valid_tol, rtol=valid_tol):
                        delta = np.abs(r[active] - expected[active])
                        worst = int(np.flatnonzero(active)[np.argmax(delta)])
                        raise ValueError(
                            f"PPV reconstruction failed | event={int(event_index)} "
                            f"representation={['raw','difference'][rep]} "
                            f"feature_column={int(columns[worst])} "
                            f"reconstructed_scaled={float(r[worst]):.7g} "
                            f"saved_scaled={float(expected[worst]):.7g} "
                            f"scale={float(scales[columns[worst]]):.7g} "
                            f"max_error={float(delta[worst_active]):.7g}")
                else:
                    if detector == 0:
                        if rep == 0:
                            first_raw = r
                        else:
                            first_diff = r
                    else:
                        baseline = first_raw if rep == 0 else first_diff
                        expected = event_features[start:end][columns]
                        active = selected_mask[columns]
                        if not np.allclose((baseline-r)[active], expected[active], atol=valid_tol, rtol=valid_tol):
                            delta = np.abs((baseline-r)[active] - expected[active])
                            worst_active = int(np.argmax(delta))
                            worst = int(np.flatnonzero(active)[worst_active])
                            raise ValueError(
                                f"Shared PPV reconstruction failed | event={int(event_index)} "
                                f"representation={['raw','difference'][rep]} "
                                f"feature_column={int(columns[worst])} "
                                f"reconstructed_difference={float((baseline-r)[worst]):.7g} "
                                f"saved_difference={float(expected[worst]):.7g} "
                                f"scale={float(scales[columns[worst]]):.7g} "
                                f"max_error={float(delta[worst]):.7g}")
                signed[rep, detector] += s
                absolute[rep, detector] += a
                counts[rep, detector] += n_hits
                unlocalized_abs[rep, detector] += ua
                unlocalized_signed[rep, detector] += us
        if logger and (pos == 0 or (pos+1) % 10 == 0 or pos+1 == len(indices)):
            logger.info("ROCKET-LRP PPV tracing | %d/%d blind events | top_features=%d",
                        pos+1, len(indices), len(selected_ppv_columns))
    n = len(indices)
    input_signed /= n
    input_absolute /= n
    input_residual /= n
    signed /= n
    absolute /= n
    counts = counts / n
    unlocalized_abs /= n
    unlocalized_signed /= n
    combined_abs = absolute.mean(axis=1).sum(axis=0)
    combined_signed = signed.mean(axis=1).sum(axis=0)
    input_total_absolute = input_absolute.mean(axis=1).sum(axis=0)
    input_total_signed = input_signed.mean(axis=1).sum(axis=0)
    input_missing = input_residual.mean(axis=1).sum()
    np.testing.assert_allclose(
        input_total_absolute.sum() + input_missing + unlocalized_abs.mean(axis=1).sum(),
        selected_total_abs / n, rtol=1e-7, atol=1e-5)
    input_fraction = (input_total_absolute / input_total_absolute.sum()
                      if input_total_absolute.sum() else np.zeros_like(input_total_absolute))
    mass = combined_abs.sum() + unlocalized_abs.mean(axis=1).sum()
    np.testing.assert_allclose(mass, selected_total_abs / n, rtol=1e-8, atol=1e-5)
    normalized = (combined_abs / combined_abs.sum() if combined_abs.sum() else
                  np.zeros_like(combined_abs))
    return {
        "time_ps": time,
        "event_indices": indices,
        "n_events": np.asarray(n),
        "model_family": np.asarray("multirocket_ppv"),
        "method": np.asarray("rocket_lrp_ppv_pooling_response_positions"),
        "method_version": np.asarray(3),
        "scope": np.asarray("PPV-only; no convolution-to-input LRP"),
        "representation_names": np.asarray(["raw", "first_difference"]),
        "detector_names": np.asarray(["joint_pair"] if not shared else ["detector_1", "detector_2"]),
        "absolute_ps": absolute,
        "signed_ps": signed,
        "positive_match_count_per_event": counts,
        "unlocalized_abs_ps": unlocalized_abs,
        "unlocalized_signed_ps": unlocalized_signed,
        "input_absolute_ps": input_absolute,
        "input_signed_ps": input_signed,
        "input_residual_abs_ps": input_residual,
        "combined_input_absolute_ps": input_total_absolute,
        "combined_input_signed_ps": input_total_signed,
        "combined_input_fraction": input_fraction,
        "input_attribution_method": np.asarray(
            "abs_input_kernel_product_with_difference_endpoint_split"),
        "combined_absolute_ps": combined_abs,
        "combined_signed_ps": combined_signed,
        "combined_fraction": normalized,
        "ppv_total_abs_ps": np.asarray(ppv_total_abs / n),
        "selected_ppv_abs_ps": np.asarray(selected_total_abs / n),
        "selected_ppv_columns": selected_ppv_columns,
        "ppv_global_importance": ppv_scores,
        "ppv_global_selection_coverage": np.asarray(importance_coverage),
        "n_ppv_features_selected": np.asarray(len(selected)),
        "n_ppv_features_available": np.asarray(len(ppv_cols)),
        "all_feature_total_abs_ps": np.asarray(all_features_abs / n),
        "ppv_feature_fraction_abs": np.asarray(ppv_total_abs / max(all_features_abs, 1e-30)),
        "onishi_fraction_localized": np.asarray(normalized[(time >= -1500) & (time <= 2000)].sum()),
    }
