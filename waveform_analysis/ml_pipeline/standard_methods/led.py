from __future__ import annotations

import numpy as np

from ..search import SearchResult, select_candidate
from ..stats import ctr_fwhm
from ..timing import leading_edge_grid, pair_delta


def select_led(config, dataset, family, training, validation, seed) -> SearchResult:
    thresholds = np.asarray(config["standard_methods"]["led_thresholds_mV"], dtype=np.float64)
    grid = leading_edge_grid(config, dataset, family, validation, thresholds)

    def fit(candidate, _data, _seed):
        return int(np.flatnonzero(thresholds == float(candidate))[0])

    def predict(_candidate, column, _data):
        residual = pair_delta(grid[:, :, int(column)]) - float(dataset.true_tof_ps)
        if not np.all(np.isfinite(residual)):
            raise ValueError("Candidate does not provide complete validation crossing coverage")
        return residual

    return select_candidate(
        thresholds.tolist(), training, validation,
        fit_candidate=fit,
        predict_candidate=predict,
        score_candidate=lambda residual: ctr_fwhm(residual, config.get("fit")).ctr_ps,
        seed=seed,
    )


def evaluate_led(config, dataset, family, indices, threshold_mV) -> np.ndarray:
    return pair_delta(
        leading_edge_grid(
            config, dataset, family, indices, np.asarray([float(threshold_mV)], dtype=np.float64)
        )[:, :, 0]
    )
