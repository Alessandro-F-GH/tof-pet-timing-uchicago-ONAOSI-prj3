from __future__ import annotations

import numpy as np

from ..search import SearchResult, select_candidate
from ..stats import ctr_fwhm
from ..timing import cfd_grid, pair_delta


def select_precomputed_cfd_times(
    energy_cfd_time_fs: np.ndarray,
    timing_cfd_time_fs: np.ndarray | None,
) -> np.ndarray:
    """Raw-preprocessing helper: keep CFD and LED on the same active family."""
    energy = np.asarray(energy_cfd_time_fs, dtype=np.int64)
    if timing_cfd_time_fs is None:
        return energy
    timing = np.asarray(timing_cfd_time_fs, dtype=np.int64)
    if timing.shape != energy.shape:
        raise ValueError("Energy/timing CFD timestamp shapes differ")
    return timing


def select_cfd(config, dataset, family, training, validation, seed) -> SearchResult:
    fractions = np.asarray(config["standard_methods"]["cfd_fractions"], dtype=np.float64)
    grid = cfd_grid(config, dataset, family, validation, fractions)

    def fit(candidate, _data, _seed):
        return int(np.flatnonzero(fractions == float(candidate))[0])

    def predict(_candidate, column, _data):
        residual = pair_delta(grid[:, :, int(column)]) - float(dataset.true_tof_ps)
        if not np.all(np.isfinite(residual)):
            raise ValueError("Candidate does not provide complete validation crossing coverage")
        return residual

    return select_candidate(
        fractions.tolist(), training, validation,
        fit_candidate=fit,
        predict_candidate=predict,
        score_candidate=lambda residual: ctr_fwhm(residual, config.get("fit")).ctr_ps,
        seed=seed,
    )


def evaluate_cfd(config, dataset, family, indices, fraction) -> np.ndarray:
    return pair_delta(
        cfd_grid(
            config, dataset, family, indices, np.asarray([float(fraction)], dtype=np.float64)
        )[:, :, 0]
    )
