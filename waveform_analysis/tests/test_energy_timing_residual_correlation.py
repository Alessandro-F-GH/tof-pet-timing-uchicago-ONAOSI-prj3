"""Tests for matched-event energy/timing residual comparison."""
import numpy as np
import pytest

from waveform_analysis.scripts.compare_energy_timing_residuals import (
    read_residuals, align_residuals, correlations,
)


def test_align_by_event_id_not_by_array_position():
    ids, energy, timing, excluded = align_residuals(
        np.array([30, 10, 20, 50]), np.array([3., 1., 2., 5.]),
        np.array([20, 30, 40, 10]), np.array([20., 30., 40., 10.]),
    )
    np.testing.assert_array_equal(ids, [10, 20, 30])
    np.testing.assert_array_equal(energy, [1., 2., 3.])
    np.testing.assert_array_equal(timing, [10., 20., 30.])
    assert excluded == 0
    assert correlations(energy, timing)["pearson_r"] == pytest.approx(1.)


def test_no_duplicate_ids(tmp_path):
    file = tmp_path / "pred.npz"
    np.savez(file, event_id=np.array([1, 1, 2]),
             corrected_ps=np.array([1., 2., 3.]))
    with pytest.raises(ValueError, match="duplicate"):
        read_residuals(file)


def test_intersection_too_small_is_rejected():
    with pytest.raises(ValueError, match="Only 1 matched"):
        align_residuals(
            np.array([1, 2, 3]), np.array([1., 2., 3.]),
            np.array([3, 4, 5]), np.array([3., 4., 5.]),
        )


def test_difference_statistics_use_paired_residuals():
    from waveform_analysis.scripts.compare_energy_timing_residuals import difference_statistics
    timing = np.array([-2., -1., 0., 1., 2.])
    energy = timing + np.array([3., -3., 3., -3., 0.])
    result = difference_statistics(energy, timing)
    assert result["difference_mean_ps"] == pytest.approx(0.)
    assert result["difference_std_ps"] == pytest.approx(np.std(energy-timing, ddof=1))
    assert result["difference_rmse_ps"] == pytest.approx(np.sqrt(np.mean((energy-timing)**2)))


def test_common_component_recovers_known_decomposition():
    from waveform_analysis.scripts.compare_energy_timing_residuals import common_component_statistics
    rng = np.random.default_rng(123)
    n = 50000
    common = rng.normal(size=n) * 4.0
    energy = common + rng.normal(size=n) * 6.0
    timing = common + rng.normal(size=n) * 2.0
    result = common_component_statistics(energy, timing)
    expected_fraction = 16.0 / (16.0 + 4.0)
    assert result["common_fraction_timing"] == pytest.approx(expected_fraction, abs=0.03)
    assert result["common_component_variances_nonnegative"] is True
    assert result["dominance_test_p_value_one_sided"] < 0.05
    assert result["timing_specific_variance_ps2"] == pytest.approx(4.0, abs=0.5)


def test_common_component_does_not_claim_majority_when_absent():
    from waveform_analysis.scripts.compare_energy_timing_residuals import common_component_statistics
    rng = np.random.default_rng(456)
    energy = rng.normal(size=30000)
    timing = rng.normal(size=30000)
    result = common_component_statistics(energy, timing)
    assert abs(result["common_fraction_timing"]) < 0.05
    assert result["dominance_reject_h0_at_0_05"] is False


def test_gaussian_component_clipping_levels_and_population_fit():
    from waveform_analysis.scripts.compare_energy_timing_residuals import gaussian_component_fit_by_central_range
    rng = np.random.default_rng(2026)
    n = 100000
    common = rng.normal(0, 4, n)
    energy = common + rng.normal(0, 6, n)
    timing = common + rng.normal(0, 2, n)
    fits = gaussian_component_fit_by_central_range(energy, timing)
    assert [f["central_percent_per_channel"] for f in fits] == [100.0, 99.9, 99.0, 95.0]
    full = fits[0]
    assert full["n_retained"] == n
    assert full["physically_admissible"] is True
    assert full["sigma_common_ps"] == pytest.approx(4, abs=0.15)
    assert full["sigma_energy_specific_ps"] == pytest.approx(6, abs=0.15)
    assert full["sigma_timing_specific_ps"] == pytest.approx(2, abs=0.15)
    assert all(f["n_retained"] <= n for f in fits)
    assert all(fits[i]["n_retained"] >= fits[i+1]["n_retained"] for i in range(3))


def test_gaussian_component_cut_is_paired_and_joint():
    from waveform_analysis.scripts.compare_energy_timing_residuals import gaussian_component_fit_by_central_range
    timing = np.arange(1001, dtype=float)
    energy = timing.copy()
    energy[500], energy[0] = energy[0], energy[500]
    fit = gaussian_component_fit_by_central_range(energy, timing, (95.0,))[0]
    assert fit["n_retained"] < 951
    assert fit["retained_fraction"] == pytest.approx(fit["n_retained"] / 1001)


def test_bivariate_gmm_recovers_two_distinct_correlations():
    from waveform_analysis.scripts.compare_energy_timing_residuals import fit_bivariate_gmm
    rng = np.random.default_rng(2027)
    core = rng.multivariate_normal([0, 0], [[400, 40], [40, 100]], size=8000)
    tail = rng.multivariate_normal([0, 0], [[40000, 15000], [15000, 10000]], size=2000)
    observations = np.vstack([core, tail])
    rng.shuffle(observations)
    result, responsibilities = fit_bivariate_gmm(
        observations[:, 0], observations[:, 1], seed=13, n_init=3)
    assert responsibilities.shape == (10000, 2)
    np.testing.assert_allclose(responsibilities.sum(axis=1), 1, atol=1e-10)
    assert result["components"][0]["label"] == "core"
    assert result["components"][1]["label"] == "tail"
    assert result["components"][0]["sigma_energy_ps"] < result["components"][1]["sigma_energy_ps"]
    assert result["components"][0]["correlation"] < result["components"][1]["correlation"]
    assert result["delta_bic_one_minus_two"] > 0


def test_gmm_waveform_groups_left_core_right():
    from waveform_analysis.scripts.compare_energy_timing_residuals import waveform_groups_from_gmm
    energy = np.array([-10., 1., 15., 2.])
    timing = np.array([-12., 0., 12., 3.])
    responsibilities = np.array([
        [0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]
    ])
    summary = {"components": [
        {"label": "core"}, {"label": "tail", "mean_energy_ps": 0., "mean_timing_ps": 0.}
    ]}
    groups = waveform_groups_from_gmm(energy, timing, summary, responsibilities)
    np.testing.assert_array_equal(groups["left tail"], [True, False, False, False])
    np.testing.assert_array_equal(groups["core"], [False, True, False, True])
    np.testing.assert_array_equal(groups["right tail"], [False, False, True, False])


def test_prepared_waveform_means_match_event_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from waveform_analysis.scripts import compare_energy_timing_residuals as analysis
    wave = np.zeros((4, 2, 3), dtype=np.float32)
    for i in range(4):
        wave[i, :, :] = i + 1
    prepared = SimpleNamespace(
        event_index=np.array([40, 10, 30, 20]),
        energy_windows=wave,
        energy_time_ps=np.array([0, 1000, 2000]),
        energy_transform=SimpleNamespace(inverse=lambda x: x * 10),
    )
    monkeypatch.setattr(analysis, "load_prepared_dataset", lambda path: prepared)
    ids = np.array([10, 20, 30, 40])
    groups = {"left tail": np.array([True, False, False, False]),
              "core": np.array([False, True, True, False]),
              "right tail": np.array([False, False, False, True])}
    times, result = analysis.prepared_waveform_means(tmp_path, "energy", ids, groups)
    np.testing.assert_array_equal(times, [0, 1000, 2000])
    np.testing.assert_allclose(result["left tail"]["mean"], np.full((2, 3), 20.0))
    np.testing.assert_allclose(result["core"]["mean"], np.full((2, 3), 35.0))
    np.testing.assert_allclose(result["right tail"]["mean"], np.full((2, 3), 10.0))


def test_cache_root_resolves_wide_blind_from_named_alias(tmp_path):
    from waveform_analysis.scripts.compare_energy_timing_residuals import resolve_prepared_from_cache
    alias = tmp_path / "datasets" / "R0" / "R1_R2" / "energy" / "wide" / "blind"
    alias.mkdir(parents=True)
    np.save(alias / "event_index.npy", np.array([1, 3, 5, 7]))
    assert resolve_prepared_from_cache(
        tmp_path, "energy", "blind", np.array([3, 7])) == alias.resolve()


def test_cache_root_resolves_hash_from_manifest(tmp_path):
    import json
    from waveform_analysis.scripts.compare_energy_timing_residuals import resolve_prepared_from_cache
    product_id = "a" * 64
    folder = tmp_path / "artifact_store" / "products" / "prepared_dataset" / product_id
    folder.mkdir(parents=True)
    np.save(folder / "event_index.npy", np.array([10, 20, 30]))
    (tmp_path / "cache_manifest.json").write_text(json.dumps({
        "prepared_runs": [{"mode": "timing_to_timing", "window_name": "wide",
                           "datasets": {"blind": product_id}}]
    }), encoding="utf-8")
    assert resolve_prepared_from_cache(
        tmp_path, "timing", "blind", np.array([20, 30])) == folder.resolve()


def test_cache_root_rejects_ambiguous_dataset(tmp_path):
    from waveform_analysis.scripts.compare_energy_timing_residuals import resolve_prepared_from_cache
    for name in ("A", "B"):
        alias = tmp_path / "datasets" / "control" / name / "energy" / "wide" / "blind"
        alias.mkdir(parents=True)
        np.save(alias / "event_index.npy", np.array([1, 2, 3]))
    with pytest.raises(ValueError, match="exactly one"):
        resolve_prepared_from_cache(tmp_path, "energy", "blind", np.array([1, 2]))


def test_led_residuals_are_aligned_to_prediction_ids(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from waveform_analysis.scripts import compare_energy_timing_residuals as analysis
    dataset = SimpleNamespace(
        event_index=np.array([30, 10, 20], dtype=np.int64),
        energy_target_ps=np.array([300., 100., 200.]),
    )
    monkeypatch.setattr(analysis, "load_prepared_dataset", lambda _: dataset)
    observed = analysis.led_residuals_from_cache(
        tmp_path, "energy", np.array([10, 20, 30]))
    np.testing.assert_array_equal(observed, [100., 200., 300.])


def test_led_vs_ml_tail_comparison_tracks_fixed_led_events():
    from waveform_analysis.scripts.compare_energy_timing_residuals import compare_led_ml
    rng = np.random.default_rng(227)
    n = 4000
    is_tail = rng.random(n) < .04
    shared = rng.normal(0, 20, n)
    shared[is_tail] = rng.normal(0, 350, is_tail.sum())
    led_e = shared + rng.normal(0, 35, n)
    led_t = shared + rng.normal(0, 15, n)
    ml_e = led_e * .8
    ml_t = led_t * .8
    comparison, led_probs, ml_probs = compare_led_ml(
        led_e, led_t, ml_e, ml_t, seed=42, n_init=2)
    assert led_probs.shape == (n, 2)
    assert ml_probs.shape == (n, 2)
    transitions = comparison["transition"]
    assert transitions["led_tail_count"] > 0
    assert (transitions["led_tail_still_ml_tail"]
            + transitions["led_tail_becomes_ml_core"]
            == transitions["led_tail_count"])
    assert (comparison["paired_group_statistics"]["initial_led_tail"]["energy"]["ml_rms_ps"]
            < comparison["paired_group_statistics"]["initial_led_tail"]["energy"]["led_rms_ps"])
