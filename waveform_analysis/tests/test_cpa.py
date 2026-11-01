"""Synthetic CPA regression checks; no real waveforms or model artifacts needed."""
import numpy as np
from waveform_analysis.engine.cpa_explainer import (
    cutoff_samples, cutoff_schedule_ns, segment_for_cutoff,
    suffix_with_imputer, reconstruct_pair, summarize_predictions, select_imputers,
)


def test_cutoffs_skip_masked_samples_without_moving_them():
    time = np.arange(-2., 26., .5)*1000.
    mask = np.ones(len(time), dtype=bool)
    mask[(time >= 4000.) & (time <= 6000.)] = False
    kept, omitted = cutoff_samples(time, mask, [-1., 0., 2., 4., 6., 8., 20.])
    assert 4. in [v for v, reason in omitted if reason == "masked"]
    assert 6. in [v for v, reason in omitted if reason == "masked"]
    assert np.all(mask[kept])
    assert 8. in (time[kept]/1000.).tolist()
    assert cutoff_schedule_ns(20) == [-1., 0., 2., 4., 6., 8., 10., 15., 20.]


def test_ar_never_crosses_gap_and_never_reveals_hidden_suffix():
    mean = np.full(22, .4)
    std = np.ones(22)
    mask = np.ones(22, bool)
    mask[9:13] = False
    waves = np.full((3, 22), .6, np.float32)
    assert segment_for_cutoff(mask, 7) == (0, 9)
    assert segment_for_cutoff(mask, 15) == (13, 22)
    choice = {"method": "mean_ar", "coefficients": np.array([.9])}
    recon = suffix_with_imputer(waves, mean, std, mask, 7, choice)
    np.testing.assert_allclose(recon[:, :7], waves[:, :7])
    assert recon[0, 7] > .4
    np.testing.assert_allclose(recon[:, 9:], .4)
    altered = waves.copy()
    altered[:, 8:] = .99
    np.testing.assert_allclose(
        suffix_with_imputer(altered, mean, std, mask, 7, choice), recon,
    )


def test_mean_fallback_and_paired_cutoff():
    rng = np.random.default_rng(9)
    pair = rng.uniform(size=(3, 2, 30)).astype(np.float32)
    model = {"detectors": [{"mean": np.full(30,.25), "std":np.ones(30)},
                           {"mean": np.full(30,.75), "std":np.ones(30)}],
             "selected": {(d,10): {"method":"mean","order":0} for d in range(2)}}
    predicted = reconstruct_pair(pair, model, np.ones(30, bool), 10)
    np.testing.assert_allclose(predicted[:,:,:10], pair[:,:,:10])
    np.testing.assert_allclose(predicted[:,0,10:], .25)
    np.testing.assert_allclose(predicted[:,1,10:], .75)


def test_reproducible_centered_width_and_zero_bootstrap():
    predictions = np.array([[3.,4.,7.,8.],[1.,2.,3.,4.]])
    target = np.array([1.,2.,3.,4.])
    rows = summarize_predictions(predictions, target, seed=12, bootstrap=0)
    assert len(rows) == 2
    assert rows[-1]["centered_rmse_ps"] == 0.
    assert np.isfinite(rows[0]["centered_rmse_ps"])


def test_candidate_selection_falls_back_to_mean_on_constant_signals():
    data = np.full((40,2,90), .5, np.float32)
    selected = select_imputers(data, np.ones(90,bool), [15,30],
                               seed=7, fit_events=24, validation_events=8,
                               orders=(4,8), local_samples=10)
    assert all(c["method"] == "mean" for c in selected["selected"].values())
    assert set(selected["selected"]) == {(0,15),(1,15),(0,30),(1,30)}


def test_readable_cpa_path_and_collision_detection(tmp_path, monkeypatch):
    from waveform_analysis.engine import cpa_explainer as cpa
    from pathlib import Path
    batch = tmp_path / "results" / "UC" / "benchmark_48V_final"
    batch.mkdir(parents=True)
    monkeypatch.setattr(cpa, "batch_root", lambda run: batch)
    output = cpa._cpa_destination(
        tmp_path / "cpa_results", "UC", batch / "run", "energy", "onishi_cnn")
    assert output == (tmp_path / "cpa_results" / "UC" /
                      "benchmark_48V_final" / "energy" / "onishi_cnn")
    assert not cpa._guard_dataset_identity(output, "cohortA")
    output.mkdir(parents=True)
    (output / "summary.json").write_text('{"dataset_identity":"cohortA"}')
    assert cpa._guard_dataset_identity(output, "cohortA")
    import pytest
    with pytest.raises(ValueError, match="collision"):
        cpa._guard_dataset_identity(output, "cohortB")


def test_cpa_discovery_ignores_unrelated_missing_manifest(tmp_path, monkeypatch):
    import json
    from waveform_analysis.engine import cpa_explainer as cpa
    root = tmp_path / "results"
    good = root / "energy" / "wide" / "onishi_cnn"
    partial = root / "energy" / "wide" / "shared_multirocket_rdst"
    for directory, model in ((good, "onishi_cnn"),
                             (partial, "shared_multirocket_rdst")):
        metadata = directory / "metadata"
        metadata.mkdir(parents=True)
        (metadata / "config.json").write_text(json.dumps({"model": model}))
    (good / "metadata" / "manifest.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(cpa, "resolve_run_config", lambda p: {
        "mode": "energy_to_energy", "window_name": "wide",
    })
    monkeypatch.setattr(cpa, "read_run", lambda p: {
        "directory": p, "manifest": {"model": "onishi_cnn"},
        "best": {"parameters": {}},
    })
    found = cpa._cpa_run_inputs(root, None, models=["onishi_cnn"])
    assert [run["directory"] for run in found] == [good]
    # A partial but relevant run can still be discovered without manifest.
    found = cpa._cpa_run_inputs(root, None, models=["shared_multirocket_rdst"])
    assert len(found) == 1
    assert found[0]["manifest"]["model"] == "shared_multirocket_rdst"
