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


def test_timing_cutoffs_restart_after_each_masked_region():
    from waveform_analysis.engine.cpa_explainer import (
        timing_cutoff_schedule_ns, masked_time_intervals, _plot_metrics,
    )
    t = np.arange(-2., 41., .1) * 1000
    mask = np.ones(len(t), bool)
    mask[(t >= 500.) & (t < 2500.)] = False
    mask[(t >= 17500.) & (t < 19500.)] = False
    cutoffs, omitted = cutoff_samples(
        t, mask, timing_cutoff_schedule_ns(t, mask))
    ns = t[cutoffs]/1000
    # Segment starts at 2.5 and 19.5 ns; requested offsets are relative.
    for expected in (3.5,5.5,7.5,9.5,11.5,16.5,20.5,22.5,24.5,26.5,28.5,33.5):
        assert np.any(np.isclose(ns, expected))
    assert np.all(mask[cutoffs])
    # No legacy absolute-grid cutoffs in post-mask segments.
    assert not np.any(np.isclose(ns, 10.))
    assert not np.any(np.isclose(ns, 15.))
    assert not np.any(np.isclose(ns, 20.))
    assert not np.any((ns >= .5) & (ns < 2.5))
    assert not np.any((ns >= 17.5) & (ns < 19.5))
    intervals = masked_time_intervals(t, mask)
    assert len(intervals) == 2


def test_cpa_performance_plot_uses_two_axes_and_mask_shading(tmp_path, monkeypatch):
    from waveform_analysis.engine import cpa_explainer as cpa
    import matplotlib.axes
    labels = []
    original = matplotlib.axes.Axes.set_ylabel

    def collect_label(self, label, *args, **kwargs):
        labels.append(label)
        return original(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_ylabel", collect_label)
    rows = []
    for cutoff, kind in ((0.0, "prefix"), (4.0, "prefix"), (20.0, "full")):
        rows.append({
            "cutoff_ns": cutoff, "kind": kind,
            "centered_rmse_ps": 50., "centered_rmse_ci_low_ps": 48.,
            "centered_rmse_ci_high_ps": 52., "ctr_ps": 75.,
            "ctr_ci_low_ps": 70., "ctr_ci_high_ps": 80.,
        })
    path = tmp_path / "timing_resolution.png"
    cpa._plot_metrics(path, rows, "test",
                      masked_intervals=[(1., 2.)])
    assert path.stat().st_size > 100
    assert len(labels) == 2
    assert "RMSE" in labels[0] and "CTR" in labels[1]


def test_timing_cutoffs_do_not_mix_absolute_and_relative_grids():
    from waveform_analysis.engine.cpa_explainer import timing_cutoff_schedule_ns
    t = np.arange(-2., 50., .025) * 1000
    mask = np.ones(len(t), bool)
    mask[(t >= .2*1000) & (t < 7.925*1000)] = False
    schedule = timing_cutoff_schedule_ns(t, mask)
    cutoffs, _ = cutoff_samples(t, mask, schedule)
    actual = t[cutoffs]/1000
    for expected in (-1., 0., 8.925, 10.925, 12.925,
                     14.925, 16.925, 21.925, 26.925):
        assert np.any(np.isclose(actual, expected, atol=.0001))
    for forbidden in (2., 4., 6., 8., 10., 15., 20., 25., 30., 35., 40., 45.):
        assert not np.any(np.isclose(actual, forbidden, atol=.0001))


def test_draw_all_blind_events_with_zero_limit():
    from waveform_analysis.engine.cpa_explainer import _draw_indices
    ids = _draw_indices(32, 0, 123)
    np.testing.assert_array_equal(ids, np.arange(32))
    np.testing.assert_array_equal(_draw_indices(32, 32, 123), ids)


def test_cpa_multi_board_cli_jobs(tmp_path):
    from waveform_analysis.cli import _parser, _cpa_jobs
    results = tmp_path / "results"
    uc_cache = tmp_path / "uc_cache"
    fbk_cache = tmp_path / "fbk_cache"
    for p in (results / "UC", results / "FBK", uc_cache, fbk_cache):
        p.mkdir(parents=True)
    cmd = ["explain-cpa", "--results", str(results),
           "--boards", "UC", "FBK", "--uc-cache-root", str(uc_cache),
           "--fbk-cache-root", str(fbk_cache), "--max-events", "0"]
    args = _parser().parse_args(cmd)
    jobs = _cpa_jobs(args)
    assert [(board, root) for _, _, board, root in jobs] == [
        ("UC", uc_cache.resolve()), ("FBK", fbk_cache.resolve())]
    assert [result for result, _, _, _ in jobs] == [
        (results / "UC").resolve(), (results / "FBK").resolve()]
    assert args.max_events == 0


def test_cpa_multi_board_rejects_missing_cache_and_preserves_single_board(tmp_path):
    from waveform_analysis.cli import _parser, _cpa_jobs
    import pytest
    results = tmp_path / "results"
    (results / "UC").mkdir(parents=True)
    args = _parser().parse_args(["explain-cpa", "--results", str(results),
                                "--boards", "UC"])
    with pytest.raises(ValueError, match="uc-cache-root"):
        _cpa_jobs(args)
    args = _parser().parse_args(["explain-cpa", "--results", str(results / "UC"),
                                "--board", "UC", "--cache-root", str(tmp_path)])
    assert _cpa_jobs(args) == [(results / "UC", None, "UC", tmp_path)]


def test_cpa_energy_kernel_defaults_only():
    from waveform_analysis.engine.cpa_explainer import DEFAULT_MODELS
    assert "direct_multirocket" in DEFAULT_MODELS["energy"]
    assert "direct_rdst" in DEFAULT_MODELS["energy"]
    assert "direct_multirocket_rdst" in DEFAULT_MODELS["energy"]
    assert "direct_rdst" not in DEFAULT_MODELS["timing"]


def test_cpa_shared_feature_blocks_and_incompatible_fallback(monkeypatch):
    from types import SimpleNamespace as Obj
    from waveform_analysis.engine.cpa_feature_sharing import (
        compatible_component, predict_shared_energy_group,
    )

    class FakeTransformer:
        def __init__(self, n):
            self.n = n
            self.parameter = (np.array([1., 2.]),)
            self.parameter1 = (np.array([3., 4.]),)
        def get_params(self, deep=False):
            return {"n_kernels": self.n, "random_state": 42}
    class Scaler:
        scale_ = np.array([1.])
        mean_ = np.array([0.])
        var_ = np.array([1.])
        n_features_in_ = 1

    mr = Obj(transformer=FakeTransformer(4), scaler=Scaler(), shared=False,
             metadata={"input_samples": 4, "normalise": False,
                       "feature_count": 2})
    same = Obj(transformer=FakeTransformer(4), scaler=Scaler(), shared=False,
               metadata=dict(mr.metadata))
    different = Obj(transformer=FakeTransformer(5), scaler=Scaler(), shared=False,
                    metadata=dict(mr.metadata))
    mask = np.ones(4, bool)
    def fitted(transform):
        return Obj(sample_mask=mask, output_max_abs_ps=1000.,
                   feature_transform=Obj(artifact=transform))
    combined = fitted(Obj(multirocket=mr,
                          metadata={"batch_size": 8,
                                    "feature_blocks": {"multirocket":[0,2],
                                                       "rdst":[2,4]}}))
    standalone = fitted(same)
    assert compatible_component(combined, standalone, "multirocket")
    assert not compatible_component(combined, fitted(different), "multirocket")
    # Grouped inference evaluates one combined transform, sharing its first
    # two feature columns with the standalone Ridge.
    calls = []
    combined.feature_transform.apply = lambda x: (
        calls.append(len(x)) or np.ones((len(x),4), np.float32))
    spec = Obj(predict=lambda model, features: np.sum(features, axis=1))
    group = {
        "direct_multirocket_rdst": (spec, combined, None),
        "direct_multirocket": (spec, standalone, "multirocket"),
    }
    pred = predict_shared_energy_group(
        group, np.ones((5,2,4), np.float32), chunk_size=3)
    assert calls == [3, 2]
    np.testing.assert_allclose(pred["direct_multirocket_rdst"], 4.)
    np.testing.assert_allclose(pred["direct_multirocket"], 2.)


def test_cpa_component_compatibility_diagnostic_reports_reason():
    from types import SimpleNamespace as Obj
    from waveform_analysis.engine.cpa_feature_sharing import component_compatibility

    class Transformer:
        parameter = (np.array([1.,2.]),)
        parameter1 = (np.array([3.,4.]),)
        def get_params(self, deep=False):
            return {"n_kernels": 42, "random_state": 3, "n_jobs": 1}
    class Scaler:
        def __init__(self, scale):
            self.scale_ = np.array([scale])
            self.mean_ = np.array([0.])
            self.var_ = np.array([scale**2])
            self.n_features_in_ = 1

    def component(scale):
        return Obj(transformer=Transformer(), scaler=Scaler(scale), shared=False,
                   metadata={"input_samples": 10, "normalise": False,
                             "feature_count": 2})
    combined = Obj(feature_transform=Obj(artifact=Obj(multirocket=component(1.))))
    same = Obj(feature_transform=Obj(artifact=component(1.)))
    different = Obj(feature_transform=Obj(artifact=component(2.)))
    ok, reason = component_compatibility(combined, same, "multirocket")
    assert ok and "match" in reason
    ok, reason = component_compatibility(combined, different, "multirocket")
    assert not ok and "scaler" in reason
