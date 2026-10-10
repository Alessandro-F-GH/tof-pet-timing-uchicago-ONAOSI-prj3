from waveform_analysis.reporting import postprocess


def test_result_only_reporting_forwards_exclusions(monkeypatch, tmp_path):
    calls = []

    def generate(root, **kwargs):
        calls.append((root, kwargs))
        return root / "report"

    monkeypatch.setattr(postprocess, "generate_report", generate)
    assert postprocess.remake_plots(tmp_path, exclude_models=["direct_mlp"]) == tmp_path / "report"
    assert calls == [(tmp_path.resolve(), {
        "logger": None, "reuse_numeric": True, "exclude_models": ["direct_mlp"],
    })]
