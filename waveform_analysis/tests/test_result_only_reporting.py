from waveform_analysis.ml_pipeline import postprocess
def test_result_only_reporting(monkeypatch,tmp_path):
    root=tmp_path/"results";root.mkdir();monkeypatch.setattr(postprocess,"load_plot_config",lambda p:{"x":1});monkeypatch.setattr(postprocess,"collect_runs",lambda p:[{"directory":root/"timing"}]);calls=[];monkeypatch.setattr(postprocess,"render_run_plots",lambda d,c:calls.append((d,c)));monkeypatch.setattr(postprocess,"generate_report",lambda p,logger=None,reuse_numeric=False:root/"report");assert postprocess.remake_plots(root)==root/"report";assert len(calls)==1
