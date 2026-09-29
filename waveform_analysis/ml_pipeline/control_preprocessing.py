from __future__ import annotations
import json,shutil
from pathlib import Path
import numpy as np
from .common import atomic_json,canonical_hash,source_signature
from .preprocessing_plots import plot_led_selection
from .event_selection import fit_selection_rules,apply_selection_rules
from .data import preprocess_selected
from .timing import led_grid,pair_delta
from .stats import ctr_estimate

CONTROL_ARTIFACT_VERSION=5

def _artifact_fingerprint(reference_file,reference_dataset,preprocessing):
    return canonical_hash({"format_version":CONTROL_ARTIFACT_VERSION,"reference":source_signature(reference_file),
        "dataset":reference_dataset,"preprocessing":preprocessing})

def _supported_modes(dataset):
    out=["energy_to_energy"]
    if (dataset.get("channels") or {}).get("timing"):out.append("timing_to_timing")
    return tuple(out)

def _family(mode):return "energy" if mode=="energy_to_energy" else "timing"

def _select_led(preprocessed,mode,reference_dataset,preprocessing,fit_cfg):
    family=_family(mode); candidates=np.asarray(preprocessing["led_selection"]["thresholds_mV"],float)
    baseline_window=preprocessing["selection"]["baseline_noise"]["window_ns"]
    grid=led_grid(preprocessed,family,np.arange(preprocessed.n_events),candidates,baseline_window_ns=baseline_window)
    true=float(reference_dataset["true_tof_ps"]);window_ps=1000.0*float(preprocessing["led_selection"]["coincidence_window_ns"])
    min_eff=float(preprocessing["led_selection"]["minimum_crossing_efficiency"]);rows=[]
    for i,thr in enumerate(candidates):
        residual=pair_delta(grid[:,:,i])-true;valid=np.isfinite(residual)&(np.abs(residual)<=window_ps)
        n=int(valid.sum());eff=n/max(1,residual.size)
        ctr=float("inf")
        if n and eff>=min_eff:
            ctr=float(ctr_estimate(residual[valid],fit_cfg,bootstrap=False).ctr_ps)
        rows.append({"threshold_mV":float(thr),"ctr_ps":ctr,"n":n,"efficiency":eff})
    finite=[r for r in rows if np.isfinite(r["ctr_ps"])]
    if not finite:raise RuntimeError(f"No LED candidate satisfies control selection for {mode}")
    best=min(finite,key=lambda r:(r["ctr_ps"],r["threshold_mV"]))
    return float(best["threshold_mV"]),rows

def load_control_artifact(directory,reference_file,reference_dataset,preprocessing):
    directory=Path(directory).resolve();m=json.loads((directory/"manifest.json").read_text())
    if m.get("fingerprint")!=_artifact_fingerprint(reference_file,reference_dataset,preprocessing):
        raise ValueError("control preprocessing artifact is stale")
    return m

def fit_control_artifact(reference_file,reference_dataset,preprocessing,fit_cfg,*,cache_root,rebuild=False,logger=None):
    fp=_artifact_fingerprint(reference_file,reference_dataset,preprocessing)
    directory=Path(cache_root).resolve()/"control"/fp[:16]
    if directory.is_dir() and not rebuild:
        try:
            m=load_control_artifact(directory,reference_file,reference_dataset,preprocessing)
            if logger:logger.info("Control preprocessing loaded from cache | %s",directory)
            return m,directory
        except (ValueError,FileNotFoundError):pass
    if directory.exists():shutil.rmtree(directory)
    directory.mkdir(parents=True)
    rules=fit_selection_rules(reference_file,reference_dataset,preprocessing,output_dir=directory/"diagnostics",logger=logger)
    selected={};led={}
    for mode in _supported_modes(reference_dataset):
        selection=apply_selection_rules(reference_file,reference_dataset,preprocessing,rules,mode,
            cache_dir=Path(cache_root)/"control_selection",rebuild=rebuild,logger=logger)
        prepared=preprocess_selected(reference_file,selection,reference_dataset,preprocessing,mode,
            cache_dir=Path(cache_root)/"control_native",rebuild=rebuild,logger=logger)
        threshold,scan=_select_led(prepared,mode,reference_dataset,preprocessing,fit_cfg)
        selected[mode]=threshold;led[mode]={"selected_threshold_mV":threshold,"candidates":scan}
        plot_led_selection(scan,threshold,directory/"diagnostics"/f"{mode}_led_selection.png",f"Reference LED selection: {mode}")
        if logger:logger.info("Control LED selected | %s | %.6g mV",mode,threshold)
    manifest={"format_version":CONTROL_ARTIFACT_VERSION,"fingerprint":fp,"reference_dataset":reference_dataset,
        "reference_source":str(Path(reference_file).resolve()),"preprocessing_fingerprint":canonical_hash(preprocessing),
        "selection_rules":rules,"selected_led_threshold_mV":selected,"led_selection":led}
    atomic_json(directory/"manifest.json",manifest)
    return manifest,directory
