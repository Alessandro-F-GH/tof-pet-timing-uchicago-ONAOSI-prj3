from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
from .common import atomic_json,canonical_hash,channel_limits,write_csv
from .dataset import DATASET_FORMAT_VERSION,load_prepared_dataset
from .timing import led_grid,anchor_grid,pair_delta
from .view import mode_family

def _input_offsets(interval_s,window_ns,subsampling):
    dt=float(interval_s)*1e9
    first=int(np.ceil(float(window_ns["start"])/dt-1e-9));last=int(np.floor(float(window_ns["end"])/dt+1e-9))
    off=np.arange(first,last+1,dtype=np.int64)[::int(subsampling)]
    if off.size<2:raise ValueError("ML input window has fewer than two samples")
    return off,off.astype(float)*dt*1000.0

def _control_mode_fingerprint(control_artifact,mode):
    return str((control_artifact.get("mode_fingerprints") or {}).get(mode,control_artifact["fingerprint"]))

def dataset_fingerprint(preprocessed,control_artifact,config):
    mode=config["mode"]
    return canonical_hash({"format_version":DATASET_FORMAT_VERSION,"preprocessed":preprocessed.manifest["fingerprint"],"control_mode":_control_mode_fingerprint(control_artifact,mode),"mode":mode,"window_ns":config["window_ns"],"subsampling":config["ml_input"]["subsampling"],"normalization":config["preprocessing"][mode_family(mode)]["vertical_scale_limit_mV"]})

def prepare_ml_dataset(preprocessed,control_artifact,config,*,cache_dir,rebuild=False,logger=None):
    fp=dataset_fingerprint(preprocessed,control_artifact,config);base=Path(cache_dir).resolve()/"prepared"/fp[:16]
    if base.is_dir() and not rebuild:
        try:
            d=load_prepared_dataset(base)
            if d.manifest.get("fingerprint")==fp:return d
        except Exception:pass
    if base.exists():shutil.rmtree(base)
    base.mkdir(parents=True)
    family=mode_family(config["mode"]);waves=preprocessed.energy_windows_mV if family=="energy" else preprocessed.timing_windows_mV
    intervals=preprocessed.energy_sample_interval_s if family=="energy" else preprocessed.timing_sample_interval_s
    if waves is None or intervals is None:raise ValueError(f"{family} waveforms unavailable")
    threshold=float(control_artifact["selected_led_threshold_mV"][config["mode"]]);baseline=config["preprocessing"]["selection"]["baseline_noise"]["window_ns"]
    led=led_grid(preprocessed,family,np.arange(preprocessed.n_events),np.asarray([threshold]),baseline_window_ns=baseline)[:,:,0]
    true=float(config["analysis"]["true_tof_ps"]);window_ps=1000.0*float(config["preprocessing"]["led_selection"]["coincidence_window_ns"])
    residual=pair_delta(led)-true;finite_led=np.all(np.isfinite(led),axis=1)&np.isfinite(residual);coincidence=finite_led&(np.abs(residual)<=window_ps)
    anchors=anchor_grid(preprocessed,family,threshold,baseline_window_ns=baseline);ref=float(np.asarray(intervals)[0,0])
    if not np.allclose(np.asarray(intervals),ref,rtol=1e-9,atol=0):raise ValueError(f"{family} sampling interval must be common")
    offsets,time_ps=_input_offsets(ref,config["window_ns"],config["ml_input"]["subsampling"]);window_valid=np.all((anchors+int(offsets[0])>=0)&(anchors+int(offsets[-1])<waves.shape[2]),axis=1)
    rows=np.flatnonzero(coincidence & window_valid)
    if not rows.size:raise RuntimeError("No events remain after frozen selection, fixed LED coincidence, and waveform-window availability")
    dropped_window=int(np.count_nonzero(coincidence & ~window_valid));out=np.empty((rows.size,2,offsets.size),np.float32)
    for oi,event in enumerate(rows):
        for d in range(2):out[oi,d]=np.asarray(waves[event,d,anchors[event,d]+offsets],np.float32)
    limits=channel_limits(config["preprocessing"][family]["vertical_scale_limit_mV"]);minimum=limits[:,0,None].astype(np.float32);maximum=limits[:,1,None].astype(np.float32)
    normalized=((out-minimum[None,:,:])/(maximum-minimum)[None,:,:]).astype(np.float32)
    np.save(base/f"{family}_windows.npy",normalized);np.save(base/f"{family}_time_ps.npy",time_ps);np.savez(base/f"{family}_transform.npz",minimum=minimum,maximum=maximum)
    np.save(base/f"{family}_led_time_ps.npy",np.asarray(led[rows],np.float64));np.save(base/f"{family}_target_ps.npy",np.asarray(residual[rows],np.float64))
    event_index=np.asarray(preprocessed.event_index)[rows];np.save(base/"event_index.npy",event_index);np.save(base/"bias_voltage_V.npy",np.asarray(preprocessed.bias_voltage_V)[rows])
    event_population_identity=canonical_hash({"source":preprocessed.manifest["source"],"mode":config["mode"],"window_ns":config["window_ns"],"event_index":event_index.tolist()})
    analysis_protocol_identity=canonical_hash({"prepared_fingerprint":fp,"event_population_identity":event_population_identity})
    manifest={"format_version":DATASET_FORMAT_VERSION,"fingerprint":fp,"analysis_source":preprocessed.manifest["source"],"event_population_identity":event_population_identity,"analysis_protocol_identity":analysis_protocol_identity,"analysis_population_identity":analysis_protocol_identity,"control_fingerprint":control_artifact["fingerprint"],"control_mode_fingerprint":_control_mode_fingerprint(control_artifact,config["mode"]),"mode":config["mode"],"family":family,"fixed_led_threshold_mV":threshold,"true_tof_ps":true,"window_ns":config["window_ns"],"subsampling":int(config["ml_input"]["subsampling"]),"n_before_fixed_led":int(preprocessed.n_events),"n_after_fixed_led":int(coincidence.sum()),"n_dropped_window":dropped_window,"n_final":int(rows.size),"population_scope":"dataset+mode+window","protocol_scope":"population+control+preprocessing+normalization+subsampling","target_definition":"fixed_LED_delta_t - true_tof (no analysis-wide calibration fitted before splitting)"}
    atomic_json(base/"manifest.json",manifest)
    write_csv(base/"preprocessing_summary.csv",[
        {"stage":"native_preprocessed","remaining":int(preprocessed.n_events),"fraction":1.0},
        {"stage":"fixed_led_crossing","remaining":int(finite_led.sum()),"fraction":float(finite_led.mean())},
        {"stage":"fixed_led_coincidence","remaining":int(coincidence.sum()),"fraction":float(coincidence.mean())},
        {"stage":"window_available","remaining":int(rows.size),"fraction":float(rows.size)/max(1,int(preprocessed.n_events))},
        {"stage":"final_ml_population","remaining":int(rows.size),"fraction":float(rows.size)/max(1,int(preprocessed.n_events))},
    ])
    if logger:logger.info("Prepared ML population | n=%d | LED=%.6g mV | window_dropped=%d | protocol=%s",rows.size,threshold,dropped_window,analysis_protocol_identity[:12])
    return load_prepared_dataset(base)
