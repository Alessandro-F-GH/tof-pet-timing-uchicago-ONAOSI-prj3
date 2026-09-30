from __future__ import annotations

import shutil
from pathlib import Path

from .common import atomic_json, canonical_hash, write_csv
from .config import BatchConfig
from .study import run_study


def _log_gpu_status(logger):
    if logger is None:return
    try:import torch
    except Exception as exc:logger.warning("GPU check | PyTorch unavailable | %s",exc);return
    if not torch.cuda.is_available():logger.info("GPU check | CUDA unavailable | PyTorch=%s | using CPU",torch.__version__);return
    try:
        device=torch.device("cuda:0");probe=torch.ones(8,device=device);result=(probe*2.0).sum();torch.cuda.synchronize(device)
        if float(result.item())!=16.0:raise RuntimeError("unexpected CUDA smoke-test result")
        props=torch.cuda.get_device_properties(device);logger.info("GPU check | CUDA OK | device=%s | capability=%d.%d | memory=%.1f GiB | PyTorch=%s | CUDA=%s",torch.cuda.get_device_name(device),props.major,props.minor,props.total_memory/(1024**3),torch.__version__,torch.version.cuda)
    except Exception as exc:logger.warning("GPU check | CUDA detected but smoke test failed | %s",exc)


def _run_index(batch: BatchConfig):
    root=Path(batch.output_dir).resolve();rows=[]
    for cfg in batch.runs:
        path=Path(cfg["output_dir"]).resolve()
        try:relative=str(path.relative_to(root))
        except ValueError:relative=str(path)
        rows.append({"run_id":cfg.get("run_id",cfg["name"]),"name":cfg["name"],"model":cfg["model"]["name"],"mode":cfg["mode"],"window":cfg.get("window_name",""),"window_start_ns":cfg["window_ns"]["start"],"window_end_ns":cfg["window_ns"]["end"],"status":"pending","path":relative,"config_fingerprint":cfg["_config_fingerprint"]})
    return rows


def _write_batch_state(batch,rows,status):
    root=Path(batch.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    manifest={"schema_version":2,"name":batch.name,"status":status,"source_config":batch.source_path,"protocol":batch.protocol,"axes":batch.axes,"comparison_unit":"repeated_holdout_replica","fit_bootstrap":False,"pairing_rule":"same analysis protocol + mode + window + replica seed","shared_artifact_root":str((root/"artifacts").resolve()),"runs":rows}
    atomic_json(root/"manifest.json",manifest);write_csv(root/"runs.csv",rows)


def _control_protocol(configs):
    by_mode={}
    for cfg in configs:
        mode=str(cfg["mode"]);fit=dict(cfg["fit"])
        if mode in by_mode and canonical_hash(by_mode[mode])!=canonical_hash(fit):raise ValueError(f"All runs in one batch must use one control CTR fit definition per mode; mismatch for {mode}")
        by_mode[mode]=fit
    return by_mode,tuple(sorted(by_mode))


def _runtime_configs(configs,batch_root,rebuild_preprocessing=False):
    fit_by_mode,modes=_control_protocol(configs);artifact_root=str((Path(batch_root).resolve()/"artifacts").resolve());out=[];seen_modes=set();seen_populations=set()
    for index,cfg in enumerate(configs):
        runtime=dict(cfg);runtime["_batch_artifact_root"]=artifact_root;runtime["_control_fit_by_mode"]={k:dict(v) for k,v in fit_by_mode.items()};runtime["_control_modes"]=modes
        mode=str(cfg["mode"]);population_key=(mode,float(cfg["window_ns"]["start"]),float(cfg["window_ns"]["end"]),canonical_hash(cfg.get("ml_input",{})))
        runtime["_rebuild_control"]=bool(rebuild_preprocessing and index==0);runtime["_rebuild_analysis"]=bool(rebuild_preprocessing and mode not in seen_modes);runtime["_rebuild_prepared"]=bool(rebuild_preprocessing and population_key not in seen_populations)
        seen_modes.add(mode);seen_populations.add(population_key);out.append(runtime)
    return out


def run_batch(batch,*,overwrite=False,resume=False,rebuild_preprocessing=False,logger=None):
    if isinstance(batch,BatchConfig):
        original=list(batch.runs);compact=bool(batch.protocol or batch.axes);configs=_runtime_configs(original,batch.output_dir,rebuild_preprocessing=rebuild_preprocessing) if compact else original
    else:configs=list(batch);batch=None;compact=False
    outputs=[];total=len(configs);_log_gpu_status(logger);rows=_run_index(batch) if batch is not None else None
    if batch is not None and compact:
        root=Path(batch.output_dir).resolve()
        if overwrite and root.exists():shutil.rmtree(root)
        _write_batch_state(batch,rows,"running")
    for i,cfg in enumerate(configs,1):
        if logger:logger.info("Batch run %d/%d | %s",i,total,cfg["name"])
        try:
            out=run_study(cfg,overwrite=overwrite,resume=resume,rebuild_preprocessing=False if compact else rebuild_preprocessing);outputs.append(out)
            if rows is not None:
                rows[i-1]["status"]="complete"
                if compact:_write_batch_state(batch,rows,"running")
            if logger:logger.info("Batch completed | %s | %s",cfg["name"],out)
        except Exception:
            if rows is not None:
                rows[i-1]["status"]="failed"
                if compact:_write_batch_state(batch,rows,"failed")
            if logger:logger.exception("Batch failed | %s",cfg["name"])
            raise
    if batch is not None and compact:_write_batch_state(batch,rows,"complete")
    return outputs
