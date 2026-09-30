from __future__ import annotations
import copy,json,os,warnings
from dataclasses import dataclass
from pathlib import Path
from .common import canonical_hash

CHANNEL_MODES=("energy_to_energy","timing_to_timing")
class ConfigError(ValueError):pass

@dataclass(frozen=True)
class BatchConfig:
    name:str
    output_dir:str
    source_path:str
    protocol:dict
    axes:dict
    runs:tuple[dict,...]

def _read(path):
    p=Path(path)
    try:v=json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:raise ConfigError(f"Configuration file not found: {p}") from e
    except json.JSONDecodeError as e:raise ConfigError(f"Invalid JSON in {p}: {e}") from e
    if not isinstance(v,dict):raise ConfigError(f"{p} must contain a JSON object")
    return v

def _rel(owner,value):
    p=Path(value).expanduser();return (owner.parent/p).resolve() if not p.is_absolute() else p.resolve()

def _project(root,value):
    p=Path(value).expanduser();return str((root/p).resolve() if not p.is_absolute() else p.resolve())

def _module(path,key=None):
    raw=_read(path)
    if key and key in raw:return copy.deepcopy(raw[key])
    return raw

def _dataset(owner,value,root):
    d=_module(_rel(owner,value)) if isinstance(value,str) else copy.deepcopy(value)
    if not isinstance(d,dict):raise ConfigError("dataset must resolve to an object")
    if "root_file" not in d:raise ConfigError("each dataset must explicitly define root_file")
    d["root_file"]=_project(root,d["root_file"])
    if "true_tof_ps" not in d or "channels" not in d:raise ConfigError("dataset requires true_tof_ps and channels")
    return d

def _preprocessing(owner,value,root):
    p=_module(_rel(owner,value),"preprocessing") if isinstance(value,str) else copy.deepcopy(value)
    if not isinstance(p,dict):raise ConfigError("preprocessing_config must resolve to an object")
    if "cache_dir" not in p:p["cache_dir"]="processed_data/ml_protocol_v2"
    p["cache_dir"]=_project(root,p["cache_dir"]);return p

def _model(owner,raw,root):
    if isinstance(raw,str):name=raw;space=_read(root/"config"/"model_spaces"/f"{name}.json")
    elif isinstance(raw,dict):
        name=str(raw.get("name",""))
        if "space_config" in raw:space=_read(_rel(owner,raw["space_config"]))
        elif "space" in raw:space=copy.deepcopy(raw["space"])
        else:space=_read(root/"config"/"model_spaces"/f"{name}.json")
    else:raise ConfigError("model must be a name or object")
    if not name:name=str(space.get("model",""))
    if str(space.get("model",name))!=name:raise ConfigError("model-space name mismatch")
    return {"name":name,"space":space}

def _window(raw):
    if isinstance(raw,dict) and set(raw)>={"start","end"}:return {"start":float(raw["start"]),"end":float(raw["end"])}
    raise ConfigError("window must resolve to one {start,end} interval")

def _resampling(raw):
    r=copy.deepcopy(raw)
    if not isinstance(r,dict):raise ConfigError("resampling must be an object")
    if "n_replicas" not in r and "n_bootstrap" in r:
        r["n_replicas"]=r.pop("n_bootstrap")
        warnings.warn("resampling.n_bootstrap is deprecated; use n_replicas",DeprecationWarning,stacklevel=3)
    return r

def _fit(raw):
    f=copy.deepcopy(raw)
    if not isinstance(f,dict):raise ConfigError("fit must be an object")
    if "bootstrap_samples" in f:
        n=int(f.pop("bootstrap_samples"))
        if n:warnings.warn("fit.bootstrap_samples is ignored; uncertainty is estimated across replicas",RuntimeWarning,stacklevel=3)
    return f

def validate_config(c):
    for key in ("reference","analysis","preprocessing","mode","model","window_ns","resampling","fit","ml_input","ml_output","output_dir"):
        if key not in c:raise ConfigError(f"Missing {key}")
    if c["mode"] not in CHANNEL_MODES:raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
    if c["reference"]["channels"]!=c["analysis"]["channels"]:raise ConfigError("reference and analysis channel definitions must match")
    if Path(c["reference"]["root_file"]).resolve()==Path(c["analysis"]["root_file"]).resolve():warnings.warn("Reference and analysis datasets point to the same file. This is allowed for pipeline testing only; use independent datasets for unbiased production studies.",RuntimeWarning,stacklevel=2)
    if c["mode"]=="timing_to_timing" and not c["analysis"]["channels"].get("timing"):raise ConfigError("timing mode requires timing channels")
    p=c["preprocessing"]
    for key in ("materialized_window_ns","energy","timing","selection","photopeak","tot_peak","led_selection","io"):
        if key not in p:raise ConfigError(f"preprocessing.{key} is required")
    clip=p["selection"].get("baseline_clipping")
    if not isinstance(clip,dict) or "margin_mV" not in clip or float(clip["margin_mV"])<0:raise ConfigError("selection.baseline_clipping.margin_mV must be non-negative")
    noise=p["selection"]["baseline_noise"]
    if len(noise["window_ns"])!=2 or float(noise["window_ns"][1])>0:raise ConfigError("baseline window must lie before trigger")
    led=p["led_selection"]
    if not led.get("thresholds_mV"):raise ConfigError("led_selection.thresholds_mV must be non-empty")
    if not 0<float(led["minimum_crossing_efficiency"])<=1:raise ConfigError("invalid LED minimum crossing efficiency")
    if float(led["coincidence_window_ns"])<=0:raise ConfigError("invalid LED coincidence window")
    r=c["resampling"]
    if str(r.get("policy","repeated_holdout"))!="repeated_holdout":raise ConfigError("only repeated_holdout is implemented")
    if "seed" not in r:raise ConfigError("resampling.seed is required")
    int(r["seed"])
    if int(r.get("n_replicas",0))<1:raise ConfigError("resampling.n_replicas must be >=1")
    vf,tf=float(r["validation_fraction"]),float(r["test_fraction"])
    if vf<=0 or tf<=0 or vf+tf>=1:raise ConfigError("validation/test fractions must be positive and sum to <1")
    if int(r["minimum_events_per_split"])<1:raise ConfigError("resampling.minimum_events_per_split must be >=1")
    if c["window_ns"]["end"]<=c["window_ns"]["start"]:raise ConfigError("window end must exceed start")
    if int(c["ml_input"].get("subsampling",1))<=0:raise ConfigError("ml_input.subsampling must be positive")
    if float(c["ml_output"]["max_abs_ps"])<=0:raise ConfigError("ml_output.max_abs_ps must be positive")
    if "bootstrap_samples" in c["fit"]:raise ConfigError("fit.bootstrap_samples is not supported; use replicas")
    if float(c["fit"]["histogram_bin_width_ps"])<=0:raise ConfigError("invalid fit settings")
    from .models import model_names
    if c["model"]["name"] not in model_names():raise ConfigError(f"unregistered model {c['model']['name']}")

def _resolve(source,raw,root):
    legacy={"extends","data_config","experiment","models","standard_methods","validation","cfd"}
    if legacy & set(raw):raise ConfigError("Old experiment schema is incompatible with the control/analysis study protocol")
    reference=_dataset(source,raw.get("reference_dataset"),root);analysis=_dataset(source,raw.get("analysis_dataset"),root)
    preprocessing=_preprocessing(source,raw.get("preprocessing_config"),root);model=_model(source,raw.get("model"),root)
    if "window" in raw:window=_window(raw["window"])
    elif "window_name" in raw and isinstance(raw.get("windows"),dict):
        try:window=_window(raw["windows"][raw["window_name"]])
        except KeyError as e:raise ConfigError("window_name not found in windows") from e
    else:raise ConfigError("study requires one window or window_name")
    c={"name":str(raw.get("name",source.stem)),"reference":reference,"analysis":analysis,"preprocessing":preprocessing,
       "mode":str(raw["mode"]),"model":model,"window_ns":window,"resampling":_resampling(raw["resampling"]),"fit":_fit(raw["fit"]),
       "ml_input":copy.deepcopy(raw.get("ml_input",{"subsampling":1})),"ml_output":copy.deepcopy(raw["ml_output"]),"output_dir":_project(root,raw["output_dir"])}
    for key in ("study_name","run_id","window_name"):
        if key in raw:c[key]=str(raw[key])
    validate_config(c);c["_config_path"]=str(source);c["_config_fingerprint"]=canonical_hash(c);return c

def load_config(path,project_root=None,defaults=None):
    source=Path(path).expanduser().resolve();root=Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1];raw=_read(source)
    if defaults:merged=copy.deepcopy(defaults);merged.update(raw);raw=merged
    return _resolve(source,raw,root)

def _mode_tag(mode):return "energy" if mode=="energy_to_energy" else "timing"

def _excluded(model,mode,window,rules):
    for rule in rules:
        if not isinstance(rule,dict):raise ConfigError("sweep.exclude entries must be objects")
        if all(rule.get(k,v)==v for k,v in (("model",model),("mode",mode),("window",window))):return True
    return False

def _protocol_value(protocol,key,mode):
    value=protocol[key]
    return copy.deepcopy(value[mode]) if isinstance(value,dict) and mode in value else copy.deepcopy(value)

def _compact_batch(source,raw,root):
    for key in ("name","reference_dataset","analysis_dataset","output_dir","protocol","sweep"):
        if key not in raw:raise ConfigError(f"compact batch config requires {key}")
    protocol=copy.deepcopy(raw["protocol"]);sweep=copy.deepcopy(raw["sweep"])
    for key in ("preprocessing_config","resampling","fit","ml_output"):
        if key not in protocol:raise ConfigError(f"protocol.{key} is required")
    models=sweep.get("models");modes=sweep.get("modes");windows=sweep.get("windows");rules=sweep.get("exclude",[])
    if not isinstance(models,list) or not models:raise ConfigError("sweep.models must be a non-empty list")
    if not isinstance(modes,list) or not modes:raise ConfigError("sweep.modes must be a non-empty list")
    if not isinstance(windows,dict) or not windows:raise ConfigError("sweep.windows must be a non-empty object")
    if not isinstance(rules,list):raise ConfigError("sweep.exclude must be a list")
    name=str(raw["name"]);out=Path(_project(root,raw["output_dir"]));runs=[]
    for model_raw in models:
        model=model_raw if isinstance(model_raw,str) else str(model_raw.get("name",""))
        if not model:raise ConfigError("every sweep model needs a name")
        for mode in modes:
            mode=str(mode)
            if mode not in CHANNEL_MODES:raise ConfigError(f"mode must be one of {CHANNEL_MODES}")
            for window_name,window in windows.items():
                if _excluded(model,mode,str(window_name),rules):continue
                run_id=f"{model}__{_mode_tag(mode)}__{window_name}"
                item={"name":f"{name}__{run_id}","study_name":name,"run_id":run_id,"window_name":str(window_name),
                      "reference_dataset":raw["reference_dataset"],"analysis_dataset":raw["analysis_dataset"],"preprocessing_config":protocol["preprocessing_config"],
                      "model":model_raw,"mode":mode,"window":window,"resampling":protocol["resampling"],"fit":_protocol_value(protocol,"fit",mode),
                      "ml_input":_protocol_value(protocol,"ml_input",mode) if "ml_input" in protocol else {"subsampling":1},
                      "ml_output":_protocol_value(protocol,"ml_output",mode),"output_dir":str(out/model/_mode_tag(mode)/str(window_name))}
                runs.append(_resolve(source,item,root))
    if not runs:raise ConfigError("batch sweep produced no runs")
    axes={"models":[m if isinstance(m,str) else str(m.get("name")) for m in models],"modes":[str(m) for m in modes],"windows":{str(k):_window(v) for k,v in windows.items()}}
    normalized={"preprocessing_config":str(protocol["preprocessing_config"]),"resampling":_resampling(protocol["resampling"]),"fit":copy.deepcopy(protocol["fit"]),
                "ml_input":copy.deepcopy(protocol.get("ml_input",{"subsampling":1})),"ml_output":copy.deepcopy(protocol["ml_output"])}
    return BatchConfig(name,str(out.resolve()),str(source),normalized,axes,tuple(runs))

def _legacy_batch(source,raw,root):
    studies=raw.get("studies")
    if not isinstance(studies,list) or not studies:raise ConfigError("batch config requires a non-empty ordered studies list")
    if "reference_dataset" not in raw or "analysis_dataset" not in raw:raise ConfigError("batch config must define shared reference_dataset and analysis_dataset")
    shared={"reference_dataset":_dataset(source,raw["reference_dataset"],root),"analysis_dataset":_dataset(source,raw["analysis_dataset"],root)}
    runs=tuple(load_config(_rel(source,item),root,defaults=shared) for item in studies);out=os.path.commonpath([c["output_dir"] for c in runs])
    return BatchConfig(str(raw.get("name",source.stem)),str(Path(out).resolve()),str(source),{}, {},runs)

def load_batch_config(path,project_root=None):
    source=Path(path).expanduser().resolve();root=Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1];raw=_read(source)
    return _compact_batch(source,raw,root) if "sweep" in raw or "protocol" in raw else _legacy_batch(source,raw,root)

def public_config(c):return {k:v for k,v in c.items() if not str(k).startswith("_")}
def public_batch_config(batch):return {"name":batch.name,"output_dir":batch.output_dir,"source_path":batch.source_path,"protocol":batch.protocol,"axes":batch.axes,"runs":[public_config(c) for c in batch.runs]}
