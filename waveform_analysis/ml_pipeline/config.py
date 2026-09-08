from __future__ import annotations

import copy,json
from pathlib import Path
from typing import Any
from .common import canonical_hash

CHANNEL_MODES=("energy_to_energy","energy_to_timing","timing_to_timing")
class ConfigError(ValueError): pass

def _read(path:Path)->dict[str,Any]:
    try:value=json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc: raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc: raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value,dict): raise ConfigError(f"Configuration {path} must contain an object")
    return value

def merge(base,override):
    result=copy.deepcopy(base)
    for key,value in override.items(): result[key]=merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else copy.deepcopy(value)
    return result

def _relative(owner:Path,value):
    path=Path(value).expanduser(); return (owner.parent/path).resolve() if not path.is_absolute() else path.resolve()
def _resolve(path:Path,stack:tuple[Path,...]=()):
    path=path.resolve()
    if path in stack: raise ConfigError("Configuration include cycle: "+" -> ".join(map(str,(*stack,path))))
    raw=_read(path); result={}; refs=raw.get("extends",[]); refs=[refs] if isinstance(refs,str) else list(refs or [])
    for ref in refs: result=merge(result,_resolve(_relative(path,ref),(*stack,path)))
    for key,section in (("data_config","data"),("preprocessing_config","preprocessing")):
        if key in raw:
            module=_read(_relative(path,raw[key])); value=module.get(section,module)
            if not isinstance(value,dict): raise ConfigError(f"{key} must resolve to an object")
            result=merge(result,{section:value})
    return merge(result,{k:copy.deepcopy(v) for k,v in raw.items() if k not in {"extends","data_config","preprocessing_config"}})
def _project_path(root:Path,value):
    path=Path(value).expanduser(); return str((root/path).resolve() if not path.is_absolute() else path.resolve())
def _load_models(config,root):
    names=config.get("models")
    if not isinstance(names,list) or not names: raise ConfigError("models must be a non-empty list")
    model_dir=Path(_project_path(root,config.pop("model_spaces_dir","config/model_spaces"))); resolved={}
    for raw_name in names:
        name=str(raw_name); path=model_dir/f"{name}.json"; model=_read(path)
        if str(model.get("model",name))!=name: raise ConfigError(f"Model file {path} must declare model={name!r}")
        resolved[name]=model
    config["models"]=resolved
def _enabled_modes(config):
    modes=config.get("modes")
    if not isinstance(modes,dict): raise ConfigError("modes must be an object")
    unknown=set(modes)-set(CHANNEL_MODES)
    if unknown: raise ConfigError(f"Unsupported channel modes: {sorted(unknown)}")
    enabled=[name for name in CHANNEL_MODES if bool((modes.get(name) or {}).get("enabled",False))]
    if not enabled: raise ConfigError("At least one channel mode must be enabled")
    return enabled
def validate_config(config):
    required={"data","preprocessing","validation","standard_methods","models","modes","ml_input","experiment"}; missing=sorted(required-set(config))
    if missing: raise ConfigError(f"Missing configuration section(s): {missing}")
    validation=config["validation"]; extra=sorted(set(validation)-{"seed","test_fraction","validation_fraction"})
    if extra: raise ConfigError(f"Unknown validation option(s): {extra}")
    for key in ("test_fraction","validation_fraction"):
        if not 0.<float(validation[key])<.5: raise ConfigError(f"validation.{key} must be in (0, 0.5)")
    ml_input=config["ml_input"]
    if set(ml_input)-{"window_ns","subsampling"}: raise ConfigError("ml_input accepts only window_ns and subsampling")
    if float(ml_input["window_ns"]["end"])<=float(ml_input["window_ns"]["start"]): raise ConfigError("ml_input.window_ns.end must exceed start")
    if int(ml_input.get("subsampling",1))<=0: raise ConfigError("ml_input.subsampling must be positive")
    preprocessing=config["preprocessing"]
    for key in ("selection_store_dir","preprocessed_dir","prepared_dir","materialized_window_ns","selection","photopeak","energy"):
        if key not in preprocessing: raise ConfigError(f"preprocessing.{key} is required")
    channels=config["data"]["channels"]
    if channels.get("timing") and "timing" not in preprocessing: raise ConfigError("preprocessing.timing is required when timing channels exist")
    for key in ("trigger_threshold_mV","vertical_scale_limit_mV"):
        if key not in preprocessing["energy"]: raise ConfigError(f"preprocessing.energy.{key} is required")
    if "rising_edge_before_trigger_ns" in preprocessing["energy"]: raise ConfigError("preprocessing.energy.rising_edge_before_trigger_ns is obsolete; energy uses the materialized window start to peak")
    if "timing" in preprocessing:
        for key in ("trigger_threshold_mV","vertical_scale_limit_mV","rising_edge_before_trigger_ns"):
            if key not in preprocessing["timing"]: raise ConfigError(f"preprocessing.timing.{key} is required")
    noise=preprocessing["selection"]["baseline_noise"]
    if bool(noise.get("enabled",False)) and (len(noise["window_ns"])!=2 or float(noise["window_ns"][1])>0.): raise ConfigError("baseline_noise.window_ns must be [start, end] before the trigger")
    if not config["standard_methods"].get("led_thresholds_mV"): raise ConfigError("LED threshold list must not be empty")
    if not config["standard_methods"].get("cfd_fractions"): raise ConfigError("CFD fraction list must not be empty")
    from .models import model_names
    unknown_models=set(config["models"])-set(model_names())
    if unknown_models: raise ConfigError(f"Unregistered model(s): {sorted(unknown_models)}")
def load_config(path:str|Path,project_root:str|Path|None=None):
    source=Path(path).expanduser().resolve(); root=Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[1]; config=_resolve(source); config["channel_modes"]=_enabled_modes(config); _load_models(config,root)
    if "root_folder" in config["data"]: config["data"]["root_folder"]=_project_path(root,config["data"]["root_folder"])
    for key in ("selection_store_dir","preprocessed_dir","prepared_dir"): config["preprocessing"][key]=_project_path(root,config["preprocessing"][key])
    config["experiment"]["output_dir"]=_project_path(root,config["experiment"]["output_dir"]); config["_config_path"]=str(source); validate_config(config); config["_config_fingerprint"]=canonical_hash({k:v for k,v in config.items() if not str(k).startswith("_")}); return config
def discover_root_files(config):
    data=config["data"]; root=Path(data["root_folder"]); pattern=str(data.get("root_glob","*.root")); files=sorted(root.rglob(pattern) if bool(data.get("recursive",False)) else root.glob(pattern)); return [p.resolve() for p in files if p.is_file()]
def public_config(config): return {k:copy.deepcopy(v) for k,v in config.items() if not str(k).startswith("_")}
