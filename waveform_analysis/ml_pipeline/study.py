from __future__ import annotations

import json, logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from .config import discover_root_files, load_config, public_config
from .data import preprocess_selected
from .event_selection import select_events
from .models import get_model
from .prepared_data import prepare_ml_dataset
from .selection_outputs import ensure_selection_outputs
from .splits import semantic_seed
from .stats import bootstrap_ctr, ctr_fwhm
from .storage import RunStore
from .train import FittedModel, predict_indices, refit_selected, save_model, search_model
from .view import anchor_delta, inverse_pair, standard_delta, target_family

@dataclass
class FinalModel:
    fitted:FittedModel
    search:Any

def _logger(run_dir:Path):
    logger=logging.getLogger(f'waveform-study:{run_dir}'); logger.setLevel(logging.INFO); logger.handlers.clear(); fmt=logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    for h in (logging.StreamHandler(),logging.FileHandler(run_dir/'study.log',encoding='utf-8')): h.setFormatter(fmt); logger.addHandler(h)
    return logger
def _metric_row(config,name,voltage,mode,method,residual,population_n,seed):
    values=np.asarray(residual,dtype=float); finite=values[np.isfinite(values)]; minimum=int((config.get('fit') or {}).get('min_events',20))
    if finite.size<minimum: raise RuntimeError(f'{name}/{mode}/{method}: only {finite.size} finite test residuals')
    nboot=int(config.get('reporting',{}).get('ctr_uncertainty_bootstrap_samples',500)); metric=bootstrap_ctr(finite,nboot,seed,config.get('fit')) if nboot else ctr_fwhm(finite,config.get('fit'))
    return {'dataset':name,'voltage_V':voltage,'mode':mode,'method':method,'stage':'test','ctr_ps':float(metric.ctr_ps),'ctr_uncertainty_ps':float(metric.uncertainty_ps),'center_ps':float(metric.center_ps),'n':int(metric.n),'population_n':int(population_n),'crossing_efficiency':float(metric.n/max(1,int(population_n)))}
def _selection_row(name,voltage,mode,method,score,parameters):
    return {'dataset':name,'voltage_V':voltage,'mode':mode,'method':method,'stage':'development_selection' if method in {'led','cfd'} else 'validation','ctr_ps':float(score),'ctr_uncertainty_ps':float('nan'),'center_ps':float('nan'),'n':0,'population_n':0,'crossing_efficiency':float('nan'),'parameters_json':json.dumps(parameters,sort_keys=True)}
def _prepare_one(root,config,rebuild,logger):
    selection=select_events(root,config,rebuild=rebuild,logger=logger); ensure_selection_outputs(root,selection,config,logger); preprocessed=preprocess_selected(root,selection,config,rebuild=rebuild,logger=logger); return prepare_ml_dataset(preprocessed,config,rebuild=rebuild,logger=logger)
def run_study(config_or_path:dict[str,Any]|str|Path,*,overwrite:bool=False,rebuild_preprocessing:bool=False)->Path:
    config=load_config(config_or_path) if not isinstance(config_or_path,dict) else config_or_path; store=RunStore(config['experiment']['output_dir'],overwrite=overwrite); logger=_logger(store.root); roots=discover_root_files(config)
    if not roots: raise FileNotFoundError('No ROOT files matched the configured source')
    datasets=[_prepare_one(root,config,rebuild_preprocessing,logger) for root in roots]; rows=[]; seed=int(config['validation']['seed']); manifest={'schema_version':2,'protocol':'raw_split_then_development_fitted_selection_then_train_validation_then_test','test_used_for_selection':False,'ctr_metric':'direct_smoothed_histogram_fwhm','config':public_config(config),'datasets':{}}
    for dataset in datasets:
        name=Path(dataset.manifest['source']).stem; voltage=float(np.nanmedian(dataset.bias_voltage_V)); store.save_split(name,dataset); final_models={}
        for mode in config['channel_modes']:
            family=target_family(mode); threshold=float(dataset.manifest['led_threshold_mV'][family]); rows.append(_selection_row(name,voltage,mode,'led',dataset.manifest['led_development_ctr_ps'][family],{'threshold_mV':threshold}))
            if bool((config['modes'].get(mode) or {}).get('cfd',True)) and family in dataset.manifest['cfd_fraction']:
                fraction=float(dataset.manifest['cfd_fraction'][family]); rows.append(_selection_row(name,voltage,mode,'cfd',dataset.manifest['cfd_development_ctr_ps'][family],{'fraction':fraction}))
            for model_name,model_config in config['models'].items():
                spec=get_model(model_name); search=search_model(spec,model_config,config,dataset,mode,seed=semantic_seed(seed,name,mode,model_name,'search')); fitted=refit_selected(spec,model_config,dataset,mode,search.best,seed=semantic_seed(seed,name,mode,model_name,'final')); save_model(spec,fitted,store.model_dir(name,mode,model_name),search.best.candidate); store.save_search(name,mode,model_name,search.as_dict()); final_models[(mode,model_name)]=FinalModel(fitted,search); rows.append(_selection_row(name,voltage,mode,model_name,search.best.score,search.best.candidate)); logger.info('Selected %s/%s | validation CTR %.3f ps | %s',mode,model_name,search.best.score,search.best.candidate)
                xai=config.get('reporting',{}).get('xai',{}) or {}
                if bool(xai.get('enabled',True)) and spec.explain is not None:
                    limit=min(dataset.development.size,int(xai.get('max_events',1024))); chosen=dataset.development[:limit]; _prediction,time_ps,normalized=predict_indices(spec,fitted,dataset,mode,chosen); importance=spec.explain(fitted.artifact,normalized); physical=inverse_pair(dataset,mode,normalized); store.save_xai(name,mode,model_name,time_ps=time_ps,importance=importance,example_pair_mV=physical[0])
        test=np.asarray(dataset.test,dtype=np.int64); true_tof=float(dataset.true_tof_ps)
        for mode in config['channel_modes']:
            led=standard_delta(dataset,mode,'led')[test]-true_tof; rows.append(_metric_row(config,name,voltage,mode,'led',led,test.size,semantic_seed(seed,name,mode,'led','test'))); store.save_residuals(name,mode,'led',led)
            if bool((config['modes'].get(mode) or {}).get('cfd',True)):
                try: cfd=standard_delta(dataset,mode,'cfd')[test]-true_tof
                except ValueError: cfd=None
                if cfd is not None: rows.append(_metric_row(config,name,voltage,mode,'cfd',cfd,test.size,semantic_seed(seed,name,mode,'cfd','test'))); store.save_residuals(name,mode,'cfd',cfd)
            anchor=anchor_delta(dataset,mode)[test]
            for model_name in config['models']:
                final=final_models[(mode,model_name)]; spec=get_model(model_name); correction,_time,_pair=predict_indices(spec,final.fitted,dataset,mode,test); residual=anchor+correction-true_tof; rows.append(_metric_row(config,name,voltage,mode,model_name,residual,test.size,semantic_seed(seed,name,mode,model_name,'test'))); store.save_residuals(name,mode,model_name,residual)
        manifest['datasets'][name]={'prepared_dir':str(dataset.directory),'split':dataset.manifest['split'],'led_threshold_mV':dataset.manifest['led_threshold_mV'],'cfd_fraction':dataset.manifest['cfd_fraction'],'subsampling':int(dataset.manifest['ml_input']['subsampling'])}; store.write_results(rows); store.write_manifest(manifest)
    logger.info('Study complete | %s',store.root); return store.root
