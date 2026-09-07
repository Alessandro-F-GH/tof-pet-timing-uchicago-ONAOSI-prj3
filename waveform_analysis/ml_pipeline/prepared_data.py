from __future__ import annotations

import json, shutil
from pathlib import Path
from typing import Any
import numpy as np
from numpy.lib.format import open_memmap
from .common import atomic_json, canonical_hash
from .data import PreprocessedData
from .dataset import DATASET_FORMAT_VERSION, load_prepared_dataset
from .splits import semantic_seed, split_training_validation
from .stats import ctr_fwhm
from .timing import anchor_grid, cfd_grid, led_grid, pair_delta


def source_family(mode:str)->str:
    if mode in {'energy_to_energy','energy_to_timing'}: return 'energy'
    if mode=='timing_to_timing': return 'timing'
    raise ValueError(f'Unknown mode: {mode}')
def target_family(mode:str)->str:
    if mode=='energy_to_energy': return 'energy'
    if mode in {'energy_to_timing','timing_to_timing'}: return 'timing'
    raise ValueError(f'Unknown mode: {mode}')
def _families(config):
    return {source_family(m) for m in config['channel_modes']},{target_family(m) for m in config['channel_modes']}
def dataset_fingerprint(preprocessed,config):
    return canonical_hash({'format_version':DATASET_FORMAT_VERSION,'preprocessed':preprocessed.manifest['fingerprint'],'true_tof_ps':config['data']['true_tof_ps'],'validation':{'seed':config['validation']['seed'],'validation_fraction':config['validation']['validation_fraction']},'standard_methods':config['standard_methods'],'ml_input':config['ml_input'],'modes':{m:config['modes'][m] for m in config['channel_modes']}})
def _best_column(grid,candidates,true_tof,fit):
    best=None; score=float('inf')
    for i,c in enumerate(candidates):
        residual=pair_delta(np.asarray(grid[:,:,i],dtype=np.float64))-float(true_tof)
        if not np.all(np.isfinite(residual)): continue
        s=float(ctr_fwhm(residual,fit).ctr_ps)
        if s<score: score=s; best=float(c)
    if best is None: raise RuntimeError('No standard-method candidate provides complete development crossing coverage')
    return best,score
def _input_offsets(interval_s,config):
    w=config['ml_input']['window_ns']; dt_ns=float(interval_s)*1e9; first=int(np.ceil(float(w['start'])/dt_ns-1e-9)); last=int(np.floor(float(w['end'])/dt_ns+1e-9)); offsets=np.arange(first,last+1,dtype=np.int64); factor=int(config['ml_input'].get('subsampling',1)); offsets=offsets[::factor]
    if offsets.size<2: raise ValueError('ML input window contains fewer than two samples')
    return offsets,offsets.astype(float)*dt_ns*1000.0
def _materialize_family(data,family,anchor_index,training,config):
    waves=data.energy_windows_mV if family=='energy' else data.timing_windows_mV; intervals=data.energy_sample_interval_s if family=='energy' else data.timing_sample_interval_s
    if waves is None or intervals is None: raise ValueError(f'{family} waveform unavailable')
    ref=float(np.asarray(intervals)[training[0],0])
    if not np.allclose(np.asarray(intervals),ref,rtol=1e-9,atol=0): raise ValueError(f'{family} sampling interval must be common')
    offsets,time_ps=_input_offsets(ref,config); output=np.empty((data.n_events,2,offsets.size),dtype=np.float32)
    for event in range(data.n_events):
        for detector in range(2):
            idx=int(anchor_index[event,detector])+offsets
            if idx[0]<0 or idx[-1]>=waves.shape[2]: raise RuntimeError(f'{family} ML window exceeds preprocessing window')
            output[event,detector]=np.asarray(waves[event,detector,idx],dtype=np.float32)
    train=np.asarray(output[training],dtype=np.float64); mean=np.mean(train,axis=0); scale=np.std(train,axis=0); scale=np.where(scale>1e-6,scale,1.0); normalized=((output-mean[None,:,:])/scale[None,:,:]).astype(np.float32); return normalized,time_ps,mean.astype(np.float32),scale.astype(np.float32)
def prepare_ml_dataset(preprocessed,config,*,rebuild,logger):
    base=Path(config['preprocessing']['prepared_dir']).resolve()/Path(preprocessed.manifest['source']).stem
    if base.is_dir() and not rebuild:
        try:
            manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
            if manifest.get('fingerprint')==dataset_fingerprint(preprocessed,config): return load_prepared_dataset(base)
        except Exception: pass
    if base.exists(): shutil.rmtree(base)
    base.mkdir(parents=True); development=np.asarray(preprocessed.development,dtype=np.int64); test=np.asarray(preprocessed.test,dtype=np.int64); split=split_training_validation(development,validation_fraction=float(config['validation']['validation_fraction']),seed=semantic_seed(int(config['validation']['seed']),Path(preprocessed.manifest['source']).name)); training,validation=split.training,split.validation; true_tof=float(config['data']['true_tof_ps']); sources,targets=_families(config); families=sorted(sources|targets); thresholds=np.asarray(config['standard_methods']['led_thresholds_mV'],dtype=float); fractions=np.asarray(config['standard_methods']['cfd_fractions'],dtype=float); led_choice={}; led_score={}; cfd_choice={}; cfd_score={}; led_times={}; cfd_times={}; anchor_idx={}; anchor_times={}
    for family in families:
        dev_led=led_grid(preprocessed,family,development,thresholds); led_choice[family],led_score[family]=_best_column(dev_led,thresholds,true_tof,config.get('fit')); logger.info('Selected %s LED | threshold %.6g mV | development FWHM CTR %.3f ps',family,led_choice[family],led_score[family]); led_times[family]=led_grid(preprocessed,family,np.arange(preprocessed.n_events),np.asarray([led_choice[family]]))[:,:,0]; anchor_idx[family],anchor_times[family]=anchor_grid(preprocessed,family,led_choice[family])
        if not np.all(np.isfinite(anchor_times[family])): raise RuntimeError(f'Unable to define {family} LED anchor for all selected events')
        need_cfd=family in targets and any(target_family(m)==family and bool((config['modes'].get(m) or {}).get('cfd',True)) for m in config['channel_modes'])
        if need_cfd:
            dev_cfd=cfd_grid(preprocessed,family,development,fractions); cfd_choice[family],cfd_score[family]=_best_column(dev_cfd,fractions,true_tof,config.get('fit')); cfd_times[family]=cfd_grid(preprocessed,family,np.arange(preprocessed.n_events),np.asarray([cfd_choice[family]]))[:,:,0]
    np.save(base/'event_index.npy',np.asarray(preprocessed.event_index,dtype=np.int64)); np.save(base/'bias_voltage_V.npy',np.asarray(preprocessed.bias_voltage_V,dtype=np.float64)); np.savez_compressed(base/'splits.npz',training=training,validation=validation,test=test); transforms={}
    for family in sorted(sources):
        normalized,time_ps,mean,scale=_materialize_family(preprocessed,family,anchor_idx[family],training,config); target=open_memmap(base/f'{family}_windows.npy',mode='w+',dtype=np.float32,shape=normalized.shape); target[:]=normalized; target.flush(); del target; np.save(base/f'{family}_time_ps.npy',time_ps); np.savez_compressed(base/f'{family}_transform.npz',mean=mean,scale=scale); transforms[family]={'mean_shape':list(mean.shape),'scale_shape':list(scale.shape),'fit_population':'training_only'}
    for family in families:
        np.save(base/f'{family}_led_time_ps.npy',led_times[family]); np.save(base/f'{family}_anchor_time_ps.npy',anchor_times[family]); np.save(base/f'{family}_target_ps.npy',true_tof-pair_delta(anchor_times[family])); np.save(base/f'{family}_anchor_offset_ps.npy',led_times[family]-anchor_times[family]);
        if family in cfd_times: np.save(base/f'{family}_cfd_time_ps.npy',cfd_times[family])
    manifest={'format_version':DATASET_FORMAT_VERSION,'fingerprint':dataset_fingerprint(preprocessed,config),'source':preprocessed.manifest['source'],'preprocessed_dir':str(preprocessed.directory),'true_tof_ps':true_tof,'n_events':preprocessed.n_events,'split':{'training':int(training.size),'validation':int(validation.size),'test':int(test.size)},'led_threshold_mV':led_choice,'led_development_ctr_ps':led_score,'cfd_fraction':cfd_choice,'cfd_development_ctr_ps':cfd_score,'ml_input':config['ml_input'],'normalization':transforms,'target_definition':'true_tof_ps - (native_anchor_time_1_ps - native_anchor_time_2_ps)','time_reference':'relative_to_native_sample_closest_to_selected_led_threshold'}; atomic_json(base/'manifest.json',manifest); logger.info('ML dataset %s | train=%d validation=%d test=%d | subsampling=%d',Path(preprocessed.manifest['source']).name,training.size,validation.size,test.size,int(config['ml_input'].get('subsampling',1))); return load_prepared_dataset(base)
