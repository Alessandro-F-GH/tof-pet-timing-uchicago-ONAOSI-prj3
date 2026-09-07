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
def _family_offsets_and_invalid(data,family,anchor_index,config):
    waves=data.energy_windows_mV if family=='energy' else data.timing_windows_mV; intervals=data.energy_sample_interval_s if family=='energy' else data.timing_sample_interval_s
    if waves is None or intervals is None: raise ValueError(f'{family} waveform unavailable')
    ref=float(np.asarray(intervals)[0,0])
    if not np.allclose(np.asarray(intervals),ref,rtol=1e-9,atol=0): raise ValueError(f'{family} sampling interval must be common')
    offsets,time_ps=_input_offsets(ref,config); idx=np.asarray(anchor_index,dtype=np.int64); invalid=np.any((idx+int(offsets[0])<0)|(idx+int(offsets[-1])>=waves.shape[2]),axis=1)
    return offsets,time_ps,invalid
def _materialize_family(data,family,anchor_index,training_old,kept_rows,config):
    waves=data.energy_windows_mV if family=='energy' else data.timing_windows_mV; intervals=data.energy_sample_interval_s if family=='energy' else data.timing_sample_interval_s
    if waves is None or intervals is None: raise ValueError(f'{family} waveform unavailable')
    ref=float(np.asarray(intervals)[0,0]); offsets,time_ps=_input_offsets(ref,config); output=np.empty((kept_rows.size,2,offsets.size),dtype=np.float32)
    for out_event,event in enumerate(kept_rows):
        for detector in range(2):
            idx=int(anchor_index[event,detector])+offsets
            output[out_event,detector]=np.asarray(waves[event,detector,idx],dtype=np.float32)
    train_positions=np.flatnonzero(np.isin(kept_rows,training_old)); train=np.asarray(output[train_positions],dtype=np.float64)
    if train.size==0: raise RuntimeError(f'No training events remain after {family} ML-window exclusion')
    mean=np.mean(train,axis=0); scale=np.std(train,axis=0); scale=np.where(scale>1e-6,scale,1.0); normalized=((output-mean[None,:,:])/scale[None,:,:]).astype(np.float32); return normalized,time_ps,mean.astype(np.float32),scale.astype(np.float32)
def _remap_split(indices,keep):
    mapping=np.full(keep.size,-1,dtype=np.int64); mapping[np.flatnonzero(keep)]=np.arange(np.count_nonzero(keep),dtype=np.int64); kept=np.asarray(indices,dtype=np.int64); kept=kept[keep[kept]]; return mapping[kept]
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

    invalid=np.zeros(preprocessed.n_events,dtype=bool)
    for family in sorted(sources):
        _offsets,_time,bad=_family_offsets_and_invalid(preprocessed,family,anchor_idx[family],config); invalid|=bad
    keep=~invalid; kept_rows=np.flatnonzero(keep); excluded_event_index=np.asarray(preprocessed.event_index,dtype=np.int64)[invalid]
    np.save(base/'excluded_ml_window_event_index.npy',excluded_event_index)
    if excluded_event_index.size:
        logger.warning('ML window exceeds preprocessing window | excluded=%d/%d | event indices saved to %s',excluded_event_index.size,preprocessed.n_events,base/'excluded_ml_window_event_index.npy')
    if not kept_rows.size: raise RuntimeError('No events remain after ML-window exclusion')
    training_new=_remap_split(training,keep); validation_new=_remap_split(validation,keep); test_new=_remap_split(test,keep)

    np.save(base/'event_index.npy',np.asarray(preprocessed.event_index,dtype=np.int64)[keep]); np.save(base/'bias_voltage_V.npy',np.asarray(preprocessed.bias_voltage_V,dtype=np.float64)[keep]); np.savez_compressed(base/'splits.npz',training=training_new,validation=validation_new,test=test_new); transforms={}
    for family in sorted(sources):
        normalized,time_ps,mean,scale=_materialize_family(preprocessed,family,anchor_idx[family],training,kept_rows,config); target=open_memmap(base/f'{family}_windows.npy',mode='w+',dtype=np.float32,shape=normalized.shape); target[:]=normalized; target.flush(); del target; np.save(base/f'{family}_time_ps.npy',time_ps); np.savez_compressed(base/f'{family}_transform.npz',mean=mean,scale=scale); transforms[family]={'mean_shape':list(mean.shape),'scale_shape':list(scale.shape),'fit_population':'training_only'}
    led_training_mean={}
    for family in families:
        led_pair=pair_delta(led_times[family][keep]); anchor_pair=pair_delta(anchor_times[family][keep]); mean_led=float(np.mean(led_pair[training_new])); led_training_mean[family]=mean_led
        np.save(base/f'{family}_led_time_ps.npy',led_times[family][keep]); np.save(base/f'{family}_anchor_time_ps.npy',anchor_times[family][keep]); np.save(base/f'{family}_target_ps.npy',mean_led-anchor_pair); np.save(base/f'{family}_anchor_offset_ps.npy',(led_times[family]-anchor_times[family])[keep]);
        if family in cfd_times: np.save(base/f'{family}_cfd_time_ps.npy',cfd_times[family][keep])
    manifest={'format_version':DATASET_FORMAT_VERSION,'fingerprint':dataset_fingerprint(preprocessed,config),'source':preprocessed.manifest['source'],'preprocessed_dir':str(preprocessed.directory),'true_tof_ps':true_tof,'n_input_events':preprocessed.n_events,'n_events':int(kept_rows.size),'excluded_ml_window_exceeds_preprocessing':int(excluded_event_index.size),'excluded_ml_window_event_index_file':'excluded_ml_window_event_index.npy','split':{'training':int(training_new.size),'validation':int(validation_new.size),'test':int(test_new.size)},'led_threshold_mV':led_choice,'led_development_ctr_ps':led_score,'led_training_mean_ps':led_training_mean,'cfd_fraction':cfd_choice,'cfd_development_ctr_ps':cfd_score,'ml_input':config['ml_input'],'normalization':transforms,'target_definition':'mean_training_led_pair_ps - native_anchor_pair_ps','time_reference':'relative_to_native_sample_closest_to_selected_led_threshold'}; atomic_json(base/'manifest.json',manifest); logger.info('ML dataset %s | train=%d validation=%d test=%d | excluded_ml_window=%d | subsampling=%d',Path(preprocessed.manifest['source']).name,training_new.size,validation_new.size,test_new.size,excluded_event_index.size,int(config['ml_input'].get('subsampling',1))); return load_prepared_dataset(base)
