from __future__ import annotations

import json, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from numpy.lib.format import open_memmap
from .common import atomic_json, canonical_hash, source_signature
from .energy_io import iterate_energy_chunks
from .event_selection import SelectionData, decode_oriented

PREPROCESS_FORMAT_VERSION=1

@dataclass(frozen=True)
class PreprocessedData:
    directory:Path; manifest:dict[str,Any]; event_index:np.ndarray; split:np.ndarray; bias_voltage_V:np.ndarray
    energy_windows_mV:np.ndarray|None; timing_windows_mV:np.ndarray|None
    energy_window_start_time_s:np.ndarray|None; timing_window_start_time_s:np.ndarray|None
    energy_sample_interval_s:np.ndarray|None; timing_sample_interval_s:np.ndarray|None
    energy_rising_start:np.ndarray|None; timing_rising_start:np.ndarray|None
    energy_rising_stop:np.ndarray|None; timing_rising_stop:np.ndarray|None
    @property
    def n_events(self): return int(self.event_index.size)
    @property
    def development(self): return np.flatnonzero(self.split==0).astype(np.int64)
    @property
    def test(self): return np.flatnonzero(self.split==1).astype(np.int64)

def used_families(config):
    modes=config['channel_modes']; out=[]
    if any('energy' in m for m in modes): out.append('energy')
    if any('timing' in m for m in modes): out.append('timing')
    return tuple(out)
def _limits(value):
    v=np.asarray(value,dtype=float)
    if v.shape==(2,): v=np.repeat(v[None,:],2,axis=0)
    if v.shape!=(2,2) or np.any(v[:,0]>=v[:,1]): raise ValueError('vertical_scale_limit_mV must be [low,high] or two detector pairs')
    return v
def _family_arrays(chunk,f):
    if f=='energy': return chunk.samples,chunk.vertical_gain_v_per_count,chunk.vertical_offset_v,chunk.horizontal_interval_s,chunk.horizontal_offset_s
    return chunk.timing_samples,chunk.timing_vertical_gain_v_per_count,chunk.timing_vertical_offset_v,chunk.timing_horizontal_interval_s,chunk.timing_horizontal_offset_s
def preprocessing_fingerprint(root,selection,config):
    return canonical_hash({'format_version':1,'source':source_signature(root),'selection':selection.manifest['fingerprint'],'families':used_families(config),'materialized_window_ns':config['preprocessing']['materialized_window_ns'],'family_config':{f:{'vertical_scale_limit_mV':config['preprocessing'][f]['vertical_scale_limit_mV'],'rising_edge_before_trigger_ns':config['preprocessing'][f]['rising_edge_before_trigger_ns']} for f in used_families(config)}})
def _optional(directory,name):
    p=directory/f'{name}.npy'; return np.load(p,mmap_mode='r') if p.is_file() else None
def load_preprocessed(directory,root,selection,config):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('fingerprint')!=preprocessing_fingerprint(root,selection,config): raise ValueError('Preprocessing fingerprint changed')
    return PreprocessedData(directory,manifest,np.load(directory/'event_index.npy',mmap_mode='r'),np.load(directory/'split.npy',mmap_mode='r'),np.load(directory/'bias_voltage_V.npy',mmap_mode='r'),_optional(directory,'energy_windows_mV'),_optional(directory,'timing_windows_mV'),_optional(directory,'energy_window_start_time_s'),_optional(directory,'timing_window_start_time_s'),_optional(directory,'energy_sample_interval_s'),_optional(directory,'timing_sample_interval_s'),_optional(directory,'energy_rising_start'),_optional(directory,'timing_rising_start'),_optional(directory,'energy_rising_stop'),_optional(directory,'timing_rising_stop'))
def _close(a):
    a.flush(); m=getattr(a,'_mmap',None)
    if m is not None:m.close()
def preprocess_selected(root_file,selection,config,*,rebuild,logger):
    root_file=Path(root_file).resolve(); base=Path(config['preprocessing']['preprocessed_dir']).resolve()/root_file.stem
    if base.is_dir() and not rebuild:
        try:return load_preprocessed(base,root_file,selection,config)
        except (FileNotFoundError,ValueError):pass
    if base.exists(): shutil.rmtree(base)
    base.mkdir(parents=True); families=used_families(config); entries=np.asarray(selection.entry_index,dtype=np.int64); lookup={int(e):i for i,e in enumerate(entries)}; n=entries.size; np.save(base/'event_index.npy',np.asarray(selection.event_index,dtype=np.int64)); np.save(base/'split.npy',np.asarray(selection.split,dtype=np.int8)); bias=open_memmap(base/'bias_voltage_V.npy',mode='w+',dtype=np.float64,shape=(n,)); bias[:]=np.nan
    targets={}; starts={}; intervals={}; rise_a={}; rise_b={}; lengths={}; written=np.zeros(n,dtype=bool); ch=config['data']['channels']; pol={'energy':np.asarray(ch.get('polarities',[1,1]),dtype=np.int8),'timing':np.asarray(ch.get('timing_polarities',[1,1]),dtype=np.int8)}; triggers={'energy':selection.main_trigger_energy,'timing':selection.main_trigger_timing}; stops={'energy':selection.main_stop_energy,'timing':selection.main_stop_timing}; mat=config['preprocessing']['materialized_window_ns']; before,after=float(mat['before']),float(mat['after']); io=config['preprocessing'].get('io',{}); max_events=int(io.get('max_events',0)); args=dict(energy_channels_one_based=tuple(map(int,ch['energy'])),timing_channels_one_based=tuple(map(int,ch['timing'])) if 'timing' in families else None,step_size=io.get('step_size','128 MB'),entry_stop=max_events if max_events>0 else None); entry=0
    for chunk in iterate_energy_chunks(root_file,**args):
        for local in range(chunk.event_index.size):
            row=lookup.get(entry); entry+=1
            if row is None: continue
            bias[row]=float(chunk.bias_voltage_V[local])
            for f in families:
                raw,gain,off,dt,hoff=_family_arrays(chunk,f); assert raw is not None and gain is not None and off is not None and dt is not None and hoff is not None and triggers[f] is not None and stops[f] is not None; lim=_limits(config['preprocessing'][f]['vertical_scale_limit_mV']); windows=[]; st=[]; iv=[]; ra=[]; rb=[]; pre=float(config['preprocessing'][f]['rising_edge_before_trigger_ns'])
                for d in range(2):
                    s=decode_oriented(np.asarray(raw[d][local],dtype=np.int16),gain[local,d],off[local,d],int(pol[f][d])); s=np.clip(s,lim[d,0],lim[d,1]); interval=float(dt[local,d]); dt_ns=interval*1e9; trig=int(triggers[f][row,d]); pulse_stop=int(stops[f][row,d]); a=trig-int(np.ceil(before/dt_ns)); b=trig+int(np.ceil(after/dt_ns))+1
                    if a<0 or b>s.size: raise RuntimeError(f'{root_file.name} event {selection.event_index[row]} {f} detector {d+1}: window exceeds trace')
                    w=np.asarray(s[a:b],dtype=np.float32); search=max(a,trig-int(np.ceil(pre/dt_ns))); preseg=s[search:trig+1]; onset=search+int(np.nanargmin(preseg)) if preseg.size else trig; peak_stop=min(s.size-1,max(trig,pulse_stop)); pulse=s[trig:peak_stop+1]; peak=trig+int(np.nanargmax(pulse)) if pulse.size else trig
                    if peak<=onset: raise RuntimeError('Unable to define rising edge interval')
                    windows.append(w); st.append(float(hoff[local,d])+a*interval); iv.append(interval); ra.append(onset-a); rb.append(peak-a)
                if f not in targets:
                    L=windows[0].size
                    if any(x.size!=L for x in windows): raise ValueError(f'{f} detector grids differ')
                    lengths[f]=L; targets[f]=open_memmap(base/f'{f}_windows_mV.npy',mode='w+',dtype=np.float32,shape=(n,2,L)); starts[f]=open_memmap(base/f'{f}_window_start_time_s.npy',mode='w+',dtype=np.float64,shape=(n,2)); intervals[f]=open_memmap(base/f'{f}_sample_interval_s.npy',mode='w+',dtype=np.float64,shape=(n,2)); rise_a[f]=open_memmap(base/f'{f}_rising_start.npy',mode='w+',dtype=np.int32,shape=(n,2)); rise_b[f]=open_memmap(base/f'{f}_rising_stop.npy',mode='w+',dtype=np.int32,shape=(n,2))
                if any(x.size!=lengths[f] for x in windows): raise ValueError(f'{f} sample interval changed')
                targets[f][row]=np.stack(windows); starts[f][row]=st; intervals[f][row]=iv; rise_a[f][row]=ra; rise_b[f][row]=rb
            written[row]=True
    if not np.all(written): raise RuntimeError(f'Failed to materialize {np.count_nonzero(~written)} selected events')
    _close(bias)
    for group in (targets,starts,intervals,rise_a,rise_b):
        for a in group.values(): _close(a)
    manifest={'format_version':1,'fingerprint':preprocessing_fingerprint(root_file,selection,config),'source':str(root_file),'selection_dir':str(selection.directory),'n_events':int(n),'families':list(families),'materialized_window_ns':{'before':before,'after':after},'time_reference':'absolute_native_acquisition_time','denoising':False}; atomic_json(base/'manifest.json',manifest); logger.info('Preprocessing %s | n=%d | families=%s',root_file.name,n,','.join(families)); return load_preprocessed(base,root_file,selection,config)
