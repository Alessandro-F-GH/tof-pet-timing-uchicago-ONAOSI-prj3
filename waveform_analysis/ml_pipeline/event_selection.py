from __future__ import annotations

import csv, json, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from ..utils.photopeak import fit_photopeak, photopeak_mask
from .common import atomic_json, canonical_hash, source_signature
from .energy_io import energy_event_count, iterate_energy_chunks
from .splits import semantic_seed, split_development_test
SELECTION_FORMAT_VERSION=1

@dataclass(frozen=True)
class SelectionData:
    directory:Path; entry_index:np.ndarray; event_index:np.ndarray; split:np.ndarray
    main_trigger_energy:np.ndarray|None; main_trigger_timing:np.ndarray|None
    main_hit_energy:np.ndarray|None; main_hit_timing:np.ndarray|None
    main_stop_energy:np.ndarray|None; main_stop_timing:np.ndarray|None
    manifest:dict[str,Any]
    @property
    def development(self): return np.flatnonzero(self.split==0).astype(np.int64)
    @property
    def test(self): return np.flatnonzero(self.split==1).astype(np.int64)

@dataclass(frozen=True)
class Hit:
    leading_index:int; stop_index:int; duration_ns:float

def robust_center_scale(values):
    x=np.asarray(values,dtype=float); x=x[np.isfinite(x)]
    if x.size==0:return float('nan'),float('nan')
    c=float(np.median(x)); mad=float(np.median(np.abs(x-c))); return c,1.4826*mad

def decode_oriented(raw,gain_v_per_count,offset_v,polarity):
    if int(polarity) not in (-1,1): raise ValueError('polarity must be +1 or -1')
    return float(polarity)*((np.asarray(raw,dtype=float)*float(gain_v_per_count)-float(offset_v))*1000.0)

def pulse_hits(signal_mV,threshold_mV,sample_interval_s):
    y=np.asarray(signal_mV,dtype=float); threshold=float(threshold_mV); dt_ns=float(sample_interval_s)*1e9
    if y.size<2 or not np.isfinite(threshold) or threshold<=0 or dt_ns<=0:return []
    y0,y1=y[:-1],y[1:]; finite=np.isfinite(y0)&np.isfinite(y1)
    rising=np.flatnonzero(finite&(y0<threshold)&(y1>=threshold)); falling=np.flatnonzero(finite&(y0>=threshold)&(y1<threshold))
    if rising.size==0:return []
    def pos(i):
        d=float(y[i+1]-y[i]); return float(i+1) if not np.isfinite(d) or d==0 else float(i)+float(np.clip((threshold-y[i])/d,0,1))
    leads=np.asarray([pos(int(i)) for i in rising]); trails=np.asarray([pos(int(i)) for i in falling]); hits=[]
    for j,(lower,lead) in enumerate(zip(rising,leads)):
        next_lead=leads[j+1] if j+1<leads.size else float(y.size-1); candidates=trails[(trails>lead)&(trails<=next_lead)]; stop=float(candidates[0]) if candidates.size else float(next_lead); stop=max(stop,lead)
        hits.append(Hit(int(lower)+1,min(y.size-1,int(np.ceil(stop))),float((stop-lead)*dt_ns)))
    return hits

def _channel_values(value,count=2):
    if isinstance(value,(int,float,np.number)): return np.full(count,float(value))
    v=np.asarray(value,dtype=float).reshape(-1)
    if v.size!=count: raise ValueError(f'expected {count} channel values')
    return v

def _families(config):
    modes=config['channel_modes']; out=[]
    if any('energy' in m for m in modes): out.append('energy')
    if any('timing' in m for m in modes): out.append('timing')
    return tuple(out)

def _io_args(config,timing):
    ch=config['data']['channels']; io=config['preprocessing'].get('io',{}); max_events=int(io.get('max_events',0))
    return dict(energy_channels_one_based=tuple(map(int,ch['energy'])),timing_channels_one_based=tuple(map(int,ch['timing'])) if timing and ch.get('timing') else None,step_size=io.get('step_size','128 MB'),entry_stop=max_events if max_events>0 else None)

def _scan_amplitudes(root,config,n):
    ch=config['data']['channels']; pol=_channel_values(ch.get('polarities',[1,1])); amps=np.full((n,2),np.nan); events=np.full(n,-1,dtype=np.int64); row=0
    for chunk in iterate_energy_chunks(root,**_io_args(config,False)):
        for local in range(chunk.event_index.size):
            if row>=n:break
            events[row]=int(chunk.event_index[local])
            for d in range(2):
                s=decode_oriented(np.asarray(chunk.samples[d][local],dtype=np.int16),chunk.vertical_gain_v_per_count[local,d],chunk.vertical_offset_v[local,d],int(pol[d])); amps[row,d]=float(np.nanmax(s)) if np.any(np.isfinite(s)) else np.nan
            row+=1
    if row!=n: raise RuntimeError(f'Expected {n} events but scanned {row}')
    return events,amps

def _scan_hits(root,config,photopeak,n):
    families=_families(config); ch=config['data']['channels']; pol={'energy':_channel_values(ch.get('polarities',[1,1])),'timing':_channel_values(ch.get('timing_polarities',[1,1]))}; thresholds={f:_channel_values(config['preprocessing'][f]['trigger_threshold_mV']) for f in families}; hits={f:[[[] for _ in range(2)] for _ in range(n)] for f in families}; row=0
    for chunk in iterate_energy_chunks(root,**_io_args(config,'timing' in families)):
        for local in range(chunk.event_index.size):
            if row>=n:break
            if photopeak[row]:
                for f in families:
                    raw=chunk.samples if f=='energy' else chunk.timing_samples; gain=chunk.vertical_gain_v_per_count if f=='energy' else chunk.timing_vertical_gain_v_per_count; off=chunk.vertical_offset_v if f=='energy' else chunk.timing_vertical_offset_v; interval=chunk.horizontal_interval_s if f=='energy' else chunk.timing_horizontal_interval_s
                    assert raw is not None and gain is not None and off is not None and interval is not None
                    for d in range(2):
                        s=decode_oriented(np.asarray(raw[d][local],dtype=np.int16),gain[local,d],off[local,d],int(pol[f][d])); hits[f][row][d]=pulse_hits(s,thresholds[f][d],interval[local,d])
            row+=1
    return hits

def _duration_limits(hits,dev,config):
    cut=config['preprocessing']['selection']['pulse_duration_mad']; left,right=float(cut['left']),float(cut['right']); out={}
    for f,events in hits.items():
        limits=np.full((2,2),np.nan)
        for d in range(2):
            primary=[max(h.duration_ns for h in events[row][d]) for row in np.flatnonzero(dev) if events[row][d]]; center,scale=robust_center_scale(primary)
            if not np.isfinite(center): raise RuntimeError(f'No development hits for {f} detector {d+1}')
            scale=max(scale if np.isfinite(scale) else 0.0,max(1e-12,abs(center)*1e-6)); limits[d]=[max(0,center-left*scale),center+right*scale]
        out[f]=limits
    return out

def _choose_main_hits(hits,limits,photopeak):
    n=photopeak.size; accepted=photopeak.copy(); main_hit={}; main_trigger={}; main_stop={}
    for f,events in hits.items():
        chosen=np.full((n,2),-1,dtype=np.int32); trigger=np.full((n,2),-1,dtype=np.int32); stop=np.full((n,2),-1,dtype=np.int32); ok=np.ones(n,dtype=bool)
        for row in np.flatnonzero(photopeak):
            for d in range(2):
                lo,hi=limits[f][d]; valid=[(i,h) for i,h in enumerate(events[row][d]) if lo<=h.duration_ns<=hi]
                if not valid: ok[row]=False; continue
                i,h=max(valid,key=lambda x:x[1].duration_ns); chosen[row,d]=i; trigger[row,d]=h.leading_index; stop[row,d]=h.stop_index
        accepted&=ok; main_hit[f]=chosen; main_trigger[f]=trigger; main_stop[f]=stop
    return accepted,main_hit,main_trigger,main_stop

def _scan_noise(root,config,candidates,triggers,n):
    families=tuple(triggers); ch=config['data']['channels']; pol={'energy':_channel_values(ch.get('polarities',[1,1])),'timing':_channel_values(ch.get('timing_polarities',[1,1]))}; start_ns,stop_ns=map(float,config['preprocessing']['selection']['baseline_noise']['window_ns']); out={f:np.full((n,2),np.nan) for f in families}; row=0
    for chunk in iterate_energy_chunks(root,**_io_args(config,'timing' in families)):
        for local in range(chunk.event_index.size):
            if row>=n:break
            if candidates[row]:
                for f in families:
                    raw=chunk.samples if f=='energy' else chunk.timing_samples; gain=chunk.vertical_gain_v_per_count if f=='energy' else chunk.timing_vertical_gain_v_per_count; off=chunk.vertical_offset_v if f=='energy' else chunk.timing_vertical_offset_v; interval=chunk.horizontal_interval_s if f=='energy' else chunk.timing_horizontal_interval_s
                    assert raw is not None and gain is not None and off is not None and interval is not None
                    for d in range(2):
                        s=decode_oriented(np.asarray(raw[d][local],dtype=np.int16),gain[local,d],off[local,d],int(pol[f][d])); dt=float(interval[local,d])*1e9; trig=int(triggers[f][row,d]); a=max(0,trig+int(np.floor(start_ns/dt))); b=min(s.size,trig+int(np.ceil(stop_ns/dt))+1); v=s[a:b]; v=v[np.isfinite(v)]
                        if v.size>=2: c=float(np.mean(v)); out[f][row,d]=float(np.sqrt(np.mean((v-c)**2)))
            row+=1
    return out

def _noise_limits(noise,dev,config):
    lam=float(config['preprocessing']['selection']['baseline_noise']['lambda_mad']); out={}
    for f,v in noise.items():
        limits=np.full(2,np.nan)
        for d in range(2):
            c,s=robust_center_scale(v[dev,d]);
            if not np.isfinite(c): raise RuntimeError(f'No development baseline RMS for {f} detector {d+1}')
            limits[d]=c+lam*max(s if np.isfinite(s) else 0,0)
        out[f]=limits
    return out

def _summary(path,masks,split):
    with path.open('w',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['criterion','split','remaining']); w.writeheader()
        for name,mask in masks:
            for code,label in ((0,'development'),(1,'test')): w.writerow({'criterion':name,'split':label,'remaining':int(np.count_nonzero(mask&(split==code)))})

def selection_fingerprint(root,config):
    return canonical_hash({'format_version':SELECTION_FORMAT_VERSION,'source':source_signature(root),'channels':config['data']['channels'],'validation':{'seed':config['validation']['seed'],'test_fraction':config['validation']['test_fraction']},'selection':config['preprocessing']['selection'],'photopeak':config['preprocessing']['photopeak'],'triggers':{f:config['preprocessing'][f]['trigger_threshold_mV'] for f in _families(config)},'max_events':int(config['preprocessing'].get('io',{}).get('max_events',0))})
def load_selection(directory,root,config):
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('fingerprint')!=selection_fingerprint(root,config): raise ValueError('Selection fingerprint changed')
    def opt(name):
        p=directory/f'{name}.npy'; return np.load(p,mmap_mode='r') if p.is_file() else None
    return SelectionData(directory,np.load(directory/'entry_index.npy',mmap_mode='r'),np.load(directory/'event_index.npy',mmap_mode='r'),np.load(directory/'split.npy',mmap_mode='r'),opt('main_trigger_energy'),opt('main_trigger_timing'),opt('main_hit_energy'),opt('main_hit_timing'),opt('main_stop_energy'),opt('main_stop_timing'),manifest)
def select_events(root_file,config,*,rebuild,logger):
    root_file=Path(root_file).resolve(); base=Path(config['preprocessing']['selection_store_dir']).resolve()/root_file.stem
    if base.is_dir() and not rebuild:
        try:return load_selection(base,root_file,config)
        except (FileNotFoundError,ValueError):pass
    if base.exists(): shutil.rmtree(base)
    base.mkdir(parents=True); total=energy_event_count(root_file); max_events=int(config['preprocessing'].get('io',{}).get('max_events',0)); n=min(total,max_events) if max_events>0 else total; entries=np.arange(n,dtype=np.int64); split0=split_development_test(entries,test_fraction=float(config['validation']['test_fraction']),seed=semantic_seed(int(config['validation']['seed']),root_file.name)); split=np.ones(n,dtype=np.int8); split[split0.development]=0
    event_index,amps=_scan_amplitudes(root_file,config,n); finite=np.all(np.isfinite(amps),axis=1); devfit=finite&(split==0); photo=finite.copy(); fits=[]
    for d,ch in enumerate(config['data']['channels']['energy']):
        result=fit_photopeak(amps[devfit,d],channel=int(ch),config=config['preprocessing']['photopeak']);
        if not result.success: raise RuntimeError(f'Photopeak fit failed for channel {ch}: {result.message}')
        fits.append(result); photo&=photopeak_mask(amps[:,d],result)
    hits=_scan_hits(root_file,config,photo,n); limits=_duration_limits(hits,(split==0)&photo,config); duration,main_hit,triggers,stops=_choose_main_hits(hits,limits,photo); selected=duration.copy(); noise=None; noise_limits=None; noise_cfg=config['preprocessing']['selection']['baseline_noise']
    if bool(noise_cfg.get('enabled',False)):
        noise=_scan_noise(root_file,config,duration,triggers,n); noise_limits=_noise_limits(noise,(split==0)&duration,config)
        for f,v in noise.items(): selected&=np.all(np.isfinite(v)&(v<=noise_limits[f][None,:]),axis=1)
    minimum=int(config['preprocessing']['selection'].get('minimum_events_per_split',50))
    for code,label in ((0,'development'),(1,'test')):
        count=int(np.count_nonzero(selected&(split==code)))
        if count<minimum: raise RuntimeError(f'Only {count} {label} events remain; need {minimum}')
    rows=np.flatnonzero(selected); np.save(base/'entry_index.npy',entries[rows]); np.save(base/'event_index.npy',event_index[rows]); np.save(base/'split.npy',split[rows])
    for f in triggers:
        np.save(base/f'main_trigger_{f}.npy',triggers[f][rows]); np.save(base/f'main_hit_{f}.npy',main_hit[f][rows]); np.save(base/f'main_stop_{f}.npy',stops[f][rows])
    _summary(base/'selection_summary.csv',[('finite_energy_amplitude',finite),('photopeak',photo),('pulse_duration',duration),('baseline_noise',selected)],split)
    manifest={'format_version':1,'fingerprint':selection_fingerprint(root_file,config),'source':str(root_file),'n_raw':n,'n_selected':int(rows.size),'n_development':int(np.count_nonzero(selected&(split==0))),'n_test':int(np.count_nonzero(selected&(split==1))),'photopeak':[f.as_dict() for f in fits],'duration_limits_ns':{f:v.tolist() for f,v in limits.items()},'baseline_noise_limits_mV':None if noise_limits is None else {f:v.tolist() for f,v in noise_limits.items()}}
    atomic_json(base/'manifest.json',manifest); logger.info('Event selection %s | %d/%d selected',root_file.name,rows.size,n); return load_selection(base,root_file,config)
