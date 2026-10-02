from __future__ import annotations
import csv,json,shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from ..utils.photopeak import fit_photopeak
from ..utils.peak import fit_histogram_peak
from .common import atomic_json,canonical_hash,source_signature,write_csv
from .preprocessing_plots import plot_photopeak,plot_baseline_noise,plot_baseline_clipping,plot_tot
from .energy_io import energy_event_count,iterate_energy_chunks

SELECTION_RULES_VERSION=10
SELECTION_APPLY_VERSION=10

@dataclass(frozen=True)
class SelectionData:
    directory:Path;entry_index:np.ndarray;event_index:np.ndarray;main_trigger:np.ndarray;main_hit:np.ndarray;main_stop:np.ndarray;manifest:dict[str,Any]
    @property
    def n_events(self):return int(self.event_index.size)
@dataclass(frozen=True)
class Hit:
    leading_index:int;stop_index:int;duration_ns:float

def robust_center_scale(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not x.size:return float("nan"),float("nan")
    c=float(np.median(x));mad=float(np.median(np.abs(x-c)));return c,1.4826*mad

def decode_oriented(raw,gain_v_per_count,offset_v,polarity):
    if int(polarity) not in (-1,1):raise ValueError("polarity must be +1 or -1")
    return float(polarity)*((np.asarray(raw,float)*float(gain_v_per_count)-float(offset_v))*1000.0)

def pulse_hits(signal_mV,threshold_mV,sample_interval_s):
    y=np.asarray(signal_mV,float);threshold=float(threshold_mV);dt=float(sample_interval_s)*1e9
    if y.size<2 or threshold<=0 or dt<=0:return []
    y0,y1=y[:-1],y[1:];finite=np.isfinite(y0)&np.isfinite(y1)
    rising=np.flatnonzero(finite&(y0<threshold)&(y1>=threshold));falling=np.flatnonzero(finite&(y0>=threshold)&(y1<threshold))
    def pos(i):
        d=float(y[i+1]-y[i]);return float(i+1) if not np.isfinite(d) or d==0 else float(i)+float(np.clip((threshold-y[i])/d,0,1))
    leads=np.asarray([pos(int(i)) for i in rising]);trails=np.asarray([pos(int(i)) for i in falling]);out=[]
    for j,(lower,lead) in enumerate(zip(rising,leads)):
        next_lead=leads[j+1] if j+1<leads.size else float(y.size-1)
        cand=trails[(trails>lead)&(trails<=next_lead)];stop=float(cand[0]) if cand.size else next_lead
        out.append(Hit(int(lower)+1,min(y.size-1,int(np.ceil(stop))),max(0.0,float(stop-lead)*dt)))
    return out

def _two(value):
    if isinstance(value,(int,float,np.number)):return np.full(2,float(value))
    a=np.asarray(value,float).reshape(-1)
    if a.size!=2:raise ValueError("expected two channel values")
    return a

def _limits(value):
    a=np.asarray(value,float)
    if a.shape==(2,):a=np.repeat(a[None,:],2,axis=0)
    if a.shape!=(2,2):raise ValueError("vertical_scale_limit_mV must be [low,high] or two pairs")
    a=np.sort(a,axis=1)
    if np.any(a[:,0]>=a[:,1]):raise ValueError("invalid vertical scale limits")
    return a

def _families(dataset):
    out=["energy"]
    if (dataset.get("channels") or {}).get("timing"):out.append("timing")
    return tuple(out)
def _io_args(dataset,preprocessing,timing):
    ch=dataset["channels"];io=preprocessing.get("io",{});mx=int(io.get("max_events",0))
    return dict(energy_channels_one_based=tuple(map(int,ch["energy"])),timing_channels_one_based=tuple(map(int,ch["timing"])) if timing else None,step_size=io.get("step_size","128 MB"),entry_stop=mx if mx>0 else None)
def _family_arrays(chunk,family):
    if family=="energy":return chunk.samples,chunk.vertical_gain_v_per_count,chunk.vertical_offset_v,chunk.horizontal_interval_s
    return chunk.timing_samples,chunk.timing_vertical_gain_v_per_count,chunk.timing_vertical_offset_v,chunk.timing_horizontal_interval_s
def _polarity(dataset,family):
    ch=dataset["channels"];return _two(ch.get("polarities",[1,1]) if family=="energy" else ch.get("timing_polarities",[1,1]))
def _scan_amplitudes(root,dataset,preprocessing,n):
    pol=_polarity(dataset,"energy");amps=np.full((n,2),np.nan);events=np.full(n,-1,np.int64);row=0
    for chunk in iterate_energy_chunks(root,**_io_args(dataset,preprocessing,False)):
        for local in range(chunk.event_index.size):
            if row>=n:break
            events[row]=int(chunk.event_index[local])
            for d in range(2):
                s=decode_oriented(chunk.samples[d][local],chunk.vertical_gain_v_per_count[local,d],chunk.vertical_offset_v[local,d],int(pol[d]))
                if np.any(np.isfinite(s)):amps[row,d]=float(np.nanmax(s))
            row+=1
    if row!=n:raise RuntimeError(f"Expected {n} events but scanned {row}")
    return events,amps
def _scan_hits(root,dataset,preprocessing,photo_mask,n,families):
    thresholds={f:_two(preprocessing[f]["trigger_threshold_mV"]) for f in families}
    hits={f:[[[] for _ in range(2)] for _ in range(n)] for f in families};row=0
    for chunk in iterate_energy_chunks(root,**_io_args(dataset,preprocessing,"timing" in families)):
        for local in range(chunk.event_index.size):
            if row>=n:break
            if photo_mask[row]:
                for f in families:
                    raw,gain,off,interval=_family_arrays(chunk,f);pol=_polarity(dataset,f)
                    assert raw is not None and gain is not None and off is not None and interval is not None
                    for d in range(2):
                        s=decode_oriented(raw[d][local],gain[local,d],off[local,d],int(pol[d]));hits[f][row][d]=pulse_hits(s,thresholds[f][d],interval[local,d])
            row+=1
    return hits
def _photo_mask(amps,rules):
    m=np.all(np.isfinite(amps),axis=1)
    for d,(lo,hi) in enumerate(rules["photopeak_intervals_mV"]):m&=(amps[:,d]>=float(lo))&(amps[:,d]<=float(hi))
    return m
def _fit_photopeak(amps,dataset,preprocessing):
    fits=[];intervals=[];finite=np.all(np.isfinite(amps),axis=1)
    for d,ch in enumerate(dataset["channels"]["energy"]):
        fit=fit_photopeak(amps[finite,d],channel=int(ch),config=preprocessing["photopeak"])
        if not fit.success:raise RuntimeError(f"Photopeak fit failed for channel {ch}: {fit.message}")
        fits.append(fit.as_dict());intervals.append([float(fit.selection_low),float(fit.selection_high)])
    return fits,intervals
def _fit_tot(hits,photo_mask,preprocessing):
    if "timing" not in hits:return None
    fits=[];limits=[];centers=[]
    for d in range(2):
        vals=np.asarray([max(h.duration_ns for h in hits["timing"][r][d]) for r in np.flatnonzero(photo_mask) if hits["timing"][r][d]],float)
        fit=fit_histogram_peak(vals,config=preprocessing["tot_peak"],unit_suffix="ns",fit_name=f"ToT detector {d+1}",value_name="pulse durations")
        if not fit.success:raise RuntimeError(f"ToT fit failed for detector {d+1}: {fit.message}")
        fits.append(fit.as_dict());limits.append([max(0.0,float(fit.selection_low)),float(fit.selection_high)]);centers.append(float(fit.mean))
    return {"fits":fits,"limits_ns":limits,"centers_ns":centers}
def _choose_hits(hits,photo_mask,family,tot_rule=None):
    n=photo_mask.size;chosen=np.full((n,2),-1,np.int32);trig=np.full((n,2),-1,np.int32);stop=np.full((n,2),-1,np.int32);ok=photo_mask.copy()
    for r in np.flatnonzero(photo_mask):
        for d in range(2):
            event_hits=hits[family][r][d]
            if family=="energy":valid=[(0,event_hits[0])] if event_hits else []
            else:
                lo,hi=tot_rule["limits_ns"][d];c=tot_rule["centers_ns"][d];valid=[(i,h) for i,h in enumerate(event_hits) if lo<=h.duration_ns<=hi]
                if valid:valid=[min(valid,key=lambda x:(abs(x[1].duration_ns-c),x[0]))]
            if not valid:ok[r]=False;continue
            i,h=valid[0];chosen[r,d]=i;trig[r,d]=h.leading_index;stop[r,d]=h.stop_index
    return ok,chosen,trig,stop

def baseline_quality(signal_mV,trigger_index,sample_interval_s,window_ns,vertical_limits_mV,clipping_margin_mV):
    y=np.asarray(signal_mV,float);dt=float(sample_interval_s)*1e9;a0,b0=map(float,window_ns)
    a=max(0,int(trigger_index)+int(np.floor(a0/dt)));b=min(y.size,int(trigger_index)+int(np.ceil(b0/dt))+1);v=y[a:b];v=v[np.isfinite(v)]
    if v.size<2:return float("nan"),False
    center=float(np.mean(v));rms=float(np.sqrt(np.mean((v-center)**2)));low,high=map(float,np.sort(np.asarray(vertical_limits_mV,float)));margin=float(clipping_margin_mV)
    return rms,bool(np.any(v<=low+margin) or np.any(v>=high-margin))
def _scan_baseline(root,dataset,preprocessing,candidate,triggers,n,family):
    rms=np.full((n,2),np.nan);clipped=np.ones((n,2),dtype=bool);row=0;selection=preprocessing["selection"];noise=selection["baseline_noise"];clip=selection["baseline_clipping"]
    window=selection["baseline_window_ns"];limits=_limits(preprocessing[family]["vertical_scale_limit_mV"]);margin=float(clip["margin_mV"])
    for chunk in iterate_energy_chunks(root,**_io_args(dataset,preprocessing,family=="timing")):
        for local in range(chunk.event_index.size):
            if row>=n:break
            if candidate[row]:
                raw,gain,off,interval=_family_arrays(chunk,family);pol=_polarity(dataset,family);assert raw is not None and gain is not None and off is not None and interval is not None
                for d in range(2):
                    s=decode_oriented(raw[d][local],gain[local,d],off[local,d],int(pol[d]));rms[row,d],clipped[row,d]=baseline_quality(s,int(triggers[row,d]),interval[local,d],window,limits[d],margin)
            row+=1
    return rms,clipped
def rules_fingerprint(reference_file,reference_dataset,preprocessing):
    return canonical_hash({"format_version":SELECTION_RULES_VERSION,"source":source_signature(reference_file),"channels":reference_dataset["channels"],"true_tof_ps":reference_dataset["true_tof_ps"],"preprocessing":preprocessing})
def fit_selection_rules(reference_file,reference_dataset,preprocessing,*,output_dir=None,logger=None):
    root=Path(reference_file).resolve();total=energy_event_count(root);mx=int(preprocessing.get("io",{}).get("max_events",0));n=min(total,mx) if mx>0 else total
    event_index,amps=_scan_amplitudes(root,reference_dataset,preprocessing,n);pp_fits,pp_intervals=_fit_photopeak(amps,reference_dataset,preprocessing);proto={"photopeak_intervals_mV":pp_intervals};photo=_photo_mask(amps,proto)
    if output_dir is not None:plot_photopeak(amps,pp_intervals,Path(output_dir)/"photopeak_selection.png","Reference photopeak selection")
    families=_families(reference_dataset);hits=_scan_hits(root,reference_dataset,preprocessing,photo,n,families);tot=_fit_tot(hits,photo,preprocessing)
    if output_dir is not None and tot is not None:plot_tot(hits["timing"],photo,tot["limits_ns"],Path(output_dir)/"timing_tot_selection.png","Reference timing ToT selection")
    noise_limits={};control_counts={}
    for family in families:
        main,chosen,triggers,stops=_choose_hits(hits,photo,family,tot if family=="timing" else None);rms,clipped=_scan_baseline(root,reference_dataset,preprocessing,main,triggers,n,family);limits=[];lam=float(preprocessing["selection"]["baseline_noise"]["lambda_mad"])
        for d in range(2):
            c,s=robust_center_scale(rms[main & ~np.any(clipped,axis=1),d])
            if not np.isfinite(c):raise RuntimeError(f"No control baseline RMS for {family} detector {d+1}")
            limits.append(float(c+lam*max(0.0,s if np.isfinite(s) else 0.0)))
        noise_limits[family]=limits
        if output_dir is not None:
            plot_baseline_noise(rms,main,limits,Path(output_dir)/f"{family}_baseline_noise.png",f"Reference {family} baseline noise");plot_baseline_clipping(clipped,main,Path(output_dir)/f"{family}_baseline_clipping.png",f"Reference {family} baseline clipping")
        final=main & np.all(np.isfinite(rms)&(rms<=np.asarray(limits)[None,:]),axis=1) & ~np.any(clipped,axis=1)
        control_counts[family]={"raw":n,"photopeak":int(photo.sum()),"main_hit":int(main.sum()),"baseline_noise":int((main&np.all(np.isfinite(rms)&(rms<=np.asarray(limits)[None,:]),axis=1)).sum()),"baseline_clipping":int(final.sum())}
    if output_dir is not None:
        rows=[]
        for family,counts in control_counts.items():
            raw=max(1,int(counts["raw"]));rows.extend({"mode_family":family,"stage":stage,"remaining":int(count),"fraction":float(count)/raw} for stage,count in counts.items() if stage!="raw")
        write_csv(Path(output_dir)/"reference_selection_summary.csv",rows)
    return {"format_version":SELECTION_RULES_VERSION,"fingerprint":rules_fingerprint(root,reference_dataset,preprocessing),"reference_source":str(root),"photopeak_fits":pp_fits,"photopeak_intervals_mV":pp_intervals,"timing_tot":tot,"baseline_noise_limits_mV":noise_limits,"baseline_clipping_rule":{"window_ns":list(preprocessing["selection"]["baseline_window_ns"]),"margin_mV":float(preprocessing["selection"]["baseline_clipping"]["margin_mV"]),"coordinate":"decode_oriented mV; both configured vertical boundaries tested"},"trigger_threshold_mV":{f:list(map(float,_two(preprocessing[f]["trigger_threshold_mV"]))) for f in families},"control_stage_counts":control_counts}
def _summary(path,counts,n):
    with path.open("w",encoding="utf-8",newline="") as s:
        w=csv.DictWriter(s,fieldnames=["stage","remaining","fraction"]);w.writeheader()
        for stage,count in counts.items():w.writerow({"stage":stage,"remaining":int(count),"fraction":float(count)/max(1,int(n))})
def application_fingerprint(root,dataset,preprocessing,rules,mode):return canonical_hash({"format_version":SELECTION_APPLY_VERSION,"source":source_signature(root),"dataset":dataset,"rules_fingerprint":rules["fingerprint"],"mode":mode,"io":preprocessing.get("io",{})})
def load_selection(directory,root,dataset,preprocessing,rules,mode):
    manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("fingerprint")!=application_fingerprint(root,dataset,preprocessing,rules,mode):raise ValueError("selection cache stale")
    return SelectionData(directory,np.load(directory/"entry_index.npy",mmap_mode="r"),np.load(directory/"event_index.npy",mmap_mode="r"),np.load(directory/"main_trigger.npy",mmap_mode="r"),np.load(directory/"main_hit.npy",mmap_mode="r"),np.load(directory/"main_stop.npy",mmap_mode="r"),manifest)
def apply_selection_rules(root_file,dataset,preprocessing,rules,mode,*,cache_dir,rebuild=False,logger=None):
    root=Path(root_file).resolve();family={"energy_to_energy":"energy","timing_to_timing":"timing"}[str(mode)];base=Path(cache_dir).resolve()/canonical_hash({"source":str(root),"rules":rules["fingerprint"],"mode":mode})[:16]
    if base.is_dir() and not rebuild:
        try:return load_selection(base,root,dataset,preprocessing,rules,mode)
        except (ValueError,FileNotFoundError):pass
    if base.exists():shutil.rmtree(base)
    base.mkdir(parents=True);total=energy_event_count(root);mx=int(preprocessing.get("io",{}).get("max_events",0));n=min(total,mx) if mx>0 else total
    event_index,amps=_scan_amplitudes(root,dataset,preprocessing,n);photo=_photo_mask(amps,rules);hits=_scan_hits(root,dataset,preprocessing,photo,n,(family,));main,chosen,triggers,stops=_choose_hits(hits,photo,family,rules.get("timing_tot") if family=="timing" else None);rms,clipped=_scan_baseline(root,dataset,preprocessing,main,triggers,n,family);lim=np.asarray(rules["baseline_noise_limits_mV"][family],float);noise=main & np.all(np.isfinite(rms)&(rms<=lim[None,:]),axis=1);final=noise & ~np.any(clipped,axis=1)
    if not np.any(final):raise RuntimeError("No events remain after frozen preprocessing")
    rows=np.flatnonzero(final);entries=np.arange(n,dtype=np.int64);np.save(base/"entry_index.npy",entries[rows]);np.save(base/"event_index.npy",event_index[rows]);np.save(base/"main_trigger.npy",triggers[rows]);np.save(base/"main_hit.npy",chosen[rows]);np.save(base/"main_stop.npy",stops[rows])
    plot_photopeak(amps,rules["photopeak_intervals_mV"],base/"photopeak_selection.png","Analysis photopeak with frozen reference limits");plot_baseline_noise(rms,main,lim,base/"baseline_noise.png","Analysis baseline noise with frozen reference limits");plot_baseline_clipping(clipped,main,base/"baseline_clipping.png","Analysis baseline clipping")
    if family=="timing" and rules.get("timing_tot") is not None:plot_tot(hits["timing"],photo,rules["timing_tot"]["limits_ns"],base/"timing_tot_selection.png","Analysis timing ToT with frozen reference limits")
    counts={"initial_valid":int(np.all(np.isfinite(amps),axis=1).sum()),"photopeak":int(photo.sum()),"main_hit_tot" if family=="timing" else "main_hit":int(main.sum()),"baseline_noise":int(noise.sum()),"baseline_clipping":int(final.sum())};_summary(base/"selection_summary.csv",counts,n)
    manifest={"format_version":SELECTION_APPLY_VERSION,"fingerprint":application_fingerprint(root,dataset,preprocessing,rules,mode),"source":str(root),"mode":mode,"family":family,"rules_fingerprint":rules["fingerprint"],"n_raw":n,"n_selected":int(rows.size),"stage_counts":counts};atomic_json(base/"manifest.json",manifest)
    if logger:logger.info("Frozen selection | %s | %s | selected=%d/%d",root.name,mode,rows.size,n)
    return load_selection(base,root,dataset,preprocessing,rules,mode)
