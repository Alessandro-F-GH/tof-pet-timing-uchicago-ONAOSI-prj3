from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .event_selection import SelectionData, _choose_main_hits, _noise_limits, _scan_amplitudes, _scan_hits, _scan_noise
from .splits import semantic_seed, split_development_test


def _photopeak_mask(amplitudes: np.ndarray, fits: list[dict[str, Any]]) -> np.ndarray:
    accepted = np.all(np.isfinite(amplitudes), axis=1)
    for detector, fit in enumerate(fits):
        accepted &= (amplitudes[:, detector] >= float(fit["selection_low_mV"])) & (amplitudes[:, detector] <= float(fit["selection_high_mV"]))
    return accepted


def _save_hits(path: Path, hits, event_index: np.ndarray) -> None:
    event, family, detector, number, leading, stop, duration = [], [], [], [], [], [], []
    for fam, events in hits.items():
        for row, pair in enumerate(events):
            for det, event_hits in enumerate(pair):
                for hit_number, hit in enumerate(event_hits):
                    event.append(int(event_index[row])); family.append(fam); detector.append(det); number.append(hit_number)
                    leading.append(hit.leading_index); stop.append(hit.stop_index); duration.append(hit.duration_ns)
    np.savez_compressed(path, event_index=np.asarray(event,dtype=np.int64), family=np.asarray(family,dtype="U8"), detector=np.asarray(detector,dtype=np.int8), hit_number=np.asarray(number,dtype=np.int16), leading_index=np.asarray(leading,dtype=np.int32), stop_index=np.asarray(stop,dtype=np.int32), duration_ns=np.asarray(duration,dtype=np.float64))


def _write_summary(path: Path, stages, split: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["criterion","split","remaining","rejected_from_previous"]); writer.writeheader()
        previous = np.ones(split.size, dtype=bool)
        for criterion, mask in stages:
            for code, label in ((0,"development"),(1,"test")):
                population = split == code
                writer.writerow({"criterion":criterion,"split":label,"remaining":int(np.count_nonzero(mask & population)),"rejected_from_previous":int(np.count_nonzero(previous & population & ~mask))})
            previous = mask.copy()


def _plots(directory, amplitudes, split, fits, photopeak, hits, duration_limits, noise, noise_limits) -> None:
    import matplotlib.pyplot as plt
    dev = split == 0
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for detector, ax in enumerate(np.atleast_1d(axes)):
        values = amplitudes[dev,detector]; values = values[np.isfinite(values)]; ax.hist(values,bins=120,histtype="step")
        fit=fits[detector]; low,high=float(fit["selection_low_mV"]),float(fit["selection_high_mV"]); ax.axvspan(low,high,alpha=.2)
        selected=int(np.count_nonzero(dev & (amplitudes[:,detector]>=low) & (amplitudes[:,detector]<=high))); ax.set_title(f"Energy ch {fit['channel']}: {selected} selected"); ax.set_xlabel("Amplitude [mV]")
    fig.suptitle(f"Development photopeak; AND selected = {int(np.count_nonzero(dev & photopeak))}"); fig.tight_layout(); fig.savefig(directory/"photopeak_selection.png",dpi=180); plt.close(fig)
    panels=2*len(hits); fig,axes=plt.subplots(panels,1,figsize=(8,2.8*panels),squeeze=False); panel=0
    for family,events in hits.items():
        for detector in range(2):
            ax=axes[panel,0]; panel+=1; values=[max(h.duration_ns for h in events[row][detector]) for row in np.flatnonzero(dev & photopeak) if events[row][detector]]
            if values: ax.hist(values,bins=100,histtype="step")
            low,high=duration_limits[family][detector]; ax.axvspan(low,high,alpha=.2); ax.set_title(f"{family} detector {detector+1} ToT"); ax.set_xlabel("Pulse duration [ns]")
    fig.tight_layout(); fig.savefig(directory/"tot_selection.png",dpi=180); plt.close(fig)
    if noise is not None and noise_limits is not None:
        panels=2*len(noise); fig,axes=plt.subplots(panels,1,figsize=(8,2.8*panels),squeeze=False); panel=0
        for family,values in noise.items():
            for detector in range(2):
                ax=axes[panel,0]; panel+=1; sample=values[dev,detector]; sample=sample[np.isfinite(sample)]
                if sample.size: ax.hist(sample,bins=100,histtype="step")
                ax.axvline(float(noise_limits[family][detector])); ax.set_title(f"{family} detector {detector+1} baseline RMS"); ax.set_xlabel("RMS [mV]")
        fig.tight_layout(); fig.savefig(directory/"baseline_noise_selection.png",dpi=180); plt.close(fig)


def ensure_selection_outputs(root_file: Path, selection: SelectionData, config: dict[str, Any], logger: Any) -> None:
    directory=selection.directory; noise_enabled=bool(config["preprocessing"]["selection"]["baseline_noise"].get("enabled",False)); expected=[directory/"hits.npz",directory/"photopeak_selection.png",directory/"tot_selection.png"]
    if noise_enabled: expected.append(directory/"baseline_noise_selection.png")
    if all(path.is_file() for path in expected): return
    n=int(selection.manifest["n_raw"]); entries=np.arange(n,dtype=np.int64); raw=split_development_test(entries,test_fraction=float(config["validation"]["test_fraction"]),seed=semantic_seed(int(config["validation"]["seed"]),Path(root_file).name)); split=np.ones(n,dtype=np.int8); split[raw.development]=0
    event_index,amplitudes=_scan_amplitudes(Path(root_file),config,n); finite=np.all(np.isfinite(amplitudes),axis=1); fits=list(selection.manifest["photopeak"]); photopeak=_photopeak_mask(amplitudes,fits); hits,_=_scan_hits(Path(root_file),config,photopeak,n); limits={family:np.asarray(value,dtype=np.float64) for family,value in selection.manifest["duration_limits_ns"].items()}; duration,_main,triggers,_stops=_choose_main_hits(hits,limits,photopeak); noise=None; noise_limits=None; final=duration.copy()
    if noise_enabled:
        noise=_scan_noise(Path(root_file),config,duration,triggers,n); stored=selection.manifest.get("baseline_noise_limits_mV") or {}; noise_limits={family:np.asarray(value,dtype=np.float64) for family,value in stored.items()}
        if not noise_limits: noise_limits=_noise_limits(noise,(split==0)&duration,config)
        for family,values in noise.items(): final &= np.all(np.isfinite(values) & (values <= noise_limits[family][None,:]),axis=1)
    _save_hits(directory/"hits.npz",hits,event_index); stages=[("finite_energy_amplitude",finite),("photopeak",photopeak),("pulse_duration",duration)]
    if noise_enabled: stages.append(("baseline_noise",final))
    _write_summary(directory/"selection_summary.csv",stages,split); _plots(directory,amplitudes,split,fits,photopeak,hits,limits,noise,noise_limits); logger.info("Selection diagnostics written | %s",directory)
