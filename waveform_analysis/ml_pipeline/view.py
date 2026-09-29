from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .dataset import PreparedDataset
MODE_FAMILY={"energy_to_energy":"energy","timing_to_timing":"timing"}
def mode_family(mode):
    try:return MODE_FAMILY[str(mode)]
    except KeyError as e:raise ValueError(f"Unsupported channel mode: {mode}") from e
source_family=mode_family; target_family=mode_family
@dataclass(frozen=True)
class WaveformView:
    pair:np.ndarray; time_ps:np.ndarray; family:str
    def materialize(self,dtype=np.float32): return np.asarray(self.pair,dtype=dtype)
def waveform_view(dataset,mode,indices):
    f=mode_family(mode); waves=dataset.energy_windows if f=="energy" else dataset.timing_windows
    time=dataset.energy_time_ps if f=="energy" else dataset.timing_time_ps
    if waves is None or time is None: raise ValueError(f"{f} ML input unavailable")
    return WaveformView(np.asarray(waves[np.asarray(indices,np.int64)],np.float32),np.asarray(time,np.float64),f)
def standard_delta(dataset,mode,method="led"):
    if method!="led": raise ValueError("Only the frozen LED timing is part of the study dataset")
    f=mode_family(mode); v=dataset.energy_led_time_ps if f=="energy" else dataset.timing_led_time_ps
    if v is None: raise ValueError(f"LED timing unavailable for {f}")
    v=np.asarray(v,np.float64); return v[:,0]-v[:,1]
def model_target(dataset,mode):
    f=mode_family(mode); v=dataset.energy_target_ps if f=="energy" else dataset.timing_target_ps
    if v is None: raise ValueError(f"{f} target unavailable")
    return np.asarray(v,np.float64)
def corrected_timing_residual(target_ps,prediction_ps):
    a=np.asarray(target_ps,np.float64); b=np.asarray(prediction_ps,np.float64)
    if a.shape!=b.shape: raise ValueError("target/prediction shape mismatch")
    return a-b
