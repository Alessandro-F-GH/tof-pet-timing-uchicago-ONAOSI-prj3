from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np

def semantic_seed(base:int,*parts:object)->int:
    payload="|".join(map(str,(int(base),*parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4],"little") & 0x7fffffff

@dataclass(frozen=True)
class ResamplingSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    def validate(self,n_events:int)->None:
        groups=[set(map(int,x)) for x in (self.train,self.validation,self.test)]
        if groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2]:
            raise AssertionError("resampling partitions overlap")
        if groups[0]|groups[1]|groups[2] != set(range(int(n_events))):
            raise AssertionError("resampling partitions must cover the prepared population exactly")

def _split(values:np.ndarray,fraction:float,rng:np.random.Generator):
    values=np.asarray(values,dtype=np.int64)
    if values.size<2: raise ValueError("Need at least two events to split")
    if not 0<float(fraction)<1: raise ValueError("split fraction must be in (0,1)")
    perm=rng.permutation(values)
    n_right=min(values.size-1,max(1,int(round(values.size*float(fraction)))))
    return np.sort(perm[n_right:]),np.sort(perm[:n_right])

def make_resampling_split(n_events:int,*,analysis_identity:str,resampling_seed:int,
                          validation_fraction:float,test_fraction:float)->ResamplingSplit:
    if n_events<3: raise ValueError("Need at least three prepared events")
    if validation_fraction<=0 or test_fraction<=0 or validation_fraction+test_fraction>=1:
        raise ValueError("validation_fraction and test_fraction must be positive and sum to <1")
    rng=np.random.default_rng(semantic_seed(int(resampling_seed),"repeated_holdout",analysis_identity))
    all_idx=np.arange(int(n_events),dtype=np.int64)
    development,test=_split(all_idx,float(test_fraction),rng)
    relative_validation=float(validation_fraction)/(1.0-float(test_fraction))
    train,validation=_split(development,relative_validation,rng)
    out=ResamplingSplit(train,validation,test); out.validate(n_events); return out
