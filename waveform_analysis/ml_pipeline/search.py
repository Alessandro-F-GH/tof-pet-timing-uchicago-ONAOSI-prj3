from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any
from .common import canonical_json

def candidate_id(candidate:dict[str,Any],length:int=12)->str:
    digest=hashlib.sha256(canonical_json(candidate).encode("utf-8")).hexdigest()
    return digest[:int(length)]

def candidate_manifest(candidates):
    out={}
    for raw in candidates:
        c=dict(raw or {}); cid=candidate_id(c)
        if cid in out and canonical_json(out[cid])!=canonical_json(c):
            raise RuntimeError(f"Candidate hash collision for {cid}")
        out[cid]=c
    if not out: raise ValueError("candidate list is empty")
    return dict(sorted(out.items()))

@dataclass(frozen=True)
class CandidateScore:
    candidate_id:str
    score:float

def choose_best(scores:list[CandidateScore])->CandidateScore:
    valid=[x for x in scores if x.score==x.score]
    if not valid: raise RuntimeError("No candidate produced a finite validation CTR")
    return min(valid,key=lambda x:(float(x.score),x.candidate_id))
