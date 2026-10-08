from __future__ import annotations
import csv, hashlib, json, os, tempfile, time
from pathlib import Path
from typing import Any
import numpy as np

def json_safe(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value

def canonical_json(value:Any)->str:
    return json.dumps(json_safe(value),sort_keys=True,separators=(",",":"))

def canonical_hash(value:Any)->str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()

def source_signature(path:Path)->dict[str,Any]:
    p=Path(path).resolve();s=p.stat()
    return {"path":str(p),"size_bytes":int(s.st_size),"mtime_ns":int(s.st_mtime_ns)}

def channel_limits(value:Any)->np.ndarray:
    a=np.asarray(value,dtype=np.float64)
    if a.shape==(2,):a=np.repeat(a[None,:],2,axis=0)
    if a.shape!=(2,2) or np.any(~np.isfinite(a)) or np.any(a[:,0]>=a[:,1]):
        raise ValueError("vertical_scale_limit_mV must be [low, high] or two detector [low, high] pairs")
    return a

def _replace_with_retry(source:Path,destination:Path)->None:
    # On Windows, transient file locks can briefly prevent atomic replacement.
    for attempt in range(5):
        try:
            os.replace(source,destination)
            return
        except PermissionError:
            if attempt==4:raise
            time.sleep(0.1*(2**attempt))

def atomic_json(path:Path,value:Any)->None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(json_safe(value),indent=2,sort_keys=True,allow_nan=False)+"\n"
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        _replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def write_csv(path:Path,rows:list[dict[str,Any]])->None:
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8",newline="") as stream:
            w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows)
            stream.flush();os.fsync(stream.fileno())
        _replace_with_retry(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
