from __future__ import annotations
from .study import run_study

def run_batch(configs,*,overwrite=False,resume=False,rebuild_preprocessing=False,logger=None):
    outputs=[]
    total=len(configs)
    for i,cfg in enumerate(configs,1):
        if logger:logger.info("Batch study %d/%d | %s",i,total,cfg["name"])
        try:
            out=run_study(cfg,overwrite=overwrite,resume=resume,rebuild_preprocessing=rebuild_preprocessing);outputs.append(out)
            if logger:logger.info("Batch completed | %s | %s",cfg["name"],out)
        except Exception:
            if logger:logger.exception("Batch failed | %s",cfg["name"])
            raise
    return outputs
