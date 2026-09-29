from __future__ import annotations

from .study import run_study


def _log_gpu_status(logger):
    """Run one lightweight CUDA smoke test and report the result once per batch."""
    if logger is None:
        return
    try:
        import torch
    except Exception as exc:
        logger.warning("GPU check | PyTorch unavailable | %s", exc)
        return

    if not torch.cuda.is_available():
        logger.info("GPU check | CUDA unavailable | PyTorch=%s | using CPU", torch.__version__)
        return

    try:
        device = torch.device("cuda:0")
        probe = torch.ones(8, device=device)
        result = (probe * 2.0).sum()
        torch.cuda.synchronize(device)
        if float(result.item()) != 16.0:
            raise RuntimeError("unexpected CUDA smoke-test result")
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU check | CUDA OK | device=%s | capability=%d.%d | memory=%.1f GiB | PyTorch=%s | CUDA=%s",
            torch.cuda.get_device_name(device),
            props.major,
            props.minor,
            props.total_memory / (1024**3),
            torch.__version__,
            torch.version.cuda,
        )
    except Exception as exc:
        logger.warning("GPU check | CUDA detected but smoke test failed | %s", exc)


def run_batch(configs,*,overwrite=False,resume=False,rebuild_preprocessing=False,logger=None):
    outputs=[]
    total=len(configs)
    _log_gpu_status(logger)
    for i,cfg in enumerate(configs,1):
        if logger:logger.info("Batch study %d/%d | %s",i,total,cfg["name"])
        try:
            out=run_study(cfg,overwrite=overwrite,resume=resume,rebuild_preprocessing=rebuild_preprocessing);outputs.append(out)
            if logger:logger.info("Batch completed | %s | %s",cfg["name"],out)
        except Exception:
            if logger:logger.exception("Batch failed | %s",cfg["name"])
            raise
    return outputs
