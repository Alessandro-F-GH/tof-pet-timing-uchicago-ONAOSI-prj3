"""Best-effort memory telemetry for fits that may be killed outside Python.

An OOM SIGKILL cannot be caught in-process. A daemon samples process/system
memory independently of fit progress, flushing log messages continuously.
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def _read_kib(path, field):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_events():
    try:
        for record in Path("/proc/self/cgroup").read_text().splitlines():
            parts = record.split(":", 2)
            if len(parts) != 3:
                continue
            hierarchy, subsystems, relative = parts
            if hierarchy == "0":
                path = Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.events"
                if path.is_file():
                    return str(path), path.read_text().strip().replace("\n", "; ")
            elif "memory" in subsystems.split(","):
                path = Path("/sys/fs/cgroup/memory") / relative.lstrip("/") / "memory.failcnt"
                if path.is_file():
                    return str(path), path.read_text().strip()
    except OSError:
        pass
    return "unavailable", "unavailable"


@contextmanager
def monitor_memory(logger, label, interval_s=2.0):
    if logger is None:
        yield
        return
    stop = threading.Event()
    process = os.getpid()

    def snapshot(stage):
        rss = _read_kib("/proc/self/status", "VmRSS")
        peak = _read_kib("/proc/self/status", "VmHWM")
        available = _read_kib("/proc/meminfo", "MemAvailable")
        path, event = _cgroup_events()
        logger.info(
            "MEMORY DIAGNOSTIC | %s | %s | pid=%d | rss_mib=%s "
            "| peak_mib=%s | available_mib=%s | cgroup_counter=%s | value=%s",
            label, stage, process,
            None if rss is None else round(rss / 1024),
            None if peak is None else round(peak / 1024),
            None if available is None else round(available / 1024),
            path, event,
        )
        for handler in logger.handlers:
            handler.flush()

    def worker():
        while not stop.wait(interval_s):
            snapshot("heartbeat")

    snapshot("start")
    thread = threading.Thread(target=worker, name="fit-memory-monitor", daemon=True)
    thread.start()
    try:
        yield
    except MemoryError:
        snapshot("python-memory-error")
        logger.exception("MemoryError during %s", label)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)
        snapshot("exit")
