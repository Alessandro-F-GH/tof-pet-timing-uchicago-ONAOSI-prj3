from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta
from time import monotonic


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


class ProgressTracker:
    """Track long-running study tasks and estimate completion time."""

    def __init__(self, logger, plan: dict[str, int]):
        self.logger = logger
        self.plan = {str(key): int(value) for key, value in plan.items() if int(value) > 0}
        self.remaining = dict(self.plan)
        self.total = int(sum(self.plan.values()))
        self.completed = 0
        self.started = monotonic()
        self.samples: dict[str, list[float]] = defaultdict(list)
        details = " | ".join(f"{name}={count}" for name, count in self.plan.items())
        self.logger.info("Execution plan | %s | total tasks=%d", details or "no tracked tasks", self.total)

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started

    @property
    def elapsed_text(self) -> str:
        return _format_duration(self.elapsed_seconds)

    def _eta_seconds(self) -> float | None:
        if self.completed <= 0:
            return None
        all_samples = [value for values in self.samples.values() for value in values if value > 0.0]
        if not all_samples:
            return None
        fallback = sum(all_samples) / len(all_samples)
        estimate = 0.0
        for category, count in self.remaining.items():
            if count <= 0:
                continue
            values = [value for value in self.samples.get(category, []) if value > 0.0]
            average = sum(values) / len(values) if values else fallback
            estimate += average * count
        return max(0.0, estimate)

    def _finish(
        self,
        category: str,
        label: str,
        *,
        duration_s: float,
        success: bool,
        note: str | None = None,
        include_sample: bool = True,
    ) -> None:
        category = str(category)
        if self.remaining.get(category, 0) > 0:
            self.remaining[category] -= 1
        self.completed += 1
        if include_sample and duration_s > 0.0:
            self.samples[category].append(float(duration_s))

        fraction = 100.0 * self.completed / max(1, self.total)
        eta = self._eta_seconds()
        if eta is None:
            timing = f"elapsed={self.elapsed_text} | ETA=pending"
        else:
            finish = datetime.now() + timedelta(seconds=eta)
            timing = (
                f"elapsed={self.elapsed_text} | ETA≈{_format_duration(eta)} "
                f"| finish≈{finish.strftime('%H:%M')}"
            )
        suffix = f" | {note}" if note else ""
        level = self.logger.info if success else self.logger.error
        level(
            "%s | %s | %d/%d (%.1f%%) | task=%s | %s%s",
            "Done" if success else "Failed",
            label,
            self.completed,
            self.total,
            fraction,
            _format_duration(duration_s),
            timing,
            suffix,
        )

    @contextmanager
    def task(self, category: str, label: str):
        self.logger.info("Start | %s", label)
        started = monotonic()
        try:
            yield
        except Exception as exc:
            self._finish(
                category,
                label,
                duration_s=monotonic() - started,
                success=False,
                note=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            self._finish(
                category,
                label,
                duration_s=monotonic() - started,
                success=True,
            )

    def complete(
        self,
        category: str,
        label: str,
        *,
        success: bool = True,
        note: str | None = None,
        include_sample: bool = False,
    ) -> None:
        self._finish(
            category,
            label,
            duration_s=0.0,
            success=success,
            note=note,
            include_sample=include_sample,
        )
