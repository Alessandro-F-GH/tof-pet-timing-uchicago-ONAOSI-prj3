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


def _display_category(category: str) -> str:
    category = str(category)
    if category == "selection":
        return "event selection"
    if category == "native_preprocess":
        return "native preprocessing"
    if category == "led_scan":
        return "LED scan"
    if category == "led_prepare":
        return "LED selection + ML preparation"
    if category.startswith("final_model:"):
        return f"model {category.split(':', 1)[1]}"
    if category.startswith("window_scan:"):
        return f"window scan {category.split(':', 1)[1]}"
    return category


class ProgressTracker:
    """Track homogeneous stages without pretending unlike tasks have equal cost."""

    def __init__(self, logger, plan: dict[str, int]):
        self.logger = logger
        self.plan = {str(key): int(value) for key, value in plan.items() if int(value) > 0}
        self.completed: dict[str, int] = {key: 0 for key in self.plan}
        self.started = monotonic()
        self.samples: dict[str, list[float]] = defaultdict(list)
        details = " | ".join(
            f"{_display_category(name)}={count}" for name, count in self.plan.items()
        )
        self.logger.info("Execution plan | %s", details or "no tracked stages")

    @property
    def elapsed_seconds(self) -> float:
        return monotonic() - self.started

    @property
    def elapsed_text(self) -> str:
        return _format_duration(self.elapsed_seconds)

    def _stage_eta_seconds(self, category: str) -> float | None:
        total = self.plan.get(category, 0)
        done = self.completed.get(category, 0)
        remaining = max(0, total - done)
        if remaining == 0:
            return 0.0
        values = [value for value in self.samples.get(category, []) if value > 0.0]
        if not values:
            return None
        return (sum(values) / len(values)) * remaining

    def stage_text(self, category: str) -> str:
        total = self.plan.get(category, 0)
        done = self.completed.get(category, 0)
        if total <= 0:
            return f"elapsed={self.elapsed_text}"
        eta = self._stage_eta_seconds(category)
        if eta is None:
            timing = "stage ETA=pending"
        elif eta <= 0.0:
            timing = "stage complete"
        else:
            finish = datetime.now() + timedelta(seconds=eta)
            timing = (
                f"stage ETA≈{_format_duration(eta)} "
                f"| stage finish≈{finish.strftime('%H:%M')}"
            )
        return f"stage={done}/{total} | elapsed={self.elapsed_text} | {timing}"

    def _record(
        self,
        category: str,
        *,
        duration_s: float,
        include_sample: bool,
    ) -> None:
        category = str(category)
        if category in self.plan:
            self.completed[category] = min(
                self.plan[category],
                self.completed.get(category, 0) + 1,
            )
        if include_sample and duration_s > 0.0:
            self.samples[category].append(float(duration_s))

    @contextmanager
    def task(
        self,
        category: str,
        label: str,
        *,
        announce_start: bool = True,
        announce_finish: bool = True,
    ):
        if announce_start:
            self.logger.info("Start | %s", label)
        started = monotonic()
        try:
            yield
        except Exception as exc:
            duration = monotonic() - started
            self._record(category, duration_s=duration, include_sample=True)
            self.logger.error(
                "Failed | %s | task=%s | %s | %s: %s",
                label,
                _format_duration(duration),
                self.stage_text(category),
                type(exc).__name__,
                exc,
            )
            raise
        else:
            duration = monotonic() - started
            self._record(category, duration_s=duration, include_sample=True)
            if announce_finish:
                self.logger.info(
                    "Done | %s | task=%s | %s",
                    label,
                    _format_duration(duration),
                    self.stage_text(category),
                )

    def complete(
        self,
        category: str,
        label: str,
        *,
        note: str | None = None,
        announce: bool = True,
    ) -> None:
        self._record(category, duration_s=0.0, include_sample=False)
        if announce:
            suffix = f" | {note}" if note else ""
            self.logger.info("Done | %s | %s%s", label, self.stage_text(category), suffix)
