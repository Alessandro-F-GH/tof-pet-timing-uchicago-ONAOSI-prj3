from __future__ import annotations
from waveform_analysis.reporting.report_engine import collect_runs, generate_report

__all__ = ["collect_runs", "generate_report"]


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.report")
