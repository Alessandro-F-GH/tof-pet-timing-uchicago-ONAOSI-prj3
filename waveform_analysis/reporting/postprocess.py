from __future__ import annotations
from pathlib import Path
from waveform_analysis.reporting.report import generate_report


def remake_plots(result_root, *, logger=None, exclude_models=()):
    root = Path(result_root).expanduser().resolve()
    return generate_report(root, logger=logger, reuse_numeric=True,
                           exclude_models=exclude_models)


# Retain serialized identities while legacy imports resolve to this module.
from waveform_analysis.core.compat import (
    preserve_legacy_identity as _preserve_legacy_identity,
)

_preserve_legacy_identity(__name__, "waveform_analysis.ml_pipeline.postprocess")
