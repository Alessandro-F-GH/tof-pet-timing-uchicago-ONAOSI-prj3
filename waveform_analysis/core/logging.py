"""Central CLI logging setup with the original format and logger name."""

from __future__ import annotations

import logging


def configure_logging() -> logging.Logger:
    """Configure the existing logging policy and return the pipeline logger."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger("waveform-pipeline")
