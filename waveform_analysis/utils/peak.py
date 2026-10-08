"""Compatibility alias for waveform_analysis.signal.peak."""
import sys as _sys
from importlib import import_module as _import_module
_sys.modules[__name__] = _import_module("waveform_analysis.signal.peak")
