"""Compatibility alias for :mod:`waveform_analysis.data.storage`."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module('waveform_analysis.data.storage')
