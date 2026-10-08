"""Compatibility alias for :mod:`waveform_analysis.models.kernel.shared_minirocket`."""
from importlib import import_module as _import_module
import sys as _sys

_sys.modules[__name__] = _import_module('waveform_analysis.models.kernel.shared_minirocket')
