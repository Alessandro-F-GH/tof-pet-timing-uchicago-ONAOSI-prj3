"""Waveform timing analysis for the TOF-PET project."""

# Physical preprocessing modules still live beside the compact ML pipeline and
# historically import shared waveform utilities as ``utils.*``. Expose the
# package utilities under that canonical name so the package CLI works without
# mutating sys.path.
import sys

try:
    from . import utils as _utils
    from .utils import photopeak as _photopeak, signal as _signal
except ImportError:  # allows isolated core/test installs without physical I/O helpers
    pass
else:
    sys.modules.setdefault("utils", _utils)
    sys.modules.setdefault("utils.photopeak", _photopeak)
    sys.modules.setdefault("utils.signal", _signal)
