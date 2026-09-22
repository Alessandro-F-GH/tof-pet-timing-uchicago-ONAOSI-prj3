"""Waveform timing analysis for the TOF-PET project.

The analysis pipeline is batch-oriented and writes figures to disk. Force a
non-interactive Matplotlib backend before any plotting module is imported so
Windows never initializes Tk from analysis/worker code.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
