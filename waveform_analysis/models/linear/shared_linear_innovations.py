"""Shared antisymmetric Ridge regression on control-fitted linear innovations."""
from waveform_analysis.models.linear.innovation_common import make_spec

MODEL_SPEC = make_spec("shared_linear_innovations", shared=True)
