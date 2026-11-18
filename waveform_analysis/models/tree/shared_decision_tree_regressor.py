"""Shared-input tree on the sample-wise waveform difference (not swap-antisymmetric)."""
from waveform_analysis.models.tree.decision_tree_common import make_spec

MODEL_SPEC = make_spec("shared_decision_tree_regressor", shared=True)
