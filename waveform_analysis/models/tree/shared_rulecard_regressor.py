"""Shared-input RuleCard-style regressor on detector waveform differences."""
from waveform_analysis.models.tree.rulecard_common import make_spec

MODEL_SPEC = make_spec("shared_rulecard_regressor", shared=True)
