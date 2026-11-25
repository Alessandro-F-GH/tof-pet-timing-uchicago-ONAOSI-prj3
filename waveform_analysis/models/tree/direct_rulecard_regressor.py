"""Direct RuleCard-style regressor on concatenated detector waveforms."""
from waveform_analysis.models.tree.rulecard_common import make_spec

MODEL_SPEC = make_spec("direct_rulecard_regressor", shared=False)
