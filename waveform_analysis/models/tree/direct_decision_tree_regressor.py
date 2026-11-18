"""Direct tree on concatenated detector waveforms."""
from waveform_analysis.models.tree.decision_tree_common import make_spec

MODEL_SPEC = make_spec("direct_decision_tree_regressor", shared=False)
