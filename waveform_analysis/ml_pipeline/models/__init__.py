"""Legacy model registry exports; implementations live in waveform_analysis.models."""
from waveform_analysis.models import FeatureTransformSpec, ModelSpec, get_model, model_names, model_registry

__all__ = ["FeatureTransformSpec", "ModelSpec", "get_model", "model_names", "model_registry"]
