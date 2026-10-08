"""Configuration errors shared by the CLI and pipeline configuration parser."""


class ConfigError(ValueError):
    """An invalid or unsupported existing batch configuration."""


# Keep artifacts or consumers that reference the former public exception path.
ConfigError.__module__ = "waveform_analysis.ml_pipeline.config"
