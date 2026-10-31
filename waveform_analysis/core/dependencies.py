"""Stage identity and invalidation definitions used by planning and execution."""
from waveform_analysis.core.io import canonical_hash

# Increment when stage-identity semantics change; the new CV pipeline invalidates old Ridge stages.
RUN_IDENTITY_VERSION = 55
STAGE_DEPENDENCIES = {
    "cv": (),
    "selection": ("cv",),
    "final_fit": ("selection",),
    "blind": ("final_fit",),
    "bootstrap": ("blind",),
    "xai": ("final_fit", "blind"),
    "plots": ("cv", "selection", "final_fit", "blind", "bootstrap", "xai"),
}
STAGE_ORDER = tuple(STAGE_DEPENDENCIES)


def _depends_on(stage, ancestor):
    return any(
        parent == ancestor or _depends_on(parent, ancestor)
        for parent in STAGE_DEPENDENCIES[stage]
    )


INVALIDATES = {
    stage: tuple(
        candidate for candidate in STAGE_ORDER
        if candidate == stage or _depends_on(candidate, stage)
    )
    for stage in STAGE_ORDER
}


def stage_fingerprint(stage, payload, *, parents=None):
    """Hash a stage's own inputs and the identities of its direct parents."""
    if stage not in STAGE_DEPENDENCIES:
        raise ValueError(f"Unknown pipeline stage: {stage}")
    parents = dict(parents or {})
    expected = set(STAGE_DEPENDENCIES[stage])
    if set(parents) != expected:
        raise ValueError(
            f"Stage {stage!r} requires parent fingerprints {sorted(expected)}; "
            f"received {sorted(parents)}"
        )
    return canonical_hash({
        "schema_version": RUN_IDENTITY_VERSION,
        "stage": stage,
        "parents": parents,
        "payload": payload,
    })


def planner_fingerprints(config):
    """Build planning identities by propagating the declared stage dependency graph."""
    from waveform_analysis.data.sample_mask import SAMPLE_MASK_VERSION
    from waveform_analysis.engine.control_preprocessing import CONTROL_ARTIFACT_VERSION
    from waveform_analysis.data.dataset import DATASET_FORMAT_VERSION

    preprocessing_identity = {
        key: value for key, value in config["preprocessing"].items()
        if key not in {"cache_dir", "artifact_root"}
    }
    preprocessing = canonical_hash(
        {
            "control": config["control"],
            "control_artifact_version": CONTROL_ARTIFACT_VERSION,
            "preprocessing": preprocessing_identity,
            "mode": config["mode"],
            "fit": config["fit"],
        }
    )
    development = canonical_hash(
        {
            "preprocessing": preprocessing,
            "development": config["development"],
            "dataset_format_version": DATASET_FORMAT_VERSION,
            "sample_mask_version": SAMPLE_MASK_VERSION,
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
        }
    )

    payloads = {
        "cv": {
            "development": development,
            "cross_validation": config["cross_validation"],
            "model": config["model"],
            "seed": config["seed"],
        },
        "selection": {},  # The selected candidate is fully determined by the CV parent.
        "final_fit": {
            "development": development,
            "model": config["model"],
            "seed": config["seed"],
        },
        "blind": {
            "preprocessing": preprocessing,
            "blind": config["blind"],
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
            "fit": config["fit"],
        },
        "bootstrap": {
            "bootstrap": config["bootstrap"],
            "seed": config["seed"],
        },
        "xai": {
            "xai": config["xai"],
            "seed": config["seed"],
            "method_version": None,
        },
        "plots": {"plot_config": config["plot_config"]},
    }

    fingerprints = {}
    for stage in STAGE_ORDER:
        parents = {
            parent: fingerprints[parent]
            for parent in STAGE_DEPENDENCIES[stage]
        }
        fingerprints[stage] = stage_fingerprint(
            stage, payloads[stage], parents=parents
        )

    return {
        "preprocessing": preprocessing,
        "development": development,
        **fingerprints,
    }
