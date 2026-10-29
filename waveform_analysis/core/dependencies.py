"""Stage identity and invalidation definitions used by planning and execution."""
from waveform_analysis.core.io import canonical_hash
from waveform_analysis.core.ridge import LINEAR_RIDGE_MODELS, RIDGE_CV_VERSION
from waveform_analysis.engine.control_preprocessing import CONTROL_ARTIFACT_VERSION
from waveform_analysis.data.dataset import DATASET_FORMAT_VERSION

# Preserve existing identity payloads so moving code does not change seeds/caches.
RUN_IDENTITY_VERSION = 53
STAGE_ORDER = ("cv", "selection", "final_fit", "blind", "bootstrap", "xai", "plots")
INVALIDATES = {
    "cv": STAGE_ORDER,
    "selection": STAGE_ORDER[1:],
    "final_fit": STAGE_ORDER[2:],
    "blind": STAGE_ORDER[3:],
    "bootstrap": ("bootstrap", "plots"),
    "xai": ("xai", "plots"),
    "plots": ("plots",),
}


def stage_fingerprint(stage, payload):
    return canonical_hash({"schema_version": RUN_IDENTITY_VERSION, "stage": stage, "payload": payload})


def planner_fingerprints(config):
    from waveform_analysis.data.sample_mask import SAMPLE_MASK_VERSION
    preprocessing_identity = {
        key: value for key, value in config["preprocessing"].items()
        if key not in {"cache_dir", "artifact_root"}
    }
    pre = canonical_hash(
        {
            "control": config["control"],
            "control_artifact_version": CONTROL_ARTIFACT_VERSION,
            "preprocessing": preprocessing_identity,
            "mode": config["mode"],
            "fit": config["fit"],
        }
    )
    dev = canonical_hash(
        {
            "preprocessing": pre,
            "development": config["development"],
            "dataset_format_version": DATASET_FORMAT_VERSION,
            "sample_mask_version": SAMPLE_MASK_VERSION,
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
        }
    )
    cv_dependencies = {
        "development": dev,
        "cross_validation": config["cross_validation"],
        "model": config["model"],
        "seed": config["seed"],
    }
    if config["model"]["name"] in LINEAR_RIDGE_MODELS:
        cv_dependencies.pop("cross_validation")
        cv_dependencies["selection_method"] = "ridge_cv"
        cv_dependencies["ridge_cv_version"] = RIDGE_CV_VERSION
    cv = canonical_hash(cv_dependencies)
    final = canonical_hash(
        {"cv": cv, "development": dev, "model": config["model"], "seed": config["seed"]}
    )
    blind = canonical_hash(
        {
            "final_fit": final,
            "preprocessing": pre,
            "blind": config["blind"],
            "window": config["window_ns"],
            "ml_input": config["ml_input"],
            "fit": config["fit"],
        }
    )
    bootstrap = canonical_hash(
        {"blind": blind, "bootstrap": config["bootstrap"], "seed": config["seed"]}
    )
    xai = canonical_hash(
        {
            "final_fit": final,
            "blind": blind,
            "xai": config["xai"],
            "seed": config["seed"],
            "method_version": None,
        }
    )
    plots = canonical_hash({"plot_config": config["plot_config"]})
    return {
        "preprocessing": pre,
        "development": dev,
        "cv": cv,
        "final_fit": final,
        "blind": blind,
        "bootstrap": bootstrap,
        "xai": xai,
        "plots": plots,
    }
