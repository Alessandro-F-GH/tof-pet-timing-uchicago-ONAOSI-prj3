"""Run the same deterministic model experiment against either repository tree.

Executed in a fresh process, so baseline and migrated modules cannot share
imports, mutable global state, RNG state or monkeypatches.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
import pickle
import sys


def main() -> None:
    repository, output = map(Path, sys.argv[1:3])
    sys.path.insert(0, str(repository))
    import numpy as np
    import torch
    from waveform_analysis.ml_pipeline.models import get_model, model_names
    from waveform_analysis.ml_pipeline.splits import make_cv_split, fold_assignment
    from waveform_analysis.ml_pipeline.stats import blind_event_bootstrap

    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(731)
    control = rng.uniform(0, 1, size=(32, 2, 32)).astype(np.float32)
    pair = rng.uniform(0, 1, size=(32, 2, 32)).astype(np.float32)
    blind = rng.uniform(0, 1, size=(12, 2, 32)).astype(np.float32)
    target = rng.normal(0, 15, size=32)

    def digest(values):
        array = np.ascontiguousarray(values)
        return {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }

    result = {}
    for name in model_names():
        spec = get_model(name)
        params = {
            "architecture": [8, 4],
            "activation": "silu",
            "learning_rate": 1e-3,
            "batch_size": 8,
            "conv_channels": [3, 4],
            "kernel_samples": [3, 3],
            "layer1_kernel_samples": 4,
            "layer1_stride_samples": 2,
            "layer2_kernel_positions": 3,
            "layer2_stride_positions": 1,
            "max_correction_ps": 50.0,
            "num_kernels": 84,
            "ridge_alpha": 0.1,
        }
        log = io.StringIO()
        logger = logging.getLogger(name)
        logger.handlers[:] = [logging.StreamHandler(log)]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        config = {
            "training": {
                "epochs": 2,
                "patience": 10,
                "device": "cpu",
                "lr_decay_epochs": [1],
                "lr_decay_factor": 0.1,
            },
            "architecture": {"channels": [3, 4], "kernels": [3, 3], "dense_units": 8},
            "transform": {"n_jobs": 1},
            "verbose": True,
            "_logger": logger,
        }
        directory = output / name
        transform = None
        fit_x, predict_x = pair, blind
        transform_metadata = None
        if spec.feature_transform is not None:
            transform_params = spec.feature_transform.parameters(params, config)
            transform, control_x = spec.feature_transform.fit_transform(
                transform_params, control, seed=173, config=config
            )
            fit_x = spec.feature_transform.transform(transform, pair)
            predict_x = spec.feature_transform.transform(transform, blind)
            transform_metadata = transform.metadata
            if spec.feature_transform.save is not None:
                spec.feature_transform.save(transform, directory)
        artifact = spec.fit(params, fit_x, target, seed=174, config=config)
        prediction = spec.predict(artifact, predict_x)
        explanation = spec.explain(artifact, predict_x) if spec.explain else None
        spec.save(artifact, directory)
        arrays = {}
        if hasattr(artifact, "model"):
            arrays = {
                key: digest(value.detach().cpu().numpy())
                for key, value in artifact.model.state_dict().items()
            }
            checkpoint = torch.load(directory / "model.pt", weights_only=False)
            assert set(checkpoint) == {"state_dict", "metadata"}
            assert checkpoint["metadata"] == artifact.metadata
            for key, value in checkpoint["state_dict"].items():
                np.testing.assert_array_equal(
                    value.numpy(), artifact.model.state_dict()[key].numpy()
                )
        else:
            arrays = {
                "coef": digest(artifact.regressor.coef_),
                "intercept": digest(artifact.regressor.intercept_),
            }
            with (directory / "model.pkl").open("rb") as stream:
                loaded = pickle.load(stream)
            np.testing.assert_array_equal(spec.predict(loaded, predict_x), prediction)
        # Load baseline pickles using migrated import paths and frozen transforms.
        if len(sys.argv) > 3 and not hasattr(artifact, "model"):
            source = Path(sys.argv[3]) / name
            with (source / "model.pkl").open("rb") as stream:
                loaded = pickle.load(stream)
            loaded_x = predict_x
            if (source / "transform.pkl").is_file():
                with (source / "transform.pkl").open("rb") as stream:
                    loaded_transform = pickle.load(stream)
                loaded_x = spec.feature_transform.transform(loaded_transform, blind)
                np.testing.assert_array_equal(loaded_x, predict_x)
            np.testing.assert_array_equal(spec.predict(loaded, loaded_x), prediction)
        result[name] = {
            "prediction": digest(prediction),
            "parameters": arrays,
            "metadata": artifact.metadata,
            "transform_metadata": transform_metadata,
            "fit_features": digest(fit_x),
            "blind_features": digest(predict_x),
            "explanation": digest(explanation) if explanation is not None else None,
            "logs": log.getvalue(),
            "files": sorted(p.name for p in directory.iterdir()),
            "formulation": spec.estimator_formulation,
            "preserve_temporal_grid": spec.preserve_temporal_grid,
            "artifact_module": type(artifact).__module__,
        }
    split = make_cv_split(
        32, population_identity="synthetic", batch_seed=175, n_folds=4, shuffle=True
    )
    residual = rng.normal(0, 100, size=512)
    summary, draws = blind_event_bootstrap(
        residual,
        residual + rng.normal(0, 20, size=512),
        {"histogram_bin_width_ps": 10.0},
        n_resamples=5,
        seed=176,
    )
    result["protocol"] = {
        "folds": digest(fold_assignment(split)),
        "bootstrap": summary,
        "draws": {k: digest(v) for k, v in draws.items()},
    }
    (output / "result.json").write_text(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
