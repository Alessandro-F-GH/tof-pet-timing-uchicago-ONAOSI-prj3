"""Global shapelet regression-tree surrogates of frozen timing models (not LASTS)."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import pickle
from pathlib import Path
import shutil
import tempfile

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, export_text

from waveform_analysis.core.io import atomic_json, canonical_hash
from waveform_analysis.data.results import discover_runs, resolve_run_config
from waveform_analysis.data.splits import semantic_seed
from waveform_analysis.data.view import model_target, waveform_view
from waveform_analysis.engine.diagnostics import diagnostic_random_state
from waveform_analysis.engine.train import load_fitted_model, predict_indices, release_training_memory, saved_model_complete
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model
from waveform_analysis.models.kernel.rdst_common import export_shapelets
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time


@dataclass
class ShapeletRegressionSurrogate:
    """One pooled univariate shapelet bank; distinct distances for each detector."""
    transformer: object
    tree: DecisionTreeRegressor
    sample_mask: np.ndarray | None

    def features(self, pair):
        pair = apply_sample_mask(pair, self.sample_mask)
        # Separate calls prevent any match across a detector boundary.
        return np.concatenate([
            np.asarray(self.transformer.transform(pair[:, d:d + 1]), dtype=np.float32)[:, ::3]
            for d in range(2)
        ], axis=1)

    def predict(self, pair):
        return self.tree.predict(self.features(pair))


def fidelity(original, surrogate):
    original, surrogate = np.asarray(original), np.asarray(surrogate)
    if original.shape != surrogate.shape or original.ndim != 1 or not np.all(np.isfinite([original, surrogate])):
        raise ValueError("Fidelity requires matching finite prediction vectors")
    return {"n_events": len(original), "mae_ps": float(mean_absolute_error(original, surrogate)),
            "rmse_ps": float(np.sqrt(mean_squared_error(original, surrogate))),
            "r2": float(r2_score(original, surrogate)) if np.var(original) > 0 else None}


def fit_surrogate(pair, predictions, *, sample_mask, seed, max_shapelets=256,
                  shapelet_lengths=(7, 11), depths=(2, 3, 4, 5), min_samples_leaf=20):
    """Select depth by development fidelity only; blind inputs never enter fitting."""
    from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform

    pair = apply_sample_mask(pair, sample_mask)
    predictions = np.asarray(predictions, dtype=float)
    if pair.ndim != 3 or pair.shape[1] != 2 or len(pair) < 10:
        raise ValueError("Surrogate fitting requires at least ten two-detector development events")
    if predictions.shape != (len(pair),) or not np.all(np.isfinite(pair)) or not np.all(np.isfinite(predictions)):
        raise ValueError("Surrogate inputs and teacher predictions must be finite and aligned")
    if max(shapelet_lengths) > pair.shape[-1]:
        raise ValueError("Shapelet lengths exceed saved model's retained grid; supply shorter --shapelet-lengths")
    train, validation = train_test_split(np.arange(len(pair)), test_size=.2, random_state=seed)
    transformer = RandomDilatedShapeletTransform(
        max_shapelets=max_shapelets, shapelet_lengths=list(shapelet_lengths),
        random_state=seed, n_jobs=1)
    # Unsupervised representation fitted only on the surrogate-training split.
    transformer.fit(np.concatenate([pair[train, :1], pair[train, 1:]], axis=0))
    surrogate = ShapeletRegressionSurrogate(transformer, None, sample_mask)
    features = surrogate.features(pair)
    selection = []
    for depth in sorted(set(depths)):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        tree.fit(features[train], predictions[train])
        selection.append({"max_depth": depth, **fidelity(predictions[validation], tree.predict(features[validation]))})
    best = min(selection, key=lambda row: (row['rmse_ps'], row['max_depth']))
    surrogate.tree = DecisionTreeRegressor(max_depth=best['max_depth'], min_samples_leaf=min_samples_leaf, random_state=seed)
    surrogate.tree.fit(features, predictions)
    return surrogate, {"selection": selection, "selected_max_depth": best['max_depth'],
                       "train_indices": train, "validation_indices": validation,
                       "feature_definition": "per-detector normalized-or-raw minimum shapelet distance; aeon RDST",
                       "transform_fit_role": "development_surrogate_training_split",
                       "validation_fraction": .2}


def _features_for_dataset(surrogate, dataset, mode, indices, chunk_size):
    return np.concatenate([surrogate.features(waveform_view(dataset, mode, indices[start:start + chunk_size]).materialize())
                           for start in range(0, len(indices), chunk_size)])


def _sample(dataset, maximum, seed):
    count = dataset.n_events if maximum is None else min(dataset.n_events, maximum)
    return np.sort(np.random.default_rng(seed).choice(dataset.n_events, count, replace=False))


def _validate_settings(max_shapelets, shapelet_lengths, depths, min_samples_leaf, max_events):
    for name, values, minimum in (("max_shapelets", [max_shapelets], 1), ("shapelet_lengths", shapelet_lengths, 2),
                                  ("depths", depths, 1), ("min_samples_leaf", [min_samples_leaf], 1),
                                  ("max_events", [] if max_events is None else [max_events], 10)):
        if (name in {"shapelet_lengths", "depths"} and not values) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values
        ):
            raise ValueError(f"{name} requires integers >= {minimum}")


def explain_regression(results, *, models=None, cache_root=None, max_shapelets=256,
                       shapelet_lengths=(7, 11), depths=(2, 3, 4, 5), min_samples_leaf=20,
                       max_events=None, logger=None):
    """Publish independent explanation artifacts; never mutate benchmark stages."""
    _validate_settings(max_shapelets, shapelet_lengths, depths, min_samples_leaf, max_events)
    runs = discover_runs(Path(results).expanduser().resolve())
    if not runs:
        raise ValueError(f"No complete canonical runs found in {results}")
    available = {r['manifest']['model'] for r in runs}
    if models is not None and set(models) - available:
        raise ValueError(f"Requested models not found: {sorted(set(models) - available)}")
    outputs, failures, lookup = [], [], PreparedCacheLookup()
    for run in runs:
        if models is not None and run['manifest']['model'] not in models:
            continue
        directory = Path(run['directory'])
        try:
            with diagnostic_random_state():
                output = _explain_run(run, lookup, cache_root, max_shapelets, shapelet_lengths,
                                      depths, min_samples_leaf, max_events, logger)
            outputs.append(output)
        except Exception as exc:
            failures.append(f"{directory}: {exc}")
            if logger:
                logger.error("Regression explanation failed | %s | %s", directory, exc)
        finally:
            release_training_memory()
    if failures:
        raise RuntimeError("Some regression explanations failed; successful outputs retained:\n" + '\n'.join(failures))
    return outputs


def _explain_run(run, lookup, cache_override, max_shapelets, lengths, depths, leaf, maximum, logger):
    from waveform_analysis.reporting.regression_explainer import plot_explanation
    from waveform_analysis.reporting.stats import metric_values
    import sklearn
    from importlib.metadata import version

    directory, manifest = Path(run['directory']), run['manifest']
    config = resolve_run_config(directory)
    spec = get_model(manifest['model'])
    model_dir = directory / 'model'
    if not saved_model_complete(spec, model_dir):
        raise FileNotFoundError("Saved final model or frozen transform is missing/incomplete")
    root = _cache_root(config, directory, cache_override, roles=('development', 'blind'))
    development = lookup.find(root, 'development', config, manifest)
    blind = lookup.find(root, 'blind', config, manifest)
    if development.n_events < 10 or blind.n_events < 5:
        raise ValueError("Need >=10 development and >=5 blind events")
    mode = config['mode']
    # Equal temporal grids and normalization are required for meaningful distances.
    family = waveform_view(development, mode, [0]).family
    time = np.asarray(getattr(development, f'{family}_time_ps'))
    if not np.array_equal(time, getattr(blind, f'{family}_time_ps')):
        raise ValueError("Development/blind time grids differ")
    for key in ('minimum', 'maximum'):
        if not np.array_equal(getattr(getattr(development, f'{family}_transform'), key),
                              getattr(getattr(blind, f'{family}_transform'), key)):
            raise ValueError("Development/blind normalization differs")
    settings = {'max_shapelets': max_shapelets, 'shapelet_lengths': list(lengths), 'depths': list(depths),
                'min_samples_leaf': leaf, 'max_events_per_role': maximum}
    identity = {'method': 'global_shapelet_regression_tree', 'settings': settings, 'seed': config['seed'],
                'final_fit': manifest['stage_fingerprints']['final_fit'],
                'development': development.manifest['analysis_protocol_identity'],
                'blind': blind.manifest['analysis_protocol_identity'],
                'aeon': version('aeon'), 'sklearn': sklearn.__version__}
    destination = directory / 'explanations' / 'regression_tree' / canonical_hash(identity)[:16]
    if (destination / 'summary.json').is_file():
        if logger:
            logger.info("Global regression surrogate | reuse | %s", destination)
        return destination
    seed = semantic_seed(config['seed'], spec.name, 'regression_explainer')
    indices = {role: _sample(data, maximum, semantic_seed(seed, role))
               for role, data in [('development', development), ('blind', blind)]}
    chunk = int(config['runtime']['prediction_chunk_size'])
    fitted = load_fitted_model(spec, model_dir, run['best']['parameters'], config)
    try:
        if logger:
            logger.info("Global regression surrogate | %s | development=%d | blind=%d", spec.name,
                        len(indices['development']), len(indices['blind']))
        teacher = {role: predict_indices(spec, fitted, data, mode, indices[role], chunk_size=chunk)
                   for role, data in [('development', development), ('blind', blind)]}
        pair = waveform_view(development, mode, indices['development']).materialize()
        surrogate, selection = fit_surrogate(pair, teacher['development'], sample_mask=fitted.sample_mask,
            seed=seed, max_shapelets=max_shapelets, shapelet_lengths=lengths, depths=depths, min_samples_leaf=leaf)
    finally:
        del fitted
    features = _features_for_dataset(surrogate, blind, mode, indices['blind'], chunk)
    predicted = surrogate.tree.predict(features)
    target = model_target(blind, mode)[indices['blind']]
    metrics = []
    for label, residual in [('LED', target), ('Original', target - teacher['blind']), ('Surrogate', target - predicted)]:
        metrics.append({'model': label, **metric_values(residual, config['fit'], seed=semantic_seed(seed, 'comparison'))})
    shapelets = export_shapelets(surrogate.transformer)
    n = len(shapelets['shapelet_id'])
    names = [f'detector_{d + 1}_shapelet_{s}_distance' for d in range(2) for s in range(n)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.regression-', dir=destination.parent))
    try:
        summary = {**identity, 'teacher_model': spec.name, 'surrogate_seed': seed,
                   'fidelity_blind': fidelity(teacher['blind'], predicted),
                   'timing_blind': metrics, 'selection': selection,
                   'evaluation': 'paired held-out blind cohort; development predictions are teacher training-population outputs',
                   'warning': 'Surrogate is an approximation, not the original model; no antisymmetry constraint is imposed.'}
        atomic_json(temporary / 'summary.json', summary)
        with (temporary / 'surrogate.pkl').open('wb') as stream:
            pickle.dump(surrogate, stream, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez_compressed(temporary / 'shapelets.npz', **shapelets,
                            time_ps=apply_sample_mask_to_time(time, surrogate.sample_mask))
        np.savez_compressed(temporary / 'predictions.npz', event_id=blind.event_index[indices['blind']],
                            development_event_id=development.event_index[indices['development']],
                            original_ps=teacher['blind'], surrogate_ps=predicted, led_residual_ps=target,
                            leaf_id=surrogate.tree.apply(features))
        (temporary / 'rules.txt').write_text(export_text(surrogate.tree, feature_names=names, decimals=6), encoding='utf-8')
        with (temporary / 'performance.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['model', 'ctr_ps', 'rmse_ps'])
            writer.writeheader()
            writer.writerows(metrics)
        atomic_json(temporary / 'tree.json', {'feature_names': names, 'children_left': surrogate.tree.tree_.children_left,
            'children_right': surrogate.tree.tree_.children_right, 'feature': surrogate.tree.tree_.feature,
            'threshold': surrogate.tree.tree_.threshold, 'prediction_ps': surrogate.tree.tree_.value.ravel(),
            'development_n_events': surrogate.tree.tree_.n_node_samples,
            'development_prediction_sd_ps': np.sqrt(np.maximum(surrogate.tree.tree_.impurity, 0)),
            'blind_leaf_counts': dict(zip(*np.unique(surrogate.tree.apply(features), return_counts=True)))})
        plot_explanation(temporary, surrogate.tree, names, teacher['blind'], predicted, metrics, shapelets)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    if logger:
        score = summary['fidelity_blind']
        logger.info("Global regression surrogate | blind fidelity | MAE=%.3f ps | RMSE=%.3f ps | R2=%s",
                    score['mae_ps'], score['rmse_ps'], score['r2'])
        for row in metrics:
            logger.info("Global regression surrogate | %s | CTR=%.3f ps | RMSE=%.3f ps",
                        row['model'], row['ctr_ps'], row['rmse_ps'])
    return destination
