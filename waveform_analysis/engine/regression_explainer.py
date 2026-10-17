"""Global explanations in the original model's frozen feature space (not LASTS)."""
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
from waveform_analysis.engine.train import load_fitted_model, _prediction_from_input, release_training_memory, saved_model_complete
from waveform_analysis.engine.xai_replay import PreparedCacheLookup, _cache_root
from waveform_analysis.models import get_model
from waveform_analysis.models.kernel.rdst_common import export_shapelets
from waveform_analysis.signal.sample_mask import apply_sample_mask, apply_sample_mask_to_time


@dataclass
class FrozenFeatureRegressionSurrogate:
    """Approximate tree; predict accepts ORIGINAL frozen feature matrices only."""
    tree: DecisionTreeRegressor
    transform_identity: str

    def predict(self, features):
        return self.tree.predict(features)


def fidelity(original, surrogate):
    original, surrogate = np.asarray(original), np.asarray(surrogate)
    if original.shape != surrogate.shape or original.ndim != 1 or not np.all(np.isfinite([original, surrogate])):
        raise ValueError("Fidelity requires matching finite prediction vectors")
    return {"n_events": len(original), "mae_ps": float(mean_absolute_error(original, surrogate)),
            "rmse_ps": float(np.sqrt(mean_squared_error(original, surrogate))),
            "r2": float(r2_score(original, surrogate)) if np.var(original) > 0 else None}


def fit_surrogate(features, predictions, *, transform_identity, seed,
                  depths=(2, 3, 4, 5), min_samples_leaf=20):
    """Fit only a tree: no transform construction, fitting, or scaling."""
    features = np.asarray(features, dtype=np.float32)
    predictions = np.asarray(predictions, dtype=float)
    if features.ndim != 2 or len(features) < 10:
        raise ValueError("Surrogate fitting requires >=10 development frozen-feature rows")
    if predictions.shape != (len(features),) or not np.all(np.isfinite(predictions)):
        raise ValueError("Teacher predictions must be finite and aligned")
    train, validation = train_test_split(np.arange(len(features)), test_size=.2, random_state=seed)
    selection = []
    for depth in sorted(set(depths)):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=min_samples_leaf, random_state=seed)
        tree.fit(features[train], predictions[train])
        selection.append({"max_depth": depth, **fidelity(predictions[validation], tree.predict(features[validation]))})
    best = min(selection, key=lambda row: (row['rmse_ps'], row['max_depth']))
    tree = DecisionTreeRegressor(max_depth=best['max_depth'], min_samples_leaf=min_samples_leaf, random_state=seed)
    tree.fit(features, predictions)
    return FrozenFeatureRegressionSurrogate(tree, transform_identity), {
        "selection": selection, "selected_max_depth": best['max_depth'],
        "train_indices": train, "validation_indices": validation,
        "feature_definition": "original saved transform output, including original scaling and detector formulation",
        "transform_fit_role": "original_saved_model; never refitted", "validation_fraction": .2}


def frozen_features(spec, fitted, dataset, config, indices, cache_root, role, scratch):
    """Read verified existing features, otherwise apply the saved transform once."""
    from waveform_analysis.data.feature_cache import _CACHE_VERSION, _transform_root
    transform = fitted.feature_transform
    identity = canonical_hash({"schema_version": _CACHE_VERSION, "transform_identity": transform.identity,
        "dataset_identity": dataset.manifest['analysis_protocol_identity'], "role": role})
    directory = _transform_root(cache_root, config['mode'], config['window_name'], spec.name, transform.identity)
    metadata_path = directory / f'{role}_{identity[:12]}.json'
    path = directory / f'{role}_{identity[:12]}.npy'
    if transform.identity != 'reloaded' and metadata_path.is_file() and path.is_file():
        import json
        matrix = None
        try:
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (metadata.get('identity') == identity and metadata.get('transform_identity') == transform.identity
                    and metadata.get('dataset_identity') == dataset.manifest['analysis_protocol_identity']
                    and metadata.get('role') == role):
                matrix = np.load(path, mmap_mode='r')
                if matrix.shape == (dataset.n_events, metadata.get('feature_count')):
                    if np.array_equal(indices, np.arange(dataset.n_events)):
                        return matrix, {'source': 'verified_existing_cache', 'path': str(path)}
                    subset = np.asarray(matrix[indices], dtype=np.float32)
                    matrix._mmap.close()
                    return subset, {'source': 'verified_existing_cache', 'path': str(path)}
                matrix._mmap.close()
        except (OSError, ValueError, TypeError, AttributeError):
            if isinstance(matrix, np.memmap):
                matrix._mmap.close()
    chunk = int(config['runtime']['prediction_chunk_size'])
    if spec.bounded_transform_batches:
        chunk = min(chunk, int(transform.artifact.metadata['batch_size']))
    matrix = None
    try:
        for start in range(0, len(indices), chunk):
            pair = waveform_view(dataset, config['mode'], indices[start:start + chunk]).materialize()
            block = np.asarray(transform.apply(apply_sample_mask(pair, fitted.sample_mask)), dtype=np.float32)
            if block.ndim != 2 or not np.all(np.isfinite(block)):
                raise ValueError('Saved transform returned invalid features')
            if matrix is None:
                matrix = np.lib.format.open_memmap(scratch / f'{role}.npy', mode='w+', dtype=np.float32,
                                                    shape=(len(indices), block.shape[1]))
            matrix[start:start + len(block)] = block
        matrix.flush()
    except BaseException:
        if isinstance(matrix, np.memmap):
            matrix._mmap.close()  # Allow temporary-file removal on Windows too.
        raise
    return matrix, {'source': 'applied_original_frozen_transform', 'transform_identity': transform.identity}


def predict_features(spec, fitted, features, chunk):
    return np.concatenate([_prediction_from_input(spec, fitted, features[start:start + chunk])
                           for start in range(0, len(features), chunk)])


def contribution_summary(fitted, features, teacher, *, group_size, chunk):
    """Exact uncentered linear contributions; aggregate signed groups BEFORE abs."""
    regressor = getattr(fitted.artifact, 'regressor', None)
    if regressor is None or not hasattr(regressor, 'coef_'):
        return None
    weights = np.asarray(regressor.coef_, dtype=np.float64)
    epsilon = np.finfo(np.result_type(features.dtype, np.asarray(regressor.coef_).dtype)).eps
    if weights.shape != (features.shape[1],) or len(weights) % group_size:
        raise ValueError('Original coefficient/frozen-feature mapping mismatch')
    intercept = float(regressor.intercept_)
    groups = len(weights) // group_size
    sums, squares, absolute, positive = [np.zeros(groups) for _ in range(4)]
    feature_sums = np.zeros(len(weights))
    feature_abs = np.zeros(len(weights))
    max_error, clipped_events = 0., 0
    for start in range(0, len(features), chunk):
        contributions = np.asarray(features[start:start + chunk], dtype=np.float64) * weights
        grouped = contributions.reshape(-1, groups, group_size).sum(axis=2)
        sums += grouped.sum(axis=0)
        squares += (grouped ** 2).sum(axis=0)
        absolute += np.abs(grouped).sum(axis=0)
        positive += (grouped > 0).sum(axis=0)
        feature_sums += contributions.sum(axis=0)
        feature_abs += np.abs(contributions).sum(axis=0)
        reconstructed = intercept + grouped.sum(axis=1)
        if fitted.output_max_abs_ps is not None:
            clipped_events += int(np.count_nonzero(np.abs(reconstructed) > fitted.output_max_abs_ps))
            reconstructed = np.clip(reconstructed, -fitted.output_max_abs_ps, fitted.output_max_abs_ps)
        expected = teacher[start:start + len(contributions)]
        max_error = max(max_error, float(np.max(np.abs(reconstructed - expected))))
        # The native head may use float32 BLAS; the explanatory sum uses float64.
        # Bound rounding by the magnitude of all terms, including cancellations.
        tolerance = 1e-4 + 8 * epsilon * (np.abs(contributions).sum(axis=1) + abs(intercept))
        if np.any(np.abs(reconstructed - expected) > tolerance):
            raise ValueError('Additive reconstruction does not match original predictions; refusing invalid explanation')
    count = len(features)
    mean = sums / count
    return {'group_id': np.arange(groups), 'mean_contribution_ps': mean,
        'std_contribution_ps': np.sqrt(np.maximum(squares / count - mean ** 2, 0)),
        'mean_absolute_contribution_ps': absolute / count, 'positive_fraction': positive / count,
        'feature_mean_contribution_ps': feature_sums / count,
        'feature_mean_absolute_contribution_ps': feature_abs / count,
        'coefficients': weights, 'intercept_ps': np.asarray(intercept),
        'ranking': np.lexsort((np.arange(groups), -absolute)),
        'group_size': np.asarray(group_size), 'n_events': np.asarray(count),
        'max_reconstruction_error_ps': np.asarray(max_error), 'clipped_events': np.asarray(clipped_events)}


def _sample(dataset, maximum, seed):
    count = dataset.n_events if maximum is None else min(dataset.n_events, maximum)
    return np.sort(np.random.default_rng(seed).choice(dataset.n_events, count, replace=False))


def _validate_settings(depths, min_samples_leaf, max_events):
    for name, values, minimum in (("depths", depths, 1), ("min_samples_leaf", [min_samples_leaf], 1),
                                  ("max_events", [] if max_events is None else [max_events], 10)):
        if (name == 'depths' and not values) or any(
            isinstance(v, bool) or not isinstance(v, int) or v < minimum for v in values
        ):
            raise ValueError(f"{name} requires integers >= {minimum}")


def explain_regression(results, *, models=None, cache_root=None, depths=(2, 3, 4, 5), min_samples_leaf=20,
                       max_events=None, logger=None):
    """Publish independent explanation artifacts; never mutate benchmark stages."""
    _validate_settings(depths, min_samples_leaf, max_events)
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
                output = _explain_run(run, lookup, cache_root, depths, min_samples_leaf, max_events, logger)
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


def _explain_run(run, lookup, cache_override, depths, leaf, maximum, logger):
    from waveform_analysis.reporting.regression_explainer import plot_explanation
    from waveform_analysis.reporting.stats import metric_values
    import sklearn
    from importlib.metadata import version

    directory, manifest = Path(run['directory']), run['manifest']
    config = resolve_run_config(directory)
    spec = get_model(manifest['model'])
    if spec.feature_transform is None:
        raise ValueError('This model has no explicit frozen feature transform; use input-level XAI for CNN/MLP')
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
    settings = {'depths': list(depths),
                'min_samples_leaf': leaf, 'max_events_per_role': maximum}
    identity = {'method': 'original_frozen_feature_regression_explanation', 'settings': settings, 'seed': config['seed'],
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
    if fitted.feature_transform is None:
        raise ValueError('Saved original frozen transform is missing; it will not be fitted again')
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrices, feature_sources = {}, {}
    try:
        with tempfile.TemporaryDirectory(prefix='.frozen-features-', dir=destination.parent) as scratch:
            try:
                for role, data in [('development', development), ('blind', blind)]:
                    if logger:
                        logger.info('Original frozen features | %s | %s | events=%d', spec.name, role, len(indices[role]))
                    matrices[role], feature_sources[role] = frozen_features(
                        spec, fitted, data, config, indices[role], root, role, Path(scratch))
                teacher = {role: predict_features(spec, fitted, matrix, chunk) for role, matrix in matrices.items()}
                surrogate, selection = fit_surrogate(matrices['development'], teacher['development'],
                    transform_identity=fitted.feature_transform.identity, seed=seed, depths=depths, min_samples_leaf=leaf)
                predicted = surrogate.predict(matrices['blind'])
                leaf_id = surrogate.tree.apply(matrices['blind'])
                is_rdst = spec.name in {'direct_rdst', 'shared_rdst'}
                shapelets = export_shapelets(fitted.feature_transform.artifact.transformer) if is_rdst else None
                contributions = contribution_summary(fitted, matrices['blind'], teacher['blind'],
                                                     group_size=3 if is_rdst else 1, chunk=chunk)
                feature_count = matrices['development'].shape[1]
                mask = fitted.sample_mask
            finally:
                for matrix in matrices.values():
                    if isinstance(matrix, np.memmap):
                        matrix._mmap.close()
                matrices.clear()
    finally:
        del fitted
    target = model_target(blind, mode)[indices['blind']]
    metrics = []
    for label, residual in [('LED', target), ('Original', target - teacher['blind']), ('Surrogate', target - predicted)]:
        metrics.append({'model': label, **metric_values(residual, config['fit'], seed=semantic_seed(seed, 'comparison'))})
    names = ([f'shapelet_{s}_{feature}' for s in shapelets['shapelet_id']
              for feature in ('minimum_distance', 'match_location', 'occurrence_count')]
             if shapelets is not None else [f'original_feature_{i}' for i in range(feature_count)])
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.regression-', dir=destination.parent))
    try:
        summary = {**identity, 'teacher_model': spec.name, 'surrogate_seed': seed,
                   'fidelity_blind': fidelity(teacher['blind'], predicted),
                   'timing_blind': metrics, 'selection': selection, 'feature_sources': feature_sources,
                   'transform_identity': surrogate.transform_identity, 'formulation': spec.estimator_formulation,
                   'exact_contributions': contributions is not None,
                   'contribution_definition': 'intercept + sum(feature * original coefficient), before output clipping',
                   'contribution_cohort': 'blind',
                   'contribution_group': 'original_shapelet' if is_rdst else 'original_feature',
                   'max_reconstruction_error_ps': None if contributions is None else float(contributions['max_reconstruction_error_ps']),
                   'clipped_events': None if contributions is None else int(contributions['clipped_events']),
                   'evaluation': 'paired held-out blind cohort; development predictions are teacher training-population outputs',
                   'warning': 'Surrogate is an approximation, not the original model; no antisymmetry constraint is imposed.'}
        atomic_json(temporary / 'summary.json', summary)
        with (temporary / 'surrogate.pkl').open('wb') as stream:
            pickle.dump(surrogate, stream, protocol=pickle.HIGHEST_PROTOCOL)
        if shapelets is not None:
            np.savez_compressed(temporary / 'shapelets.npz', **shapelets,
                                time_ps=apply_sample_mask_to_time(time, mask))
        if contributions is not None:
            np.savez_compressed(temporary / 'contributions.npz', **contributions)
            with (temporary / 'contributions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['rank', 'original_shapelet_id' if shapelets is not None else 'original_feature_id',
                                 'mean_absolute_net_contribution_ps', 'mean_contribution_ps', 'population_sd_ps', 'positive_fraction'])
                for rank, group in enumerate(contributions['ranking'], 1):
                    writer.writerow([rank, group, contributions['mean_absolute_contribution_ps'][group],
                                     contributions['mean_contribution_ps'][group], contributions['std_contribution_ps'][group],
                                     contributions['positive_fraction'][group]])
        np.savez_compressed(temporary / 'predictions.npz', event_id=blind.event_index[indices['blind']],
                            development_event_id=development.event_index[indices['development']],
                            original_ps=teacher['blind'], surrogate_ps=predicted, led_residual_ps=target,
                            leaf_id=leaf_id)
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
            'blind_leaf_counts': dict(zip(*np.unique(leaf_id, return_counts=True)))})
        plot_explanation(temporary, surrogate.tree, names, teacher['blind'], predicted, metrics, shapelets, contributions, spec.estimator_formulation)
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
