"""Canonical hierarchical result configuration and side-effect-free readers.

Only this layout is read by the pipeline. Historical layout recovery belongs to
scripts/convert_results.py and never runs implicitly during reporting.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

FORMAT = 'waveform-results'
ROLES = ('control', 'development', 'blind')
BATCH_SETTINGS = ('preprocessing', 'seed', 'cross_validation', 'bootstrap', 'runtime', 'xai', 'save_model')
CONTEXT_SETTINGS = ('fit', 'ml_input', 'ml_output')
CONTEXT_METADATA = (
    'dataset_populations', 'control_artifact', 'led_control_mean_ps',
    'led_centering_fit_role', 'sample_mask_identity', 'sample_mask_fit_role',
    'input_samples_before_mask', 'input_samples_after_mask',
)
PROTOCOL_METADATA = {
    'preprocessing_fit_role': 'control', 'model_selection_role': 'development',
    'blind_role': 'final_one_time_evaluation_only', 'blind_used_in_selection': False,
    'bootstrap_unit': 'blind_event',
    'blind_bootstrap_interpretation': 'event-level uncertainty conditional on final fitted model; no retraining',
}


def read_json(path, default=None):
    path = Path(path)
    if not path.is_file() and default is not None:
        return copy.deepcopy(default)
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read JSON {path}: {exc}') from exc
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _write(path, value):
    # Imported lazily to keep read-only/file-only reporting independent of numpy.
    from waveform_analysis.core.io import atomic_json
    path = Path(path)
    if not path.is_file() or read_json(path) != value:
        atomic_json(path, value)


def batch_root(run):
    run = Path(run).resolve()
    for parent in (run, *run.parents):
        path = parent / 'config.json'
        if path.is_file() and read_json(path).get('format') == FORMAT:
            return parent
    raise ValueError(f'No canonical batch configuration above {run}; convert old results with scripts/convert_results.py')



def write_run_config(config, *, managed_runs=None, validate_only=False):
    """Persist each setting once; expanded dictionaries are runtime-only."""
    run = Path(config['output_dir']).resolve()
    root = Path(config.get('batch_output_dir', run.parents[2])).resolve()
    if not run.is_relative_to(root):
        raise ValueError(f'Run must be inside its batch: {run}')
    managed = {Path(p).resolve() for p in (managed_runs or [run])}
    existing = read_json(root / 'config.json', {})
    if existing and existing.get('format') != FORMAT:
        raise ValueError(f'Noncanonical batch {root}; convert old results first')
    channels = [config[role]['channels'] for role in ROLES]
    if not all(c == channels[0] for c in channels):
        raise ValueError('Dataset channels/polarities must agree within one batch')
    datasets = {role: {k: v for k, v in config[role].items() if k != 'channels'} for role in ROLES}
    batch = {'format': FORMAT, 'name': config.get('study_name') or root.name,
             'datasets': datasets, 'channels': channels[0],
             'board': config.get('_board', existing.get('board')),
             'settings': {k: config[k] for k in BATCH_SETTINGS if k in config}}
    context_dir = run.parent if run != root else root
    mode_dir = context_dir.parent if context_dir.parent.is_relative_to(root) else root
    mode = {'mode': config['mode'], 'settings': {k: config[k] for k in CONTEXT_SETTINGS if k in config}}
    context = {'window_name': config.get('window_name', run.parent.name), 'window_ns': config['window_ns']}
    model = config['model']
    paths = [(root / 'config.json', batch), (root / 'plots.json', config.get('plot_config', {})),
             (mode_dir / 'mode.json', mode), (context_dir / 'context.json', context),
             (root / 'model_spaces' / f"{model['name']}.json", model['space'])]
    # Validate every owner before writing any shared document.
    for path, value in paths:
        if path.is_file() and read_json(path) != value:
            scope = root if path.parent.name == 'model_spaces' else path.parent
            retained = {p.parent.parent.resolve() for p in scope.rglob('metadata/config.json')} - managed
            if retained:
                raise ValueError(f'Changing {path} would relabel retained runs; use a new batch folder')
    if validate_only:
        return
    for path, value in paths:
        _write(path, value)
    _write(run / 'metadata/config.json', {'model': model['name']})


def resolve_run_config(run):
    run = Path(run).resolve()
    root = batch_root(run)
    batch = read_json(root / 'config.json')
    local = read_json(run / 'metadata/config.json')
    if set(local) != {'model'}:
        raise ValueError(f'Noncanonical model configuration in {run}; run scripts/convert_results.py')
    context_dir = run.parent if run != root else root
    mode_dir = context_dir.parent if context_dir.parent.is_relative_to(root) else root
    mode = read_json(mode_dir / 'mode.json')
    context = read_json(context_dir / 'context.json')
    name = local['model']
    tag = 'energy' if mode['mode'] == 'energy_to_energy' else 'timing'
    window = context['window_name']
    return {
        **copy.deepcopy(batch['settings']), **copy.deepcopy(mode['settings']),
        **{role: {**batch['datasets'][role], 'channels': copy.deepcopy(batch['channels'])} for role in ROLES},
        'model': {'name': name, 'space': read_json(root / 'model_spaces' / f'{name}.json')},
        'mode': mode['mode'], 'window_ns': context['window_ns'], 'window_name': window,
        'study_name': batch['name'], 'name': f"{batch['name']}__{tag}__{window}__{name}",
        'run_id': f'{tag}__{window}__{name}', 'output_dir': str(run), 'batch_output_dir': str(root),
        'plot_config': read_json(root / 'plots.json'),
    }


def write_run_manifest(run, manifest):
    run = Path(run).resolve()
    config = resolve_run_config(run)
    context_path = (run.parent if run != batch_root(run) else run) / 'population.json'
    shared = {k: manifest[k] for k in CONTEXT_METADATA if manifest.get(k) is not None}
    # Historical model-specific masks stay model-owned; the converter supplies
    # an explicit override rather than pretending they were shared.
    if 'dataset_populations' in shared and shared['dataset_populations']:
        shared['dataset_populations'] = {role: {k: v for k, v in data.items() if k != 'source'}
                                         for role, data in shared['dataset_populations'].items() if data}
    overrides = manifest.get('input_policy', {})
    for key in overrides:
        shared.pop(key, None)
    if context_path.is_file():
        previous = read_json(context_path)
        for key, value in shared.items():
            if key in previous and previous[key] != value:
                raise ValueError(f'Conflicting shared {key} in {context_path}; separate experiment contexts are required')
        shared = {**previous, **shared}
    _write(context_path, shared)
    derived = {'name', 'study_name', 'run_id', 'mode', 'window_name', 'window_ns', 'model',
               'control_dataset', 'development_dataset', 'blind_dataset', 'cv', 'single_configured_seed',
               'selected_candidate_id', 'selected_hyperparameters', 'validation_ctr_mean_ps',
               'validation_ctr_std_ps', 'validation_rmse_mean_ps', 'validation_rmse_std_ps'}
    local = {k: v for k, v in manifest.items() if k not in derived and k not in CONTEXT_METADATA and k not in PROTOCOL_METADATA}
    local.pop('schema_version', None)
    _write(run / 'metadata/manifest.json', local)


def read_run_manifest(run):
    run = Path(run).resolve()
    config = resolve_run_config(run)
    local = read_json(run / 'metadata/manifest.json')
    population = read_json((run.parent if run != batch_root(run) else run) / 'population.json', {})
    best = read_json(run / 'metadata/best.json', {})
    if population.get('dataset_populations'):
        population['dataset_populations'] = {role: {**data, 'source': config[role]['root_file']}
                                            for role, data in population['dataset_populations'].items()}
    method = local.get('selection_method', best.get('selection_method', 'development_cv'))
    return {**PROTOCOL_METADATA, **population, **local, **local.get('input_policy', {}),
            **{k: config[k] for k in ('name', 'study_name', 'run_id', 'mode', 'window_name', 'window_ns')},
            'model': config['model']['name'],
            **{f'{role}_dataset': config[role] for role in ROLES},
            'cv': config.get('cross_validation') if method != 'ridge_cv' else {'enabled': False, 'selection_method': method, 'internal_cv': best.get('internal_cv')},
            'single_configured_seed': config.get('seed'),
            'selected_candidate_id': best.get('candidate_id'), 'selected_hyperparameters': best.get('parameters'),
            **{k: best[k] for k in best if k.startswith('validation_')}}


def read_run(run):
    run = Path(run).resolve()
    return {'directory': run, 'manifest': read_run_manifest(run),
            **{key: read_json(run / 'metadata' / f'{key}.json', {}) for key in ('best', 'blind', 'bootstrap')}}


def discover_runs(root, *, complete=True):
    """One discovery path for reporting/grouping/XAI, without changing sources."""
    root = Path(root).expanduser().resolve()
    runs = []
    for path in sorted(root.rglob('metadata/config.json')):
        run = path.parent.parent
        # Grouped reports reference original runs; they contain no copied runs.
        result = read_run(run)
        if not complete or result['manifest'].get('status') == 'complete':
            runs.append(result)
    return runs
