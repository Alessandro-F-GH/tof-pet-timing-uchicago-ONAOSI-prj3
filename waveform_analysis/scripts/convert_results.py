"""One-time recovery of trusted historical results into the canonical hierarchy.

This file is deliberately self-contained and removable after conversion. Normal
pipeline modules never import it. Sources are read-only; no fitting, prediction,
selection, preprocessing or statistical estimation is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from waveform_analysis.data.results import (
    BATCH_SETTINGS, CONTEXT_SETTINGS, CONTEXT_METADATA, FORMAT, PROTOCOL_METADATA,
    ROLES, read_json, read_run, write_run_config, write_run_manifest,
)

# This is the only historical Python import map. It can be deleted with this file.
OLD_MODULES = {
    "waveform_analysis.ml_pipeline.models": "waveform_analysis.models",
    "waveform_analysis.ml_pipeline.config": "waveform_analysis.core.config",
    "waveform_analysis.ml_pipeline.common": "waveform_analysis.core.io",
    "waveform_analysis.ml_pipeline.progress": "waveform_analysis.core.progress",
    "waveform_analysis.ml_pipeline.data": "waveform_analysis.data.preprocessing",
    "waveform_analysis.ml_pipeline.dataset": "waveform_analysis.data.dataset",
    "waveform_analysis.ml_pipeline.energy_io": "waveform_analysis.data.root_io",
    "waveform_analysis.ml_pipeline.feature_cache": "waveform_analysis.data.feature_cache",
    "waveform_analysis.ml_pipeline.prepared_data": "waveform_analysis.data.preparation",
    "waveform_analysis.ml_pipeline.shared_artifacts": "waveform_analysis.data.shared_artifacts",
    "waveform_analysis.ml_pipeline.splits": "waveform_analysis.data.splits",
    "waveform_analysis.ml_pipeline.storage": "waveform_analysis.data.storage",
    "waveform_analysis.ml_pipeline.view": "waveform_analysis.data.view",
    "waveform_analysis.ml_pipeline.timing": "waveform_analysis.data.timing_adapter",
    "waveform_analysis.ml_pipeline.sample_mask": "waveform_analysis.signal.sample_mask",
    "waveform_analysis.ml_pipeline.batch": "waveform_analysis.engine.batch",
    "waveform_analysis.ml_pipeline.control_preprocessing": "waveform_analysis.engine.control_preprocessing",
    "waveform_analysis.ml_pipeline.event_selection": "waveform_analysis.engine.event_selection",
    "waveform_analysis.ml_pipeline.preprocessing": "waveform_analysis.engine.preprocessing",
    "waveform_analysis.ml_pipeline.search": "waveform_analysis.engine.search",
    "waveform_analysis.ml_pipeline.study": "waveform_analysis.engine.study",
    "waveform_analysis.ml_pipeline.train": "waveform_analysis.engine.train",
    "waveform_analysis.ml_pipeline.validation": "waveform_analysis.engine.validation",
    "waveform_analysis.ml_pipeline.xai": "waveform_analysis.engine.xai",
    "waveform_analysis.ml_pipeline.plotting": "waveform_analysis.reporting.plotting",
    "waveform_analysis.ml_pipeline.preprocessing_plots": "waveform_analysis.reporting.preprocessing_plots",
    "waveform_analysis.ml_pipeline.stats": "waveform_analysis.reporting.stats",
    "waveform_analysis.ml_pipeline.report": "waveform_analysis.reporting.report",
    "waveform_analysis.ml_pipeline.report_engine": "waveform_analysis.reporting.report_engine",
    "waveform_analysis.ml_pipeline.postprocess": "waveform_analysis.reporting.postprocess",
    "waveform_analysis.ml_pipeline.models.shared_linear_ridge": "waveform_analysis.models.linear.shared_linear_ridge",
    "waveform_analysis.ml_pipeline.models._torch_common": "waveform_analysis.models.torch_runtime",
    "waveform_analysis.ml_pipeline.models.shared_cnn1d": "waveform_analysis.models.neural.shared_cnn1d",
    "waveform_analysis.ml_pipeline.models.locally_connected_mlp": "waveform_analysis.models.neural.locally_connected_mlp",
    "waveform_analysis.ml_pipeline.models.spec": "waveform_analysis.models.spec",
    "waveform_analysis.ml_pipeline.models.direct_mlp": "waveform_analysis.models.neural.direct_mlp",
    "waveform_analysis.ml_pipeline.models.direct_minirocket": "waveform_analysis.models.kernel.direct_minirocket",
    "waveform_analysis.ml_pipeline.models.shared_minirocket": "waveform_analysis.models.kernel.shared_minirocket",
    "waveform_analysis.ml_pipeline.models.independent_cnn1d": "waveform_analysis.models.neural.independent_cnn1d",
    "waveform_analysis.ml_pipeline.models.onishi_cnn": "waveform_analysis.models.neural.onishi_cnn",
    "waveform_analysis.ml_pipeline.models.registry": "waveform_analysis.models.registry",
    "waveform_analysis.ml_pipeline.models._linear_ridge_common": "waveform_analysis.models.linear.linear_ridge_common",
    "waveform_analysis.ml_pipeline.models.antisymmetric_mlp": "waveform_analysis.models.neural.antisymmetric_mlp",
    "waveform_analysis.ml_pipeline.models._cnn1d_common": "waveform_analysis.models.neural.cnn1d_common",
    "waveform_analysis.ml_pipeline.models._mlp_common": "waveform_analysis.engine.neural_training",
    "waveform_analysis.ml_pipeline.models.direct_linear_ridge": "waveform_analysis.models.linear.direct_linear_ridge",
    "waveform_analysis.utils.peak": "waveform_analysis.signal.peak",
    "waveform_analysis.utils.photopeak": "waveform_analysis.signal.photopeak"
}


class HistoricalUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(OLD_MODULES.get(module, module), name)


def load_historical(stream):
    return HistoricalUnpickler(stream).load()


def _torch_pickle_module():
    module = types.ModuleType('conversion_pickle')
    module.__dict__.update(pickle.__dict__)
    module.Unpickler = HistoricalUnpickler
    module.load = load_historical
    module.loads = lambda data, **kwargs: HistoricalUnpickler(io.BytesIO(data), **kwargs).load()
    return module


def rewrite_serialized(path):
    """Change Python import references, retaining fitted objects and tensor values."""
    if path.suffix == '.pkl':
        with path.open('rb') as stream:
            value = load_historical(stream)
        with path.open('wb') as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elif path.suffix == '.pt':
        import torch
        value = torch.load(path, map_location='cpu', pickle_module=_torch_pickle_module(), weights_only=False)
        torch.save(value, path)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(value):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value)).strip('._') or 'unknown'


def _saved(root, name):
    modern = root / 'metadata' / f'{name}.json'
    return modern if modern.is_file() else root / f'{name}.json'


def _old_batch(run):
    for parent in run.parents:
        config = read_json(parent / 'config.json', {})
        if ('protocol' in config or 'datasets' in config or
                (parent / 'tables/runs.csv').is_file() or (parent / 'runs.csv').is_file()):
            return parent, config
    return run.parent, {}


def _object(value, *, owner):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        path = Path(value)
        path = path if path.is_absolute() else owner / path
        if path.suffix.lower() == '.json' and path.is_file():
            return read_json(path)
    return {}


@dataclass
class OldRun:
    source: Path
    batch: Path
    config: dict
    manifest: dict
    destination: str = ''


def recover_config(run, manifest):
    saved = read_json(_saved(run, 'config'), {})
    batch, common = _old_batch(run)
    protocol = _object(common.get('protocol'), owner=batch)
    expanded = [entry for entry in common.get('runs', []) if isinstance(entry, dict)
                and isinstance(entry.get('model'), dict)
                and entry['model'].get('name') == manifest.get('model')
                and entry.get('mode') == manifest.get('mode')
                and entry.get('window_ns') == manifest.get('window_ns')]
    if len(expanded) == 1:
        saved = {**expanded[0], **saved}
    mode = saved.get('mode') or manifest.get('mode')
    window = saved.get('window_ns') or manifest.get('window_ns')
    if mode not in {'energy_to_energy', 'timing_to_timing'} or not isinstance(window, dict) or not {'start', 'end'} <= window.keys():
        raise ValueError('Missing waveform mode or numerical window bounds')
    if not all(math.isfinite(float(window[k])) for k in ('start', 'end')) or float(window['end']) <= float(window['start']):
        raise ValueError('Invalid numerical window bounds')
    model = manifest.get('model')
    if not isinstance(model, str) or not model:
        raise ValueError('Missing model name')
    datasets = {}
    for role in ROLES:
        value = saved.get(role) or manifest.get(f'{role}_dataset') or common.get('datasets', {}).get(role) or common.get(f'{role}_dataset')
        value = {'root_file': value} if isinstance(value, str) and not value.lower().endswith('.json') else _object(value, owner=batch)
        if not value.get('root_file'):
            population = (manifest.get('dataset_populations') or {}).get(role) or {}
            value['root_file'] = population.get('source')
        if not value.get('root_file'):
            raise ValueError(f'Missing {role} dataset; cannot reconstruct a three-role experiment')
        value.setdefault('true_tof_ps', None)
        value.setdefault('channels', None)
        datasets[role] = value
    channels = [value['channels'] for value in datasets.values()]
    if not all(c == channels[0] for c in channels):
        raise ValueError('Role-specific channel definitions differ; a canonical experiment requires one channel definition')
    settings = {key: saved[key] if key in saved else protocol.get(key) for key in BATCH_SETTINGS}
    preprocessing = settings.get('preprocessing') or saved.get('preprocessing') or protocol.get('preprocessing_config')
    preprocessing = _object(preprocessing, owner=batch)
    settings['preprocessing'] = preprocessing.get('preprocessing', preprocessing)
    for key in CONTEXT_SETTINGS:
        value = saved.get(key, protocol.get(key))
        settings[key] = value.get(mode) if isinstance(value, dict) and mode in value else value
    if (not isinstance(settings.get('ml_input'), dict) or
            isinstance(settings['ml_input'].get('subsampling'), bool) or
            not isinstance(settings['ml_input'].get('subsampling'), int)):
        raise ValueError('Missing recorded subsampling; it is never guessed')
    if settings['ml_input']['subsampling'] < 1:
        raise ValueError('Invalid recorded subsampling')
    settings['fit'] = settings.get('fit') or {}
    settings['runtime'] = settings.get('runtime') or {'prediction_chunk_size': 4096}
    settings['xai'] = settings.get('xai') or {'enabled': False, 'group_size_samples': 8, 'max_events': 512}
    model_config = saved.get('model') if isinstance(saved.get('model'), dict) else {}
    space = model_config.get('space') or common.get('models', {}).get(model) or {}
    labels = f'{run.as_posix()} {common.get("results", {}).get("folder", "")}'
    boards = set(re.findall(r'(?:^|[/\\ ])(UC|FBK)(?=$|[/\\ ])', labels.upper()))
    board = next(iter(boards)) if len(boards) == 1 else None
    return OldRun(run, batch, {
        **settings, **datasets, 'model': {'name': model, 'space': space},
        'mode': mode, 'window_name': str(saved.get('window_name') or manifest.get('window_name') or f"window_{window['start']}_{window['end']}"),
        'window_ns': window, 'study_name': str(saved.get('study_name') or common.get('name') or batch.name),
        'plot_config': saved.get('plot_config') or read_json(batch / 'plots.json', {}), '_board': board,
    }, manifest)


def discover(sources, output):
    runs, skipped, seen = [], [], set()
    for source in sources:
        source = Path(source).expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f'Source does not exist: {source}')
        for path in sorted(source.rglob('manifest.json')):
            if path.is_relative_to(output):
                continue
            run = path.parent.parent if path.parent.name == 'metadata' else path.parent
            if run in seen or any((p / 'grouping_manifest.json').is_file() for p in (run, *run.parents)):
                continue
            manifest = read_json(_saved(run, 'manifest'), {})
            if not isinstance(manifest.get('model'), str):
                continue
            seen.add(run)
            if manifest.get('status') != 'complete':
                skipped.append({'source': str(run), 'reason': 'Run is not complete'})
                continue
            try:
                runs.append(recover_config(run, manifest))
            except (ValueError, TypeError, KeyError) as exc:
                skipped.append({'source': str(run), 'reason': str(exc)})
    return runs, skipped


def assign_destinations(runs):
    """Partition conflicting settings instead of silently picking one value."""
    batches = defaultdict(list)
    for run in runs:
        batches[run.batch].append(run)
    names = {}
    for batch, members in sorted(batches.items()):
        packs = []
        for run in members:
            config = run.config
            shared = {key: config.get(key) for key in (*BATCH_SETTINGS, *ROLES, 'plot_config')}
            context = (config['mode'], config['window_name'])
            mode_settings = {key: config.get(key) for key in CONTEXT_SETTINGS}
            for pack in packs:
                compatible = (pack['shared'] == shared and
                              pack['modes'].get(config['mode'], mode_settings) == mode_settings and
                              pack['windows'].get(context, config['window_ns']) == config['window_ns'] and
                              pack['spaces'].get(config['model']['name'], config['model']['space']) == config['model']['space'] and
                              (*context, config['model']['name']) not in pack['keys'])
                if compatible:
                    break
            else:
                pack = {'shared': shared, 'modes': {}, 'windows': {}, 'spaces': {}, 'keys': set(), 'runs': []}
                packs.append(pack)
            pack['modes'][config['mode']] = mode_settings
            pack['windows'][context] = config['window_ns']
            pack['spaces'][config['model']['name']] = config['model']['space']
            pack['keys'].add((*context, config['model']['name']))
            pack['runs'].append(run)
        for number, pack in enumerate(packs, 1):
            first = pack['runs'][0]
            name = f"{first.config['_board'] or 'UNKNOWN'}/{_safe(batch.name)}"
            if len(packs) > 1:
                name += f'_variant{number}'
            if name in names and names[name] != batch:
                name += '_' + _digest(str(batch))
            names[name] = batch
            for run in pack['runs']:
                cfg = run.config
                mode = 'energy' if cfg['mode'] == 'energy_to_energy' else 'timing'
                run.destination = f"{name}/{mode}/{_safe(cfg['window_name'])}/{_safe(cfg['model']['name'])}"


def _copy_run(run, destination, *, metrics_only):
    copied = []
    metadata_names = {'manifest', 'state', 'config', 'candidates', 'best', 'final_fit', 'blind', 'bootstrap', 'xai'}
    for source in sorted(run.source.rglob('*')):
        if not source.is_file() or source.is_symlink():
            continue
        relative = source.relative_to(run.source)
        if len(relative.parts) == 1 and (run.source / 'metadata' / source.name).is_file():
            continue
        if metrics_only and ('model' in relative.parts or source.suffix in {'.pkl', '.pt', '.db'}):
            continue
        if len(relative.parts) == 1:
            if source.suffix == '.json' and source.stem in metadata_names:
                relative = Path('metadata') / relative
            elif source.name in {'folds.csv', 'cv.csv'}:
                relative = Path('tables') / relative
            elif source.suffix == '.npz' and source.name in {'pred.npz', 'development_pred.npz', 'bootstrap.npz', 'xai.npz'}:
                relative = Path('artifacts') / relative
            elif source.suffix in {'.png', '.svg', '.pdf', '.eps', '.jpg'}:
                relative = Path('plots') / relative
            elif source.name in {'model.pkl', 'model.pt', 'sample_mask.npy'}:
                relative = Path('model') / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        before = _file_hash(source)
        if target.suffix in {'.pkl', '.pt'}:
            try:
                rewrite_serialized(target)
            except Exception as exc:
                raise ValueError(f'Cannot convert {source}: {exc}. Use --metrics-only to omit serialized models.') from exc
        after = _file_hash(target)
        if target.suffix not in {'.pkl', '.pt'} and before != after:
            raise ValueError(f'File changed while copying: {source}')
        copied.append({'source': str(source), 'destination': relative.as_posix(),
                       'source_sha256': before, 'converted_sha256': after})
    return copied


def _policies(runs):
    contexts = defaultdict(list)
    for run in runs:
        contexts[str(Path(run.destination).parent)].append(run)
    for members in contexts.values():
        for run in members:
            manifest = dict(run.manifest)
            override = dict(manifest.get('input_policy', {}))
            for key in CONTEXT_METADATA:
                values = [r.manifest.get(key) for r in members]
                if any(v != values[0] for v in values):
                    override[key] = manifest.get(key)
            for key, default in PROTOCOL_METADATA.items():
                if key in manifest and manifest[key] != default:
                    override[key] = manifest[key]
            if override:
                manifest['input_policy'] = override
            yield run, manifest


def convert_results(sources, output, *, check=False, metrics_only=False, cache_source=None):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise ValueError(f'Destination must be new; originals and earlier conversions are never overwritten: {output}')
    for source in sources:
        if Path(source).expanduser().resolve().is_relative_to(output):
            raise ValueError('Destination must not contain a source')
    runs, skipped = discover(sources, output)
    assign_destinations(runs)
    if any(output.is_relative_to(run.source) for run in runs):
        raise ValueError('Destination must not be inside a source model run')
    if cache_source is not None:
        cache_source = Path(cache_source).expanduser().resolve()
        if output.is_relative_to(cache_source) or cache_source.is_relative_to(output):
            raise ValueError('Cache source and conversion destination must be separate trees')
    plan = {'runs': [{'source': str(r.source), 'destination': r.destination} for r in runs], 'skipped': skipped}
    if check:
        return plan
    if not runs:
        raise ValueError('No recoverable complete runs; use --check for the skipped reasons')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-conversion-', dir=output.parent))
    receipts = []
    try:
        for run, manifest in _policies(runs):
            target = stage / run.destination
            files = _copy_run(run, target, metrics_only=metrics_only)
            cfg = {**run.config, 'output_dir': str(target), 'batch_output_dir': str(target.parents[2])}
            if cache_source is not None:
                cfg['preprocessing'] = {**cfg['preprocessing'], 'cache_dir': str(output / '_caches')}
            write_run_config(cfg)
            manifest['conversion'] = {'source': str(run.source), 'original_schema': run.manifest.get('schema_version'),
                                      'models_recovered': not metrics_only}
            write_run_manifest(target, manifest)
            state_path = target / 'metadata/state.json'
            if state_path.is_file():
                state = read_json(state_path)
                state.pop('schema_version', None)
                state_path.write_text(json.dumps(state, indent=2) + '\n')
            # Recovered metrics are read, not recalculated. Validate the hierarchy
            # while it is still staged and leave the source bytes untouched.
            restored = read_run(target)
            for metric in ('best', 'blind', 'bootstrap'):
                expected = read_json(_saved(run.source, metric), {})
                if _digest(restored[metric]) != _digest(expected):
                    raise ValueError(f'Recovered {metric} differs: {run.source}')
            for item in files:
                item['converted_sha256'] = _file_hash(target / item['destination'])
            receipts.append({'source': str(run.source), 'destination': run.destination, 'files': files})
        batches = {}
        for run in runs:
            batches.setdefault(Path(run.destination).parents[2].as_posix(), run.batch)
        for relative, source_batch in batches.items():
            target_batch = stage / relative
            for name in ('artifacts', 'preprocessing'):
                source = source_batch / name
                if source.is_dir():
                    shutil.copytree(source, target_batch / name, dirs_exist_ok=True)
                    for path in (target_batch / name).rglob('*.pkl'):
                        rewrite_serialized(path)
            index = []
            for run in runs:
                if Path(run.destination).parents[2].as_posix() != relative:
                    continue
                target = stage / run.destination
                config = read_run(target)['manifest']
                index.append({'run_id': config['run_id'], 'model': config['model'],
                              'mode': config['mode'], 'window': config['window_name'],
                              'status': 'complete', 'path': target.relative_to(target_batch).as_posix()})
            import csv
            (target_batch / 'tables').mkdir(exist_ok=True)
            with (target_batch / 'tables/runs.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(index[0]))
                writer.writeheader()
                writer.writerows(index)
            (target_batch / 'manifest.json').write_text(json.dumps({'status': 'converted',
                'conversion_source': str(source_batch)}, indent=2) + '\n')
        if cache_source is not None:
            cache_source = Path(cache_source).expanduser().resolve()
            if not cache_source.is_dir():
                raise ValueError(f'Cache source does not exist: {cache_source}')
            shutil.copytree(cache_source, stage / '_caches')
            for path in (stage / '_caches').rglob('*'):
                if path.suffix in {'.pkl', '.pt'} and path.is_file():
                    rewrite_serialized(path)
                elif path.name == 'manifest.json':
                    metadata = read_json(path)
                    if metadata.get('format_version') in {33, 34} and (path.parent / 'event_index.npy').is_file():
                        metadata['conversion_original_format'] = metadata['format_version']
                        metadata['format_version'] = 35  # same full-grid normalized array layout
                        path.write_text(json.dumps(metadata, indent=2) + '\n')
        report = {'tool': 'waveform_analysis.convert_results', 'runs': receipts, 'skipped': skipped,
                  'metrics_only': metrics_only, 'cache_source': str(cache_source) if cache_source else None}
        (stage / 'conversion_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        if output.exists():
            raise ValueError('Destination appeared during conversion; refusing to replace it')
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**plan, 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, nargs='+', required=True, help='Old batch folders or parents to scan')
    parser.add_argument('--output', type=Path, required=True, help='New destination tree, never an existing directory')
    parser.add_argument('--check', action='store_true', help='Inventory recoverable/skipped runs without writing or loading models')
    parser.add_argument('--metrics-only', action='store_true', help='Recover saved metrics/arrays/plots, omitting model/checkpoint/sampler files')
    parser.add_argument('--cache-source', type=Path, help='Explicitly copy/convert the original cache tree and bind recovered configurations to it')
    args = parser.parse_args(argv)
    try:
        result = convert_results(args.sources, args.output, check=args.check, metrics_only=args.metrics_only, cache_source=args.cache_source)
    except (ValueError, OSError, ImportError, pickle.UnpicklingError) as exc:
        parser.exit(2, f'Conversion failed: {exc}\nNo destination was published. Source results are unchanged.\n')
    print(f"{len(result['runs'])} recoverable runs | {len(result['skipped'])} skipped")
    for run in result['runs']:
        print(f"  {run['source']} -> {run['destination']}")
    for item in result['skipped']:
        print(f"  SKIPPED {item['source']}: {item['reason']}")
    if args.check:
        print('No files written. Remove --check to convert.')
    else:
        print(f"Saved to: {result['output']}")
    return 0 if result['runs'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
