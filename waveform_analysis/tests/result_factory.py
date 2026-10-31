"""Canonical result fixtures; historical fixtures belong to converter tests."""
from copy import deepcopy
from pathlib import Path
from waveform_analysis.data.results import write_run_config, write_run_manifest


def persist_result(directory, config=None, manifest=None):
    directory = Path(directory)
    config, manifest = deepcopy(config or {}), deepcopy(manifest or {})
    channels = {'energy': [1, 2], 'timing': [3, 4], 'polarities': [1, 1], 'timing_polarities': [1, 1]}
    name = manifest.get('model', 'direct_linear_ridge')
    defaults = {
        'study_name': 'test', 'mode': manifest.get('mode', 'timing_to_timing'),
        'window_name': manifest.get('window_name', 'short'),
        'window_ns': manifest.get('window_ns', {'start': -1.5, 'end': 2.}),
        'model': {'name': name, 'space': {}}, 'seed': 1001,
        'cross_validation': {'metric': 'ctr'}, 'bootstrap': {'n_resamples': 3},
        'runtime': {'prediction_chunk_size': 16}, 'xai': {'enabled': False},
        'save_model': True, 'preprocessing': {}, 'fit': {'histogram_bin_width_ps': 5},
        'ml_input': {'subsampling': 1}, 'ml_output': {'max_abs_ps': 2000.},
        'plot_config': {}, 'output_dir': str(directory), 'batch_output_dir': str(directory),
    }
    for role in ('control', 'development', 'blind'):
        defaults[role] = {'root_file': f'{role}.root', 'true_tof_ps': 0., 'channels': channels}
        defaults[role].update(manifest.get(f'{role}_dataset', {}))
        defaults[role].update(config.pop(role, {}))
    defaults.update(config)
    defaults.setdefault('preprocessing', {})
    defaults['preprocessing'].setdefault(
        'artifact_root', str(directory / 'artifact_store')
    )
    write_run_config(defaults)
    if manifest:
        write_run_manifest(directory, manifest)
    return defaults
