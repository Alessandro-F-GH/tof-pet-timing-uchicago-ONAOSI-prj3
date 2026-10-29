"""Shared ownership, nonmutating reads, and strict canonical format contracts."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from waveform_analysis.core.config import load_batch_config
from waveform_analysis.data.results import (
    read_run, read_run_manifest, resolve_run_config, write_run_config, write_run_manifest,
)


def configs(tmp_path):
    batch = load_batch_config(Path(__file__).parents[1] / 'config/batches/test_ridge.json')
    result = []
    for original in batch.runs:
        config = deepcopy(original)
        config['batch_output_dir'] = str(tmp_path)
        config['output_dir'] = str(tmp_path / 'timing/onishi' / config['model']['name'])
        config['_board'] = 'FBK'
        result.append(config)
    return result


def contents(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_settings_have_one_owner_and_resolve_exactly(tmp_path):
    runs = configs(tmp_path)
    managed = [c['output_dir'] for c in runs]
    for config in runs:
        write_run_config(config, managed_runs=managed)
    raw = json.loads((tmp_path / 'config.json').read_text())
    assert 'runs' not in raw and 'model' not in raw
    assert 'cross_validation' in raw['settings']
    assert 'channels' in raw
    assert all('channels' not in d for d in raw['datasets'].values())
    mode = json.loads((tmp_path / 'timing/mode.json').read_text())
    assert mode['settings']['ml_input'] == runs[0]['ml_input']
    context = json.loads((tmp_path / 'timing/onishi/context.json').read_text())
    assert set(context) == {'window_name', 'window_ns'}
    for config in runs:
        directory = Path(config['output_dir'])
        local = json.loads((directory / 'metadata/config.json').read_text())
        assert local == {'model': config['model']['name']}
        restored = resolve_run_config(directory)
        for key, value in restored.items():
            if key in config:
                assert value == config[key], key
    before = contents(tmp_path)
    for config in runs:
        resolve_run_config(config['output_dir'])
    assert contents(tmp_path) == before


def test_manifest_owns_only_model_information(tmp_path):
    config = configs(tmp_path)[0]
    write_run_config(config)
    directory = Path(config['output_dir'])
    population = {'control': {'source': 'control.root', 'n_selected': 25, 'population_identity': 'cohort'}}
    write_run_manifest(directory, {'status': 'complete', 'model': config['model']['name'],
                       'mode': config['mode'], 'window_ns': config['window_ns'],
                       'dataset_populations': population, 'led_control_mean_ps': 4.,
                       'blind_used_in_selection': False, 'stage_fingerprints': {'final_fit': 'weights'}})
    raw = json.loads((directory / 'metadata/manifest.json').read_text())
    assert set(raw) == {'status', 'stage_fingerprints'}
    shared = json.loads((directory.parent / 'population.json').read_text())
    assert shared['dataset_populations']['control']['n_selected'] == 25
    assert 'source' not in shared['dataset_populations']['control']
    before = contents(tmp_path)
    result = read_run(directory)
    assert result['manifest']['blind_used_in_selection'] is False
    assert result['manifest']['led_control_mean_ps'] == 4.
    assert result['manifest']['model'] == config['model']['name']
    assert contents(tmp_path) == before


def test_changing_shared_settings_cannot_relabel_retained_runs(tmp_path):
    runs = configs(tmp_path)
    for config in runs:
        write_run_config(config)
    before = contents(tmp_path)
    changed = deepcopy(runs[0])
    changed['bootstrap']['n_resamples'] += 1
    with pytest.raises(ValueError, match='retained runs'):
        write_run_config(changed)
    assert contents(tmp_path) == before


def test_old_layout_rejected_without_side_effects(tmp_path):
    directory = tmp_path / 'old'
    directory.mkdir()
    (directory / 'config.json').write_text('{"control": {}}')
    (directory / 'manifest.json').write_text('{"status":"complete"}')
    before = contents(tmp_path)
    with pytest.raises(ValueError, match='historical result compatibility'):
        read_run(directory)
    assert contents(tmp_path) == before


def test_metadata_imports_do_not_load_training_frameworks():
    completed = subprocess.run([sys.executable, '-c',
        "import sys; import waveform_analysis.data.results; from waveform_analysis.models import model_names; "
        "assert len(model_names()) == 18; assert not {'torch','aeon','sktime','uproot'} & sys.modules.keys()"],
        capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
