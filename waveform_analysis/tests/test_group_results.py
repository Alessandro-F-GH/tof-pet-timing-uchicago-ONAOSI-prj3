"""Grouping saved artifacts must never migrate or evaluate the original runs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from waveform_analysis.scripts.group_results import group_results


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def run(root, batch, model, *, subsampling=1, legacy=False, window=(-1.5, 2),
        window_name='onishi', acquisition='48V-R1.root', mask='control'):
    base = root / 'results' / 'UC' / batch
    directory = base / 'timing' / window_name / model
    (base / 'tables').mkdir(parents=True, exist_ok=True)
    (base / 'tables/runs.csv').write_text('path\n', encoding='utf-8')
    metadata = directory if legacy else directory / 'metadata'
    datasets = {role: {'root_file': f'C:\\repo\\processed_data\\{source}',
                      'channels': {'timing': [3, 4]}, 'true_tof_ps': 0.0}
                for role, source in [('control', '48V-R0.root'), ('development', acquisition), ('blind', '48V-R2.root')]}
    populations = {role: {'source': value['root_file'], 'population_identity': role + acquisition,
                          'bias_voltage_V': [48.0]} for role, value in datasets.items()}
    manifest = {'status': 'complete', 'schema_version': 30 if legacy else 53,
                'model': model, 'mode': 'timing_to_timing', 'window_name': window_name,
                'window_ns': dict(zip(('start', 'end'), window)),
                'dataset_populations': populations, 'sample_mask_fit_role': mask,
                'led_centering_fit_role': 'control', 'led_control_mean_ps': 4.5,
                **{f'{role}_dataset': value for role, value in datasets.items()}}
    config = {'ml_input': {'subsampling': subsampling}, 'fit': {'histogram_bin_width_ps': 5.0},
              'results': {'folder': f'UC/{batch}'}, **datasets}
    write_json(metadata / 'manifest.json', manifest)
    write_json(metadata / 'config.json', config)
    write_json(metadata / 'blind.json', {'ctr_ps': 100., 'rmse_ps': 50., 'led_ctr_ps': 140., 'led_rmse_ps': 80., 'n_events': 1000})
    write_json(metadata / 'bootstrap.json', {'ctr_bootstrap_std_ps': 2., 'rmse_bootstrap_std_ps': 1.})
    write_json(metadata / 'best.json', {'candidate_id': 'best', 'validation_ctr_mean_ps': 110.})
    (directory / 'plots').mkdir(parents=True)
    (directory / 'plots/blind.png').write_bytes(b'existing-figure')
    (directory / 'tables').mkdir()
    (directory / 'tables/cv.csv').write_text('fold,ctr_ps\n1,110\n')
    (directory / 'model').mkdir()
    (directory / 'model/model.pkl').write_bytes(b'NEVER LOAD OR COPY THIS')
    (directory / 'artifacts').mkdir()
    (directory / 'artifacts/pred.npz').write_bytes(b'COPY ONLY, NEVER READ AS ARRAYS')
    return directory


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_grouping_separates_subsampling_and_preserves_historical_sources(tmp_path):
    first = run(tmp_path, 'ridge', 'direct_linear_ridge')
    second = run(tmp_path, 'old', 'shared_rdst', legacy=True, mask='model_specific')
    third = run(tmp_path, 'subsampled', 'direct_linear_ridge', subsampling=4)
    sources = tmp_path / 'results'
    before = snapshot(sources)
    output = tmp_path / 'grouped'
    result = group_results([sources], output, copy_predictions=True)
    assert {info['group'] for info in result['groups']} == {'UC_48V_sub1', 'UC_48V_sub4'}
    for info in result['groups']:
        assert {row['subsampling'] for row in info['runs']} == {info['subsampling']}
    leaf = output / 'UC_48V_sub1/timing_to_timing/window_-1.5_2ns'
    assert (leaf / 'runs/shared_rdst/metadata/manifest.json').is_file()
    assert (leaf / 'runs/shared_rdst/plots/blind.png').read_bytes() == b'existing-figure'
    assert (leaf / 'runs/shared_rdst/artifacts/pred.npz').read_bytes().startswith(b'COPY ONLY')
    assert (leaf / 'blind_ctr.png').is_file()
    assert (leaf / 'blind_rmse.png').is_file()
    assert not list(output.rglob('*.pkl'))
    assert snapshot(sources) == before
    assert not (second / 'metadata').exists()  # No RunStore legacy migration.
    assert 'model_specific' in (leaf / 'summary.csv').read_text()


def test_duplicate_models_require_explicit_choice_before_any_output(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    run(tmp_path, 'b', 'direct_rdst')
    source, output = tmp_path / 'results', tmp_path / 'grouped'
    plan = group_results([source, source / 'UC/a'], output, check=True)
    assert len(plan['runs']) == 2  # Overlapping discovery roots deduplicate.
    assert len(plan['conflicts']) == 1
    assert not output.exists()
    with pytest.raises(ValueError, match='explicit'):
        group_results([source], output)
    chosen = plan['conflicts'][0]['candidates'][0]['id']
    result = group_results([source], output, selected=[chosen], summary_plots=False)
    assert len(result['selected']) == 1
    with pytest.raises(ValueError, match='Unknown selected'):
        group_results([source], tmp_path / 'other', selected=['typo'])


def test_numeric_windows_and_datasets_define_groups_not_names(tmp_path):
    run(tmp_path, 'a', 'direct_rdst', window_name='onishi')
    run(tmp_path, 'b', 'shared_rdst', window_name='same_range')
    run(tmp_path, 'wide', 'direct_rdst', window=(-2., 30.), window_name='onishi')
    run(tmp_path, 'other', 'direct_rdst', acquisition='48V-another.root')
    result = group_results([tmp_path / 'results'], tmp_path / 'grouped', summary_plots=False)
    assert len(result['groups']) == 3
    assert len({info['group'] for info in result['groups']}) == 2
    matching = [info for info in result['groups'] if len(info['runs']) == 2]
    assert len(matching) == 1
    assert all(info['group'].startswith('UC_48V_sub1_') for info in result['groups'])


def test_missing_subsampling_is_not_guessed_and_missing_plots_are_reported(tmp_path):
    invalid = run(tmp_path, 'missing_sub', 'direct_rdst')
    config = invalid / 'metadata/config.json'
    value = json.loads(config.read_text())
    value.pop('ml_input')
    write_json(config, value)
    valid = run(tmp_path, 'valid', 'shared_rdst')
    (valid / 'plots/blind.png').unlink()
    result = group_results([tmp_path / 'results'], tmp_path / 'grouped', summary_plots=False)
    assert len(result['skipped']) == 1 and 'subsampling' in result['skipped'][0]['reason']
    assert any(item['missing'] == 'saved plots' for item in result['groups'][0]['missing'])


def test_incompatible_legacy_calibration_does_not_block_copying(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    old = run(tmp_path, 'b', 'shared_rdst', legacy=True)
    path = old / 'manifest.json'
    data = json.loads(path.read_text())
    data.pop('led_centering_fit_role')
    write_json(path, data)
    output = tmp_path / 'grouped'
    result = group_results([tmp_path / 'results'], output)
    assert 'calibration' in ' '.join(result['groups'][0]['comparison_plot_skip_reasons'])
    assert len(list(output.rglob('plots/blind.png'))) == 2
    assert not list(output.rglob('blind_ctr.png'))


def test_overwrite_only_owned_output_and_exclude_it_from_discovery(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    sources = tmp_path / 'results'
    output = sources / 'grouped'
    group_results([sources], output, summary_plots=False)
    with pytest.raises(ValueError, match='Output exists'):
        group_results([sources], output, summary_plots=False)
    result = group_results([sources], output, summary_plots=False, overwrite=True)
    assert len(result['runs']) == 1
    unrelated = tmp_path / 'unrelated'
    unrelated.mkdir()
    (unrelated / 'keep.txt').write_text('user file')
    with pytest.raises(ValueError, match='previous output'):
        group_results([sources], unrelated, overwrite=True)
    assert (unrelated / 'keep.txt').read_text() == 'user file'


def test_cli_check_is_standalone_and_does_not_import_models(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    script = Path(__file__).resolve().parents[1] / 'scripts/group_results.py'
    code = (
        "import importlib.util, sys; "
        f"spec=importlib.util.spec_from_file_location('standalone_group', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; "
        "spec.loader.exec_module(module); "
        "result=module.main(['--sources',sys.argv[1],'--output',sys.argv[2],'--check']); "
        "assert not any(k in sys.modules for k in ('torch','aeon','waveform_analysis.models')); "
        "raise SystemExit(result)"
    )
    completed = subprocess.run([sys.executable, '-c', code, str(tmp_path / 'results'),
                                str(tmp_path / 'grouped')], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    # Machine-readable output remains available explicitly.
    completed = subprocess.run([sys.executable, str(script), '--sources', str(tmp_path / 'results'),
                                '--output', str(tmp_path / 'grouped'), '--check', '--json'], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)['runs'][0]['group'] == 'UC_48V_sub1'
    assert not (tmp_path / 'grouped').exists()



def test_explicit_dataset_aliases_support_moved_acquisitions(tmp_path):
    run(tmp_path, 'old', 'direct_rdst')
    moved = run(tmp_path, 'new', 'shared_rdst')
    path = moved / 'metadata/manifest.json'
    manifest = json.loads(path.read_text())
    aliases = {}
    for role in ('control', 'development', 'blind'):
        old = manifest[f'{role}_dataset']['root_file']
        new = old.replace('C:\\repo', 'D:\\new_repo')
        aliases[old] = aliases[new] = f'UC/{role}'
        manifest[f'{role}_dataset']['root_file'] = new
    write_json(path, manifest)
    sources = tmp_path / 'results'
    separate = group_results([sources], tmp_path / 'separate', check=True)
    assert len({r['group'] for r in separate['runs']}) == 2
    combined = group_results([sources], tmp_path / 'combined', aliases=aliases, summary_plots=False)
    assert len(combined['groups']) == 1
    assert len(combined['groups'][0]['runs']) == 2


def test_null_legacy_population_and_unresolved_fit_still_copy_results(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    old = run(tmp_path, 'b', 'shared_rdst', legacy=True)
    path = old / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['dataset_populations'] = None
    write_json(path, manifest)
    path = old / 'config.json'
    config = json.loads(path.read_text())
    config['fit'] = 'legacy-fit-reference.json'
    write_json(path, config)
    output = tmp_path / 'grouped'
    result = group_results([tmp_path / 'results'], output)
    assert len(result['groups']) == 1
    assert len(result['selected']) == 2
    assert result['groups'][0]['comparison_plot_skip_reasons']
    assert len(list(output.rglob('plots/blind.png'))) == 2


def test_cli_lists_compact_duplicate_choices_without_requiring_check(tmp_path, capsys):
    from waveform_analysis.scripts.group_results import main
    run(tmp_path, 'batch_a', 'direct_rdst')
    run(tmp_path, 'batch_b', 'direct_rdst')
    run(tmp_path, 'batch_c', 'shared_rdst')
    output = tmp_path / 'grouped'
    args = ['--sources', str(tmp_path / 'results'), '--output', str(output)]
    assert main(args) == 2
    text = capsys.readouterr().out
    assert '3 runs | 1 groups | 1 choices needed' in text
    assert 'Choice 1: UC_48V_sub1 | timing_to_timing | -1.5 to 2 ns | direct_rdst' in text
    assert 'batch_a/timing/onishi/direct_rdst' in text
    assert 'batch_b/timing/onishi/direct_rdst' in text
    assert text.count('--select ') == 3  # Two candidate options and one instruction.
    assert '"candidates"' not in text
    assert not output.exists()
    plan = group_results([tmp_path / 'results'], output, check=True)
    chosen = plan['conflicts'][0]['candidates'][0]['id']
    assert main(args + ['--select', chosen, '--no-summary-plots']) == 0
    text = capsys.readouterr().out
    assert '0 choices needed' in text and 'Saved to:' in text
    assert len(list(output.rglob('plots/blind.png'))) == 2


def test_grouped_bars_reuse_ranked_reporting_style_and_saved_led(tmp_path, monkeypatch):
    import csv
    import matplotlib.colors as colors
    import numpy as np
    from waveform_analysis.reporting import plotting
    first = run(tmp_path, 'a', 'direct_rdst')
    second = run(tmp_path, 'b', 'shared_rdst')
    write_json(first / 'metadata/blind.json', {'ctr_ps': 120., 'rmse_ps': 40.,
               'led_ctr_ps': 140., 'led_rmse_ps': 80.})
    write_json(second / 'metadata/blind.json', {'ctr_ps': 90., 'rmse_ps': 60.,
               'led_ctr_ps': 140., 'led_rmse_ps': 80.})
    # Missing historical uncertainties remain unknown, rather than invented zeros.
    (second / 'metadata/bootstrap.json').unlink()
    cfg = json.loads((Path(__file__).parents[1] / 'config/plots/default.json').read_text())
    cfg['output']['format'] = 'svg'
    cfg['formulations']['shared']['color'] = '#123456'
    config_path = tmp_path / 'style.json'
    write_json(config_path, cfg)
    captured = []
    original_save = plotting._save
    def inspect(fig, path):
        ax = fig.axes[0]
        captured.append({
            'codes': [t.get_text() for t in ax.get_xticklabels()],
            'heights': [p.get_height() for p in ax.patches],
            'colors': [p.get_facecolor() for p in ax.patches],
            'led': next(line.get_ydata()[0] for line in ax.lines if line.get_label() == 'LED reference'),
            'legend': [t.get_text() for t in ax.get_legend().get_texts()],
        })
        return original_save(fig, path)
    monkeypatch.setattr(plotting, '_save', inspect)
    monkeypatch.setattr(np, 'load', lambda *a, **kw: pytest.fail('Must not load prediction arrays'))
    output = tmp_path / 'grouped'
    result = group_results([tmp_path / 'results'], output, plot_config=config_path)
    assert not result['groups'][0]['comparison_plot_skip_reasons']
    ctr, rmse = captured
    assert ctr['codes'] == ['S-RDST', 'D-RDST']
    assert rmse['codes'] == ['D-RDST', 'S-RDST']
    assert ctr['heights'] == [90., 120.] and rmse['heights'] == [40., 60.]
    assert ctr['colors'] == [colors.to_rgba(cfg['formulations'][f]['color']) for f in ('shared', 'direct')]
    assert ctr['led'] == 140. and rmse['led'] == 80.
    assert ctr['legend'][0] == 'S-RDST: Shared dilated shapelets + ridge'
    leaf = output / result['groups'][0]['group'] / result['groups'][0]['leaf']
    assert (leaf / 'blind_ctr.svg').is_file()
    with (leaf / 'blind_rmse.csv').open() as stream:
        assert [r['code'] for r in csv.DictReader(stream)] == ['D-RDST', 'S-RDST']
    assert 'LED & n/a & $140.0$ & $80.0$' in (leaf / 'metrics.tex').read_text()
    assert not (leaf / 'blind_ctr.tex').exists()
    assert not (leaf / 'blind_rmse.tex').exists()
    assert json.loads((output / 'plots.json').read_text()) == cfg


def test_missing_or_different_led_reference_skips_only_affected_metric(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    second = run(tmp_path, 'b', 'shared_rdst')
    path = second / 'metadata/blind.json'
    blind = json.loads(path.read_text())
    blind['led_ctr_ps'] = 150.
    blind.pop('led_rmse_ps')
    write_json(path, blind)
    output = tmp_path / 'grouped'
    result = group_results([tmp_path / 'results'], output)
    reasons = result['groups'][0]['comparison_plot_skip_reasons']
    assert 'ctr: saved LED reference differs between runs' in reasons
    assert any('rmse:' in reason and 'missing' in reason for reason in reasons)
    assert not list(output.rglob('blind_ctr.png'))
    assert not list(output.rglob('blind_rmse.png'))
    assert len(list(output.rglob('plots/blind.png'))) == 2


def test_split_explanation_shows_only_differing_fields(tmp_path, capsys):
    from waveform_analysis.scripts.group_results import main
    run(tmp_path, 'a', 'direct_rdst')
    changed = run(tmp_path, 'b', 'shared_rdst', acquisition='48V-different.root')
    path = changed / 'metadata/manifest.json'
    manifest = json.loads(path.read_text())
    manifest['control_dataset']['channels']['timing'] = [5, 6]
    manifest['blind_dataset']['true_tof_ps'] = 10.
    write_json(path, manifest)
    run(tmp_path, 'c', 'direct_rdst', subsampling=4)
    output = tmp_path / 'grouped'
    assert main(['--sources', str(tmp_path / 'results'), '--output', str(output), '--check']) == 0
    printed = capsys.readouterr().out
    assert 'Dataset split: UC_48V_sub1 (same subsampling; dataset metadata differs)' in printed
    assert 'control.channels.timing:' in printed
    assert '[3, 4]' in printed and '[5, 6]' in printed
    assert 'development.source:' in printed and '48v-different.root' in printed.lower()
    assert 'blind.true_tof_ps:' in printed
    assert 'control.source:' not in printed
    assert 'Dataset split: UC_48V_sub4' not in printed
    assert '--dataset-aliases' in printed
    result = group_results([tmp_path / 'results'], output, summary_plots=False)
    splits = result['dataset_splits']
    assert len(splits) == 1
    assert {d['field'] for d in splits[0]['differences']} == {
        'control.channels.timing', 'development.source', 'blind.true_tof_ps'}
    assert json.loads((output / 'grouping_manifest.json').read_text())['dataset_splits'] == splits


def test_equivalent_dataset_aliases_remove_split_explanation(tmp_path):
    run(tmp_path, 'a', 'direct_rdst')
    run(tmp_path, 'b', 'shared_rdst', acquisition='48V-different.root')
    sources = tmp_path / 'results'
    plan = group_results([sources], tmp_path / 'grouped', check=True)
    paths = next(d['values'] for d in plan['dataset_splits'][0]['differences'] if d['field'] == 'development.source')
    aliases = {p: 'same-development-acquisition' for p in paths.values()}
    combined = group_results([sources], tmp_path / 'grouped', check=True, aliases=aliases)
    assert combined['dataset_splits'] == []
    assert {r['group'] for r in combined['runs']} == {'UC_48V_sub1'}


def test_append_new_subsampling_and_windows_preserves_existing_results(tmp_path):
    first = run(tmp_path, 'a', 'direct_rdst')
    output = tmp_path / 'grouped'
    initial = group_results([first], output, summary_plots=False)
    leaf = output / initial['groups'][0]['group'] / initial['groups'][0]['leaf']
    before = snapshot(leaf)
    second = run(tmp_path, 'b', 'direct_rdst', subsampling=4)
    appended = group_results([second], output, summary_plots=False)
    assert {g['group'] for g in appended['groups']} == {'UC_48V_sub1', 'UC_48V_sub4'}
    assert snapshot(leaf) == before
    wide = run(tmp_path, 'c', 'shared_rdst', window=(-2., 30.))
    result = group_results([wide], output, summary_plots=False)
    assert len(result['groups']) == 3
    assert snapshot(leaf) == before
    assert len(list(output.rglob('plots/blind.png'))) == 3
    assert len(json.loads((output / 'grouping_manifest.json').read_text())['groups']) == 3
    import csv
    with (output / 'sources.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_append_distinct_datasets_and_reuse_assigned_group_names(tmp_path):
    first = run(tmp_path, 'a', 'direct_rdst')
    second = run(tmp_path, 'b', 'shared_rdst', acquisition='48V-other.root')
    output = tmp_path / 'grouped'
    group_results([first], output, summary_plots=False)
    before = snapshot(output / 'UC_48V_sub1')
    result = group_results([second], output, summary_plots=False)
    names = {g['group'] for g in result['groups']}
    assert len(names) == 2
    assert 'UC_48V_sub1' in names
    second_name = (names - {'UC_48V_sub1'}).pop()
    assert second_name.startswith('UC_48V_sub1_')
    assert snapshot(output / 'UC_48V_sub1') == before
    # Even a later scan containing both datasets preserves the original names.
    plan = group_results([first, second], output, check=True)
    assert {r['group'] for r in plan['runs']} == names
    assert len(plan['output_conflicts']) == 2
    before_all = snapshot(output)
    with pytest.raises(ValueError, match='use --overwrite'):
        group_results([second], output, summary_plots=False)
    assert snapshot(output) == before_all
    result = group_results([second], output, summary_plots=False, overwrite=True)
    assert {g['group'] for g in result['groups']} == names
    assert snapshot(output / 'UC_48V_sub1') == before


def test_targeted_overwrite_preserves_other_groups_and_removes_stale_models(tmp_path):
    first = run(tmp_path, 'a', 'direct_rdst')
    other = run(tmp_path, 'b', 'shared_rdst', subsampling=4)
    replacement = run(tmp_path, 'c', 'shared_hydra')
    output = tmp_path / 'grouped'
    group_results([first, other], output, summary_plots=False)
    before = snapshot(output / 'UC_48V_sub4')
    group_results([replacement], output, summary_plots=False, overwrite=True)
    assert snapshot(output / 'UC_48V_sub4') == before
    assert not list((output / 'UC_48V_sub1').rglob('runs/direct_rdst'))
    assert list((output / 'UC_48V_sub1').rglob('runs/shared_hydra'))


def test_failed_append_leaves_previous_output_unchanged(tmp_path, monkeypatch):
    from waveform_analysis.scripts import group_results as module
    first = run(tmp_path, 'a', 'direct_rdst')
    output = tmp_path / 'grouped'
    group_results([first], output, summary_plots=False)
    before = snapshot(output)
    second = run(tmp_path, 'b', 'shared_rdst', subsampling=4)
    def fail(*args, **kwargs):
        raise RuntimeError('plot failure')
    monkeypatch.setattr(module, '_summary_plots', fail)
    with pytest.raises(RuntimeError, match='plot failure'):
        group_results([second], output)
    assert snapshot(output) == before


def test_cli_reports_why_comparison_plots_are_missing(tmp_path, capsys):
    from waveform_analysis.scripts.group_results import main
    run(tmp_path, 'a', 'direct_rdst')
    old = run(tmp_path, 'b', 'shared_rdst')
    path = old / 'metadata/manifest.json'
    manifest = json.loads(path.read_text())
    manifest.pop('led_centering_fit_role')
    write_json(path, manifest)
    output = tmp_path / 'grouped'
    assert main(['--sources', str(tmp_path / 'results'), '--output', str(output)]) == 0
    text = capsys.readouterr().out
    assert 'Comparison plots skipped or incomplete: UC_48V_sub1/timing_to_timing/window_-1.5_2ns' in text
    assert 'control LED calibration missing or different' in text
    assert 'Comparison plots: ' not in text
    assert len(list(output.rglob('plots/blind.png'))) == 2


@pytest.mark.parametrize('missing', [False, True])
def test_population_ids_warn_without_blocking_plots(tmp_path, capsys, missing):
    from waveform_analysis.scripts.group_results import main
    run(tmp_path, 'a', 'direct_rdst')
    second = run(tmp_path, 'b', 'shared_rdst')
    path = second / 'metadata/manifest.json'
    manifest = json.loads(path.read_text())
    if missing:
        manifest['dataset_populations'] = None
    else:
        manifest['dataset_populations']['control']['population_identity'] = 'from-another-machine'
    write_json(path, manifest)
    output = tmp_path / 'grouped'
    assert main(['--sources', str(tmp_path / 'results'), '--output', str(output)]) == 0
    text = capsys.readouterr().out
    assert 'WARNING:' in text and 'Population IDs missing or different' in text
    assert 'Comparison plots skipped' not in text
    leaf = output / 'UC_48V_sub1/timing_to_timing/window_-1.5_2ns'
    assert (leaf / 'blind_ctr.png').is_file()
    assert (leaf / 'blind_rmse.png').is_file()
    info = json.loads((leaf / 'group.json').read_text())
    assert info['comparison_plot_skip_reasons'] == []
    assert len(info['comparison_plot_warnings']) == 1
    assert 'control' in info['comparison_plot_warnings'][0]


def test_combined_table_cv_led_uncertainties_order_and_no_caption(tmp_path):
    first = run(tmp_path, 'a', 'direct_rdst')
    second = run(tmp_path, 'b', 'shared_rdst')
    ridge = run(tmp_path, 'c', 'direct_linear_ridge')
    for directory, ctr in ((first, 120.44), (second, 120.43), (ridge, 130.)):
        write_json(directory / 'metadata/best.json', {
            'candidate_id': 'selected', 'folds': 2,
            'validation_ctr_mean_ps': 118.6, 'validation_ctr_std_ps': 1.7,
            'selection_method': 'ridge_cv' if directory == ridge else 'development_cv'})
        write_json(directory / 'metadata/blind.json', {
            'ctr_ps': ctr, 'rmse_ps': 61.5, 'led_ctr_ps': 147.3, 'led_rmse_ps': 70.9})
        write_json(directory / 'metadata/bootstrap.json', {
            'ctr_bootstrap_std_ps': 1.2, 'rmse_bootstrap_std_ps': 1.1,
            'led_ctr_bootstrap_std_ps': 1.8 if directory == first else 1.9,
            'led_rmse_bootstrap_std_ps': 1.0})
    # Different candidate rows must not be averaged into the LED CV reference.
    (first / 'tables/folds.csv').write_text(
        'candidate_id,fold_id,led_ctr_ps\nselected,1,142\nselected,2,148\npruned,1,999\n')
    output = tmp_path / 'grouped'
    result = group_results([tmp_path / 'results'], output, summary_plots=False)
    leaf = output / result['groups'][0]['group'] / result['groups'][0]['leaf']
    table = (leaf / 'metrics.tex').read_text()
    assert r'\begin{table}[tbp]' in table
    assert r'\resizebox{\columnwidth}{!}' in table
    assert 'Model & CV CTR [ps] & Test CTR [ps] & Test RMSE [ps]' in table
    assert r'LED & $145.0 \pm 4.2$ & $147.3 \pm 1.9$ & $70.9 \pm 1.0$' in table
    assert r'D-RDST & $118.6 \pm 1.7$ & $120.4 \pm 1.2$ & $61.5 \pm 1.1$' in table
    assert r'D-LIN & n/a & $130.0 \pm 1.2$' in table
    assert table.index('S-RDST &') < table.index('D-RDST &') < table.index('D-LIN &')
    assert r'\caption' not in table and r'\label' not in table
    assert not (leaf / 'blind_ctr.tex').exists()
    assert not (leaf / 'blind_rmse.tex').exists()
    info = json.loads((leaf / 'group.json').read_text())
    reference = info['metrics_table_reference']
    assert reference['led_cv_reference_run'] == next(r['id'] for r in result['runs'] if r['model'] == 'direct_rdst')
    assert reference['led_test_reference_run'] == next(r['id'] for r in result['runs'] if r['model'] == 'direct_linear_ridge')
    # Old separate tables disappear when an entry is explicitly rebuilt.
    (leaf / 'blind_ctr.tex').write_text('old table')
    group_results([tmp_path / 'results'], output, summary_plots=False, overwrite=True)
    assert not (leaf / 'blind_ctr.tex').exists()
    assert (leaf / 'metrics.tex').read_text() == table
