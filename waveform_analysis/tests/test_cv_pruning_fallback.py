"""LED pruning applies at startup; only the best partial fallback is completed."""
import logging
import json
from types import SimpleNamespace

import pytest

from waveform_analysis.data.storage import RunStore
from waveform_analysis.engine import study
from waveform_analysis.engine.validation import best_complete_candidate


def setup_search(tmp_path, monkeypatch, values, *, strategy='grid', metric='ctr'):
    store = RunStore(tmp_path)
    config = {'seed': 1001, 'mode': 'timing_to_timing', 'cross_validation': {
        'folds': 4, 'metric': metric, 'pruning': {
            'enabled': True, 'startup_complete_candidates': 2,
            'min_folds_before_prune': 1, 'max_degradation_ps': 5.,
            'prune_if_worse_than_led': True, 'led_max_degradation_ps': 0.,
        }}}
    space = {'optimization': {'strategy': strategy}, 'parameters': {
        'score': {'type': 'categorical', 'choices': values}}}
    if strategy == 'optuna':
        space['optimization'].update(n_trials=5, n_startup_trials=1)
    elif strategy == 'fixed':
        space['parameters']['score'] = {'type': 'fixed', 'value': values[0]}
    shared = SimpleNamespace(split=SimpleNamespace(folds=[SimpleNamespace(fold_id=i) for i in range(1, 5)]))
    calls = []

    def evaluator(*args):
        def fold(parameters, fold):
            score = parameters['score']
            calls.append((score, fold.fold_id))
            # Remaining folds can worsen the fallback: don't compare its full
            # score to other candidates' incomplete scores during selection.
            value = score if fold.fold_id == 1 else score + 50.
            return {'ctr_ps': value if metric == 'ctr' else 1.,
                    'rmse_ps': value if metric == 'rmse' else 1.,
                    'led_ctr_ps': 100., 'led_rmse_ps': 100.}
        return fold

    monkeypatch.setattr(study, '_fold_evaluator', evaluator)
    args = (store, SimpleNamespace(name='direct_mlp'), space, config, None,
            shared, logging.getLogger('test-pruning'))
    return store, config, calls, args


@pytest.mark.parametrize('strategy', ['fixed', 'grid', 'optuna'])
@pytest.mark.parametrize('metric', ['ctr', 'rmse'])
def test_all_pruned_finish_only_best_partial(tmp_path, monkeypatch, strategy, metric):
    store, config, calls, args = setup_search(tmp_path, monkeypatch, [150., 110., 130.], strategy=strategy, metric=metric)
    candidates = study._search(*args)
    rows = store.read_candidate_rows()
    complete = best_complete_candidate(rows, metric)
    assert complete is not None
    parameters = candidates[complete['candidate_id']]
    assert parameters['score'] == min(p['score'] for p in candidates.values())
    assert sum(not study._bool(r['pruned']) for r in rows) == 1
    for row in rows:
        assert int(row['completed_folds']) == (4 if row['candidate_id'] == complete['candidate_id'] else 1)
    for score in {p['score'] for p in candidates.values()}:
        assert calls.count((score, 1)) == 1
    assert sum(fold > 1 for _, fold in calls) == 3
    selected = study._select(store, candidates, config)
    assert selected['candidate_id'] == complete['candidate_id']
    assert selected[f'validation_{metric}_mean_ps'] == parameters['score'] + 37.5
    before = list(calls)
    study._search(*args)
    assert calls == before  # Resume completed CV without refitting folds.


def test_completed_candidate_prevents_fallback(tmp_path, monkeypatch):
    store, config, calls, args = setup_search(tmp_path, monkeypatch, [50., 110., 150.])
    candidates = study._search(*args)
    best = study._select(store, candidates, config)
    assert best['parameters']['score'] == 50.
    assert calls == [(50., i) for i in range(1, 5)] + [(110., 1), (150., 1)]


def test_interrupted_fallback_reuses_saved_folds(tmp_path, monkeypatch):
    store, config, calls, args = setup_search(tmp_path, monkeypatch, [150., 110., 130.])
    original = study._fold_evaluator
    interrupted = False

    def evaluator(*args):
        evaluate = original(*args)
        def fold(params, fold):
            nonlocal interrupted
            if fold.fold_id == 3 and not interrupted:
                interrupted = True
                raise RuntimeError('interrupted')
            return evaluate(params, fold)
        return fold

    monkeypatch.setattr(study, '_fold_evaluator', evaluator)
    with pytest.raises(RuntimeError, match='interrupted'):
        study._search(*args)
    candidates = study._search(*args)
    assert study._select(store, candidates, config)['parameters']['score'] == 110.
    assert calls.count((110., 1)) == 1
    assert calls.count((110., 2)) == 1
    assert calls.count((110., 3)) == 1
    assert calls.count((110., 4)) == 1


def test_startup_still_exempts_incumbent_pruning(tmp_path, monkeypatch):
    store, config, calls, args = setup_search(tmp_path, monkeypatch, [50., 90., 150.])
    config['cross_validation']['pruning']['prune_if_worse_than_led'] = False
    study._search(*args)
    assert calls == [(score, i) for score in (50., 90.) for i in range(1, 5)] + [(150., 1)]
    assert next(r for r in store.read_candidate_rows() if float(json.loads(r['parameters'])['score']) == 150.)['pruning_reason'] == 'incumbent'
