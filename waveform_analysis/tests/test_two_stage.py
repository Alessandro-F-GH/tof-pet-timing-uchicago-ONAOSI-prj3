import numpy as np
import pytest

from types import SimpleNamespace\n\nfrom waveform_analysis.engine.two_stage import make_development_split\nfrom waveform_analysis.engine.train import resolve_target_vector, target_fingerprint


def test_two_stage_split_is_deterministic_disjoint_and_exhaustive():
    first, second = make_development_split(101, seed=1234, stage1_fraction=0.25)
    first_again, second_again = make_development_split(101, seed=1234, stage1_fraction=0.25)
    np.testing.assert_array_equal(first, first_again)
    np.testing.assert_array_equal(second, second_again)
    assert first.size == 25
    assert second.size == 76
    assert not np.intersect1d(first, second).size
    np.testing.assert_array_equal(np.sort(np.concatenate([first, second])), np.arange(101))


def test_two_stage_split_changes_with_seed():
    first, second = make_development_split(100, seed=1)
    other_first, other_second = make_development_split(100, seed=2)
    assert not (np.array_equal(first, other_first) and np.array_equal(second, other_second))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_two_stage_split_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        make_development_split(100, seed=1, stage1_fraction=fraction)


def test_two_stage_split_requires_two_events():
    with pytest.raises(ValueError):
        make_development_split(1, seed=1)



def test_target_override_is_event_aligned_and_does_not_mutate_dataset():
    dataset = SimpleNamespace(
        n_events=3,
        energy_target_ps=np.asarray([1.0, 2.0, 3.0]),
    )
    override = np.asarray([0.5, 1.5, 2.5])
    resolved = resolve_target_vector(dataset, "energy_to_energy", override)
    np.testing.assert_array_equal(resolved, override)
    np.testing.assert_array_equal(dataset.energy_target_ps, [1.0, 2.0, 3.0])
    assert target_fingerprint(override) != target_fingerprint(dataset.energy_target_ps)


def test_target_override_rejects_wrong_length_and_nonfinite_values():
    dataset = SimpleNamespace(n_events=3, energy_target_ps=np.asarray([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError):
        resolve_target_vector(dataset, "energy_to_energy", [1.0, 2.0])
    with pytest.raises(ValueError):
        resolve_target_vector(dataset, "energy_to_energy", [1.0, np.nan, 3.0])
