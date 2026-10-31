import numpy as np
import pytest

from waveform_analysis.engine.two_stage import make_development_split


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
