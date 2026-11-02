"""Checks for development-only progressive temporal feature assembly."""
import numpy as np
from waveform_analysis.scripts.evaluate_progressive_windows import temporal_blocks, fingerprint


def test_temporal_blocks_are_disjoint_and_cumulative():
    time = np.arange(-2000, 4000, 12.5)
    mask = np.ones(len(time), bool)
    blocks = temporal_blocks(time, mask, -2, 4, 1)
    assert len(blocks) == 6
    all_indices = np.concatenate([block[2] for block in blocks])
    assert len(np.unique(all_indices)) == len(all_indices)
    assert np.array_equal(all_indices, np.arange(len(time)))
    assert [round(b[1]) for b in blocks] == [-1, 0, 1, 2, 3, 4]


def test_masked_segments_are_not_used_as_features():
    time = np.arange(-2000, 4000, 12.5)
    mask = np.ones(len(time), bool)
    mask[(time >= 1000) & (time < 2000)] = False
    blocks = temporal_blocks(time, mask, -2, 4, 1)
    assert len(blocks) == 5
    assert all(mask[indices].all() for _, _, indices in blocks)


def test_feature_fingerprint_independent_of_argument_order():
    assert fingerprint({"a": 1, "b": [2,3]}) == fingerprint({"b": [2,3], "a": 1})
