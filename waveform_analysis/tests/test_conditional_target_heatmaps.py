"""Binned conditional teacher-output heatmaps: support and derivatives."""
import numpy as np

from waveform_analysis.scripts.plot_conditional_target_heatmaps import accumulate_heatmaps


def test_heatmap_weighted_means_and_paired_derivative():
    waves = np.zeros((4, 2, 5), dtype=np.float32)
    waves[:, 0, :] = np.array([
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4],
        [0, 2, 4, 6, 8],
        [0, 2, 4, 6, 8],
    ])
    waves[:, 1, :] = 0
    labels = np.array([10., 30., 50., 70.])
    mean = np.zeros((2, 5), dtype=np.float32)
    scale = np.ones((2, 5), dtype=np.float32)
    edges, sums, counts = accumulate_heatmaps(
        waves, labels, np.arange(4), np.array([0, 1, 3, 4]),
        mean, scale, mean, scale, bins=16, z_limit=10, chunk_size=2)
    # At t=1, detector 1 value 1 has labels 10 and 30.
    bin_at_one = int(np.searchsorted(edges, 1, side="right") - 1)
    assert counts[0, 0, bin_at_one, 1] == 2
    assert sums[0, 0, bin_at_one, 1] == 40.
    # Same binned values for derivative at t=1 (dx=1 and dx=2).
    assert counts[1, 0, bin_at_one, 1] == 2
    assert sums[1, 0, bin_at_one, 1] == 40.
    # Difference detector2 - detector1 equals -1 at t=1.
    bin_minus_one = int(np.searchsorted(edges, -1, side="right") - 1)
    assert counts[0, 2, bin_minus_one, 1] == 2
    assert sums[0, 2, bin_minus_one, 1] == 40.
    # Absolute train-only targets: no target binning or percentiles.
    assert int(counts[0, 0].sum()) == 4 * 4


def test_heatmap_excludes_out_of_range_values():
    waves = np.zeros((2, 2, 3), dtype=np.float32)
    waves[0, 0, 2] = 10
    labels = np.array([-1., 1.])
    mean = np.zeros((2, 3), dtype=np.float32)
    scale = np.ones((2, 3), dtype=np.float32)
    _, _, counts = accumulate_heatmaps(
        waves, labels, np.arange(2), np.arange(3),
        mean, scale, mean, scale, bins=8, z_limit=3, chunk_size=2)
    assert int(counts[0, 0].sum()) == 5
