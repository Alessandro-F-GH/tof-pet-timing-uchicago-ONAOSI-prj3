"""Portable cache settings must not depend on machine-specific directories."""
import pytest

from waveform_analysis.data.cache_settings import portable_settings, verify_settings


def test_portable_source_paths_and_cache_roots():
    linux = {
        "dataset": {"root_file": "/home/alessandro/data/48V-R1.root"},
        "preprocessing": {"cache_dir": "/tmp/linux/cache", "filter": {"cutoff_GHz": 2}},
        "mode": "energy_to_energy",
    }
    windows = {
        "dataset": {"root_file": r"C:\Users\aless\data\48V-R1.root"},
        "preprocessing": {"cache_dir": r"C:\Users\aless\cache", "filter": {"cutoff_GHz": 2}},
        "mode": "energy_to_energy",
    }
    assert portable_settings(linux) == portable_settings(windows)
    verify_settings({"cache_settings": portable_settings(linux)}, windows, label="dataset")


def test_relevant_settings_mismatch_is_explicit():
    original = {"filter": {"cutoff_GHz": 2}, "mode": "energy_to_energy"}
    modified = {"filter": {"cutoff_GHz": 1.5}, "mode": "energy_to_energy"}
    with pytest.raises(ValueError, match="filter"):
        verify_settings({"cache_settings": original}, modified, label="preprocessing")


def test_legacy_manifest_is_accepted_without_sha_validation():
    verify_settings(
        {"fingerprint": "old-machine-dependent-value"},
        {"mode": "energy_to_energy"},
        label="legacy",
    )
