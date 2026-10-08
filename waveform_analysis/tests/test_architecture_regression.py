"""Compare unchanged registered models with the untouched Git baseline, exactly."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

BASELINE = "59a095a4cbc8b699c5bce16f656d4156483ce86b"
ROOT = Path(__file__).resolve().parents[2]
PROBE = Path(__file__).parent / "fixtures" / "model_probe.py"
MODELS = (
    "antisymmetric_mlp",
    "direct_minirocket",
    "direct_mlp",
    "independent_cnn1d",
    "locally_connected_mlp",
    "onishi_cnn",
    "shared_cnn1d",
    "shared_minirocket",
    "protocol",
)


@pytest.fixture(scope="module")
def differential_results(tmp_path_factory):
    """Run isolated reference/current fits using identical installed dependencies."""
    directory = tmp_path_factory.mktemp("architecture-regression")
    reference = directory / "reference"
    reference.mkdir()
    archived = subprocess.run(
        ["git", "archive", BASELINE, "waveform_analysis", "utils_fit"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        archive.extractall(reference, filter="data")
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": str(directory / "matplotlib"),
            "XDG_CACHE_HOME": str(directory / "cache"),
            "NUMBA_CACHE_DIR": str(directory / "numba"),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
        }
    )
    before, after = directory / "before", directory / "after"
    for repo, destination, extra in [
        (reference, before, []),
        (ROOT, after, [str(before)]),
    ]:
        completed = subprocess.run(
            [sys.executable, str(PROBE), str(repo), str(destination), *extra],
            cwd=repo,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads((before / "result.json").read_text()), json.loads(
        (after / "result.json").read_text()
    )


@pytest.mark.parametrize("name", MODELS)
def test_model_and_protocol_outputs_are_exact(name, differential_results):
    """Verify bytes of numeric arrays, weights, metadata, logs and file schemas."""
    before, after = differential_results
    assert after[name] == before[name]
