import numpy as np
import torch

from waveform_analysis.ml_pipeline.models import get_model, model_names
from waveform_analysis.ml_pipeline.models._cnn1d_common import DirectCNN1D, SharedCNN1D
from waveform_analysis.ml_pipeline.models.direct_mlp import DirectPairMLP
from waveform_analysis.ml_pipeline.models.minirocket import candidates as minirocket_candidates


def test_benchmark_registry_contains_all_model_families():
    expected={"mlp","locally_connected_mlp","shared_cnn1d","direct_mlp","direct_cnn1d","onishi_cnn","minirocket"}
    assert expected <= set(model_names())
    assert get_model("mlp").estimator_formulation=="shared"
    assert get_model("locally_connected_mlp").estimator_formulation=="shared"
    assert get_model("shared_cnn1d").estimator_formulation=="shared"
    assert get_model("direct_mlp").estimator_formulation=="direct"
    assert get_model("direct_cnn1d").estimator_formulation=="direct"
    assert get_model("onishi_cnn").estimator_formulation=="direct"
    assert get_model("minirocket").estimator_formulation=="direct"


def test_shared_cnn1d_is_exactly_antisymmetric():
    model=SharedCNN1D(32,[8],"silu",conv_channels=[4,8],kernel_samples=[5,3])
    pair=torch.randn(6,2,32)
    forward=model(pair);reverse=model(pair[:,[1,0],:])
    torch.testing.assert_close(forward,-reverse)


def test_direct_cnn1d_preserves_pair_shape_contract():
    model=DirectCNN1D(32,[8],"silu",conv_channels=[4,8],kernel_samples=[5,3])
    pair=torch.randn(6,2,32)
    assert model(pair).shape==(6,)


def test_direct_mlp_preserves_pair_shape_contract():
    model=DirectPairMLP(24,[8,4],"silu")
    pair=torch.randn(5,2,24)
    assert model(pair).shape==(5,)


def test_minirocket_candidates_are_explicit_and_small():
    rows=minirocket_candidates({"parameters":{"num_kernels":[10000],"ridge_alpha":[1.0],"n_jobs":[1]}})
    assert rows==[{"num_kernels":10000,"ridge_alpha":1.0,"n_jobs":1}]
