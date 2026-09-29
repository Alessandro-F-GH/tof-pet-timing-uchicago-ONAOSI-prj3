import numpy as np
import torch

from waveform_analysis.ml_pipeline.models import get_model, model_names
from waveform_analysis.ml_pipeline.models._cnn1d_common import DirectCNN1D, SharedCNN1D
from waveform_analysis.ml_pipeline.models.direct_mlp import DirectPairMLP
from waveform_analysis.ml_pipeline.models.minirocket import candidates as minirocket_candidates
from waveform_analysis.ml_pipeline.models.spec import FeatureTransformSpec
from waveform_analysis.ml_pipeline.train import FeatureTransformCache


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
    assert get_model("minirocket").feature_transform is not None


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


def test_minirocket_candidates_separate_transform_and_ridge_parameters():
    rows=minirocket_candidates({"parameters":{"num_kernels":[10000],"ridge_alpha":[0.1,1.0,10.0]},"transform":{"n_jobs":-1}})
    assert rows==[
        {"num_kernels":10000,"ridge_alpha":0.1},
        {"num_kernels":10000,"ridge_alpha":1.0},
        {"num_kernels":10000,"ridge_alpha":10.0},
    ]
    transform=get_model("minirocket").feature_transform
    assert transform.parameters(rows[0],{"transform":{"n_jobs":-1}})=={"num_kernels":10000,"n_jobs":-1}
    assert transform.parameters(rows[2],{"transform":{"n_jobs":-1}})=={"num_kernels":10000,"n_jobs":-1}


def test_feature_transform_cache_reuses_same_transform_across_downstream_candidates():
    calls={"fit":0,"apply":0}
    def parameters(candidate,config):return {"scale":candidate["transform_scale"]}
    def fit_transform(params,x,*,seed,config):
        calls["fit"]+=1;artifact={"scale":params["scale"]};return artifact,np.asarray(x)*params["scale"]
    def transform(artifact,x):calls["apply"]+=1;return np.asarray(x)*artifact["scale"]
    spec=FeatureTransformSpec("fake",parameters,fit_transform,transform)
    cache=FeatureTransformCache();x=np.arange(24,dtype=np.float32).reshape(3,2,4);cfg={}
    a,xa=cache.prepare(spec,{"transform_scale":2,"alpha":0.1},x,seed_base=7,scope_key="train",config=cfg)
    b,xb=cache.prepare(spec,{"transform_scale":2,"alpha":10.0},x,seed_base=7,scope_key="train",config=cfg)
    assert a is b and calls["fit"]==1
    np.testing.assert_array_equal(xa,xb)
    val=np.ones((2,2,4),dtype=np.float32)
    first=a.apply(val,"validation");second=b.apply(val,"validation")
    assert calls["apply"]==1
    np.testing.assert_array_equal(first,second)


def test_feature_transform_cache_refits_when_transform_parameters_change():
    calls={"fit":0}
    def parameters(candidate,config):return {"scale":candidate["transform_scale"]}
    def fit_transform(params,x,*,seed,config):calls["fit"]+=1;return dict(params),np.asarray(x)*params["scale"]
    spec=FeatureTransformSpec("fake",parameters,fit_transform,lambda artifact,x:np.asarray(x)*artifact["scale"])
    cache=FeatureTransformCache();x=np.ones((3,2,4),dtype=np.float32)
    cache.prepare(spec,{"transform_scale":2},x,seed_base=7,scope_key="train",config={})
    cache.prepare(spec,{"transform_scale":3},x,seed_base=7,scope_key="train",config={})
    assert calls["fit"]==2
