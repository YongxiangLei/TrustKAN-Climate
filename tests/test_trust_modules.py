import numpy as np
import torch

from src.training.trust_engine import pinball_loss
from src.drift.scores import mahalanobis_shift, percentile_to_reliability
from src.reliability.fusion import fuse_reliability, choose_threshold_on_calibration
from src.interpretability.kan_curves import extract_trustkan_curves
from src.interpretability.stability import curve_correlation, normalized_curve_distance
from src.models.trustkan import TrustKAN


def test_pinball_loss_zero_for_exact_quantiles():
    y=torch.tensor([[1.0,2.0]])
    pred=y.unsqueeze(-1).repeat(1,1,3)
    assert float(pinball_loss(pred,y,(0.1,0.5,0.9))) == 0.0


def test_mahalanobis_shift_is_larger_far_away():
    rng=np.random.default_rng(0)
    ref=rng.normal(size=(200,3))
    near=np.zeros((1,3)); far=np.full((1,3),10.0)
    d=mahalanobis_shift(ref,np.vstack([near,far]))
    assert d[1] > d[0]


def test_percentile_reliability_decreases_with_score():
    ref=np.arange(10.0)
    r=percentile_to_reliability(np.array([0.0,9.0]),ref)
    assert r[0] > r[1]


def test_fusion_penalizes_low_component():
    a=np.array([0.9,0.9]); b=np.array([0.9,0.1])
    r=fuse_reliability(a,b)
    assert r[0] > r[1]


def test_threshold_selected_without_test_information():
    y=np.array([0.,0.,0.,0.]); p=np.array([0.,.1,2.,3.]); r=np.array([.9,.8,.2,.1])
    selected=choose_threshold_on_calibration(y,p,r,min_coverage=.5)
    assert selected is not None
    assert selected['coverage'] >= .5


def test_threshold_selection_retains_multihorizon_origins_together():
    y=np.zeros((4,2)); p=np.array([[0.,0.],[.1,.1],[2.,2.],[3.,3.]])
    r=np.array([.9,.8,.2,.1])
    selected=choose_threshold_on_calibration(y,p,r,min_coverage=.5)
    assert selected is not None
    assert selected["coverage"]>=.5


def test_curve_stability_identical_curves():
    a=np.array([[0.,1.,2.],[2.,1.,0.]])
    assert np.allclose(curve_correlation(a,a),1.0)
    assert np.allclose(normalized_curve_distance(a,a),0.0)


def test_trustkan_curves_are_seed_stable_and_change_after_perturbation():
    torch.manual_seed(0)
    model_a = TrustKAN(n_features=1, horizon=1, hidden_dim=4, grid_size=3)
    torch.manual_seed(0)
    model_b = TrustKAN(n_features=1, horizon=1, hidden_dim=4, grid_size=3)
    curves_a = extract_trustkan_curves(model_a)
    curves_b = extract_trustkan_curves(model_b)
    assert curves_a["curves"].shape[0] == 16
    assert np.allclose(curves_a["curves"], curves_b["curves"])
    with torch.no_grad():
        model_b.encoder.kan.coeff.add_(0.5)
    curves_c = extract_trustkan_curves(model_b)
    distance = normalized_curve_distance(curves_a["curves"], curves_c["curves"])
    assert float(distance.mean()) > 0.0
