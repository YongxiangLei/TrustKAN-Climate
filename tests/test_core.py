import numpy as np
import torch

from src.metrics.forecast import rmse, risk_coverage_curve, sample_risk_coverage_curve
from src.models.trustkan import TrustKAN
from src.uncertainty.conformal import (
    apply_conformal,
    conformal_radius,
    horizonwise_conformal_radii,
    joint_interval_coverage,
    mean_interval_score,
    simultaneous_conformal_radius,
)


def test_trustkan_shapes():
    model = TrustKAN(n_features=3, horizon=4, hidden_dim=16)
    x = torch.randn(5, 24, 3)
    out = model(x)
    assert out["point"].shape == (5, 4)
    assert out["quantiles"].shape == (5, 4, 3)
    assert out["embedding"].shape == (5, 16)


def test_quantiles_are_ordered():
    model = TrustKAN(n_features=1, horizon=2, hidden_dim=8)
    q = model(torch.randn(3, 12, 1))["quantiles"]
    assert torch.all(q[..., 1:] >= q[..., :-1])


def test_conformal_expands_interval():
    y = np.array([0., 1., 2., 3.])
    lo = np.array([0.1, 0.8, 1.8, 2.8])
    hi = np.array([0.2, 1.2, 2.2, 3.2])
    r = conformal_radius(y, lo, hi, alpha=0.1)
    new_lo, new_hi = apply_conformal(lo, hi, r)
    assert np.all(new_lo <= lo)
    assert np.all(new_hi >= hi)


def test_selective_curve_lengths():
    y = np.array([0., 1., 2.])
    yhat = np.array([0., 1.5, 4.])
    reliability = np.array([0.9, 0.8, 0.1])
    coverage, risk = risk_coverage_curve(y, yhat, reliability)
    assert len(coverage) == len(risk) == 3
    assert rmse(y[:1], yhat[:1]) == 0.0


def test_multihorizon_conformal_separates_marginal_and_joint_calibration():
    y=np.array([[0.,0.],[0.,10.],[0.,0.],[0.,0.]])
    lo=np.full_like(y,-1.); hi=np.full_like(y,1.)
    radii=horizonwise_conformal_radii(y,lo,hi,alpha=.25)
    simultaneous=simultaneous_conformal_radius(y,lo,hi,alpha=.25)
    assert radii.shape==(2,)
    assert simultaneous>=radii.max()
    expanded=apply_conformal(lo,hi,simultaneous)
    assert joint_interval_coverage(y,*expanded)>=.75
    assert mean_interval_score(y,*expanded,alpha=.25)>=0


def test_sample_risk_curve_keeps_horizon_vectors_together():
    y=np.zeros((3,2)); pred=np.array([[0.,0.],[1.,1.],[3.,3.]])
    coverage,risk=sample_risk_coverage_curve(y,pred,[.9,.8,.1])
    assert len(coverage)==len(risk)==3
    assert risk[0]==0.0
