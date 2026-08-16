import numpy as np
from src.uncertainty.adaptive import conformity_score,rolling_conformal,adaptive_conformal,rolling_coverage
from src.reliability.calibration import sample_rmse,reliability_error_bins,reliability_error_association,top_error_detection


def test_rolling_conformal_is_causal_shape_safe():
    ycal=np.array([0.,1.,2.,3.]); lo=np.array([-.1,.8,1.8,2.7]); hi=np.array([.1,1.2,2.2,3.1])
    scores=conformity_score(ycal,lo,hi)
    y=np.array([4.,5.,8.]); base_lo=np.array([3.8,4.8,5.8]); base_hi=np.array([4.2,5.2,6.2])
    L,U,r=rolling_conformal(y,base_lo,base_hi,scores,alpha=.1,window=4)
    assert L.shape==U.shape==r.shape==y.shape
    assert np.all(L<=base_lo) and np.all(U>=base_hi)


def test_adaptive_alpha_responds_after_miss():
    scores=np.zeros(20); y=np.array([0.,10.,0.]); lo=np.zeros(3); hi=np.zeros(3)
    L,U,tr=adaptive_conformal(y,lo,hi,scores,alpha=.1,gamma=.05,window=20)
    a=tr['alpha'].reshape(-1)
    assert a[2] < a[1]
    assert len(tr['miss'].reshape(-1))==3


def test_rolling_coverage():
    y=np.arange(5.); lo=y-.1; hi=y+.1
    rc=rolling_coverage(y,lo,hi,window=3)
    assert np.isnan(rc[:2]).all()
    assert np.allclose(rc[2:],1.0)


def test_reliability_error_diagnostics():
    y=np.zeros((10,2)); p=np.arange(20,dtype=float).reshape(10,2)/20
    e=sample_rmse(y,p); r=np.linspace(1,.1,10)
    bins=reliability_error_bins(r,e,5); assoc=reliability_error_association(r,e); det=top_error_detection(r,e,.8)
    assert len(e)==10 and len(bins)>1
    assert assoc['spearman_rho'] < 0
    assert 0 <= det['auroc'] <= 1
