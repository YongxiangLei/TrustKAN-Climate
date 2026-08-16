"""Reliability fusion and abstention rules for TrustKAN."""
from __future__ import annotations
import numpy as np


def normalize_interval_width(width, calibration_widths, eps=1e-8):
    """Convert prediction interval width into [0,1] reliability using calibration ECDF."""
    w=np.asarray(width,float).reshape(-1)
    ref=np.sort(np.asarray(calibration_widths,float).reshape(-1))
    if len(ref)==0: raise ValueError("calibration_widths must be non-empty")
    rank=np.searchsorted(ref,w,side="right")/len(ref)
    return np.clip(1.0-rank,0.0,1.0)


def fuse_reliability(*components, weights=None, floor=1e-8):
    """Weighted geometric mean of reliability components in [0,1].

    Geometric fusion penalizes a sample if any trust signal is poor, which is
    appropriate for conservative selective forecasting. Weights must be fixed
    on validation/calibration data, not chosen after inspecting the test set.
    """
    if not components: raise ValueError("At least one component is required")
    xs=[np.clip(np.asarray(c,float).reshape(-1),floor,1.0) for c in components]
    n=len(xs[0])
    if any(len(x)!=n for x in xs): raise ValueError("All reliability components must have equal length")
    if weights is None: weights=np.ones(len(xs))/len(xs)
    weights=np.asarray(weights,float)
    if len(weights)!=len(xs) or np.any(weights<0) or weights.sum()<=0: raise ValueError("Invalid weights")
    weights=weights/weights.sum()
    logs=sum(w*np.log(x) for w,x in zip(weights,xs))
    return np.exp(logs)


def selective_mask(reliability, threshold):
    r=np.asarray(reliability,float).reshape(-1)
    return r>=float(threshold)


def choose_threshold_on_calibration(y_true,y_pred,reliability,max_risk=None,min_coverage=0.5):
    """Choose a threshold using calibration data only.

    If max_risk is given, select the lowest threshold satisfying RMSE<=max_risk
    while maximizing coverage. Otherwise select the threshold minimizing RMSE
    subject to coverage>=min_coverage.
    """
    y=np.asarray(y_true,float).reshape(-1); p=np.asarray(y_pred,float).reshape(-1); r=np.asarray(reliability,float).reshape(-1)
    if not(len(y)==len(p)==len(r)): raise ValueError("Inputs must have equal length")
    candidates=np.unique(r)
    best=None
    for t in candidates:
        m=r>=t; cov=m.mean()
        if not m.any() or cov<min_coverage: continue
        risk=float(np.sqrt(np.mean((y[m]-p[m])**2)))
        if max_risk is not None:
            if risk<=max_risk and (best is None or cov>best["coverage"]): best={"threshold":float(t),"coverage":float(cov),"rmse":risk}
        else:
            if best is None or risk<best["rmse"]: best={"threshold":float(t),"coverage":float(cov),"rmse":risk}
    return best
