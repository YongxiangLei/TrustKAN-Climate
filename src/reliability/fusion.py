"""Reliability fusion and abstention rules for TrustKAN."""
from __future__ import annotations
import numpy as np

from src.metrics.forecast import aurc, sample_risk_coverage_curve


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


def choose_fusion_weights_on_calibration(y_true, y_pred, components, grid=21):
    """Pick fusion weights on calibration data by risk-coverage area.

    Freezing the weights at equal mass let the fused score be beaten by its own
    best single component, because one component was anti-correlated with error
    at the long horizons while still carrying half of the score. Searching a
    grid whose endpoints are the single-component scores makes the fused score
    no worse than its best component on the calibration split by construction,
    and lets the data withhold mass from a component that does not earn it.

    Only calibration targets and predictions are read, so the choice never sees
    a test label. Returns the weights and the calibration curve behind them, so
    a reader can see how much the fusion actually gained.
    """
    components = [np.asarray(c, float).reshape(-1) for c in components]
    if len(components) != 2:
        raise ValueError("weight selection is implemented for two components")
    if len({len(c) for c in components}) != 1:
        raise ValueError("All reliability components must have equal length")
    if int(grid) < 3:
        raise ValueError("grid must admit both endpoints and an interior point")
    searched = []
    for weight in np.linspace(0.0, 1.0, int(grid)):
        weights = (weight, 1.0 - weight)
        fused = fuse_reliability(*components, weights=weights)
        coverage, risk = sample_risk_coverage_curve(y_true, y_pred, fused)
        searched.append({"weights": weights, "calibration_aurc": float(aurc(coverage, risk))})
    best = min(searched, key=lambda item: item["calibration_aurc"])
    return {
        "weights": tuple(float(w) for w in best["weights"]),
        "calibration_aurc": best["calibration_aurc"],
        "searched": searched,
        "endpoint_aurc": {
            "first_component_only": searched[-1]["calibration_aurc"],
            "second_component_only": searched[0]["calibration_aurc"],
        },
    }


def selective_mask(reliability, threshold):
    r=np.asarray(reliability,float).reshape(-1)
    return r>=float(threshold)


def choose_threshold_on_calibration(y_true,y_pred,reliability,max_risk=None,min_coverage=0.5):
    """Choose a threshold using calibration data only.

    If max_risk is given, select the lowest threshold satisfying RMSE<=max_risk
    while maximizing coverage. Otherwise select the threshold minimizing RMSE
    subject to coverage>=min_coverage.
    """
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); r=np.asarray(reliability,float).reshape(-1)
    if y.shape!=p.shape: raise ValueError("Target and prediction shapes must match")
    if y.ndim==0 or y.shape[0]!=len(r):
        raise ValueError("Reliability must provide one score per forecast origin")
    axes=tuple(range(1,y.ndim))
    squared=(y-p)**2
    sample_mse=squared.mean(axis=axes) if axes else squared
    candidates=np.unique(r)
    best=None
    for t in candidates:
        m=r>=t; cov=m.mean()
        if not m.any() or cov<min_coverage: continue
        risk=float(np.sqrt(np.mean(sample_mse[m])))
        if max_risk is not None:
            if risk<=max_risk and (best is None or cov>best["coverage"]): best={"threshold":float(t),"coverage":float(cov),"rmse":risk}
        else:
            if best is None or risk<best["rmse"]: best={"threshold":float(t),"coverage":float(cov),"rmse":risk}
    return best
