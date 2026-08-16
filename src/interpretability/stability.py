"""Interpretability stability diagnostics for KAN-style representations."""
from __future__ import annotations
import numpy as np


def curve_correlation(curves_a, curves_b):
    """Per-curve Pearson correlation for arrays [n_curves, n_grid]."""
    a=np.asarray(curves_a,float); b=np.asarray(curves_b,float)
    if a.shape!=b.shape or a.ndim!=2: raise ValueError("curves must have equal [n_curves,n_grid] shape")
    out=[]
    for x,y in zip(a,b):
        sx=x.std(); sy=y.std()
        out.append(float(np.corrcoef(x,y)[0,1]) if sx>0 and sy>0 else float(np.allclose(x,y)))
    return np.asarray(out)


def normalized_curve_distance(curves_a, curves_b, eps=1e-8):
    """Relative L2 distance per learned univariate curve; lower is more stable."""
    a=np.asarray(curves_a,float); b=np.asarray(curves_b,float)
    if a.shape!=b.shape or a.ndim!=2: raise ValueError("curves must have equal [n_curves,n_grid] shape")
    num=np.linalg.norm(a-b,axis=1)
    den=0.5*(np.linalg.norm(a,axis=1)+np.linalg.norm(b,axis=1))
    return num/np.maximum(den,eps)


def aggregate_stability(curve_correlations, curve_distances):
    """Return descriptive explanation-stability summary without inventing a threshold."""
    c=np.asarray(curve_correlations,float); d=np.asarray(curve_distances,float)
    return {
        "correlation_mean":float(np.nanmean(c)),
        "correlation_median":float(np.nanmedian(c)),
        "distance_mean":float(np.nanmean(d)),
        "distance_median":float(np.nanmedian(d)),
    }
