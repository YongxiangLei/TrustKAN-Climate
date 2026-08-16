"""Paired statistical analysis for publication-grade model comparison."""
from __future__ import annotations
import numpy as np
from scipy import stats


def paired_absolute_errors(y_true, pred_a, pred_b):
    y=np.asarray(y_true).reshape(-1)
    a=np.asarray(pred_a).reshape(-1)
    b=np.asarray(pred_b).reshape(-1)
    if not (len(y)==len(a)==len(b)):
        raise ValueError("Arrays must have equal flattened length")
    return np.abs(y-a), np.abs(y-b)


def wilcoxon_paired(y_true, pred_a, pred_b):
    ea, eb = paired_absolute_errors(y_true, pred_a, pred_b)
    diff = ea-eb
    if np.allclose(diff, 0):
        return {"statistic": 0.0, "pvalue": 1.0, "median_error_difference": 0.0}
    res=stats.wilcoxon(ea, eb, zero_method="pratt", alternative="two-sided")
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue), "median_error_difference": float(np.median(diff))}


def paired_bootstrap_mae_difference(y_true, pred_a, pred_b, n_boot=5000, confidence=0.95, seed=0):
    """Bootstrap MAE(A)-MAE(B); negative values favor model A."""
    ea, eb = paired_absolute_errors(y_true, pred_a, pred_b)
    rng=np.random.default_rng(seed); n=len(ea)
    draws=np.empty(n_boot)
    for i in range(n_boot):
        idx=rng.integers(0,n,n)
        draws[i]=ea[idx].mean()-eb[idx].mean()
    alpha=(1-confidence)/2
    return {
        "mean_difference": float((ea-eb).mean()),
        "ci_low": float(np.quantile(draws,alpha)),
        "ci_high": float(np.quantile(draws,1-alpha)),
        "confidence": confidence,
        "n_boot": n_boot,
    }


def paired_cohens_d(y_true, pred_a, pred_b):
    ea, eb=paired_absolute_errors(y_true,pred_a,pred_b); d=ea-eb
    sd=d.std(ddof=1)
    return float(d.mean()/sd) if sd>0 else 0.0
