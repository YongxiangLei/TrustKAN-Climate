"""Distribution-shift scores used by TrustKAN reliability analysis."""
from __future__ import annotations
import numpy as np


def _regularized_cov(x, eps=1e-6):
    x=np.asarray(x,float)
    c=np.cov(x,rowvar=False)
    if np.ndim(c)==0: c=np.array([[float(c)]])
    return c + eps*np.eye(c.shape[0])


def mahalanobis_shift(reference_embeddings, query_embeddings, eps=1e-6):
    """Mahalanobis distance of query embeddings from reference distribution."""
    ref=np.asarray(reference_embeddings,float); q=np.asarray(query_embeddings,float)
    mu=ref.mean(axis=0); inv=np.linalg.pinv(_regularized_cov(ref,eps))
    d=q-mu
    return np.sqrt(np.maximum(0.0,np.einsum("ni,ij,nj->n",d,inv,d)))


def residual_zscore(calibration_residuals, query_residuals, eps=1e-8):
    """Absolute standardized residual score based only on calibration residuals."""
    cal=np.asarray(calibration_residuals,float).reshape(-1)
    q=np.asarray(query_residuals,float).reshape(-1)
    mu=cal.mean(); sd=cal.std(ddof=1) if len(cal)>1 else 0.0
    return np.abs(q-mu)/max(sd,eps)


def percentile_to_reliability(scores, reference_scores):
    """Map a shift score to [0,1], where 1 is most in-distribution.

    Uses the empirical survival percentile of reference/calibration scores.
    """
    s=np.asarray(scores,float).reshape(-1); ref=np.sort(np.asarray(reference_scores,float).reshape(-1))
    if len(ref)==0: raise ValueError("reference_scores must be non-empty")
    rank=np.searchsorted(ref,s,side="right")/len(ref)
    return 1.0-rank
