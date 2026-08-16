"""Diagnostics connecting model reliability scores to realized forecast error."""
from __future__ import annotations
import numpy as np
from scipy import stats


def sample_rmse(y_true,y_pred):
    y=np.asarray(y_true); p=np.asarray(y_pred)
    if y.shape!=p.shape: raise ValueError("Shapes must match")
    if y.ndim==1: return np.abs(y-p)
    return np.sqrt(np.mean((y-p)**2,axis=tuple(range(1,y.ndim))))


def reliability_error_bins(reliability,error,n_bins=10):
    r=np.asarray(reliability,dtype=float).reshape(-1); e=np.asarray(error,dtype=float).reshape(-1)
    if len(r)!=len(e): raise ValueError("Lengths must match")
    edges=np.linspace(0,1,n_bins+1); rows=[]
    for i in range(n_bins):
        left,right=edges[i],edges[i+1]; mask=(r>=left)&((r<right) if i<n_bins-1 else (r<=right))
        if mask.any(): rows.append({"bin":i,"left":left,"right":right,"n":int(mask.sum()),"mean_reliability":float(r[mask].mean()),"mean_error":float(e[mask].mean()),"median_error":float(np.median(e[mask]))})
    return rows


def reliability_error_association(reliability,error):
    r=np.asarray(reliability,dtype=float).reshape(-1); e=np.asarray(error,dtype=float).reshape(-1)
    if len(r)!=len(e): raise ValueError("Lengths must match")
    sp=stats.spearmanr(r,e); pe=stats.pearsonr(r,e)
    return {"spearman_rho":float(sp.statistic),"spearman_p":float(sp.pvalue),"pearson_r":float(pe.statistic),"pearson_p":float(pe.pvalue)}


def top_error_detection(reliability,error,quantile=0.9):
    """AUROC/AUPRC for using low reliability to identify the largest errors."""
    from sklearn.metrics import roc_auc_score,average_precision_score
    r=np.asarray(reliability).reshape(-1); e=np.asarray(error).reshape(-1)
    threshold=float(np.quantile(e,quantile)); labels=(e>=threshold).astype(int); score=1-r
    if len(np.unique(labels))<2: return {"error_threshold":threshold,"auroc":None,"auprc":None}
    return {"error_threshold":threshold,"auroc":float(roc_auc_score(labels,score)),"auprc":float(average_precision_score(labels,score))}


def monotonicity_score(bin_rows):
    """Spearman association between increasing reliability bins and mean error."""
    if len(bin_rows)<3: return None
    x=np.asarray([r["mean_reliability"] for r in bin_rows]); y=np.asarray([r["mean_error"] for r in bin_rows])
    res=stats.spearmanr(x,y); return float(res.statistic)
