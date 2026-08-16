"""Adaptive conformal prediction for temporally ordered forecasts.

The routines operate sequentially and never use future labels to calibrate the
current prediction interval. They are intended for drift experiments, not as a
claim of exchangeability under non-stationarity.
"""
from __future__ import annotations
from collections import deque
import numpy as np


def conformity_score(y, lower, upper):
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    return np.maximum(np.maximum(lower-y, y-upper),0.0)


def higher_quantile(values, q: float):
    values=np.asarray(values,dtype=float).reshape(-1)
    if len(values)==0: raise ValueError("Calibration scores must be non-empty")
    q=float(np.clip(q,0.0,1.0))
    return float(np.quantile(values,q,method="higher"))


def rolling_conformal(y, lower, upper, initial_scores, *, alpha=0.1, window=256):
    """Sequential rolling-window conformal expansion.

    For time t, the radius is computed only from scores available before seeing
    y[t]. The score for t is appended afterward.
    """
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    if y.shape!=lower.shape or y.shape!=upper.shape: raise ValueError("Shapes must match")
    hist=deque(np.asarray(initial_scores).reshape(-1).tolist(),maxlen=window)
    if not hist: raise ValueError("initial_scores must be non-empty")
    out_lo=np.empty_like(lower,dtype=float); out_hi=np.empty_like(upper,dtype=float); radii=[]
    flat_y,flat_lo,flat_hi=y.reshape(-1),lower.reshape(-1),upper.reshape(-1)
    for i,(yt,lo,hi) in enumerate(zip(flat_y,flat_lo,flat_hi)):
        n=len(hist); level=min(1.0,np.ceil((n+1)*(1-alpha))/n)
        r=higher_quantile(hist,level); radii.append(r)
        out_lo.reshape(-1)[i]=lo-r; out_hi.reshape(-1)[i]=hi+r
        hist.append(float(conformity_score(yt,lo,hi)))
    return out_lo,out_hi,np.asarray(radii).reshape(y.shape)


def adaptive_conformal(y, lower, upper, initial_scores, *, alpha=0.1, gamma=0.01,
                       window=256, alpha_min=0.005, alpha_max=0.5):
    """Adaptive conformal inference (ACI-style alpha update).

    alpha_t is chosen before observing y_t. After observing whether the interval
    missed, alpha is updated as alpha_{t+1}=alpha_t+gamma*(alpha-miss_t).
    A rolling score window is used to remain responsive to local scale changes.
    """
    y=np.asarray(y); lower=np.asarray(lower); upper=np.asarray(upper)
    if y.shape!=lower.shape or y.shape!=upper.shape: raise ValueError("Shapes must match")
    hist=deque(np.asarray(initial_scores).reshape(-1).tolist(),maxlen=window)
    if not hist: raise ValueError("initial_scores must be non-empty")
    current=float(alpha); out_lo=np.empty_like(lower,dtype=float); out_hi=np.empty_like(upper,dtype=float)
    radii=[]; alpha_path=[]; misses=[]
    fy,fl,fu=y.reshape(-1),lower.reshape(-1),upper.reshape(-1)
    olo,ohi=out_lo.reshape(-1),out_hi.reshape(-1)
    for i,(yt,lo,hi) in enumerate(zip(fy,fl,fu)):
        n=len(hist); level=min(1.0,np.ceil((n+1)*(1-current))/n)
        r=higher_quantile(hist,level); L,U=lo-r,hi+r
        miss=float(not (L<=yt<=U)); olo[i]=L; ohi[i]=U
        radii.append(r); alpha_path.append(current); misses.append(miss)
        hist.append(float(conformity_score(yt,lo,hi)))
        current=float(np.clip(current+gamma*(alpha-miss),alpha_min,alpha_max))
    shape=y.shape
    return out_lo,out_hi,{"radius":np.asarray(radii).reshape(shape),"alpha":np.asarray(alpha_path).reshape(shape),"miss":np.asarray(misses).reshape(shape)}


def rolling_coverage(y, lower, upper, window=100):
    y=np.asarray(y).reshape(-1); lo=np.asarray(lower).reshape(-1); hi=np.asarray(upper).reshape(-1)
    hit=((y>=lo)&(y<=hi)).astype(float); out=np.full(len(hit),np.nan)
    for i in range(window-1,len(hit)): out[i]=hit[i-window+1:i+1].mean()
    return out
