"""Compare static, rolling and adaptive conformal intervals on ordered test data.

Input NPZ must contain y_cal, cal_lower, cal_upper, y_test, test_lower,
test_upper. Arrays may be [N,H]; evaluation follows their flattened temporal
order, so for multi-horizon work prefer one horizon per experiment unless the
ordering semantics are explicitly justified.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.uncertainty.conformal import conformal_radius,apply_conformal,interval_coverage,mean_interval_width
from src.uncertainty.adaptive import conformity_score,rolling_conformal,adaptive_conformal,rolling_coverage


def summarize(y,lo,hi):
    return {"coverage":interval_coverage(y,lo,hi),"mean_interval_width":mean_interval_width(lo,hi)}

def main(a):
    z=np.load(a.input); ycal=z['y_cal']; clo=z['cal_lower']; chi=z['cal_upper']; yt=z['y_test']; tlo=z['test_lower']; thi=z['test_upper']
    scores=conformity_score(ycal,clo,chi).reshape(-1)
    r=conformal_radius(ycal,clo,chi,a.alpha); slo,shi=apply_conformal(tlo,thi,r)
    rlo,rhi,rr=rolling_conformal(yt,tlo,thi,scores,alpha=a.alpha,window=a.window)
    alo,ahi,trace=adaptive_conformal(yt,tlo,thi,scores,alpha=a.alpha,gamma=a.gamma,window=a.window)
    target=1-a.alpha
    out={"target_coverage":target,"static":summarize(yt,slo,shi),"rolling":summarize(yt,rlo,rhi),"adaptive":summarize(yt,alo,ahi),"static_radius":r,
         "rolling_coverage_mae":float(np.nanmean(np.abs(rolling_coverage(yt,rlo,rhi,a.coverage_window)-target))),
         "adaptive_coverage_mae":float(np.nanmean(np.abs(rolling_coverage(yt,alo,ahi,a.coverage_window)-target))),
         "adaptive_final_alpha":float(trace['alpha'].reshape(-1)[-1])}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2),encoding='utf-8')
    np.savez_compressed(p.with_suffix('.npz'),y=yt,static_lower=slo,static_upper=shi,rolling_lower=rlo,rolling_upper=rhi,adaptive_lower=alo,adaptive_upper=ahi,rolling_radius=rr,adaptive_radius=trace['radius'],adaptive_alpha=trace['alpha'],adaptive_miss=trace['miss'],rolling_coverage=rolling_coverage(yt,rlo,rhi,a.coverage_window),adaptive_coverage=rolling_coverage(yt,alo,ahi,a.coverage_window))
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',default='results/drift/adaptive_conformal.json');p.add_argument('--alpha',type=float,default=.1);p.add_argument('--gamma',type=float,default=.01);p.add_argument('--window',type=int,default=256);p.add_argument('--coverage-window',type=int,default=100);main(p.parse_args())
