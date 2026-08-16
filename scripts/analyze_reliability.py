"""Analyse whether reported reliability tracks realized forecast error."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from src.reliability.calibration import sample_rmse,reliability_error_bins,reliability_error_association,top_error_detection,monotonicity_score


def main(a):
    z=np.load(a.input); y=z['target']; pred=z['prediction']; rel=z['reliability']
    err=sample_rmse(y,pred)
    bins=reliability_error_bins(rel,err,a.bins)
    out={"association":reliability_error_association(rel,err),"top_error_detection":top_error_detection(rel,err,a.error_quantile),"monotonicity":monotonicity_score(bins),"mean_error":float(err.mean()),"n":int(len(err))}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2),encoding='utf-8')
    pd.DataFrame(bins).to_csv(p.with_name(p.stem+'_bins.csv'),index=False)
    np.savez_compressed(p.with_name(p.stem+'_samples.npz'),reliability=rel,error=err)
    print(json.dumps(out,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True,help='NPZ with target, prediction, reliability');p.add_argument('--out',default='results/reliability/reliability_error.json');p.add_argument('--bins',type=int,default=10);p.add_argument('--error-quantile',type=float,default=.9);main(p.parse_args())
