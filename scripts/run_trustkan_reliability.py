"""Run TrustKAN calibration, shift scoring and selective forecasting on saved splits.

Threshold selection is calibration-only. The script also persists arrays needed
for adaptive conformal, reliability-error analysis, and paper artifact generation.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch
import _bootstrap  # noqa: F401  # repository-root import setup
from src.models.trustkan import TrustKAN
from src.training.trust_engine import train_trustkan, predict_trustkan
from src.uncertainty.conformal import conformal_radius, apply_conformal, interval_coverage, mean_interval_width
from src.drift.scores import mahalanobis_shift, percentile_to_reliability
from src.reliability.fusion import normalize_interval_width, fuse_reliability, choose_threshold_on_calibration, selective_mask
from src.metrics.forecast import rmse, risk_coverage_curve, aurc


def load_split(path):
    z=np.load(path); return {k:z[k] for k in z.files}
def loader(x,y,batch=64,shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)),batch_size=batch,shuffle=shuffle)
def main(args):
    data=load_split(args.split_file)
    tr=loader(data['x_train'],data['y_train'],args.batch,True); va=loader(data['x_val'],data['y_val'],args.batch); ca=loader(data['x_cal'],data['y_cal'],args.batch); te=loader(data['x_test'],data['y_test'],args.batch)
    model=TrustKAN(data['x_train'].shape[-1],horizon=data['y_train'].shape[-1],hidden_dim=args.hidden,grid_size=args.grid)
    model,history,seconds=train_trustkan(model,tr,va,epochs=args.epochs,lr=args.lr,patience=args.patience)
    cal=predict_trustkan(model,ca); test=predict_trustkan(model,te)
    lo_idx,hi_idx=0,len(model.quantiles)-1
    cal_lo,cal_hi=cal['quantiles'][...,lo_idx],cal['quantiles'][...,hi_idx]; test_lo,test_hi=test['quantiles'][...,lo_idx],test['quantiles'][...,hi_idx]
    radius=conformal_radius(cal['target'],cal_lo,cal_hi,alpha=args.alpha); cal_lo_c,cal_hi_c=apply_conformal(cal_lo,cal_hi,radius); test_lo_c,test_hi_c=apply_conformal(test_lo,test_hi,radius)
    cal_width=(cal_hi_c-cal_lo_c).mean(axis=1); test_width=(test_hi_c-test_lo_c).mean(axis=1)
    width_rel_cal=normalize_interval_width(cal_width,cal_width); width_rel_test=normalize_interval_width(test_width,cal_width)
    cal_shift=mahalanobis_shift(cal['embedding'],cal['embedding']); test_shift=mahalanobis_shift(cal['embedding'],test['embedding'])
    shift_rel_cal=percentile_to_reliability(cal_shift,cal_shift); shift_rel_test=percentile_to_reliability(test_shift,cal_shift)
    rel_cal=fuse_reliability(width_rel_cal,shift_rel_cal); rel_test=fuse_reliability(width_rel_test,shift_rel_test)
    selected=choose_threshold_on_calibration(cal['target'],cal['point'],rel_cal,min_coverage=args.min_coverage)
    if selected is None: raise RuntimeError('No valid reliability threshold found on calibration data')
    mask=selective_mask(rel_test,selected['threshold']); cov_curve,risk_curve=risk_coverage_curve(test['target'],test['point'],np.repeat(rel_test,test['target'].shape[1]))
    result={'train_seconds':seconds,'conformal_alpha':args.alpha,'conformal_radius':radius,'calibration_coverage':interval_coverage(cal['target'],cal_lo_c,cal_hi_c),'test_coverage':interval_coverage(test['target'],test_lo_c,test_hi_c),'test_mean_interval_width':mean_interval_width(test_lo_c,test_hi_c),'test_rmse_all':rmse(test['target'],test['point']),'threshold_from_calibration':selected,'test_sample_coverage_after_abstention':float(mask.mean()),'test_rmse_selected':rmse(test['target'][mask],test['point'][mask]) if mask.any() else None,'aurc':aurc(cov_curve,risk_curve)}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    stem=out.with_suffix('')
    np.savez_compressed(str(stem)+'_reliability.npz',target=test['target'],prediction=test['point'],reliability=rel_test,width_reliability=width_rel_test,shift_reliability=shift_rel_test,selected_mask=mask,embedding=test['embedding'])
    np.savez_compressed(str(stem)+'_conformal_input.npz',y_cal=cal['target'],cal_lower=cal_lo,cal_upper=cal_hi,y_test=test['target'],test_lower=test_lo,test_upper=test_hi)
    np.savez_compressed(str(stem)+'_risk_coverage.npz',coverage=cov_curve,risk=risk_curve)
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--split-file',required=True); p.add_argument('--out',default='results/reliability/trustkan.json'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=10); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--batch',type=int,default=64); p.add_argument('--hidden',type=int,default=64); p.add_argument('--grid',type=int,default=8); p.add_argument('--alpha',type=float,default=.1); p.add_argument('--min-coverage',type=float,default=.5); main(p.parse_args())
