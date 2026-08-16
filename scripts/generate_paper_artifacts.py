"""Generate manuscript-ready tables and figures strictly from saved results.

The script never invents missing values. Artifacts are written under paper/
and results/figures using machine-readable experiment outputs.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dirs():
    for p in [Path('paper/tables'),Path('paper/figures'),Path('results/figures')]: p.mkdir(parents=True,exist_ok=True)

def benchmark_table(path):
    df=pd.read_csv(path); req={'dataset','model','horizon','status','rmse','mae'}
    missing=req-set(df.columns)
    if missing: raise ValueError(f'benchmark missing {sorted(missing)}')
    ok=df[df.status.eq('ok')].copy(); group=['dataset','model','horizon']
    s=ok.groupby(group,as_index=False).agg(n=('seed','count'),rmse_mean=('rmse','mean'),rmse_sd=('rmse','std'),mae_mean=('mae','mean'),mae_sd=('mae','std'))
    s.to_csv('paper/tables/deterministic_forecasting.csv',index=False)
    view=s.copy(); view['RMSE']=view.apply(lambda r:f"{r.rmse_mean:.3f} ± {r.rmse_sd:.3f}" if pd.notna(r.rmse_sd) else f"{r.rmse_mean:.3f}",axis=1); view['MAE']=view.apply(lambda r:f"{r.mae_mean:.3f} ± {r.mae_sd:.3f}" if pd.notna(r.mae_sd) else f"{r.mae_mean:.3f}",axis=1)
    view[['dataset','model','horizon','n','RMSE','MAE']].to_latex('paper/tables/deterministic_forecasting.tex',index=False,escape=True)
    return s

def plot_benchmark(s):
    for dataset in s.dataset.unique():
        sub=s[s.dataset.eq(dataset)]
        fig,ax=plt.subplots(figsize=(7,4.2))
        for model,g in sub.groupby('model'):
            g=g.sort_values('horizon'); ax.plot(g.horizon,g.rmse_mean,marker='o',label=model)
        ax.set_xlabel('Forecast horizon');ax.set_ylabel('RMSE');ax.set_title(f'{dataset}: multi-horizon forecasting');ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
        for ext in ['pdf','png']: fig.savefig(f'paper/figures/{dataset.lower().replace(" ","_")}_rmse_horizon.{ext}',dpi=300 if ext=='png' else None)
        plt.close(fig)

def drift_artifacts(path):
    z=np.load(path); y=z['y'].reshape(-1); target=None
    fig,ax=plt.subplots(figsize=(7,4.2))
    for key,label in [('rolling_coverage','Rolling conformal'),('adaptive_coverage','Adaptive conformal')]:
        if key in z.files: ax.plot(z[key].reshape(-1),label=label)
    ax.set_xlabel('Ordered test prediction');ax.set_ylabel('Rolling empirical coverage');ax.set_ylim(0,1.02);ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(f'paper/figures/rolling_coverage.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig)
    if 'adaptive_alpha' in z.files:
        fig,ax=plt.subplots(figsize=(7,3.8));ax.plot(z['adaptive_alpha'].reshape(-1));ax.set_xlabel('Ordered test prediction');ax.set_ylabel('Adaptive alpha');ax.grid(alpha=.25);fig.tight_layout()
        for ext in ['pdf','png']: fig.savefig(f'paper/figures/adaptive_alpha.{ext}',dpi=300 if ext=='png' else None)
        plt.close(fig)

def reliability_artifacts(samples_path,bins_path):
    z=np.load(samples_path); r=z['reliability'].reshape(-1); e=z['error'].reshape(-1)
    fig,ax=plt.subplots(figsize=(5.5,4.2));ax.scatter(r,e,s=10,alpha=.35);ax.set_xlabel('Reliability score');ax.set_ylabel('Realized sample error');ax.grid(alpha=.25);fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(f'paper/figures/reliability_vs_error.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig)
    b=pd.read_csv(bins_path); fig,ax=plt.subplots(figsize=(5.5,4.2));ax.plot(b.mean_reliability,b.mean_error,marker='o');ax.set_xlabel('Mean reliability in bin');ax.set_ylabel('Mean realized error');ax.grid(alpha=.25);fig.tight_layout()
    for ext in ['pdf','png']: fig.savefig(f'paper/figures/reliability_calibration.{ext}',dpi=300 if ext=='png' else None)
    plt.close(fig)

def write_manifest(args):
    manifest={'benchmark':args.benchmark,'drift_npz':args.drift_npz,'reliability_samples':args.reliability_samples,'reliability_bins':args.reliability_bins}
    Path('paper/artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

def main(a):
    ensure_dirs();
    if a.benchmark:
        s=benchmark_table(a.benchmark);plot_benchmark(s)
    if a.drift_npz: drift_artifacts(a.drift_npz)
    if a.reliability_samples and a.reliability_bins: reliability_artifacts(a.reliability_samples,a.reliability_bins)
    write_manifest(a)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--benchmark',default=None);p.add_argument('--drift-npz',default=None);p.add_argument('--reliability-samples',default=None);p.add_argument('--reliability-bins',default=None);main(p.parse_args())
