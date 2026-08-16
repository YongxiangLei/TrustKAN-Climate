"""Aggregate immutable per-run CSV results into publication tables."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

REQUIRED={"dataset","model","horizon","seed","status","rmse","mae"}

def main(path,outdir):
    df=pd.read_csv(path); missing=REQUIRED-set(df.columns)
    if missing: raise ValueError(f"Missing result columns: {sorted(missing)}")
    ok=df[df.status.eq("ok")].copy()
    if ok.empty: raise ValueError("No successful runs to aggregate")
    group=["dataset","model","horizon"]
    summary=ok.groupby(group,as_index=False).agg(
        n=("seed","count"), rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"),
        mae_mean=("mae","mean"),mae_sd=("mae","std"),
        train_seconds_mean=("train_seconds","mean") if "train_seconds" in ok else ("rmse","size"))
    out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/"benchmark_summary.csv",index=False)
    latex=summary.copy()
    latex["RMSE"] = latex.apply(lambda r: f'{r.rmse_mean:.4f} ± {r.rmse_sd:.4f}' if pd.notna(r.rmse_sd) else f'{r.rmse_mean:.4f}',axis=1)
    latex["MAE"] = latex.apply(lambda r: f'{r.mae_mean:.4f} ± {r.mae_sd:.4f}' if pd.notna(r.mae_sd) else f'{r.mae_mean:.4f}',axis=1)
    latex[["dataset","model","horizon","n","RMSE","MAE"]].to_latex(out/"benchmark_summary.tex",index=False,escape=True)
    failed=df[~df.status.eq("ok")]
    failed.to_csv(out/"failed_runs.csv",index=False)
    print(summary.to_string(index=False))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",default="results/aggregated/cet_runs.csv"); p.add_argument("--outdir",default="results/tables"); a=p.parse_args(); main(a.input,a.outdir)
