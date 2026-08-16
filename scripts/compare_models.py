"""Compare two saved model prediction files with paired tests."""
from __future__ import annotations
import argparse, json
import numpy as np
from src.statistics.paired_tests import wilcoxon_paired, paired_bootstrap_mae_difference, paired_cohens_d


def main(a_path,b_path,out):
    a=np.load(a_path); b=np.load(b_path)
    ya=a["target"]; yb=b["target"]
    if ya.shape!=yb.shape or not np.allclose(ya,yb,equal_nan=True):
        raise ValueError("Saved result files do not share identical targets; paired comparison is invalid")
    result={
        "wilcoxon_absolute_error": wilcoxon_paired(ya,a["prediction"],b["prediction"]),
        "bootstrap_mae_difference_a_minus_b": paired_bootstrap_mae_difference(ya,a["prediction"],b["prediction"]),
        "paired_cohens_d": paired_cohens_d(ya,a["prediction"],b["prediction"]),
        "interpretation": "Negative MAE difference/effect favors model A; positive favors model B. Statistical significance does not by itself imply practical importance."
    }
    with open(out,"w",encoding="utf-8") as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--a",required=True); p.add_argument("--b",required=True); p.add_argument("--out",default="results/statistical_tests/comparison.json"); args=p.parse_args()
    from pathlib import Path; Path(args.out).parent.mkdir(parents=True,exist_ok=True); main(args.a,args.b,args.out)
