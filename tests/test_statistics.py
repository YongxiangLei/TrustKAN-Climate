import numpy as np
from src.statistics.paired_tests import wilcoxon_paired, paired_bootstrap_mae_difference, paired_cohens_d


def test_paired_statistics_identical_predictions():
    y=np.array([1.,2.,3.,4.]); p=y.copy()
    w=wilcoxon_paired(y,p,p)
    assert w["pvalue"]==1.0
    b=paired_bootstrap_mae_difference(y,p,p,n_boot=100,seed=1)
    assert b["mean_difference"]==0.0
    assert paired_cohens_d(y,p,p)==0.0


def test_bootstrap_direction():
    y=np.array([0.,1.,2.,3.,4.])
    better=y+0.1
    worse=y+1.0
    b=paired_bootstrap_mae_difference(y,better,worse,n_boot=200,seed=2)
    assert b["mean_difference"] < 0
