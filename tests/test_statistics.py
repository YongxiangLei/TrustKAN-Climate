import numpy as np
import pytest

from src.statistics.paired_tests import (
    adjust_pvalues,
    circular_block_indices,
    paired_block_bootstrap_difference,
    paired_bootstrap_mae_difference,
    paired_cohens_d,
    wilcoxon_paired,
)


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
    assert b["resampling_unit"] == "forecast_origin"
    assert b["method"] == "circular_moving_block_bootstrap"


def test_multi_horizon_block_bootstrap_preserves_origin_unit():
    y=np.arange(12,dtype=float).reshape(6,2)
    better=y+0.1
    worse=y+1.0
    result=paired_block_bootstrap_difference(
        y,better,worse,metric="rmse",n_boot=200,block_length=3,seed=4
    )
    assert result["mean_difference"] < 0
    assert result["n_origins"] == 6
    assert result["block_length"] == 3


def test_default_block_is_not_shorter_than_overlapping_horizon():
    y=np.arange(60,dtype=float).reshape(12,5)
    result=paired_block_bootstrap_difference(y,y+0.1,y+0.2,n_boot=100,seed=8)
    assert result["block_length"] >= 5
    assert result["block_length_rule"].startswith("max(horizon_width")


def test_circular_blocks_have_valid_fixed_length_indices():
    indices=circular_block_indices(7,3,np.random.default_rng(5))
    assert indices.shape == (7,)
    assert np.all((0 <= indices) & (indices < 7))


def test_pvalue_adjustments_match_known_family():
    p=np.array([0.01,0.04,0.03])
    assert np.allclose(adjust_pvalues(p,"holm"),[0.03,0.06,0.06])
    assert np.allclose(adjust_pvalues(p,"bh"),[0.03,0.04,0.04])


def test_block_bootstrap_rejects_invalid_block_length():
    y=np.arange(5,dtype=float)
    with pytest.raises(ValueError,match="cannot exceed"):
        paired_block_bootstrap_difference(y,y,y,n_boot=100,block_length=6)
