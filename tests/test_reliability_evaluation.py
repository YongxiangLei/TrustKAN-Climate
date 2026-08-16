from __future__ import annotations

import copy

import numpy as np
from pathlib import Path
import yaml

from scripts.run_ghcn_reliability import validate_reliability_config
from src.data.timeseries import TrainOnlyStandardizer
from src.reliability.evaluation import evaluate_calibrated_reliability


def predictions(target, seed):
    rng=np.random.default_rng(seed)
    target=np.asarray(target,dtype=float)
    point=target+rng.normal(0,.2,size=target.shape)
    quantiles=np.stack([point-.3,point,point+.3],axis=-1)
    return {
        "target":target,
        "point":point,
        "quantiles":quantiles,
        "embedding":rng.normal(size=(len(target),3)),
    }


def test_test_labels_cannot_change_calibration_radius_or_threshold():
    scaler=TrainOnlyStandardizer().fit(np.arange(20,dtype=float))
    cal=predictions(np.zeros((30,2)),1)
    test=predictions(np.zeros((20,2)),2)
    reference=np.random.default_rng(3).normal(size=(50,3))
    metrics_a,arrays_a,state_a=evaluate_calibrated_reliability(
        cal,test,reference,scaler,quantile_levels=(.05,.5,.95)
    )
    changed=copy.deepcopy(test)
    changed["target"]=changed["target"]+100
    metrics_b,arrays_b,state_b=evaluate_calibrated_reliability(
        cal,changed,reference,scaler,quantile_levels=(.05,.5,.95)
    )
    assert state_a==state_b
    assert np.array_equal(
        arrays_a["marginal_radii_standardized"],
        arrays_b["marginal_radii_standardized"],
    )
    assert metrics_a["point"]["rmse"]!=metrics_b["point"]["rmse"]


def test_reliability_evaluation_reports_marginal_and_simultaneous_intervals():
    scaler=TrainOnlyStandardizer().fit(np.arange(20,dtype=float))
    cal=predictions(np.zeros((30,3)),4)
    test=predictions(np.zeros((20,3)),5)
    reference=np.random.default_rng(6).normal(size=(50,3))
    metrics,arrays,state=evaluate_calibrated_reliability(
        cal,test,reference,scaler,quantile_levels=(.05,.5,.95)
    )
    assert len(metrics["horizonwise_conformal"]["test_horizonwise_coverage"])==3
    assert arrays["fused_reliability"].shape==(20,)
    assert len(state["marginal_radii_standardized"])==3


def test_full_reliability_config_keeps_frozen_publication_requirements():
    path=Path(__file__).parents[1]/"configs"/"ghcn_reliability.yaml"
    with open(path,"r",encoding="utf-8") as handle:
        config=yaml.safe_load(handle)
    assert "station_regions" not in config["dataset"]
    assert config["window"]=={"history":30,"horizons":[1,7,30]}
    assert len(config["training"]["seeds"])==5
    assert config["model"]["quantiles"]==[.05,.5,.95]
    assert config["conformal"]["alpha"]==.1
    assert config["reliability"]["fusion_weights"]==[.5,.5]


def test_reliability_config_rejects_invalid_calibration_policy():
    path=Path(__file__).parents[1]/"configs"/"ghcn_reliability.yaml"
    with open(path,"r",encoding="utf-8") as handle:
        config=yaml.safe_load(handle)
    invalid=copy.deepcopy(config)
    invalid["reliability"]["min_calibration_coverage"]=0
    with np.testing.assert_raises_regex(ValueError,"min_calibration_coverage"):
        validate_reliability_config(invalid,path)
    invalid=copy.deepcopy(config)
    invalid["model"]["quantiles"]=[.1,.5,.95]
    with np.testing.assert_raises_regex(ValueError,"Outer model quantiles"):
        validate_reliability_config(invalid,path)
