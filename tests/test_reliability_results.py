from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_reliability import validate_ledger
from src.data.provenance import file_sha256
from src.metrics.forecast import aurc, mae, rmse, sample_risk_coverage_curve
from src.uncertainty.conformal import mean_interval_score


def artifact_and_row(path, *, overlap=False, stale_curve=False, stale_mask=False):
    target=np.arange(8,dtype=float).reshape(4,2)
    prediction=target+.2
    lower=target-1
    upper=target+1
    reliability=np.array([1.,.8,.6,.4])
    coverage,risk=sample_risk_coverage_curve(target,prediction,reliability)
    if stale_curve:
        risk=risk+1
    mask=np.array([True,True,False,False])
    if stale_mask:
        mask=~mask
    target_time=np.datetime64("2020-02-01")+np.arange(8).reshape(4,2).astype("timedelta64[D]")
    calibration_time=(
        target_time[:3]
        if overlap
        else np.datetime64("2020-01-01")+np.arange(6).reshape(3,2).astype("timedelta64[D]")
    )
    metadata={
        "dataset":np.asarray("GHCN_test"),
        "protocol":np.asarray("within_station"),
        "target_region":np.asarray("region"),
        "target_station":np.asarray("station"),
        "model":np.asarray("trustkan"),
        "horizon":np.asarray(2),
        "seed":np.asarray(11),
        "split":np.asarray("test"),
        "source_regions_json":np.asarray('["region"]'),
        "source_stations_json":np.asarray('["station"]'),
        "source_pooling":np.asarray("single_station"),
        "normalization":np.asarray("per_station_training_period"),
        "config_sha256":np.asarray("config"),
        "code_sha256":np.asarray("code"),
        "dataset_sha256":np.asarray("data"),
    }
    np.savez_compressed(
        path,
        target=target,
        prediction=prediction,
        target_time=target_time,
        target_origin=np.arange(10,14),
        calibration_target=np.zeros((3,2)),
        calibration_target_time=calibration_time,
        calibration_target_origin=np.arange(3),
        test_lower_horizonwise=lower,
        test_upper_horizonwise=upper,
        test_lower_simultaneous=lower,
        test_upper_simultaneous=upper,
        fused_reliability=reliability,
        fused_mask=mask,
        fused_coverage=coverage,
        fused_risk=risk,
        conformal_alpha=np.asarray(.1),
        calibration_state_json=np.asarray(
            json.dumps({"thresholds":{"fused":{"threshold":.8}}})
        ),
        requested_device=np.asarray("cpu"),
        environment_json=np.asarray(json.dumps({"device":"cpu"})),
        **metadata,
    )
    selected_rmse=rmse(target[mask],prediction[mask])
    return {
        "dataset":"GHCN_test",
        "protocol":"within_station",
        "target_region":"region",
        "target_station":"station",
        "source_regions":'["region"]',
        "source_stations":'["station"]',
        "source_pooling":"single_station",
        "normalization":"per_station_training_period",
        "model":"trustkan",
        "horizon":2,
        "seed":11,
        "split":"test",
        "status":"ok",
        "nominal_coverage":.9,
        "n_calibration":3,
        "n_test":4,
        "rmse":rmse(target,prediction),
        "mae":mae(target,prediction),
        "conformal_marginal_coverage":1.0,
        "conformal_joint_coverage":1.0,
        "conformal_mean_width":2.0,
        "conformal_interval_score":mean_interval_score(target,lower,upper,.1),
        "simultaneous_joint_coverage":1.0,
        "simultaneous_mean_width":2.0,
        "simultaneous_interval_score":mean_interval_score(target,lower,upper,.1),
        "fused_aurc":aurc(coverage,risk),
        "fused_selected_coverage":.5,
        "fused_selected_rmse":selected_rmse,
        "config_sha256":"config",
        "code_sha256":"code",
        "dataset_sha256":"data",
        "artifact_sha256":file_sha256(path),
        "artifact_path":str(path),
        "requested_device":"cpu",
        "device":"cpu",
    }


def test_reliability_ledger_recomputes_saved_metrics(tmp_path):
    path=tmp_path/"reliability.npz"
    row=artifact_and_row(path)
    validated=validate_ledger(pd.DataFrame([row]))
    assert len(validated)==1


def test_reliability_ledger_rejects_calibration_test_overlap(tmp_path):
    path=tmp_path/"reliability.npz"
    row=artifact_and_row(path,overlap=True)
    with pytest.raises(ValueError,match="calibration timestamps overlap"):
        validate_ledger(pd.DataFrame([row]))


@pytest.mark.parametrize(
    ("option", "message"),
    [("stale_curve", "stale fused risk curve"), ("stale_mask", "fused mask")],
)
def test_reliability_ledger_rejects_stale_selective_arrays(
    tmp_path, option, message
):
    path=tmp_path/"reliability.npz"
    row=artifact_and_row(path,**{option:True})
    with pytest.raises(ValueError,match=message):
        validate_ledger(pd.DataFrame([row]))
