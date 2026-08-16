from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import (
    regional_macro_summary,
    validate_ledger,
    validate_region_coverage,
)


def write_artifact(
    path,
    *,
    model="mlp",
    seed=11,
    config_sha256="abc",
    code_sha256="source",
    protocol=None,
):
    target = np.array([[1.0], [2.0]])
    np.savez_compressed(
        path,
        prediction=target + 0.1,
        target=target,
        target_time=np.array(
            [["2020-01-01"], ["2020-01-02"]], dtype="datetime64[ns]"
        ),
        target_origin=np.array([10, 11]),
        dataset=np.asarray("CET"),
        model=np.asarray(model),
        horizon=np.asarray(1),
        seed=np.asarray(seed),
        split=np.asarray("test"),
        config_sha256=np.asarray(config_sha256),
        code_sha256=np.asarray(code_sha256),
        model_selection_json=np.asarray("[]"),
        selected_hyperparameters_json=np.asarray("{}"),
        **({"protocol": np.asarray(protocol)} if protocol is not None else {}),
    )


def row(
    path,
    *,
    model="mlp",
    seed=11,
    config_sha256="abc",
    code_sha256="source",
    protocol=None,
    artifact_sha256=None,
):
    result = {
        "dataset": "CET",
        "model": model,
        "horizon": 1,
        "seed": seed,
        "split": "test",
        "status": "ok",
        "rmse": 0.1,
        "mae": 0.1,
        "parameters": 10,
        "train_seconds": 1.0,
        "inference_ms": 0.01,
        "validation_rmse": np.nan,
        "search_seconds": 0.0,
        "selected_hyperparameters": "{}",
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "artifact_path": str(path),
    }
    if protocol is not None:
        result["protocol"] = protocol
    if artifact_sha256 is not None:
        result["artifact_sha256"] = artifact_sha256
    return result


def test_valid_run_ledger_checks_raw_artifact(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact)
    valid = validate_ledger(pd.DataFrame([row(artifact)]))
    assert len(valid) == 1


def test_ledger_rejects_artifact_metadata_mismatch(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact, config_sha256="from-artifact")
    with pytest.raises(ValueError, match="metadata mismatch for config_sha256"):
        validate_ledger(pd.DataFrame([row(artifact, config_sha256="from-ledger")]))


def test_ledger_rejects_code_fingerprint_mismatch(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact, code_sha256="old-code")
    with pytest.raises(ValueError, match="metadata mismatch for code_sha256"):
        validate_ledger(pd.DataFrame([row(artifact, code_sha256="new-code")]))


def test_ledger_rejects_duplicate_run_keys(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact)
    duplicate = row(artifact)
    with pytest.raises(ValueError, match="Duplicate run keys"):
        validate_ledger(pd.DataFrame([duplicate, duplicate]))


def test_ledger_enforces_minimum_stochastic_seeds(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact)
    with pytest.raises(ValueError, match="required 5 unique seeds"):
        validate_ledger(pd.DataFrame([row(artifact)]), min_seeds=5)


def test_ledger_rejects_cross_protocol_metadata_drift(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact,protocol="within_station")
    with pytest.raises(ValueError,match="metadata mismatch for protocol"):
        validate_ledger(
            pd.DataFrame([row(artifact,protocol="leave_one_region_out")])
        )


def test_ledger_rejects_artifact_checksum_drift(tmp_path):
    artifact = tmp_path / "run.npz"
    write_artifact(artifact)
    with pytest.raises(ValueError,match="Artifact checksum mismatch"):
        validate_ledger(pd.DataFrame([row(artifact,artifact_sha256="0"*64)]))


def test_regional_macro_summary_weights_regions_equally():
    frame=pd.DataFrame(
        [
            {"protocol":"within_station","target_region":"a","model":"m","horizon":1,"seed":11,"rmse":1.0,"mae":0.5},
            {"protocol":"within_station","target_region":"a","model":"m","horizon":1,"seed":22,"rmse":3.0,"mae":1.5},
            {"protocol":"within_station","target_region":"b","model":"m","horizon":1,"seed":11,"rmse":6.0,"mae":4.0},
            {"protocol":"within_station","target_region":"b","model":"m","horizon":1,"seed":22,"rmse":8.0,"mae":6.0},
        ]
    )
    summary=regional_macro_summary(frame)
    assert summary.loc[0,"n_regions"]==2
    assert summary.loc[0,"rmse_macro_mean"]==4.5
    assert summary.loc[0,"mae_macro_mean"]==3.0


def test_region_coverage_gate_rejects_incomplete_model_panel():
    frame=pd.DataFrame(
        [
            {"protocol":"within_station","target_region":"a","model":"m","horizon":1},
            {"protocol":"within_station","target_region":"b","model":"m","horizon":1},
        ]
    )
    with pytest.raises(ValueError,match="required 5 target regions"):
        validate_region_coverage(frame,5)


def test_region_coverage_gate_counts_all_failed_group_as_zero():
    frame=pd.DataFrame(
        [
            {"protocol":"within_station","target_region":"a","model":"failed_model","horizon":1,"status":"failed"},
            {"protocol":"within_station","target_region":"b","model":"working_model","horizon":1,"status":"ok"},
        ]
    )
    with pytest.raises(ValueError,match="failed_model"):
        validate_region_coverage(frame,1)
