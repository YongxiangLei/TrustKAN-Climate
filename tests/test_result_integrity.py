from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import validate_ledger


def write_artifact(path, *, model="mlp", seed=11, config_sha256="abc", code_sha256="source"):
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
    )


def row(path, *, model="mlp", seed=11, config_sha256="abc", code_sha256="source"):
    return {
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
