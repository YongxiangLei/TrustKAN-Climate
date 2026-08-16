from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_artifact(
    path, *, model, seed, prediction_offset, time_offset=0, protocol="within_station"
):
    target = np.arange(12, dtype=float).reshape(6, 2)
    times = np.arange(12, dtype="timedelta64[D]").reshape(6, 2)
    times = np.datetime64("2020-01-01") + times + np.timedelta64(time_offset, "D")
    np.savez_compressed(
        path,
        target=target,
        prediction=target + prediction_offset,
        target_time=times,
        target_origin=np.arange(6),
        dataset=np.asarray("CET"),
        model=np.asarray(model),
        horizon=np.asarray(2),
        seed=np.asarray(seed),
        split=np.asarray("test"),
        config_sha256=np.asarray("config"),
        code_sha256=np.asarray("code"),
        protocol=np.asarray(protocol),
    )


def command(root, artifact_a, artifact_b, output):
    return [
        sys.executable,
        str(root / "scripts" / "compare_models.py"),
        "--a",
        str(artifact_a),
        "--b",
        str(artifact_b),
        "--out",
        str(output),
        "--n-boot",
        "100",
        "--block-length",
        "2",
        "--seed",
        "9",
    ]


def test_comparison_script_writes_dependence_aware_provenance(tmp_path):
    root = Path(__file__).resolve().parents[1]
    artifact_a, artifact_b = tmp_path / "a.npz", tmp_path / "b.npz"
    output = tmp_path / "comparison.json"
    write_artifact(artifact_a, model="trustkan", seed=11, prediction_offset=0.1)
    write_artifact(artifact_b, model="transformer", seed=11, prediction_offset=1.0)
    result = subprocess.run(
        command(root, artifact_a, artifact_b, output),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(output.read_text(encoding="utf-8"))
    primary = saved["primary_block_bootstrap"]["mae_difference_a_minus_b"]
    assert primary["mean_difference"] < 0
    assert primary["block_length"] == 2
    assert saved["provenance"]["model_a"] == "trustkan"


def test_comparison_script_rejects_different_target_timestamps(tmp_path):
    root = Path(__file__).resolve().parents[1]
    artifact_a, artifact_b = tmp_path / "a.npz", tmp_path / "b.npz"
    write_artifact(artifact_a, model="a", seed=11, prediction_offset=0.1)
    write_artifact(artifact_b, model="b", seed=11, prediction_offset=0.2, time_offset=1)
    result = subprocess.run(
        command(root, artifact_a, artifact_b, tmp_path / "comparison.json"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "identical target timestamps" in result.stderr


def test_comparison_script_rejects_cross_protocol_pairing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    artifact_a, artifact_b = tmp_path / "a.npz", tmp_path / "b.npz"
    write_artifact(
        artifact_a,model="a",seed=11,prediction_offset=0.1,protocol="within_station"
    )
    write_artifact(
        artifact_b,
        model="b",
        seed=11,
        prediction_offset=0.2,
        protocol="leave_one_region_out",
    )
    result = subprocess.run(
        command(root, artifact_a, artifact_b, tmp_path / "comparison.json"),
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "metadata mismatch for protocol" in result.stderr
