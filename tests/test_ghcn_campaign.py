from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_ghcn_campaign import verify_records
from scripts.plan_ghcn_campaign import build_campaign
from scripts.run_ghcn_benchmark import collect_current_records
from src.data.provenance import file_sha256


ROOT = Path(__file__).parents[1]


def test_publication_campaign_has_complete_frozen_matrix():
    campaign = build_campaign(
        ROOT / "configs" / "ghcn.yaml",
        ROOT / "configs" / "ghcn_reliability.yaml",
    )
    matrix = campaign["matrix"]
    assert campaign["publication_campaign"] is True
    assert all(campaign["publication_checks"].values())
    assert matrix["benchmark_shards"] == 126
    assert matrix["benchmark_atomic_runs"] == 1260
    assert matrix["reliability_shards"] == 15
    assert matrix["reliability_atomic_runs"] == 150
    assert matrix["total_atomic_runs"] == 1410
    jobs = [*campaign["benchmark_jobs"], *campaign["reliability_jobs"]]
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert all("--defer-collection" in job["argv"] for job in jobs)


def test_smoke_campaign_requires_explicit_engineering_mode():
    benchmark = ROOT / "configs" / "ghcn_smoke.yaml"
    reliability = ROOT / "configs" / "ghcn_reliability_smoke.yaml"
    with pytest.raises(ValueError, match="Publication campaign checks failed"):
        build_campaign(benchmark, reliability)
    campaign = build_campaign(benchmark, reliability, publication=False)
    assert campaign["publication_campaign"] is False
    assert campaign["matrix"]["benchmark_atomic_runs"] == 16
    assert campaign["matrix"]["reliability_atomic_runs"] == 4


def test_record_collection_excludes_stale_config_or_code(tmp_path):
    records = [
        {"name": "current", "config_sha256": "cfg", "code_sha256": "code"},
        {"name": "old-config", "config_sha256": "old", "code_sha256": "code"},
        {"name": "old-code", "config_sha256": "cfg", "code_sha256": "old"},
    ]
    for index, record in enumerate(records):
        (tmp_path / f"{index}.json").write_text(json.dumps(record), encoding="utf-8")
    collected = collect_current_records(tmp_path, "cfg", "code")
    assert [record["name"] for record in collected] == ["current"]


def test_campaign_audit_verifies_current_artifact_and_progress(tmp_path):
    artifact=tmp_path/"artifact.npz"
    artifact.write_bytes(b"immutable artifact")
    record={
        "dataset":"GHCN_test",
        "model":"trustkan",
        "horizon":1,
        "seed":11,
        "split":"test",
        "status":"ok",
        "train_seconds":12.5,
        "config_sha256":"cfg",
        "code_sha256":"code",
        "artifact_path":str(artifact),
        "artifact_sha256":file_sha256(artifact),
    }
    (tmp_path/"record.json").write_text(json.dumps(record),encoding="utf-8")
    progress=verify_records(tmp_path,"cfg","code",2,"benchmark")
    assert progress["successful_runs"]==1
    assert progress["verified_artifacts"]==1
    assert progress["completion_fraction"]==.5
    assert progress["complete"] is False
    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="checksum mismatch"):
        verify_records(tmp_path,"cfg","code",2,"benchmark")
