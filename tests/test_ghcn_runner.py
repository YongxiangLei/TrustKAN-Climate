from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.run_ghcn_benchmark import resumable_record
from src.data.provenance import file_sha256


def test_resume_requires_exact_artifact_checksum(tmp_path):
    artifact=tmp_path/"prediction.npz"
    artifact.write_bytes(b"prediction-v1")
    record=tmp_path/"run.json"
    record.write_text(
        json.dumps(
            {
                "status":"ok",
                "config_sha256":"config",
                "code_sha256":"code",
                "dataset_sha256":"data",
                "artifact_sha256":file_sha256(artifact),
            }
        ),
        encoding="utf-8",
    )
    assert resumable_record(record,artifact,"config","code","data") is not None
    artifact.write_bytes(b"prediction-v2")
    assert resumable_record(record,artifact,"config","code","data") is None


def test_full_ghcn_config_keeps_publication_panel_and_seed_requirements():
    path=Path(__file__).parents[1]/"configs"/"ghcn.yaml"
    with open(path,"r",encoding="utf-8") as handle:
        config=yaml.safe_load(handle)
    assert "station_regions" not in config["dataset"]
    assert config["protocols"]==["within_station","leave_one_region_out"]
    assert config["window"]=={"history":30,"horizons":[1,7,30]}
    assert len(config["training"]["seeds"])==5
    assert config["training"]["deterministic_algorithms"] is True
    assert config["training"]["deterministic_warn_only"] is False
