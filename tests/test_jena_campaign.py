from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from scripts.plan_jena_campaign import build_campaign, write_campaign
from scripts.run_jena_benchmark import validate_execution_filters
from scripts.run_jena_reliability import (
    validate_execution_filters as validate_reliability_filters,
)
from scripts.run_jena_reliability import validate_reliability_config


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def scratch_dir():
    root = Path(__file__).resolve().parent / "_scratch"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="jena_campaign_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_publication_jena_campaign_has_complete_frozen_matrix():
    campaign = build_campaign(
        ROOT / "configs" / "jena.yaml",
        ROOT / "configs" / "jena_reliability.yaml",
    )
    matrix = campaign["matrix"]
    assert campaign["publication_campaign"] is True
    assert all(campaign["publication_checks"].values())
    assert matrix["benchmark_atomic_runs"] == 126
    assert matrix["reliability_atomic_runs"] == 15
    assert matrix["total_atomic_runs"] == 141
    assert matrix["cpu_shards"] == 21
    assert matrix["gpu_shards"] == 120
    jobs = [*campaign["benchmark_jobs"], *campaign["reliability_jobs"]]
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert all("--defer-collection" in job["argv"] for job in jobs)
    assert all(
        any(
            str(token).endswith("run_jena_benchmark.py")
            or str(token).endswith("run_jena_reliability.py")
            for token in job["argv"]
        )
        for job in jobs
    )


def test_smoke_jena_campaign_requires_explicit_engineering_mode():
    with pytest.raises(ValueError, match="Publication campaign checks failed"):
        build_campaign(
            ROOT / "configs" / "jena_smoke.yaml",
            ROOT / "configs" / "jena_reliability_smoke.yaml",
        )
    campaign = build_campaign(
        ROOT / "configs" / "jena_smoke.yaml",
        ROOT / "configs" / "jena_reliability_smoke.yaml",
        publication=False,
    )
    assert campaign["publication_campaign"] is False
    assert campaign["matrix"]["benchmark_atomic_runs"] == 4
    assert campaign["matrix"]["reliability_atomic_runs"] == 1


def test_jena_campaign_writer_separates_cpu_and_gpu_jobs():
    campaign = build_campaign(
        ROOT / "configs" / "jena.yaml",
        ROOT / "configs" / "jena_reliability.yaml",
    )
    with scratch_dir() as path:
        write_campaign(campaign, path)
        cpu_script = (path / "run_cpu.ps1").read_text(encoding="utf-8")
        gpu_script = (path / "run_gpu.ps1").read_text(encoding="utf-8")
        assert "check_gpu_environment.py" not in cpu_script
        assert "run_jena_benchmark.py" in cpu_script
        assert "check_gpu_environment.py" in gpu_script
        assert "run_jena_reliability.py" in gpu_script
        assert (path / "campaign.json").is_file()


def test_jena_execution_filters_reject_unknown_horizon():
    with open(ROOT / "configs" / "jena.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_execution_filters(cfg, horizons=[1], models=["persistence"], seeds=[-1])
    with pytest.raises(ValueError, match="horizon"):
        validate_execution_filters(cfg, horizons=[2])


def test_jena_reliability_config_keeps_publication_gates():
    path = ROOT / "configs" / "jena_reliability.yaml"
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert validate_reliability_config(cfg, path) == "jena_reliability_full"
    validate_reliability_filters(cfg, horizons=[1], seeds=[11])
    with pytest.raises(ValueError, match="horizon"):
        validate_reliability_filters(cfg, horizons=[12])
