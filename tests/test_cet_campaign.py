from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from scripts.plan_cet_campaign import build_campaign, write_campaign
from scripts.run_cet_benchmark import validate_execution_filters


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def scratch_dir():
    root = Path(__file__).resolve().parent / "_scratch"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="cet_campaign_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_publication_cet_campaign_has_complete_frozen_matrix():
    campaign = build_campaign(
        ROOT / "configs" / "cet.yaml",
        ROOT / "configs" / "cet_reliability.yaml",
    )
    matrix = campaign["matrix"]
    assert campaign["publication_campaign"] is True
    assert all(campaign["publication_checks"].values())
    assert matrix["benchmark_atomic_runs"] == 168
    assert matrix["reliability_atomic_runs"] == 20
    assert matrix["total_atomic_runs"] == 188
    assert matrix["cpu_shards"] == 28
    assert matrix["gpu_shards"] == 160
    jobs = [*campaign["benchmark_jobs"], *campaign["reliability_jobs"]]
    assert len({job["job_id"] for job in jobs}) == len(jobs)
    assert all("--defer-collection" in job["argv"] for job in jobs)
    assert all(
        any(
            str(token).endswith("run_cet_benchmark.py")
            or str(token).endswith("run_cet_reliability.py")
            for token in job["argv"]
        )
        for job in jobs
    )


def test_smoke_cet_campaign_requires_explicit_engineering_mode():
    with pytest.raises(ValueError, match="Publication campaign checks failed"):
        build_campaign(
            ROOT / "configs" / "cet_smoke.yaml",
            ROOT / "configs" / "cet_reliability_smoke.yaml",
        )
    campaign = build_campaign(
        ROOT / "configs" / "cet_smoke.yaml",
        ROOT / "configs" / "cet_reliability_smoke.yaml",
        publication=False,
    )
    assert campaign["publication_campaign"] is False
    assert campaign["matrix"]["benchmark_atomic_runs"] == 9
    assert campaign["matrix"]["reliability_atomic_runs"] == 1


def test_cet_campaign_writer_separates_cpu_and_gpu_jobs():
    campaign = build_campaign(
        ROOT / "configs" / "cet.yaml",
        ROOT / "configs" / "cet_reliability.yaml",
    )
    with scratch_dir() as path:
        write_campaign(campaign, path)
        cpu_script = (path / "run_cpu.ps1").read_text(encoding="utf-8")
        gpu_script = (path / "run_gpu.ps1").read_text(encoding="utf-8")
        assert "check_gpu_environment.py" not in cpu_script
        assert "run_cet_benchmark.py" in cpu_script
        assert "check_gpu_environment.py" in gpu_script
        assert "run_cet_reliability.py" in gpu_script
        assert (path / "campaign.json").is_file()


def test_cet_execution_filters_reject_unknown_horizon():
    with open(ROOT / "configs" / "cet.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    validate_execution_filters(cfg, horizons=[1], models=["persistence"], seeds=[-1])
    with pytest.raises(ValueError, match="horizon"):
        validate_execution_filters(cfg, horizons=[2])
