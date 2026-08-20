"""Audit progress and artifact integrity for a planned Jena campaign."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from scripts.audit_ghcn_campaign import atomic_json, verify_gpu_preflight, verify_records
from scripts.run_cet_benchmark import NEURAL
from scripts.run_jena_benchmark import load_config, validate_config
from scripts.run_jena_benchmark import code_sha256 as benchmark_code_sha256
from scripts.run_jena_reliability import code_sha256 as reliability_code_sha256
from scripts.run_jena_reliability import validate_reliability_config
from src.data.provenance import file_sha256


def audit_campaign(campaign_path):
    campaign_path = Path(campaign_path)
    with open(campaign_path, "r", encoding="utf-8") as handle:
        campaign = json.load(handle)
    inputs = campaign["inputs"]
    benchmark_config_path = Path(inputs["benchmark_config"])
    reliability_config_path = Path(inputs["reliability_config"])
    benchmark_config = load_config(benchmark_config_path)
    reliability_config = load_config(reliability_config_path)
    benchmark_name = validate_config(benchmark_config, benchmark_config_path)
    reliability_name = validate_reliability_config(
        reliability_config, reliability_config_path
    )
    current = {
        "benchmark_config_sha256": file_sha256(benchmark_config_path),
        "benchmark_code_sha256": benchmark_code_sha256(),
        "reliability_config_sha256": file_sha256(reliability_config_path),
        "reliability_code_sha256": reliability_code_sha256(),
        "panel_sha256": file_sha256(inputs["panel"]),
    }
    for name, observed in current.items():
        if observed != inputs[name]:
            raise ValueError(f"Campaign is stale for {name}: {observed} != {inputs[name]}")
    matrix = campaign["matrix"]
    benchmark = verify_records(
        Path("results/runs") / benchmark_name,
        inputs["benchmark_combined_config_sha256"],
        inputs["benchmark_code_sha256"],
        matrix["benchmark_atomic_runs"],
        "benchmark",
        gpu_models=NEURAL if campaign["publication_campaign"] else (),
    )
    reliability = verify_records(
        Path("results/reliability/runs") / reliability_name,
        inputs["reliability_combined_config_sha256"],
        inputs["reliability_code_sha256"],
        matrix["reliability_atomic_runs"],
        "reliability",
        require_gpu=campaign["publication_campaign"],
    )
    preflight_required = campaign["publication_campaign"] and str(
        campaign["execution"]["gpu_device"]
    ).startswith("cuda")
    gpu_preflight = verify_gpu_preflight(
        campaign_path.parent / "gpu_environment.json",
        campaign["execution"]["gpu_device"],
        required=preflight_required,
    )
    gpu_gate = gpu_preflight["eligible"] if preflight_required else True
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign_path.as_posix(),
        "campaign_sha256": file_sha256(campaign_path),
        "publication_campaign": campaign["publication_campaign"],
        "current_inputs_verified": True,
        "benchmark": benchmark,
        "reliability": reliability,
        "gpu_preflight": gpu_preflight,
        "campaign_complete": bool(
            benchmark["complete"] and reliability["complete"] and gpu_gate
        ),
    }


def main(args):
    audit = audit_campaign(args.campaign)
    atomic_json(args.out, audit)
    benchmark = audit["benchmark"]
    reliability = audit["reliability"]
    print(
        f"Benchmark: {benchmark['successful_runs']}/{benchmark['expected_runs']} ok; "
        f"reliability: {reliability['successful_runs']}/"
        f"{reliability['expected_runs']} ok; complete={audit['campaign_complete']}"
    )
    if args.require_complete and not audit["campaign_complete"]:
        raise RuntimeError("Jena campaign is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", default="results/campaigns/jena_publication/campaign.json"
    )
    parser.add_argument(
        "--out", default="results/campaigns/jena_publication/progress.json"
    )
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args())
