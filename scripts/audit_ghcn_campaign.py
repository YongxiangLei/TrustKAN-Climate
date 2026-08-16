"""Audit progress and artifact integrity for a planned GHCN campaign."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_ghcn_benchmark import (
    code_sha256 as benchmark_code_sha256,
    load_config,
    validate_config,
)
from scripts.run_ghcn_reliability import (
    code_sha256 as reliability_code_sha256,
    validate_reliability_config,
)
from src.data.provenance import file_sha256


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_records(record_dir, config_hash, code_hash, expected, kind):
    records = []
    for path in sorted(Path(record_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        if (
            record.get("config_sha256") != config_hash
            or record.get("code_sha256") != code_hash
        ):
            continue
        record["_record_path"] = path.as_posix()
        records.append(record)
    keys = [
        (
            row.get("dataset"),
            row.get("model"),
            row.get("horizon"),
            row.get("seed"),
            row.get("split"),
        )
        for row in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate current {kind} run keys detected")
    if len(records) > expected:
        raise ValueError(f"Current {kind} records exceed the planned matrix")
    successful = [row for row in records if row.get("status") == "ok"]
    failed = [row for row in records if row.get("status") != "ok"]
    verified = 0
    for row in successful:
        artifact = Path(row["artifact_path"])
        if not artifact.is_file():
            raise ValueError(f"Missing artifact for {row['_record_path']}: {artifact}")
        if file_sha256(artifact) != row.get("artifact_sha256"):
            raise ValueError(f"Artifact checksum mismatch for {artifact}")
        verified += 1
    runtimes = {}
    for row in successful:
        seconds = row.get("train_seconds")
        if seconds is None or not np.isfinite(float(seconds)):
            continue
        runtimes.setdefault(str(row.get("model", kind)), []).append(float(seconds))
    runtime_summary = {
        model: {
            "n": len(values),
            "mean_seconds": float(np.mean(values)),
            "median_seconds": float(np.median(values)),
            "max_seconds": float(np.max(values)),
        }
        for model, values in sorted(runtimes.items())
    }
    complete = len(successful) == expected and not failed
    return {
        "expected_runs": int(expected),
        "current_records": len(records),
        "successful_runs": len(successful),
        "failed_runs": len(failed),
        "verified_artifacts": verified,
        "completion_fraction": len(successful) / expected,
        "complete": complete,
        "observed_train_seconds": float(
            sum(values for group in runtimes.values() for values in group)
        ),
        "runtime_by_model": runtime_summary,
        "failures": [
            {
                "record_path": row["_record_path"],
                "dataset": row.get("dataset"),
                "model": row.get("model"),
                "horizon": row.get("horizon"),
                "seed": row.get("seed"),
                "error": row.get("error"),
            }
            for row in failed
        ],
    }


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
        expected = inputs[name]
        if observed != expected:
            raise ValueError(
                f"Campaign is stale for {name}: {observed} != {expected}"
            )
    matrix = campaign["matrix"]
    benchmark = verify_records(
        Path("results/runs") / benchmark_name,
        inputs["benchmark_combined_config_sha256"],
        inputs["benchmark_code_sha256"],
        matrix["benchmark_atomic_runs"],
        "benchmark",
    )
    reliability = verify_records(
        Path("results/reliability/runs") / reliability_name,
        inputs["reliability_combined_config_sha256"],
        inputs["reliability_code_sha256"],
        matrix["reliability_atomic_runs"],
        "reliability",
    )
    return {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": campaign_path.as_posix(),
        "campaign_sha256": file_sha256(campaign_path),
        "publication_campaign": campaign["publication_campaign"],
        "current_inputs_verified": True,
        "benchmark": benchmark,
        "reliability": reliability,
        "campaign_complete": benchmark["complete"] and reliability["complete"],
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
    print(f"Audit: {args.out}")
    if args.require_complete and not audit["campaign_complete"]:
        raise RuntimeError("Campaign is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign", default="results/campaigns/ghcn_publication/campaign.json"
    )
    parser.add_argument(
        "--out", default="results/campaigns/ghcn_publication/progress.json"
    )
    parser.add_argument("--require-complete", action="store_true")
    main(parser.parse_args())
