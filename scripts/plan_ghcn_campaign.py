"""Plan a resumable, shardable GHCN publication experiment campaign."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import DETERMINISTIC
from scripts.run_ghcn_benchmark import (
    code_sha256 as benchmark_code_sha256,
    combined_config_sha256,
    load_config,
    validate_config,
)
from scripts.run_ghcn_reliability import (
    code_sha256 as reliability_code_sha256,
    validate_reliability_config,
)
from src.data.provenance import file_sha256


REQUIRED_MODELS = {
    "persistence",
    "svr",
    "random_forest",
    "mlp",
    "lstm",
    "gru",
    "tcn",
    "transformer",
    "kan",
    "trustkan",
}
REQUIRED_PROTOCOLS = {"within_station", "leave_one_region_out"}
FROZEN_SPLIT = {
    "train": 0.60,
    "validation": 0.15,
    "calibration": 0.10,
    "test": 0.15,
}


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _selected_regions(config, panel):
    available = [station["region"] for station in panel["stations"]]
    selected = config["dataset"].get("station_regions")
    return available if selected is None else [item for item in available if item in selected]


def publication_checks(benchmark, reliability, panel, benchmark_name, reliability_name):
    regions = _selected_regions(benchmark, panel)
    reliability_regions = _selected_regions(reliability, panel)
    return {
        "non_smoke_run_names": "smoke" not in benchmark_name.lower()
        and "smoke" not in reliability_name.lower(),
        "shared_frozen_panel": benchmark["dataset"]["panel_config"]
        == reliability["dataset"]["panel_config"],
        "all_five_regions": len(regions) == 5 and regions == reliability_regions,
        "no_tail_subset": benchmark["dataset"].get("max_observations") is None
        and reliability["dataset"].get("max_observations") is None,
        "both_protocols": set(benchmark["protocols"]) == REQUIRED_PROTOCOLS
        and set(reliability["protocols"]) == REQUIRED_PROTOCOLS,
        "frozen_windows": benchmark["window"] == {"history": 30, "horizons": [1, 7, 30]}
        and reliability["window"] == {"history": 30, "horizons": [1, 7, 30]},
        "frozen_split": benchmark["split"] == FROZEN_SPLIT
        and reliability["split"] == FROZEN_SPLIT,
        "five_unique_seeds": len(set(benchmark["training"]["seeds"])) >= 5
        and len(set(reliability["training"]["seeds"])) >= 5,
        "aligned_seeds": benchmark["training"]["seeds"]
        == reliability["training"]["seeds"],
        "required_models": REQUIRED_MODELS.issubset(benchmark["models"]),
        "frozen_reliability_policy": reliability["model"]["quantiles"]
        == [0.05, 0.5, 0.95]
        and reliability["conformal"]["alpha"] == 0.10
        and reliability["reliability"]["fusion_weights"] == [0.5, 0.5],
    }


def _powershell(tokens):
    def quote(value):
        value = str(value).replace("'", "''")
        return f"'{value}'"

    return "& " + " ".join(quote(token) for token in tokens)


def _job(job_id, kind, tokens, atomic_runs):
    return {
        "job_id": job_id,
        "kind": kind,
        "atomic_runs": int(atomic_runs),
        "argv": [str(token) for token in tokens],
        "powershell": _powershell(tokens),
    }


def build_campaign(benchmark_path, reliability_path, *, python="python", publication=True):
    benchmark_path = Path(benchmark_path)
    reliability_path = Path(reliability_path)
    benchmark = load_config(benchmark_path)
    reliability = load_config(reliability_path)
    benchmark_name = validate_config(benchmark, benchmark_path)
    reliability_name = validate_reliability_config(reliability, reliability_path)
    panel_path = Path(benchmark["dataset"]["panel_config"])
    reliability_panel_path = Path(reliability["dataset"]["panel_config"])
    if panel_path.resolve() != reliability_panel_path.resolve():
        raise ValueError("Benchmark and reliability configs must use the same frozen panel")
    panel = load_config(panel_path)
    checks = publication_checks(
        benchmark, reliability, panel, benchmark_name, reliability_name
    )
    if publication and not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Publication campaign checks failed: {failed}")
    regions = _selected_regions(benchmark, panel)
    reliability_regions = _selected_regions(reliability, panel)
    benchmark_targets_per_group = len(regions) * len(benchmark["protocols"])
    reliability_targets_per_group = len(reliability_regions) * len(
        reliability["protocols"]
    )

    benchmark_jobs = []
    for horizon in benchmark["window"]["horizons"]:
        for model in benchmark["models"]:
            model_seeds = [-1] if model in DETERMINISTIC else benchmark["training"]["seeds"]
            for seed in model_seeds:
                tokens = [
                    python,
                    "scripts/run_ghcn_benchmark.py",
                    "--config",
                    benchmark_path.as_posix(),
                    "--resume",
                    "--defer-collection",
                    "--horizon",
                    horizon,
                    "--model",
                    model,
                    "--seed",
                    seed,
                ]
                benchmark_jobs.append(
                    _job(
                        f"benchmark_h{horizon}_{model}_s{seed}",
                        "benchmark",
                        tokens,
                        benchmark_targets_per_group,
                    )
                )

    reliability_jobs = []
    for horizon in reliability["window"]["horizons"]:
        for seed in reliability["training"]["seeds"]:
            tokens = [
                python,
                "scripts/run_ghcn_reliability.py",
                "--config",
                reliability_path.as_posix(),
                "--resume",
                "--defer-collection",
                "--horizon",
                horizon,
                "--seed",
                seed,
            ]
            reliability_jobs.append(
                _job(
                    f"reliability_h{horizon}_s{seed}",
                    "reliability",
                    tokens,
                    reliability_targets_per_group,
                )
            )

    job_ids = [
        job["job_id"] for job in [*benchmark_jobs, *reliability_jobs]
    ]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Campaign contains duplicate shard identifiers")
    benchmark_minimum_seeds = (
        5 if publication else len(set(benchmark["training"]["seeds"]))
    )
    reliability_minimum_seeds = (
        5 if publication else len(set(reliability["training"]["seeds"]))
    )
    benchmark_minimum_regions = 5 if publication else len(regions)
    reliability_minimum_regions = 5 if publication else len(reliability_regions)
    collection = [
        _powershell(
            [
                python,
                "scripts/run_ghcn_benchmark.py",
                "--config",
                benchmark_path.as_posix(),
                "--collect-only",
            ]
        ),
        _powershell(
            [
                python,
                "scripts/aggregate_results.py",
                "--input",
                f"results/aggregated/{benchmark_name}_runs.csv",
                "--outdir",
                f"results/tables/{benchmark_name}",
                "--min-seeds",
                benchmark_minimum_seeds,
                "--min-regions",
                benchmark_minimum_regions,
            ]
        ),
        _powershell(
            [
                python,
                "scripts/run_ghcn_reliability.py",
                "--config",
                reliability_path.as_posix(),
                "--collect-only",
            ]
        ),
        _powershell(
            [
                python,
                "scripts/aggregate_reliability.py",
                "--input",
                f"results/reliability/aggregated/{reliability_name}_runs.csv",
                "--outdir",
                f"results/tables/{reliability_name}",
                "--min-seeds",
                reliability_minimum_seeds,
                "--min-regions",
                reliability_minimum_regions,
            ]
        ),
    ]
    benchmark_runs = sum(job["atomic_runs"] for job in benchmark_jobs)
    reliability_runs = sum(job["atomic_runs"] for job in reliability_jobs)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication_campaign": bool(publication and all(checks.values())),
        "publication_checks": checks,
        "inputs": {
            "benchmark_config": benchmark_path.as_posix(),
            "benchmark_config_sha256": file_sha256(benchmark_path),
            "benchmark_combined_config_sha256": combined_config_sha256(
                benchmark_path, panel_path
            ),
            "benchmark_code_sha256": benchmark_code_sha256(),
            "reliability_config": reliability_path.as_posix(),
            "reliability_config_sha256": file_sha256(reliability_path),
            "reliability_combined_config_sha256": combined_config_sha256(
                reliability_path, panel_path
            ),
            "reliability_code_sha256": reliability_code_sha256(),
            "panel": panel_path.as_posix(),
            "panel_sha256": file_sha256(panel_path),
        },
        "matrix": {
            "regions": regions,
            "reliability_regions": reliability_regions,
            "protocols": benchmark["protocols"],
            "horizons": benchmark["window"]["horizons"],
            "benchmark_models": benchmark["models"],
            "seeds": benchmark["training"]["seeds"],
            "benchmark_shards": len(benchmark_jobs),
            "benchmark_atomic_runs": benchmark_runs,
            "reliability_shards": len(reliability_jobs),
            "reliability_atomic_runs": reliability_runs,
            "total_atomic_runs": benchmark_runs + reliability_runs,
        },
        "benchmark_jobs": benchmark_jobs,
        "reliability_jobs": reliability_jobs,
        "collection_commands": collection,
    }


def write_campaign(campaign, outdir):
    outdir = Path(outdir)
    atomic_json(outdir / "campaign.json", campaign)
    commands = [
        "Set-StrictMode -Version Latest",
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for job in [*campaign["benchmark_jobs"], *campaign["reliability_jobs"]]:
        commands.extend(
            [
                f"Write-Host 'START {job['job_id']}'",
                job["powershell"],
                "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }",
            ]
        )
    commands.append("")
    commands.append("Write-Host 'COLLECT AND VALIDATE'")
    for command in campaign["collection_commands"]:
        commands.extend([command, "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"])
    python = campaign["benchmark_jobs"][0]["argv"][0]
    audit_command = _powershell(
        [
            python,
            "scripts/audit_ghcn_campaign.py",
            "--campaign",
            (outdir / "campaign.json").as_posix(),
            "--out",
            (outdir / "progress.json").as_posix(),
            "--require-complete",
        ]
    )
    commands.extend(
        [audit_command, "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"]
    )
    atomic_text(outdir / "run_all.ps1", "\n".join(commands) + "\n")


def main(args):
    campaign = build_campaign(
        args.benchmark_config,
        args.reliability_config,
        python=args.python,
        publication=not args.engineering,
    )
    write_campaign(campaign, args.outdir)
    matrix = campaign["matrix"]
    print(
        f"Planned {matrix['benchmark_shards']} benchmark shards and "
        f"{matrix['reliability_shards']} reliability shards "
        f"({matrix['total_atomic_runs']} atomic runs)."
    )
    print(f"Campaign: {Path(args.outdir) / 'campaign.json'}")
    print(f"Sequential runner: {Path(args.outdir) / 'run_all.ps1'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-config", default="configs/ghcn.yaml")
    parser.add_argument(
        "--reliability-config", default="configs/ghcn_reliability.yaml"
    )
    parser.add_argument("--outdir", default="results/campaigns/ghcn_publication")
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--engineering",
        action="store_true",
        help="Allow non-publication smoke configs while retaining hashes and counts.",
    )
    main(parser.parse_args())
