"""Execute campaign shards, judging success by written records not exit codes.

PyTorch's cuDNN RNN teardown can abort the interpreter on Windows (0xC0000409)
*after* a shard has already written a verified `status: ok` record. A fail-fast
runner that trusts the exit code therefore stops a campaign whose experiments
all succeeded. This driver treats a shard as successful only when the run
ledger actually grew, so genuine failures still stop the campaign.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def record_directories(campaign):
    """Locate the ledger directories this campaign writes into."""
    import yaml

    inputs = campaign["inputs"]
    directories = {}
    for kind, key, root in (
        ("benchmark", "benchmark_config", Path("results/runs")),
        ("reliability", "reliability_config", Path("results/reliability/runs")),
    ):
        config_path = inputs.get(key)
        if not config_path:
            continue
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        name = cfg.get("experiment", {}).get("name", Path(config_path).stem)
        directories[kind] = root / name
    return directories


def count_ok(directory, config_hash=None, code_hash=None):
    directory = Path(directory)
    if not directory.is_dir():
        return 0
    total = 0
    for path in directory.glob("*.json"):
        try:
            row = _load(path)
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("status") != "ok":
            continue
        if config_hash is not None and row.get("config_sha256") != config_hash:
            continue
        if code_hash is not None and row.get("code_sha256") != code_hash:
            continue
        total += 1
    return total


def run_jobs(campaign_path, *, accelerator=None, kinds=None, dry_run=False, limit=None):
    campaign_path = Path(campaign_path)
    campaign = _load(campaign_path)
    directories = record_directories(campaign)
    inputs = campaign["inputs"]
    hashes = {
        "benchmark": inputs.get("benchmark_code_sha256"),
        "reliability": inputs.get("reliability_code_sha256"),
    }
    jobs = []
    for kind in ("benchmark", "reliability"):
        if kinds is not None and kind not in kinds:
            continue
        jobs.extend(campaign.get(f"{kind}_jobs", []))
    if accelerator is not None:
        jobs = [job for job in jobs if job["accelerator"] == accelerator]
    if limit is not None:
        jobs = jobs[:limit]
    if not jobs:
        raise SystemExit("No campaign shards matched the selection")

    total = len(jobs)
    completed = 0
    tolerated = []
    failures = []
    started = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        kind = job["kind"]
        directory = directories.get(kind)
        code_hash = hashes.get(kind)
        before = count_ok(directory, code_hash=code_hash)
        argv = [str(token) for token in job["argv"]]
        elapsed = time.perf_counter() - started
        rate = elapsed / max(1, index - 1) if index > 1 else 0.0
        remaining = rate * (total - index + 1)
        print(
            f"[{index}/{total}] {job['job_id']} "
            f"(elapsed {elapsed/60:.1f}m, eta {remaining/60:.1f}m)",
            flush=True,
        )
        if dry_run:
            continue
        result = subprocess.run(argv, check=False)
        after = count_ok(directory, code_hash=code_hash)
        if result.returncode == 0:
            completed += 1
            continue
        if after > before:
            # The shard wrote a verified record and then died during teardown.
            tolerated.append({"job_id": job["job_id"], "returncode": result.returncode})
            completed += 1
            print(
                f"  tolerated teardown crash (exit {result.returncode}); "
                "ledger grew so the shard is complete",
                flush=True,
            )
            continue
        failures.append({"job_id": job["job_id"], "returncode": result.returncode})
        print(f"  FAILED {job['job_id']} exit {result.returncode}", flush=True)
        break

    summary = {
        "campaign": campaign_path.as_posix(),
        "selected_shards": total,
        "completed_shards": completed,
        "tolerated_teardown_crashes": tolerated,
        "failures": failures,
        "wall_seconds": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu"], default=None)
    parser.add_argument("--kind", action="append", dest="kinds", choices=["benchmark", "reliability"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_jobs(
        args.campaign,
        accelerator=args.accelerator,
        kinds=args.kinds,
        dry_run=args.dry_run,
        limit=args.limit,
    )
