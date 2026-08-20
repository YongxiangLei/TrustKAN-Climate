"""Quality-control and eligibility-audit one official Jena archive set."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from src.data.jena import (
    REQUIRED_COLUMNS,
    jena_continuity_and_eligibility,
    prepare_jena_hourly,
    read_jena_archives,
    verify_source_manifest,
)
from src.data.provenance import file_sha256


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_paths(args):
    if args.manifest:
        manifest = pd.read_csv(args.manifest)
        records = verify_source_manifest(manifest, args.archive_dir)
        return [item["path"] for item in records], records
    if not args.input:
        raise ValueError("Provide --input or --manifest")
    path = Path(args.input)
    return [path.as_posix()], [
        {"path": path.as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}
    ]


def main(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)["dataset"]
    period = cfg["period"]
    paths, sources = load_paths(args)
    raw = read_jena_archives(paths)
    hourly = prepare_jena_hourly(
        raw,
        start=period["start"],
        end=period["end"],
        min_valid_slots=cfg["quality_control"]["min_valid_10min_slots_per_hour"],
    )
    audit = jena_continuity_and_eligibility(
        hourly["date"],
        cfg["eligibility"],
        start=period["start"],
        end=period["end"],
    )
    payload = {
        "dataset": "jena_beutenberg",
        "protocol": cfg["key"],
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "raw_10min_rows": int(len(raw)),
        "hourly_rows": int(len(hourly)),
        "required_columns": list(REQUIRED_COLUMNS),
        "period": period,
        "model_performance_used": False,
        **audit,
    }
    if args.require_eligible and not audit["eligibility"]["eligible"]:
        raise SystemExit(
            "Jena series failed pre-registered eligibility: "
            + ", ".join(audit["eligibility"]["failed_checks"])
        )
    atomic_json(args.out, payload)
    print(json.dumps({"eligible": audit["eligibility"]["eligible"], "out": args.out}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/jena_frozen.yaml")
    parser.add_argument("--input", default=None, help="Single official 10-minute CSV")
    parser.add_argument("--manifest", default=None, help="CSV with path,sha256")
    parser.add_argument("--archive-dir", default="data/raw/jena")
    parser.add_argument("--out", default="results/dataset_audits/jena_frozen.json")
    parser.add_argument("--require-eligible", action="store_true")
    main(parser.parse_args())
