"""Create a deterministic, performance-blind GHCN candidate manifest."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

import _bootstrap  # noqa: F401  # repository-root import setup
from src.data.ghcn import (
    INVENTORY_URL,
    download_ghcn_inventory,
    read_ghcn_inventory,
    select_temperature_candidates,
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


def main(config_path, out):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset = cfg["dataset"]
    inventory_path = download_ghcn_inventory()
    inventory = read_ghcn_inventory(inventory_path)
    candidates = select_temperature_candidates(
        inventory,
        cfg["regions"],
        required_start_year=int(dataset["required_start_year"]),
        required_end_year=int(dataset["required_end_year"]),
        candidates_per_region=int(dataset["candidates_per_region"]),
    )
    payload = {
        "dataset": "GHCN-Daily",
        "selection_rule": (
            "Within each fixed geographic stratum, require both TMAX and TMIN coverage over "
            "the configured year bounds, then rank by earliest paired start, latest paired "
            "end and station ID. No model results are used."
        ),
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "inventory": {
            "url": INVENTORY_URL,
            "path": inventory_path.as_posix(),
            "bytes": inventory_path.stat().st_size,
            "sha256": file_sha256(inventory_path),
            "rows": int(len(inventory)),
        },
        "criteria": dataset,
        "regions": cfg["regions"],
        "candidates": candidates.to_dict(orient="records"),
    }
    atomic_json(out, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ghcn_selection.yaml")
    parser.add_argument("--out", default="results/dataset_audits/ghcn_candidates.json")
    args = parser.parse_args()
    main(args.config, args.out)
