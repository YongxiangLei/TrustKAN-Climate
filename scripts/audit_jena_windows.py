"""Audit Jena hourly window feasibility without fitting a model."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from scripts.run_jena_benchmark import load_prepared_jena
from src.data.provenance import file_sha256
from src.experiments.jena import build_jena_windows


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


def main(args):
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    _, _, artifact, dates, features = load_prepared_jena(cfg)
    counts = []
    for horizon in cfg["window"]["horizons"]:
        bundle = build_jena_windows(
            dates,
            features,
            cfg["split"],
            cfg["window"]["history"],
            int(horizon),
            expected_step=cfg["dataset"].get("frequency", "1h"),
            max_observations=cfg["dataset"].get("max_observations"),
        )
        counts.append(
            {
                "horizon": int(horizon),
                "windows": {
                    split: int(len(arrays[0])) for split, arrays in bundle.sets.items()
                },
                "n_features": int(features.shape[1]),
            }
        )
    frozen_window = {"history": 168, "horizons": [1, 6, 24]}
    protocol_checks = {
        "no_tail_subset": cfg["dataset"].get("max_observations") is None,
        "frozen_window_design": cfg["window"] == frozen_window,
        "four_way_split": cfg["split"]
        == {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15},
        "five_unique_seeds": len(set(cfg.get("training", {}).get("seeds", []))) >= 5,
        "multivariate_inputs": int(features.shape[1]) == 4,
    }
    payload = {
        "experiment": cfg.get("experiment", {}).get("name"),
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": Path(args.config).as_posix(),
        "artifact": artifact.as_posix(),
        "artifact_sha256": file_sha256(artifact),
        "paper_protocol": bool(all(protocol_checks.values())),
        "paper_protocol_checks": protocol_checks,
        "window_counts": counts,
        "n_hourly": int(len(dates)),
    }
    atomic_json(args.out, payload)
    print(json.dumps({"paper_protocol": payload["paper_protocol"], "out": args.out}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jena.yaml")
    parser.add_argument("--out", default="results/dataset_audits/jena_full_windows.json")
    main(parser.parse_args())
