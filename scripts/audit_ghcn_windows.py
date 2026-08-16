"""Audit full GHCN window and transfer-pool feasibility without fitting models."""
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

from scripts.run_ghcn_benchmark import (
    code_sha256,
    combined_config_sha256,
    load_config,
    load_station_panel,
    validate_config,
)
from src.data.provenance import file_sha256
from src.experiments.ghcn import build_evaluation_specs, build_station_windows


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
    cfg = load_config(config_path)
    run_name = validate_config(cfg, config_path)
    panel_path, _, series = load_station_panel(cfg)
    station_counts = []
    evaluation_counts = []
    for horizon in cfg["window"]["horizons"]:
        bundles = [
            build_station_windows(
                item,
                cfg["split"],
                cfg["window"]["history"],
                horizon,
                expected_step=cfg["dataset"].get("frequency", "1D"),
                max_observations=cfg["dataset"].get("max_observations"),
            )
            for item in series
        ]
        for bundle in bundles:
            station_counts.append(
                {
                    "region": bundle.series.region,
                    "station_id": bundle.series.station_id,
                    "horizon": int(horizon),
                    "windows": {
                        split: int(len(arrays[0]))
                        for split, arrays in bundle.sets.items()
                    },
                }
            )
        for spec in build_evaluation_specs(bundles, cfg["protocols"]):
            evaluation_counts.append(
                {
                    "protocol": spec.protocol,
                    "target_region": spec.target_region,
                    "target_station": spec.target_station,
                    "horizon": int(horizon),
                    "source_regions": list(spec.source_regions),
                    "source_pooling": spec.source_pooling,
                    "n_train": int(len(spec.train_set[0])),
                    "n_validation": int(len(spec.validation_set[0])),
                    "n_test": int(len(spec.test_set[0])),
                }
            )
    required_models = {
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
    protocol_checks = {
        "all_five_regions": len(series) == 5,
        "no_tail_subset": cfg["dataset"].get("max_observations") is None,
        "both_evaluation_protocols": set(cfg["protocols"])
        == {"within_station", "leave_one_region_out"},
        "frozen_window_design": cfg["window"]
        == {"history": 30, "horizons": [1, 7, 30]},
        "four_way_split": cfg["split"]
        == {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15},
        "five_unique_seeds": len(set(cfg["training"]["seeds"])) >= 5,
        "required_baselines": required_models.issubset(cfg["models"]),
    }
    payload = {
        "experiment": run_name,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": Path(config_path).as_posix(),
        "panel_path": panel_path.as_posix(),
        "combined_config_sha256": combined_config_sha256(config_path, panel_path),
        "panel_sha256": file_sha256(panel_path),
        "benchmark_code_sha256": code_sha256(),
        "paper_protocol": bool(all(protocol_checks.values())),
        "paper_protocol_checks": protocol_checks,
        "station_window_counts": station_counts,
        "evaluation_counts": evaluation_counts,
    }
    atomic_json(out, payload)
    print(
        f"Audited {len(series)} stations, {len(cfg['window']['horizons'])} horizons, "
        f"and {len(evaluation_counts)} evaluation targets; paper_protocol="
        f"{payload['paper_protocol']}"
    )
    print(f"Audit: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ghcn.yaml")
    parser.add_argument(
        "--out", default="results/dataset_audits/ghcn_full_windows.json"
    )
    args = parser.parse_args()
    main(args.config, args.out)
