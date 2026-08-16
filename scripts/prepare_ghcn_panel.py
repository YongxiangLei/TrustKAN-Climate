"""Build checksum-locked model-ready arrays for the frozen GHCN panel."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import _bootstrap  # noqa: F401  # repository-root import setup
from src.data.ghcn import (
    download_ghcn_archive,
    prepare_ghcn_temperature_pair,
    read_ghcn_archive,
)
from src.data.provenance import (
    file_sha256,
    fixed_window_continuity_summary,
    verify_continuity_evidence,
)


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "src" / "data" / "ghcn.py", root / "src" / "data" / "provenance.py"]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def main(config_path, out_dir, manifest_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset = cfg["dataset"]
    start = dataset["period"]["start"]
    end = dataset["period"]["end"]
    config_hash = file_sha256(config_path)
    source_hash = code_sha256()
    artifacts = []
    for expected in cfg["stations"]:
        station = expected["station_id"]
        archive = download_ghcn_archive(station)
        observed_raw_hash = file_sha256(archive)
        if observed_raw_hash != expected["raw_sha256"]:
            raise ValueError(
                f"Raw archive hash mismatch for {station}: expected "
                f'{expected["raw_sha256"]}, got {observed_raw_hash}'
            )
        raw = read_ghcn_archive(archive)
        paired = prepare_ghcn_temperature_pair(raw, station, start=start, end=end)
        continuity = fixed_window_continuity_summary(
            paired["date"], "daily", start, end
        )
        verification = verify_continuity_evidence(continuity, expected)
        artifact = Path(out_dir) / f'{expected["region"]}_{station}.npz'
        atomic_savez(
            artifact,
            date=paired["date"].to_numpy(dtype="datetime64[ns]"),
            tmax=paired["tmax"].to_numpy(dtype=np.float32),
            tmin=paired["tmin"].to_numpy(dtype=np.float32),
            target=paired["target"].to_numpy(dtype=np.float32),
            region=np.asarray(expected["region"]),
            station_id=np.asarray(station),
            raw_sha256=np.asarray(observed_raw_hash),
            frozen_config_sha256=np.asarray(config_hash),
            preparation_code_sha256=np.asarray(source_hash),
        )
        entry = {
            "region": expected["region"],
            "station_id": station,
            "raw_archive_path": archive.as_posix(),
            "raw_sha256": observed_raw_hash,
            "prepared_path": artifact.as_posix(),
            "prepared_sha256": file_sha256(artifact),
            "continuity": continuity,
            "verification": verification,
        }
        artifacts.append(entry)
        print(
            f'{expected["region"]}: {station} -> {artifact} '
            f'({continuity["observations"]} rows)',
            flush=True,
        )
    manifest = {
        "dataset": dataset["key"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_config": Path(config_path).as_posix(),
        "frozen_config_sha256": config_hash,
        "preparation_code_sha256": source_hash,
        "period": {
            "start": start.isoformat() if hasattr(start, "isoformat") else str(start),
            "end": end.isoformat() if hasattr(end, "isoformat") else str(end),
        },
        "target": dataset["target"],
        "artifacts": artifacts,
    }
    atomic_json(manifest_path, manifest)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ghcn_frozen.yaml")
    parser.add_argument("--outdir", default="results/prepared/ghcn_frozen_v1")
    parser.add_argument(
        "--manifest", default="results/dataset_audits/ghcn_prepared_manifest.json"
    )
    args = parser.parse_args()
    main(args.config, args.outdir, args.manifest)
