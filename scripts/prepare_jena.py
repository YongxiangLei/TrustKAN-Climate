"""Build a checksum-locked hourly Jena artifact from official 10-minute files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from src.data.jena import (
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    jena_continuity_and_eligibility,
    prepare_jena_hourly,
    prepared_series_from_hourly,
    read_jena_archives,
    verify_source_manifest,
)
from src.data.provenance import file_sha256


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src" / "data" / "jena.py",
        root / "src" / "data" / "provenance.py",
        root / "src" / "data" / "common.py",
    ]
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


def main(args):
    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset = cfg["dataset"]
    period = dataset["period"]
    manifest = pd.read_csv(args.manifest)
    sources = verify_source_manifest(manifest, args.archive_dir)
    raw = read_jena_archives([item["path"] for item in sources])
    hourly = prepare_jena_hourly(
        raw,
        start=period["start"],
        end=period["end"],
        min_valid_slots=dataset["quality_control"]["min_valid_10min_slots_per_hour"],
    )
    audit = jena_continuity_and_eligibility(
        hourly["date"],
        dataset["eligibility"],
        start=period["start"],
        end=period["end"],
    )
    if args.require_eligible and not audit["eligibility"]["eligible"]:
        raise SystemExit(
            "Jena series failed pre-registered eligibility: "
            + ", ".join(audit["eligibility"]["failed_checks"])
        )
    series = prepared_series_from_hourly(hourly)
    out_dir = Path(args.out_dir)
    artifact = out_dir / "jena_hourly.npz"
    atomic_savez(
        artifact,
        dates=series.dates.to_numpy(dtype="datetime64[ns]"),
        features=series.features,
        target=series.target,
        feature_names=np.asarray(series.feature_names),
        target_name=np.asarray(TARGET_COLUMN),
        required_columns=np.asarray(REQUIRED_COLUMNS),
        config_sha256=np.asarray(file_sha256(config_path)),
        code_sha256=np.asarray(code_sha256()),
    )
    hourly_csv = out_dir / "jena_hourly.csv"
    hourly.to_csv(hourly_csv, index=False)
    payload = {
        "dataset": dataset["key"],
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact.as_posix(),
        "hourly_csv": hourly_csv.as_posix(),
        "artifact_sha256": file_sha256(artifact),
        "config_sha256": file_sha256(config_path),
        "code_sha256": code_sha256(),
        "sources": sources,
        "n_hourly": int(len(hourly)),
        "model_performance_used": False,
        **audit,
    }
    atomic_json(args.manifest_out, payload)
    print(json.dumps({"artifact": artifact.as_posix(), "eligible": audit["eligibility"]["eligible"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/jena_frozen.yaml")
    parser.add_argument("--manifest", default="data/manifests/jena_sources.csv")
    parser.add_argument("--archive-dir", default="data/raw/jena")
    parser.add_argument("--out-dir", default="results/prepared/jena_frozen_v1")
    parser.add_argument("--manifest-out", default="results/dataset_audits/jena_prepared.json")
    parser.add_argument("--require-eligible", action="store_true")
    main(parser.parse_args())
