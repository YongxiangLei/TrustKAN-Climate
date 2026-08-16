"""Download, quality-control and fingerprint one pre-registered GHCN station."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

import _bootstrap  # noqa: F401  # repository-root import setup
from src.data.ghcn import (
    MISSING_SENTINEL,
    download_ghcn_archive,
    ghcn_station_url,
    normalize_station_id,
    prepare_ghcn_element,
    read_ghcn_archive,
)
from src.data.provenance import (
    evaluate_temporal_eligibility,
    file_sha256,
    temporal_continuity_summary,
)


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


def main(config_path, out, require_eligible=False):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    dataset = cfg["dataset"]
    station = normalize_station_id(dataset["station_id"])
    element = dataset["element"].upper()
    period_start = dataset.get("start")
    period_end = dataset.get("end")
    period_start_text = period_start.isoformat() if hasattr(period_start, "isoformat") else period_start
    period_end_text = period_end.isoformat() if hasattr(period_end, "isoformat") else period_end
    archive = download_ghcn_archive(station)
    raw = read_ghcn_archive(archive)
    element_rows = raw[raw["ID"].eq(station) & raw["ELEMENT"].eq(element)].copy()
    numeric = pd.to_numeric(element_rows["DATA_VALUE"], errors="coerce")
    dates = pd.to_datetime(element_rows["DATE"], format="%Y%m%d", errors="coerce")
    flagged = element_rows["Q_FLAG"].notna() & element_rows["Q_FLAG"].str.strip().ne("")
    prepared = prepare_ghcn_element(
        raw,
        station,
        element,
        reject_quality_flags=True,
        start=period_start,
        end=period_end,
    )
    continuity = temporal_continuity_summary(prepared["date"], dataset["temporal_resolution"])
    eligibility = evaluate_temporal_eligibility(continuity, dataset["eligibility"])
    result = {
        "dataset": "GHCN-Daily",
        "station_id": station,
        "element": element,
        "unit": dataset.get("target_unit"),
        "selection_policy": dataset.get("selection_policy"),
        "period_filter": {"start": period_start_text, "end": period_end_text},
        "source_url": ghcn_station_url(station),
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_archive": {
            "path": archive.as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
        },
        "quality_control": {
            "element_rows": int(len(element_rows)),
            "invalid_dates": int(dates.isna().sum()),
            "invalid_numeric_values": int(numeric.isna().sum()),
            "missing_sentinel_values": int(numeric.eq(MISSING_SENTINEL).sum()),
            "nonempty_quality_flags_rejected": int(flagged.sum()),
            "retained_period_rows": int(len(prepared)),
        },
        "temporal_continuity": continuity,
        "eligibility": eligibility,
    }
    atomic_json(out, result)
    print(json.dumps(result, indent=2))
    if require_eligible and not eligibility["eligible"]:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ghcn_example.yaml")
    parser.add_argument("--out", default="results/dataset_audits/ghcn_example.json")
    parser.add_argument("--require-eligible", action="store_true")
    args = parser.parse_args()
    main(args.config, args.out, args.require_eligible)
