"""Audit ranked GHCN candidates and freeze the first eligible station per region."""
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
    download_ghcn_archive,
    ghcn_station_url,
    prepare_ghcn_element,
    prepare_ghcn_temperature_pair,
    read_ghcn_archive,
    verify_frozen_station_panel,
)
from src.data.provenance import (
    evaluate_temporal_eligibility,
    file_sha256,
    fixed_window_continuity_summary,
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


def quality_counts(raw, station, start, end):
    counts = {}
    prepared = {}
    for element in ("TMAX", "TMIN"):
        rows = raw[raw["ID"].eq(station) & raw["ELEMENT"].eq(element)].copy()
        dates = pd.to_datetime(rows["DATE"], format="%Y%m%d", errors="coerce")
        period_rows = rows.loc[
            dates.between(pd.Timestamp(start), pd.Timestamp(end))
        ]
        flagged = period_rows["Q_FLAG"].notna() & period_rows["Q_FLAG"].str.strip().ne("")
        clean = prepare_ghcn_element(
            raw, station, element, reject_quality_flags=True, start=start, end=end
        )
        counts[element] = {
            "raw_period_rows": int(len(period_rows)),
            "nonempty_quality_flags_rejected": int(flagged.sum()),
            "retained_period_rows": int(len(clean)),
        }
        prepared[element] = clean[["date", "value"]].rename(
            columns={"value": element.lower()}
        )
    joined = prepared["TMAX"].merge(prepared["TMIN"], on="date", how="inner")
    counts["paired_before_physical_check"] = int(len(joined))
    counts["inverted_extrema_rejected"] = int((joined["tmax"] < joined["tmin"]).sum())
    return counts


def audit_candidate(candidate, start, end, criteria):
    station = candidate["ID"]
    archive = download_ghcn_archive(station)
    raw = read_ghcn_archive(archive)
    quality = quality_counts(raw, station, start, end)
    paired = prepare_ghcn_temperature_pair(raw, station, start=start, end=end)
    if paired.empty:
        continuity = None
        eligibility = {
            "eligible": False,
            "criteria": criteria,
            "checks": {"nonempty_paired_series": False},
            "failed_checks": ["nonempty_paired_series"],
        }
    else:
        continuity = fixed_window_continuity_summary(paired["date"], "daily", start, end)
        eligibility = evaluate_temporal_eligibility(continuity, criteria)
    return {
        "candidate": candidate,
        "source_url": ghcn_station_url(station),
        "raw_archive": {
            "path": archive.as_posix(),
            "bytes": archive.stat().st_size,
            "sha256": file_sha256(archive),
        },
        "quality_control": quality,
        "temporal_continuity": continuity,
        "eligibility": eligibility,
    }


def main(config_path, manifest_path, out, require_complete=False, frozen_config_path=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    dataset = cfg["dataset"]
    start = f'{int(dataset["required_start_year"]):04d}-01-01'
    end = f'{int(dataset["required_end_year"]):04d}-12-31'
    configured_criteria = dataset["eligibility"]
    criteria = {
        "minimum_span_years": configured_criteria["minimum_span_years"],
        "minimum_completeness": configured_criteria["minimum_paired_completeness"],
        "maximum_gap_steps": configured_criteria.get("maximum_gap_steps"),
    }
    candidates = manifest["candidates"]
    audits = []
    selected = []
    for region in cfg["regions"]:
        ranked = sorted(
            (item for item in candidates if item["REGION"] == region["name"]),
            key=lambda item: item["CANDIDATE_RANK"],
        )
        for candidate in ranked:
            audit = audit_candidate(candidate, start, end, criteria)
            audits.append(audit)
            print(
                f'{region["name"]}: rank {candidate["CANDIDATE_RANK"]} '
                f'{candidate["ID"]} eligible={audit["eligibility"]["eligible"]}',
                flush=True,
            )
            if audit["eligibility"]["eligible"]:
                selected.append(
                    {
                        "region": region["name"],
                        "station_id": candidate["ID"],
                        "candidate_rank": candidate["CANDIDATE_RANK"],
                        "latitude": candidate["LATITUDE"],
                        "longitude": candidate["LONGITUDE"],
                        "raw_sha256": audit["raw_archive"]["sha256"],
                        "observations": audit["temporal_continuity"]["observations"],
                        "completeness_fraction": audit["temporal_continuity"][
                            "completeness_fraction"
                        ],
                        "max_gap_steps": audit["temporal_continuity"]["max_gap_steps"],
                    }
                )
                break
    selected_regions = {item["region"] for item in selected}
    missing_regions = [
        region["name"] for region in cfg["regions"] if region["name"] not in selected_regions
    ]
    frozen_verification = None
    if frozen_config_path is not None:
        with open(frozen_config_path, "r", encoding="utf-8") as handle:
            frozen_config = yaml.safe_load(handle)
        frozen_verification = verify_frozen_station_panel(
            frozen_config, manifest["inventory"]["sha256"], selected
        )
    payload = {
        "dataset": "GHCN-Daily",
        "derived_target": dataset["derived_target"],
        "period": {"start": start, "end": end},
        "selection_rule": (
            "Audit candidates in pre-registered rank order and freeze the first station "
            "per region passing paired-data eligibility. No model result is consulted."
        ),
        "candidate_manifest": {
            "path": Path(manifest_path).as_posix(),
            "sha256": file_sha256(manifest_path),
            "inventory_sha256": manifest["inventory"]["sha256"],
        },
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "eligibility_criteria": criteria,
        "selected": selected,
        "missing_regions": missing_regions,
        "frozen_config_verification": frozen_verification,
        "audits": audits,
    }
    atomic_json(out, payload)
    print(json.dumps({"selected": selected, "missing_regions": missing_regions}, indent=2))
    if require_complete and missing_regions:
        raise SystemExit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/ghcn_selection.yaml")
    parser.add_argument(
        "--manifest", default="results/dataset_audits/ghcn_candidates.json"
    )
    parser.add_argument(
        "--out", default="results/dataset_audits/ghcn_frozen_selection.json"
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--frozen-config", default="configs/datasets/ghcn_frozen.yaml"
    )
    args = parser.parse_args()
    main(
        args.config,
        args.manifest,
        args.out,
        args.require_complete,
        args.frozen_config,
    )
