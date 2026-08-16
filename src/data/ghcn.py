"""Leakage-neutral NOAA/NCEI GHCN-Daily station preparation."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz"
INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-inventory.txt"
GHCN_COLUMNS = [
    "ID",
    "DATE",
    "ELEMENT",
    "DATA_VALUE",
    "M_FLAG",
    "Q_FLAG",
    "S_FLAG",
    "OBS_TIME",
]
TEMPERATURE_ELEMENTS = {"TAVG", "TMAX", "TMIN", "TOBS"}
MISSING_SENTINEL = -9999


def normalize_station_id(station: str) -> str:
    station = station.strip().upper()
    if not station or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for character in station):
        raise ValueError("Invalid GHCN station id")
    return station


def ghcn_station_url(station: str) -> str:
    return BASE_URL.format(station=normalize_station_id(station))


def download_ghcn_archive(station: str, cache_dir: str | Path = "data/raw/ghcn") -> Path:
    station = normalize_station_id(station)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{station}.csv.gz"
    if path.exists():
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.download")
    try:
        urllib.request.urlretrieve(ghcn_station_url(station), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def download_ghcn_inventory(cache_dir: str | Path = "data/raw/ghcn") -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "ghcnd-inventory.txt"
    if path.exists():
        return path
    temporary = path.with_name(f".{path.name}.{os.getpid()}.download")
    try:
        urllib.request.urlretrieve(INVENTORY_URL, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def read_ghcn_inventory(path: str | Path) -> pd.DataFrame:
    """Parse the official fixed-width inventory file."""
    frame = pd.read_fwf(
        path,
        colspecs=[(0, 11), (12, 20), (21, 30), (31, 35), (36, 40), (41, 45)],
        names=["ID", "LATITUDE", "LONGITUDE", "ELEMENT", "FIRST_YEAR", "LAST_YEAR"],
        dtype={"ID": "string", "ELEMENT": "string"},
    )
    for column in ("LATITUDE", "LONGITUDE", "FIRST_YEAR", "LAST_YEAR"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna().reset_index(drop=True)


def select_temperature_candidates(
    inventory: pd.DataFrame,
    regions: list[dict],
    *,
    required_start_year: int,
    required_end_year: int,
    candidates_per_region: int = 5,
) -> pd.DataFrame:
    """Rank TMAX+TMIN stations without using any model performance."""
    if candidates_per_region <= 0:
        raise ValueError("candidates_per_region must be positive")
    required = {"ID", "LATITUDE", "LONGITUDE", "ELEMENT", "FIRST_YEAR", "LAST_YEAR"}
    missing = required - set(inventory.columns)
    if missing:
        raise ValueError(f"Inventory is missing columns: {sorted(missing)}")
    temperature = inventory[inventory["ELEMENT"].isin(["TMAX", "TMIN"])].copy()
    coverage = temperature.pivot_table(
        index=["ID", "LATITUDE", "LONGITUDE"],
        columns="ELEMENT",
        values=["FIRST_YEAR", "LAST_YEAR"],
        aggfunc={"FIRST_YEAR": "min", "LAST_YEAR": "max"},
    )
    coverage.columns = [f"{measure}_{element}" for measure, element in coverage.columns]
    coverage = coverage.reset_index()
    coverage_columns = ("FIRST_YEAR_TMAX", "FIRST_YEAR_TMIN", "LAST_YEAR_TMAX", "LAST_YEAR_TMIN")
    for column in coverage_columns:
        if column not in coverage:
            return pd.DataFrame()
    coverage = coverage.dropna(subset=list(coverage_columns))
    coverage["PAIR_FIRST_YEAR"] = coverage[["FIRST_YEAR_TMAX", "FIRST_YEAR_TMIN"]].max(axis=1)
    coverage["PAIR_LAST_YEAR"] = coverage[["LAST_YEAR_TMAX", "LAST_YEAR_TMIN"]].min(axis=1)
    coverage = coverage[
        (coverage["PAIR_FIRST_YEAR"] <= required_start_year)
        & (coverage["PAIR_LAST_YEAR"] >= required_end_year)
    ]

    selected = []
    for region in regions:
        subset = coverage[
            coverage["LATITUDE"].between(region["latitude_min"], region["latitude_max"])
            & coverage["LONGITUDE"].between(region["longitude_min"], region["longitude_max"])
        ].copy()
        subset = subset.sort_values(
            ["PAIR_FIRST_YEAR", "PAIR_LAST_YEAR", "ID"],
            ascending=[True, False, True],
        ).head(candidates_per_region)
        subset.insert(0, "REGION", region["name"])
        subset.insert(1, "CANDIDATE_RANK", range(1, len(subset) + 1))
        selected.append(subset)
    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def verify_frozen_station_panel(frozen_config: dict, inventory_sha256: str, selected: list[dict]):
    """Fail closed when a reproduced panel differs from its tracked freeze."""
    expected_inventory = frozen_config["dataset"]["inventory_sha256"]
    if inventory_sha256 != expected_inventory:
        raise ValueError(
            f"GHCN inventory hash mismatch: expected {expected_inventory}, got {inventory_sha256}"
        )
    expected = {item["region"]: item for item in frozen_config["stations"]}
    actual = {item["region"]: item for item in selected}
    if set(actual) != set(expected):
        raise ValueError(
            f"Frozen region mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    for region, expected_station in expected.items():
        observed = actual[region]
        for field in ("station_id", "candidate_rank", "raw_sha256"):
            if observed[field] != expected_station[field]:
                raise ValueError(
                    f"Frozen {region} {field} mismatch: expected "
                    f"{expected_station[field]}, got {observed[field]}"
                )
    return {
        "verified": True,
        "inventory_sha256": inventory_sha256,
        "regions": sorted(expected),
    }


def read_ghcn_archive(path: str | Path) -> pd.DataFrame:
    """Read the official headerless eight-column by-station archive."""
    return pd.read_csv(
        path,
        compression="gzip",
        header=None,
        names=GHCN_COLUMNS,
        dtype={
            "ID": "string",
            "DATE": "string",
            "ELEMENT": "string",
            "DATA_VALUE": "string",
            "M_FLAG": "string",
            "Q_FLAG": "string",
            "S_FLAG": "string",
            "OBS_TIME": "string",
        },
        keep_default_na=True,
        na_values=[""],
    )


def prepare_ghcn_element(
    frame: pd.DataFrame,
    station: str,
    element: str = "TAVG",
    *,
    reject_quality_flags: bool = True,
    start=None,
    end=None,
) -> pd.DataFrame:
    station = normalize_station_id(station)
    element = element.strip().upper()
    missing = set(GHCN_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Unexpected GHCN by-station format; missing {sorted(missing)}")
    subset = frame[frame["ID"].eq(station) & frame["ELEMENT"].eq(element)].copy()
    subset["date"] = pd.to_datetime(subset["DATE"], format="%Y%m%d", errors="coerce")
    subset["raw_value"] = pd.to_numeric(subset["DATA_VALUE"], errors="coerce")
    valid = subset["date"].notna() & subset["raw_value"].notna()
    valid &= subset["raw_value"].ne(MISSING_SENTINEL)
    if reject_quality_flags:
        valid &= subset["Q_FLAG"].isna() | subset["Q_FLAG"].str.strip().eq("")
    subset = subset.loc[valid].copy()
    factor = 0.1 if element in TEMPERATURE_ELEMENTS else 1.0
    subset["value"] = subset["raw_value"] * factor
    if start is not None:
        subset = subset[subset["date"] >= pd.Timestamp(start)]
    if end is not None:
        subset = subset[subset["date"] <= pd.Timestamp(end)]
    if subset["date"].duplicated().any():
        raise ValueError(f"Duplicate {element} dates found for station {station}")
    columns = ["date", "value", "M_FLAG", "Q_FLAG", "S_FLAG", "OBS_TIME"]
    return subset[columns].sort_values("date").reset_index(drop=True)


def prepare_ghcn_temperature_pair(
    frame: pd.DataFrame,
    station: str,
    *,
    reject_quality_flags: bool = True,
    reject_inconsistent: bool = True,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Inner-join quality-controlled TMAX/TMIN and derive daily midrange."""
    maximum = prepare_ghcn_element(
        frame,
        station,
        "TMAX",
        reject_quality_flags=reject_quality_flags,
        start=start,
        end=end,
    )[["date", "value"]].rename(columns={"value": "tmax"})
    minimum = prepare_ghcn_element(
        frame,
        station,
        "TMIN",
        reject_quality_flags=reject_quality_flags,
        start=start,
        end=end,
    )[["date", "value"]].rename(columns={"value": "tmin"})
    paired = maximum.merge(minimum, on="date", how="inner", validate="one_to_one")
    if reject_inconsistent:
        paired = paired[paired["tmax"] >= paired["tmin"]].copy()
    paired["target"] = (paired["tmax"] + paired["tmin"]) / 2.0
    return paired.sort_values("date").reset_index(drop=True)


def load_ghcn_station(
    station: str,
    element: str = "TAVG",
    cache_dir: str | Path = "data/raw/ghcn",
    *,
    reject_quality_flags: bool = True,
    start=None,
    end=None,
) -> pd.DataFrame:
    path = download_ghcn_archive(station, cache_dir)
    frame = read_ghcn_archive(path)
    return prepare_ghcn_element(
        frame,
        station,
        element,
        reject_quality_flags=reject_quality_flags,
        start=start,
        end=end,
    )
