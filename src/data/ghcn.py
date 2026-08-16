"""Leakage-neutral NOAA/NCEI GHCN-Daily station preparation."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import pandas as pd


BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz"
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
