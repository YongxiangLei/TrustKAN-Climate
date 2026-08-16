"""NOAA/NCEI GHCN-Daily station loader.

Uses the official NCEI by-station CSV archive. The caller must explicitly choose
station IDs before headline experiments to avoid performance-driven station
selection.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BASE_URL = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/{station}.csv.gz"


def ghcn_station_url(station: str) -> str:
    station = station.strip().upper()
    if not station or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" for c in station):
        raise ValueError("Invalid GHCN station id")
    return BASE_URL.format(station=station)


def load_ghcn_station(station: str, element: str = "TAVG", cache_dir: str | Path = "data/raw/ghcn") -> pd.DataFrame:
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{station.upper()}.csv.gz"
    if not path.exists():
        df = pd.read_csv(ghcn_station_url(station), compression="gzip")
        df.to_csv(path, index=False, compression="gzip")
    else:
        df = pd.read_csv(path, compression="gzip")
    required = {"ID", "DATE", "ELEMENT", "DATA_VALUE"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected GHCN by-station format; missing {sorted(missing)}")
    sub = df[df["ELEMENT"].eq(element)].copy()
    sub["date"] = pd.to_datetime(sub["DATE"], format="%Y%m%d", errors="coerce")
    # GHCN temperature elements are stored in tenths of degrees C.
    factor = 0.1 if element in {"TAVG", "TMAX", "TMIN", "TOBS"} else 1.0
    sub["value"] = pd.to_numeric(sub["DATA_VALUE"], errors="coerce") * factor
    sub = sub[["date", "value"]].dropna().sort_values("date").drop_duplicates("date")
    return sub.reset_index(drop=True)
