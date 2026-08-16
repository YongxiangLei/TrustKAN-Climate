"""ERA5 prepared time-series interface.

ERA5 access is handled outside model code because CDS requests require user
credentials and can be large. This module defines a deterministic request
payload helper plus a prepared CSV loader for downstream experiments.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def era5_timeseries_request(variable: str, year: int, month: int, latitude: float, longitude: float) -> dict:
    """Return a CDS request payload for one point/month from ERA5 time-series.

    The exact CDS dataset name/API contract should be checked against the current
    CDS page before running downloads because service schemas can change.
    """
    return {
        "variable": [variable],
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "location": {"latitude": float(latitude), "longitude": float(longitude)},
        "data_format": "csv",
    }


def load_era5_prepared(path: str | Path, time_column: str = "valid_time", target_column: str = "2m_temperature") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prepared ERA5 file not found: {path}")
    df = pd.read_csv(path)
    if time_column not in df or target_column not in df:
        raise ValueError(f"Expected ERA5 columns {time_column!r} and {target_column!r}")
    out = df.copy()
    out["date"] = pd.to_datetime(out[time_column], errors="coerce", utc=True)
    out["target"] = pd.to_numeric(out[target_column], errors="coerce")
    # ERA5 2m temperature is commonly returned in Kelvin; convert only when the
    # data clearly appear to be Kelvin, leaving other variables untouched.
    if target_column == "2m_temperature" and out["target"].median(skipna=True) > 150:
        out["target"] = out["target"] - 273.15
    return out.dropna(subset=["date", "target"]).sort_values("date").drop_duplicates("date").reset_index(drop=True)
