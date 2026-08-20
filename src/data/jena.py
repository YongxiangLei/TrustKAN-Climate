"""Leakage-neutral MPI-BGC Jena Beutenberg preparation.

Official 10-minute station archives are quality-controlled and aggregated to
hourly means before any model sees the series. Derived thermodynamic variables
are excluded so the multivariate inputs remain physically distinct observables.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.common import PreparedSeries
from src.data.provenance import (
    evaluate_temporal_eligibility,
    file_sha256,
    fixed_window_continuity_summary,
    temporal_continuity_summary,
)


MISSING_SENTINEL = -9999.0
DATE_COLUMN = "Date Time"
TARGET_COLUMN = "T (degC)"
REQUIRED_COLUMNS = (TARGET_COLUMN, "p (mbar)", "rh (%)", "wv (m/s)")
PHYSICAL_RANGES = {
    TARGET_COLUMN: (-40.0, 45.0),
    "p (mbar)": (900.0, 1100.0),
    "rh (%)": (0.0, 100.0),
    "wv (m/s)": (0.0, 60.0),
}
DEFAULT_MIN_VALID_SLOTS = 4


def official_column_map():
    return {
        "date": DATE_COLUMN,
        "target": TARGET_COLUMN,
        "required": list(REQUIRED_COLUMNS),
    }


def read_jena_csv(path: str | Path, date_column: str = DATE_COLUMN) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Jena archive not found: {path}. Download official MPI-BGC ZIP/CSV "
            "files and record them in the source manifest before preparation."
        )
    frame = pd.read_csv(path)
    if date_column not in frame.columns:
        raise ValueError(f"Expected Jena timestamp column {date_column!r}")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Jena archive is missing required columns: {missing}")
    out = frame[[date_column, *REQUIRED_COLUMNS]].copy()
    out["date"] = pd.to_datetime(out[date_column], dayfirst=True, errors="coerce")
    return out.drop(columns=[date_column])


def read_jena_archives(paths) -> pd.DataFrame:
    frames = [read_jena_csv(path) for path in paths]
    if not frames:
        raise ValueError("At least one Jena archive path is required")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["date"]).sort_values("date")
    combined = combined.drop_duplicates("date", keep="first").reset_index(drop=True)
    return combined


def quality_control_10min(frame: pd.DataFrame, columns=REQUIRED_COLUMNS) -> pd.DataFrame:
    """Replace sentinels and physically impossible 10-minute values with NA."""
    out = frame.copy()
    if "date" not in out.columns:
        raise ValueError("quality_control_10min requires a date column")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for column in columns:
        if column not in out.columns:
            raise ValueError(f"Missing Jena column {column!r}")
        values = pd.to_numeric(out[column], errors="coerce")
        values = values.mask(values == MISSING_SENTINEL)
        low, high = PHYSICAL_RANGES[column]
        values = values.where(values.between(low, high))
        out[column] = values
    return out.dropna(subset=["date"])


def aggregate_hourly(
    frame: pd.DataFrame,
    columns=REQUIRED_COLUMNS,
    min_valid_slots: int = DEFAULT_MIN_VALID_SLOTS,
) -> pd.DataFrame:
    """Mean-aggregate valid 10-minute samples; do not interpolate missing hours."""
    if min_valid_slots <= 0:
        raise ValueError("min_valid_slots must be positive")
    work = frame.dropna(subset=["date"]).copy()
    work["hour"] = work["date"].dt.floor("h")
    if work.empty:
        return pd.DataFrame(columns=["date", *columns])
    grouped = work.set_index("hour")[list(columns)]
    counts = grouped.groupby(level=0).count()
    means = grouped.groupby(level=0).mean()
    valid = (counts >= min_valid_slots).all(axis=1)
    out = means.loc[valid].reset_index().rename(columns={"hour": "date"})
    return out.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def filter_period(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    dates = pd.to_datetime(frame["date"])
    mask = (dates >= start) & (dates <= end)
    return frame.loc[mask].reset_index(drop=True)


def prepare_jena_hourly(
    frame: pd.DataFrame,
    *,
    start,
    end,
    min_valid_slots: int = DEFAULT_MIN_VALID_SLOTS,
    columns=REQUIRED_COLUMNS,
) -> pd.DataFrame:
    cleaned = quality_control_10min(frame, columns)
    hourly = aggregate_hourly(cleaned, columns, min_valid_slots=min_valid_slots)
    return filter_period(hourly, start, end)


def jena_continuity_and_eligibility(dates, criteria: dict, start=None, end=None) -> dict:
    if start is None or end is None:
        continuity = temporal_continuity_summary(dates, "hourly")
    else:
        continuity = fixed_window_continuity_summary(dates, "hourly", start, end)
    eligibility = evaluate_temporal_eligibility(continuity, criteria)
    return {"continuity": continuity, "eligibility": eligibility}


def prepared_series_from_hourly(frame: pd.DataFrame, name: str = "jena") -> PreparedSeries:
    if frame.empty:
        raise ValueError("Hourly Jena frame is empty after quality control")
    features = frame.loc[:, list(REQUIRED_COLUMNS)].to_numpy(float)
    target = frame[TARGET_COLUMN].to_numpy(float)
    dates = pd.DatetimeIndex(pd.to_datetime(frame["date"]))
    return PreparedSeries(
        name,
        dates,
        features,
        target,
        list(REQUIRED_COLUMNS),
        TARGET_COLUMN,
    ).validate()


def verify_source_manifest(manifest: pd.DataFrame, archive_dir: str | Path) -> list[dict]:
    required = {"path", "sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Jena source manifest is missing {sorted(missing)}")
    archive_dir = Path(archive_dir)
    records = []
    for row in manifest.itertuples(index=False):
        path = Path(row.path)
        if not path.is_absolute():
            path = archive_dir / path
        observed = file_sha256(path)
        expected = str(row.sha256).lower()
        if observed != expected:
            raise ValueError(f"Jena source checksum mismatch for {path}: {observed} != {expected}")
        records.append({"path": path.as_posix(), "sha256": observed, "bytes": path.stat().st_size})
    return records
