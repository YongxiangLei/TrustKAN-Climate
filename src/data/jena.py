"""Jena Beutenberg weather data loader.

The official MPI-BGC service distributes date-range ZIP archives. To keep this
research repository deterministic and respectful of upstream changes, this
module consumes a locally prepared CSV produced from those official archives.
The preparation manifest should record every source ZIP and checksum.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_jena_prepared(path: str | Path, date_column: str = "Date Time", target_column: str = "T (degC)") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared Jena file not found: {path}. Download official MPI-BGC archives "
            "and record them in data/manifests/jena_sources.csv before preparation."
        )
    df = pd.read_csv(path)
    if date_column not in df or target_column not in df:
        raise ValueError(f"Expected columns {date_column!r} and {target_column!r}")
    out = df.copy()
    out["date"] = pd.to_datetime(out[date_column], dayfirst=True, errors="coerce")
    out["target"] = pd.to_numeric(out[target_column], errors="coerce")
    out = out.dropna(subset=["date", "target"]).sort_values("date").drop_duplicates("date")
    return out.reset_index(drop=True)
