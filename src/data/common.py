"""Common dataset contract for multi-dataset forecasting experiments."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class PreparedSeries:
    name: str
    dates: pd.DatetimeIndex
    features: np.ndarray
    target: np.ndarray
    feature_names: list[str]
    target_name: str

    def validate(self):
        n=len(self.target)
        if self.features.ndim != 2 or len(self.features) != n or len(self.dates) != n:
            raise ValueError("PreparedSeries arrays must have aligned first dimension")
        if not self.dates.is_monotonic_increasing:
            raise ValueError("Dates must be chronologically sorted")
        if self.dates.has_duplicates:
            raise ValueError("Duplicate timestamps are not allowed")
        if not np.isfinite(self.features).all() or not np.isfinite(self.target).all():
            raise ValueError("PreparedSeries contains NaN/Inf after preparation")
        return self


def from_dataframe(name: str, df: pd.DataFrame, date_col: str, feature_cols: list[str], target_col: str) -> PreparedSeries:
    cols=[date_col,*feature_cols,target_col]
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    work=df[cols].copy(); work[date_col]=pd.to_datetime(work[date_col],errors="coerce",utc=True)
    for c in [*feature_cols,target_col]: work[c]=pd.to_numeric(work[c],errors="coerce")
    work=work.dropna().sort_values(date_col).drop_duplicates(date_col)
    return PreparedSeries(name,pd.DatetimeIndex(work[date_col]),work[feature_cols].to_numpy(float),work[target_col].to_numpy(float),feature_cols,target_col).validate()
