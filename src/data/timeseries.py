"""Leakage-safe utilities for chronological climate forecasting datasets."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TemporalSplit:
    train: slice
    val: slice
    calibration: slice
    test: slice


def chronological_split(n: int, train=0.60, val=0.15, calibration=0.10) -> TemporalSplit:
    """Return contiguous non-overlapping chronological slices.

    The remaining observations form the final test segment.
    """
    if n < 8:
        raise ValueError("Time series is too short for four-way splitting")
    if min(train, val, calibration) <= 0 or train + val + calibration >= 1:
        raise ValueError("Invalid split fractions")
    i1 = int(n * train)
    i2 = i1 + int(n * val)
    i3 = i2 + int(n * calibration)
    if not (0 < i1 < i2 < i3 < n):
        raise ValueError("Split fractions produce an empty chronological segment")
    return TemporalSplit(slice(0, i1), slice(i1, i2), slice(i2, i3), slice(i3, n))


def sliding_windows(values, history: int, horizon: int):
    """Create time-ordered history/target windows without shuffling."""
    a = np.asarray(values, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    if history <= 0 or horizon <= 0 or len(a) < history + horizon:
        raise ValueError("Invalid history/horizon for available series")
    xs, ys, origins = [], [], []
    for origin in range(history, len(a) - horizon + 1):
        xs.append(a[origin-history:origin])
        ys.append(a[origin:origin+horizon, 0])
        origins.append(origin)
    return np.stack(xs), np.stack(ys), np.asarray(origins)


class TrainOnlyStandardizer:
    """Standardize features using training observations only."""
    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, train_values):
        a = np.asarray(train_values)
        if a.ndim == 1:
            a = a[:, None]
        self.scaler.fit(a)
        self._fitted = True
        return self

    def transform(self, values):
        if not self._fitted:
            raise RuntimeError("Standardizer must be fitted on training data first")
        a = np.asarray(values)
        original_shape = a.shape
        if a.ndim == 1:
            a = a[:, None]
        out = self.scaler.transform(a)
        return out.reshape(original_shape)


def assign_windows_by_target_origin(origins, split: TemporalSplit, horizon: int = 1):
    """Assign windows whose complete target lies within one split.

    ``origins`` contains the first forecasted timestamp. Requiring the exclusive
    target end to be at or before the split end prevents a multi-step target from
    crossing a train/validation/calibration/test boundary.
    """
    origins = np.asarray(origins)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    def mask(s: slice):
        return (origins >= s.start) & (origins + horizon <= s.stop)

    return {"train": mask(split.train), "val": mask(split.val), "calibration": mask(split.calibration), "test": mask(split.test)}
