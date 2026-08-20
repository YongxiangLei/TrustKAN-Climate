"""Leakage-safe window construction for the frozen hourly Jena series."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)


@dataclass
class JenaWindows:
    dates: np.ndarray
    features_raw: np.ndarray
    target_raw: np.ndarray
    scaler: TrainOnlyStandardizer
    sets: dict[str, tuple[np.ndarray, np.ndarray]]
    train_target_raw: np.ndarray
    calibration_target_raw: np.ndarray
    calibration_origins: np.ndarray
    calibration_target_times: np.ndarray
    test_target_raw: np.ndarray
    test_origins: np.ndarray
    test_target_times: np.ndarray


def maybe_tail(dates, features, max_observations):
    dates = np.asarray(dates)
    features = np.asarray(features, dtype=float)
    if max_observations is None:
        return dates, features
    if not isinstance(max_observations, int) or max_observations <= 0:
        raise ValueError("max_observations must be a positive integer")
    return dates[-max_observations:], features[-max_observations:]


def build_jena_windows(
    dates,
    features,
    split_fractions: dict,
    history: int,
    horizon: int,
    *,
    expected_step="1h",
    max_observations=None,
) -> JenaWindows:
    dates, features = maybe_tail(dates, features, max_observations)
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError("Jena features must be a two-dimensional array")
    if len(dates) != len(features):
        raise ValueError("Jena dates and features must be aligned")
    if not np.isfinite(features).all():
        raise ValueError("Jena features contain non-finite values")
    split = chronological_split(
        len(features),
        split_fractions["train"],
        split_fractions["validation"],
        split_fractions["calibration"],
    )
    scaler = TrainOnlyStandardizer().fit(features[split.train])
    standardized = scaler.transform(features)
    step = pd.to_timedelta(expected_step).to_timedelta64()
    x, y, origins = sliding_windows(
        standardized,
        history,
        horizon,
        timestamps=dates,
        expected_step=step,
    )
    masks = assign_windows_by_target_origin(origins, split, horizon)
    sets = {name: (x[mask], y[mask]) for name, mask in masks.items()}
    for name, arrays in sets.items():
        if len(arrays[0]) == 0:
            raise ValueError(f"Jena split {name!r} has no valid windows at horizon {horizon}")

    def raw_targets(mask):
        selected = origins[mask]
        standardized_y = y[mask]
        return scaler.inverse_column(standardized_y, 0), selected, np.stack(
            [dates[origin : origin + horizon] for origin in selected]
        )

    cal_y, cal_origins, cal_times = raw_targets(masks["calibration"])
    test_y, test_origins, test_times = raw_targets(masks["test"])
    return JenaWindows(
        dates=np.asarray(dates),
        features_raw=features,
        target_raw=features[:, 0],
        scaler=scaler,
        sets=sets,
        train_target_raw=features[split.train, 0],
        calibration_target_raw=cal_y,
        calibration_origins=cal_origins,
        calibration_target_times=cal_times,
        test_target_raw=test_y,
        test_origins=test_origins,
        test_target_times=test_times,
    )
