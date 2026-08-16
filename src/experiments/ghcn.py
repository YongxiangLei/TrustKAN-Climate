"""Leakage-safe experiment construction for the frozen GHCN station panel."""
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


@dataclass(frozen=True)
class StationSeries:
    region: str
    station_id: str
    dates: np.ndarray
    target: np.ndarray
    raw_sha256: str

    def validate(self):
        dates = np.asarray(self.dates)
        target = np.asarray(self.target)
        if dates.ndim != 1 or target.ndim != 1 or len(dates) != len(target):
            raise ValueError("Station dates and target must be aligned one-dimensional arrays")
        if len(target) < 8 or not np.isfinite(target).all():
            raise ValueError("Station target is too short or contains non-finite values")
        if not np.issubdtype(dates.dtype, np.datetime64):
            raise ValueError("Station dates must use a numpy datetime64 dtype")
        if np.any(np.diff(dates) <= np.timedelta64(0, "ns")):
            raise ValueError("Station dates must be strictly increasing")
        return self


@dataclass
class StationWindows:
    series: StationSeries
    scaler: TrainOnlyStandardizer
    sets: dict[str, tuple[np.ndarray, np.ndarray]]
    test_target_raw: np.ndarray
    test_origins: np.ndarray
    test_target_times: np.ndarray


@dataclass(frozen=True)
class EvaluationSpec:
    protocol: str
    dataset: str
    target_region: str
    target_station: str
    source_regions: tuple[str, ...]
    source_stations: tuple[str, ...]
    source_pooling: str
    train_set: tuple[np.ndarray, np.ndarray]
    validation_set: tuple[np.ndarray, np.ndarray]
    test_set: tuple[np.ndarray, np.ndarray]
    target_scaler: TrainOnlyStandardizer
    test_target_raw: np.ndarray
    test_origins: np.ndarray
    test_target_times: np.ndarray
    raw_hashes: tuple[str, ...]


def build_station_windows(
    series: StationSeries,
    split_fractions: dict,
    history: int,
    horizon: int,
    *,
    expected_step="1D",
    max_observations=None,
) -> StationWindows:
    series.validate()
    dates = np.asarray(series.dates)
    target = np.asarray(series.target, dtype=float)
    if max_observations is not None:
        if not isinstance(max_observations, int) or max_observations <= 0:
            raise ValueError("max_observations must be a positive integer")
        dates = dates[-max_observations:]
        target = target[-max_observations:]
    split = chronological_split(
        len(target),
        split_fractions["train"],
        split_fractions["validation"],
        split_fractions["calibration"],
    )
    scaler = TrainOnlyStandardizer().fit(target[split.train])
    standardized = scaler.transform(target)
    x, y, origins = sliding_windows(
        standardized,
        history,
        horizon,
        timestamps=dates,
        expected_step=pd.to_timedelta(expected_step).to_timedelta64(),
    )
    masks = assign_windows_by_target_origin(origins, split, horizon)
    sets = {name: (x[mask], y[mask]) for name, mask in masks.items()}
    for name in ("train", "val", "test"):
        if len(sets[name][0]) == 0:
            raise ValueError(
                f"No {name} windows for {series.region}/{series.station_id} at horizon {horizon}"
            )
    test_origins = origins[masks["test"]]
    target_times = np.stack(
        [dates[origin : origin + horizon] for origin in test_origins]
    )
    test_target = sets["test"][1]
    shape = test_target.shape
    test_target_raw = scaler.scaler.inverse_transform(
        test_target.reshape(-1, 1)
    ).reshape(shape)
    return StationWindows(
        series=series,
        scaler=scaler,
        sets=sets,
        test_target_raw=test_target_raw,
        test_origins=test_origins,
        test_target_times=target_times,
    )


def _pool_equal_regions(bundles: list[StationWindows], split_name: str):
    if not bundles:
        raise ValueError("At least one source station is required")
    count = min(len(bundle.sets[split_name][0]) for bundle in bundles)
    if count <= 0:
        raise ValueError(f"A source region has no {split_name} windows")
    sampled = []
    for bundle in bundles:
        available = len(bundle.sets[split_name][0])
        indices = np.rint(np.linspace(0, available - 1, count)).astype(int)
        if len(np.unique(indices)) != count:
            raise RuntimeError("Equal-region sampling produced duplicate indices")
        sampled.append(
            (bundle.sets[split_name][0][indices], bundle.sets[split_name][1][indices])
        )
    return (
        np.concatenate([item[0] for item in sampled]),
        np.concatenate([item[1] for item in sampled]),
    )


def build_evaluation_specs(
    bundles: list[StationWindows], protocols: list[str]
) -> list[EvaluationSpec]:
    supported = {"within_station", "leave_one_region_out"}
    unknown = set(protocols) - supported
    if unknown:
        raise ValueError(f"Unknown GHCN evaluation protocols: {sorted(unknown)}")
    if len({bundle.series.region for bundle in bundles}) != len(bundles):
        raise ValueError("The frozen panel must contain one station per region")
    specs = []
    if "within_station" in protocols:
        for bundle in bundles:
            series = bundle.series
            specs.append(
                EvaluationSpec(
                    protocol="within_station",
                    dataset=f"GHCN_within_{series.region}_{series.station_id}",
                    target_region=series.region,
                    target_station=series.station_id,
                    source_regions=(series.region,),
                    source_stations=(series.station_id,),
                    source_pooling="single_station",
                    train_set=bundle.sets["train"],
                    validation_set=bundle.sets["val"],
                    test_set=bundle.sets["test"],
                    target_scaler=bundle.scaler,
                    test_target_raw=bundle.test_target_raw,
                    test_origins=bundle.test_origins,
                    test_target_times=bundle.test_target_times,
                    raw_hashes=(series.raw_sha256,),
                )
            )
    if "leave_one_region_out" in protocols:
        if len(bundles) < 2:
            raise ValueError("leave_one_region_out requires at least two regions")
        for target in bundles:
            sources = [bundle for bundle in bundles if bundle is not target]
            source_regions = tuple(bundle.series.region for bundle in sources)
            source_stations = tuple(bundle.series.station_id for bundle in sources)
            specs.append(
                EvaluationSpec(
                    protocol="leave_one_region_out",
                    dataset=(
                        f"GHCN_transfer_{target.series.region}_{target.series.station_id}"
                    ),
                    target_region=target.series.region,
                    target_station=target.series.station_id,
                    source_regions=source_regions,
                    source_stations=source_stations,
                    source_pooling="equal_region_deterministic_subsample",
                    train_set=_pool_equal_regions(sources, "train"),
                    validation_set=_pool_equal_regions(sources, "val"),
                    test_set=target.sets["test"],
                    target_scaler=target.scaler,
                    test_target_raw=target.test_target_raw,
                    test_origins=target.test_origins,
                    test_target_times=target.test_target_times,
                    raw_hashes=tuple(bundle.series.raw_sha256 for bundle in sources + [target]),
                )
            )
    return specs
