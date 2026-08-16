from __future__ import annotations

import numpy as np

from src.experiments.ghcn import (
    StationSeries,
    build_evaluation_specs,
    build_station_windows,
)


def station(region, offset=0.0):
    dates=np.arange(
        np.datetime64("2000-01-01"),np.datetime64("2001-02-04"),np.timedelta64(1,"D")
    )
    target=np.arange(len(dates),dtype=float)+offset
    return StationSeries(region,region.upper(),dates,target,f"hash-{region}")


def bundle(region, offset=0.0):
    return build_station_windows(
        station(region,offset),
        {"train":0.5,"validation":0.2,"calibration":0.1},
        history=10,
        horizon=2,
    )


def test_station_normalization_is_fit_on_training_period_only():
    built=bundle("north")
    assert built.scaler.scaler.mean_[0] < built.series.target.mean()
    assert built.test_target_times.shape==built.test_target_raw.shape
    assert built.calibration_target_times.shape==built.calibration_target_raw.shape
    assert built.test_origins.min() >= int(len(built.series.target)*0.8)


def test_leave_one_region_out_never_pools_target_training_windows():
    bundles=[bundle("north",0),bundle("south",1000),bundle("tropics",2000)]
    specs=build_evaluation_specs(
        bundles,["within_station","leave_one_region_out"]
    )
    assert len(specs)==6
    held_out=next(
        spec for spec in specs
        if spec.protocol=="leave_one_region_out" and spec.target_region=="north"
    )
    assert "north" not in held_out.source_regions
    assert set(held_out.source_regions)=={"south","tropics"}
    expected=2*min(len(item.sets["train"][0]) for item in bundles[1:])
    assert len(held_out.train_set[0])==expected
    assert held_out.source_pooling=="equal_region_deterministic_subsample"


def test_windows_do_not_cross_missing_calendar_days():
    original=station("north")
    keep=np.arange(len(original.dates))!=350
    gapped=StationSeries(
        original.region,
        original.station_id,
        original.dates[keep],
        original.target[keep],
        original.raw_sha256,
    )
    built=build_station_windows(
        gapped,
        {"train":0.5,"validation":0.2,"calibration":0.1},
        history=10,
        horizon=2,
    )
    assert len(built.test_origins)>0
    for origin in built.test_origins:
        span=gapped.dates[origin-10:origin+2]
        assert np.all(np.diff(span)==np.timedelta64(1,"D"))
