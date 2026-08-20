from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.jena import (
    MISSING_SENTINEL,
    REQUIRED_COLUMNS,
    aggregate_hourly,
    jena_continuity_and_eligibility,
    prepare_jena_hourly,
    quality_control_10min,
    read_jena_csv,
    verify_source_manifest,
)
from src.data.provenance import file_sha256
from src.experiments.jena import build_jena_windows
from src.reliability.evaluation import inverse_standardized


@contextmanager
def scratch_dir():
    root = Path(__file__).resolve().parent / "_scratch"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="jena_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _ten_minute_frame(hours=6, start="2010-01-01 00:00"):
    stamps = pd.date_range(start, periods=hours * 6, freq="10min")
    frame = pd.DataFrame(
        {
            "date": stamps,
            "T (degC)": 10.0 + np.linspace(0, 1, len(stamps)),
            "p (mbar)": 1010.0,
            "rh (%)": 70.0,
            "wv (m/s)": 2.0,
        }
    )
    return frame


def test_quality_control_rejects_sentinel_and_impossible_wind():
    frame = _ten_minute_frame(1)
    frame.loc[0, "T (degC)"] = MISSING_SENTINEL
    frame.loc[1, "wv (m/s)"] = -9999
    frame.loc[2, "rh (%)"] = 140
    cleaned = quality_control_10min(frame)
    assert pd.isna(cleaned.loc[0, "T (degC)"])
    assert pd.isna(cleaned.loc[1, "wv (m/s)"])
    assert pd.isna(cleaned.loc[2, "rh (%)"])
    assert cleaned.loc[3, "T (degC)"] == pytest.approx(frame.loc[3, "T (degC)"])


def test_hourly_aggregation_requires_enough_valid_slots_and_does_not_fill_gaps():
    frame = _ten_minute_frame(3)
    frame.loc[0:4, "T (degC)"] = np.nan
    hourly = aggregate_hourly(frame, min_valid_slots=4)
    hours = pd.to_datetime(hourly["date"])
    assert pd.Timestamp("2010-01-01 00:00") not in set(hours)
    assert pd.Timestamp("2010-01-01 01:00") in set(hours)
    gap = frame.iloc[6:12].copy()
    gap["date"] = pd.date_range("2010-01-01 03:00", periods=6, freq="10min")
    with_gap = pd.concat([frame.iloc[:6], gap], ignore_index=True)
    cleaned = quality_control_10min(with_gap)
    hourly_gap = aggregate_hourly(cleaned)
    assert pd.Timestamp("2010-01-01 02:00") not in set(pd.to_datetime(hourly_gap["date"]))


def test_prepare_jena_hourly_filters_frozen_period():
    inside = _ten_minute_frame(2, start="2012-06-01 00:00")
    outside = _ten_minute_frame(1, start="2009-12-31 22:00")
    raw = pd.concat([outside, inside], ignore_index=True)
    hourly = prepare_jena_hourly(raw, start="2010-01-01", end="2020-12-31 23:00")
    assert pd.to_datetime(hourly["date"]).min() >= pd.Timestamp("2010-01-01")
    assert set(hourly.columns) == {"date", *REQUIRED_COLUMNS}


def test_read_jena_csv_uses_dayfirst_and_required_schema():
    with scratch_dir() as path:
        csv_path = path / "jena.csv"
        csv_path.write_text(
            "Date Time,p (mbar),T (degC),rh (%),wv (m/s)\n"
            "01.01.2010 00:10:00,1012.0,1.5,80.0,1.2\n",
            encoding="utf-8",
        )
        frame = read_jena_csv(csv_path)
        assert frame.loc[0, "date"] == pd.Timestamp("2010-01-01 00:10:00")
        assert frame.loc[0, "T (degC)"] == 1.5


def test_source_manifest_fails_closed_on_checksum():
    with scratch_dir() as path:
        archive = path / "raw.csv"
        archive.write_text("ok", encoding="utf-8")
        manifest = pd.DataFrame([{"path": archive.name, "sha256": "0" * 64}])
        with pytest.raises(ValueError, match="checksum mismatch"):
            verify_source_manifest(manifest, path)
        manifest = pd.DataFrame([{"path": archive.name, "sha256": file_sha256(archive)}])
        records = verify_source_manifest(manifest, path)
        assert records[0]["bytes"] == 2


def test_jena_windows_are_multivariate_and_leakage_safe():
    hours = pd.date_range("2010-01-01", periods=400, freq="h")
    features = np.column_stack(
        [
            np.linspace(-2, 8, 400),
            np.full(400, 1010.0),
            np.full(400, 70.0),
            np.full(400, 2.0),
        ]
    )
    split = {"train": 0.60, "validation": 0.15, "calibration": 0.10, "test": 0.15}
    bundle = build_jena_windows(hours, features, split, history=24, horizon=6, expected_step="1h")
    assert bundle.sets["train"][0].shape[-1] == 4
    train_end = int(400 * 0.60)
    assert bundle.scaler.scaler.mean_[0] == pytest.approx(features[:train_end, 0].mean())
    assert np.allclose(bundle.train_target_raw, features[:train_end, 0])
    assert np.all(bundle.test_origins + 6 <= 400)
    raw = bundle.scaler.inverse_column(bundle.sets["test"][1], 0)
    assert np.allclose(raw, bundle.test_target_raw)
    inverted = inverse_standardized(bundle.scaler, bundle.sets["calibration"][1])
    assert np.allclose(inverted, bundle.calibration_target_raw)


def test_frozen_jena_config_matches_protocol():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs" / "datasets" / "jena_frozen.yaml", encoding="utf-8") as handle:
        frozen = yaml.safe_load(handle)
    assert frozen["dataset"]["period"]["start"].startswith("2010-01-01")
    assert frozen["window"]["horizons"] == [1, 6, 24]
    assert frozen["dataset"]["selection"]["model_performance_used"] is False
    eligibility = jena_continuity_and_eligibility(
        pd.date_range("2010-01-01", periods=48, freq="h"),
        frozen["dataset"]["eligibility"],
        start="2010-01-01",
        end="2020-12-31 23:00",
    )
    assert not eligibility["eligibility"]["eligible"]
    assert "minimum_completeness" in eligibility["eligibility"]["failed_checks"]
