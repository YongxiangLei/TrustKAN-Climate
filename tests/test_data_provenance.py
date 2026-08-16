from __future__ import annotations

import pandas as pd
import pytest

from src.data.provenance import (
    evaluate_temporal_eligibility,
    fixed_window_continuity_summary,
    temporal_continuity_summary,
)


def test_temporal_summary_quantifies_daily_gap():
    dates=pd.to_datetime(["2020-01-01","2020-01-02","2020-01-04"])
    summary=temporal_continuity_summary(dates,"1D")
    assert summary["observations"]==3
    assert summary["expected_steps_in_span"]==4
    assert summary["estimated_missing_steps"]==1
    assert summary["non_regular_intervals"]==1
    assert summary["max_gap_steps"]==2.0
    assert summary["completeness_fraction"]==0.75


def test_fixed_window_summary_counts_missing_edges():
    dates=pd.to_datetime(["2020-01-02","2020-01-03"])
    summary=fixed_window_continuity_summary(
        dates,"daily","2020-01-01","2020-01-04"
    )
    assert summary["expected_steps_in_required_window"]==4
    assert summary["estimated_missing_steps_in_required_window"]==2
    assert summary["completeness_fraction"]==0.5


def test_temporal_summary_accepts_semantic_frequency_alias():
    dates=pd.to_datetime(["2020-01-01","2020-01-02"])
    summary=temporal_continuity_summary(dates,"daily")
    assert summary["non_regular_intervals"]==0


def test_temporal_eligibility_reports_failed_pre_registered_check():
    summary=temporal_continuity_summary(
        pd.to_datetime(["2000-01-01","2000-01-02","2000-01-03"]),"daily"
    )
    eligibility=evaluate_temporal_eligibility(
        summary,
        {"minimum_span_years":30,"minimum_completeness":0.95,"maximum_gap_steps":31},
    )
    assert not eligibility["eligible"]
    assert eligibility["failed_checks"]==["minimum_span_years"]


def test_temporal_summary_rejects_unsorted_or_duplicate_dates():
    with pytest.raises(ValueError,match="sorted and unique"):
        temporal_continuity_summary(pd.to_datetime(["2020-01-02","2020-01-01"]),"1D")
    with pytest.raises(ValueError,match="sorted and unique"):
        temporal_continuity_summary(pd.to_datetime(["2020-01-01","2020-01-01"]),"1D")
