"""Dataset fingerprints and temporal-continuity summaries."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path: str | Path, chunk_size=1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def temporal_continuity_summary(dates, expected_frequency: str) -> dict:
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
    if len(index) == 0:
        raise ValueError("Cannot summarize an empty timestamp sequence")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("Timestamps must be sorted and unique")
    aliases = {"daily": "1D", "hourly": "1h", "weekly": "7D"}
    normalized_frequency = aliases.get(str(expected_frequency).lower(), expected_frequency)
    step = pd.to_timedelta(normalized_frequency)
    if step <= pd.Timedelta(0):
        raise ValueError("expected_frequency must be positive")
    deltas = index.to_series(index=np.arange(len(index))).diff().dropna()
    ratios = deltas / step
    non_regular = ~np.isclose(ratios, 1.0)
    missing_steps = np.maximum(np.rint(ratios[ratios > 1]).astype(int) - 1, 0).sum()
    span_steps = int((index[-1] - index[0]) / step) + 1
    return {
        "start": index[0].isoformat(),
        "end": index[-1].isoformat(),
        "observations": int(len(index)),
        "expected_frequency": str(expected_frequency),
        "expected_steps_in_span": span_steps,
        "completeness_fraction": float(len(index) / span_steps),
        "non_regular_intervals": int(non_regular.sum()),
        "estimated_missing_steps": int(missing_steps),
        "max_gap_steps": float(ratios.max()) if len(ratios) else 0.0,
    }


def fixed_window_continuity_summary(dates, expected_frequency: str, start, end) -> dict:
    """Measure completeness against pre-specified bounds, including edge gaps."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        raise ValueError("end must not precede start")
    summary = temporal_continuity_summary(dates, expected_frequency)
    observed_start = pd.Timestamp(summary["start"])
    observed_end = pd.Timestamp(summary["end"])
    if observed_start < start or observed_end > end:
        raise ValueError("Timestamps must lie inside the fixed window")
    aliases = {"daily": "1D", "hourly": "1h", "weekly": "7D"}
    normalized_frequency = aliases.get(str(expected_frequency).lower(), expected_frequency)
    step = pd.to_timedelta(normalized_frequency)
    expected = int((end - start) / step) + 1
    summary["required_window_start"] = start.isoformat()
    summary["required_window_end"] = end.isoformat()
    summary["expected_steps_in_required_window"] = expected
    summary["completeness_fraction"] = float(summary["observations"] / expected)
    summary["estimated_missing_steps_in_required_window"] = int(
        expected - summary["observations"]
    )
    return summary


def evaluate_temporal_eligibility(summary: dict, criteria: dict) -> dict:
    start = pd.Timestamp(summary["start"])
    end = pd.Timestamp(summary["end"])
    span_years = (end - start).total_seconds() / (365.2425 * 24 * 3600)
    checks = {
        "minimum_span_years": span_years >= float(criteria["minimum_span_years"]),
        "minimum_completeness": summary["completeness_fraction"]
        >= float(criteria["minimum_completeness"]),
    }
    if criteria.get("maximum_gap_steps") is not None:
        checks["maximum_gap_steps"] = summary["max_gap_steps"] <= float(
            criteria["maximum_gap_steps"]
        )
    return {
        "eligible": bool(all(checks.values())),
        "observed_span_years": float(span_years),
        "criteria": criteria,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
