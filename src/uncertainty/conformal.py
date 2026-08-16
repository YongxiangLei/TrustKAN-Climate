"""Split-conformal utilities for forecast intervals."""
from __future__ import annotations

import numpy as np


def conformal_radius(y_true, lower, upper, alpha: float = 0.1):
    """Return finite-sample split-conformal interval expansion radius."""
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)
    scores = np.maximum(lower - y_true, y_true - upper)
    scores = np.maximum(scores, 0.0).reshape(-1)
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set must be non-empty")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal(lower, upper, radius: float):
    return np.asarray(lower) - radius, np.asarray(upper) + radius


def interval_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mean_interval_width(lower, upper) -> float:
    return float(np.mean(np.asarray(upper) - np.asarray(lower)))
