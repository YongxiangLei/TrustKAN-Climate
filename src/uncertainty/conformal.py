"""Split-conformal utilities for forecast intervals."""
from __future__ import annotations

import numpy as np


def _validate_interval_inputs(y_true, lower, upper):
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if y_true.shape != lower.shape or y_true.shape != upper.shape:
        raise ValueError("Target and interval arrays must have identical shapes")
    if y_true.size == 0 or not np.isfinite(y_true).all():
        raise ValueError("Target array must be non-empty and finite")
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Interval arrays must be finite")
    if np.any(lower > upper):
        raise ValueError("Lower interval bounds must not exceed upper bounds")
    return y_true, lower, upper


def _finite_sample_quantile(scores, alpha, axis=None):
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0] if axis == 0 else scores.size
    if n == 0:
        raise ValueError("Calibration set must be non-empty")
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return np.quantile(scores, level, axis=axis, method="higher")


def conformal_radius(y_true, lower, upper, alpha: float = 0.1):
    """Return a legacy pooled finite-sample expansion radius."""
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    scores = np.maximum(lower - y_true, y_true - upper)
    scores = np.maximum(scores, 0.0).reshape(-1)
    return float(_finite_sample_quantile(scores, alpha))


def horizonwise_conformal_radii(y_true, lower, upper, alpha: float = 0.1):
    """Calibrate one marginal radius per lead using forecast origins as samples."""
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    if y_true.ndim == 1:
        y_true, lower, upper = y_true[:, None], lower[:, None], upper[:, None]
    if y_true.ndim != 2:
        raise ValueError("Horizonwise conformal calibration expects [origins, horizon]")
    scores = np.maximum(np.maximum(lower - y_true, y_true - upper), 0.0)
    return np.asarray(_finite_sample_quantile(scores, alpha, axis=0), dtype=float)


def simultaneous_conformal_radius(y_true, lower, upper, alpha: float = 0.1):
    """Calibrate trajectory-wise coverage from the maximum lead score per origin."""
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    if y_true.ndim == 1:
        y_true, lower, upper = y_true[:, None], lower[:, None], upper[:, None]
    if y_true.ndim != 2:
        raise ValueError("Simultaneous conformal calibration expects [origins, horizon]")
    scores = np.maximum(np.maximum(lower - y_true, y_true - upper), 0.0)
    return float(_finite_sample_quantile(scores.max(axis=1), alpha))


def apply_conformal(lower, upper, radius: float):
    return np.asarray(lower) - radius, np.asarray(upper) + radius


def interval_coverage(y_true, lower, upper) -> float:
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def horizonwise_interval_coverage(y_true, lower, upper):
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    if y_true.ndim == 1:
        y_true, lower, upper = y_true[:, None], lower[:, None], upper[:, None]
    return np.mean((y_true >= lower) & (y_true <= upper), axis=0)


def joint_interval_coverage(y_true, lower, upper) -> float:
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    if y_true.ndim == 1:
        y_true, lower, upper = y_true[:, None], lower[:, None], upper[:, None]
    hit = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(np.all(hit, axis=1)))


def mean_interval_width(lower, upper) -> float:
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != upper.shape or np.any(lower > upper):
        raise ValueError("Interval bounds must be aligned and ordered")
    return float(np.mean(upper - lower))


def mean_interval_score(y_true, lower, upper, alpha: float = 0.1) -> float:
    """Mean Winkler interval score; lower is better."""
    y_true, lower, upper = _validate_interval_inputs(y_true, lower, upper)
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    score = upper - lower
    score += (2.0 / alpha) * (lower - y_true) * (y_true < lower)
    score += (2.0 / alpha) * (y_true - upper) * (y_true > upper)
    return float(np.mean(score))
