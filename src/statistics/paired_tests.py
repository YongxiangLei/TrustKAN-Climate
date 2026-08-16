"""Dependence-aware paired analysis for ordered forecast cases."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_arrays(y_true, pred_a, pred_b):
    y = np.asarray(y_true, dtype=float)
    a = np.asarray(pred_a, dtype=float)
    b = np.asarray(pred_b, dtype=float)
    if y.shape != a.shape or y.shape != b.shape:
        raise ValueError("Target and paired predictions must have identical shapes")
    if y.ndim == 0 or len(y) < 2:
        raise ValueError("At least two ordered forecast cases are required")
    if not np.isfinite(y).all() or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Paired comparison inputs must be finite")
    if y.ndim == 1:
        y, a, b = y[:, None], a[:, None], b[:, None]
    return y, a, b


def origin_absolute_errors(y_true, pred_a, pred_b):
    """Return one horizon-averaged absolute error per forecast origin."""
    y, a, b = paired_arrays(y_true, pred_a, pred_b)
    axes = tuple(range(1, y.ndim))
    return np.mean(np.abs(y - a), axis=axes), np.mean(np.abs(y - b), axis=axes)


def paired_absolute_errors(y_true, pred_a, pred_b):
    """Backward-compatible alias returning origin-level absolute errors."""
    return origin_absolute_errors(y_true, pred_a, pred_b)


def wilcoxon_paired(y_true, pred_a, pred_b):
    """Origin-level Wilcoxon sensitivity analysis.

    The test still assumes independent paired origins and is therefore not the
    primary inference method for overlapping time-series forecasts.
    """
    error_a, error_b = origin_absolute_errors(y_true, pred_a, pred_b)
    difference = error_a - error_b
    if np.allclose(difference, 0):
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "median_error_difference": 0.0,
            "n_origins": len(difference),
            "role": "sensitivity_only",
        }
    result = stats.wilcoxon(error_a, error_b, zero_method="pratt", alternative="two-sided")
    return {
        "statistic": float(result.statistic),
        "pvalue": float(result.pvalue),
        "median_error_difference": float(np.median(difference)),
        "n_origins": len(difference),
        "role": "sensitivity_only",
    }


def metric_difference(y, a, b, metric):
    if metric == "mae":
        return float(np.mean(np.abs(y - a)) - np.mean(np.abs(y - b)))
    if metric == "rmse":
        return float(np.sqrt(np.mean((y - a) ** 2)) - np.sqrt(np.mean((y - b) ** 2)))
    raise ValueError("metric must be 'mae' or 'rmse'")


def circular_block_indices(n_cases, block_length, rng):
    if not isinstance(block_length, (int, np.integer)) or not 1 <= block_length <= n_cases:
        raise ValueError("block_length must be an integer between 1 and n_cases")
    n_blocks = int(np.ceil(n_cases / block_length))
    starts = rng.integers(0, n_cases, size=n_blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % n_cases).reshape(-1)[:n_cases]


def paired_block_bootstrap_difference(
    y_true,
    pred_a,
    pred_b,
    *,
    metric="mae",
    n_boot=5000,
    confidence=0.95,
    block_length=None,
    seed=0,
):
    """Circular moving-block bootstrap for metric(A)-metric(B).

    Forecast origins, rather than flattened horizon elements, are resampled in
    contiguous blocks. Negative differences favor model A.
    """
    y, a, b = paired_arrays(y_true, pred_a, pred_b)
    n_cases = len(y)
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 100:
        raise ValueError("n_boot must be an integer of at least 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if block_length is None:
        horizon_width = y.shape[1]
        block_length = min(
            n_cases,
            max(horizon_width, int(np.ceil(n_cases ** (1 / 3)))),
        )
        block_length_rule = "max(horizon_width,ceil(n_origins^(1/3)))"
    else:
        block_length_rule = "user_supplied"
    if block_length > n_cases:
        raise ValueError("block_length cannot exceed the number of forecast origins")

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sampled = circular_block_indices(n_cases, block_length, rng)
        draws[index] = metric_difference(y[sampled], a[sampled], b[sampled], metric)
    alpha = (1 - confidence) / 2
    observed = metric_difference(y, a, b, metric)
    return {
        "metric": metric,
        "mean_difference": observed,
        "ci_low": float(np.quantile(draws, alpha)),
        "ci_high": float(np.quantile(draws, 1 - alpha)),
        "confidence": confidence,
        "n_boot": int(n_boot),
        "block_length": int(block_length),
        "block_length_rule": block_length_rule,
        "n_origins": int(n_cases),
        "resampling_unit": "forecast_origin",
        "method": "circular_moving_block_bootstrap",
        "direction": "negative_favors_a",
    }


def paired_bootstrap_mae_difference(
    y_true, pred_a, pred_b, n_boot=5000, confidence=0.95, seed=0, block_length=None
):
    """Compatibility wrapper using dependence-aware block resampling."""
    return paired_block_bootstrap_difference(
        y_true,
        pred_a,
        pred_b,
        metric="mae",
        n_boot=n_boot,
        confidence=confidence,
        seed=seed,
        block_length=block_length,
    )


def paired_cohens_d(y_true, pred_a, pred_b):
    """Descriptive standardized mean of origin-level MAE differences."""
    error_a, error_b = origin_absolute_errors(y_true, pred_a, pred_b)
    difference = error_a - error_b
    standard_deviation = difference.std(ddof=1)
    return float(difference.mean() / standard_deviation) if standard_deviation > 0 else 0.0


def adjust_pvalues(pvalues, method="holm"):
    """Adjust a pre-specified family of p-values using Holm or BH."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be finite values in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    count = len(values)
    if method == "holm":
        adjusted_ranked = np.maximum.accumulate((count - np.arange(count)) * ranked)
    elif method in {"bh", "fdr_bh"}:
        raw = ranked * count / np.arange(1, count + 1)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
    else:
        raise ValueError("method must be 'holm' or 'bh'")
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0, 1)
    return adjusted
