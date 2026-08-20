"""Extreme-event subsets defined from training-period target quantiles.

Thresholds are never estimated from validation, calibration, or test targets.
An origin is labelled from its first forecast lead unless a caller explicitly
requests the any-lead diagnostic.
"""
from __future__ import annotations

import numpy as np

from src.metrics.forecast import mae, rmse


DEFAULT_LOWER_QUANTILE = 0.05
DEFAULT_UPPER_QUANTILE = 0.95
MIN_ORIGINS = 30


def training_extreme_thresholds(
    train_target,
    *,
    lower_quantile=DEFAULT_LOWER_QUANTILE,
    upper_quantile=DEFAULT_UPPER_QUANTILE,
):
    values = np.asarray(train_target, dtype=float).reshape(-1)
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("Training targets must be finite and non-empty")
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("Extreme quantiles must satisfy 0 < lower < upper < 1")
    return {
        "lower": float(np.quantile(values, lower_quantile, method="linear")),
        "upper": float(np.quantile(values, upper_quantile, method="linear")),
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "n_train": int(values.size),
    }


def first_lead(target):
    target = np.asarray(target, dtype=float)
    if target.ndim == 1:
        return target
    if target.ndim != 2:
        raise ValueError("target must be [n_origins] or [n_origins, horizon]")
    return target[:, 0]


def extreme_masks(target, thresholds, *, definition: str = "first_lead"):
    if definition == "first_lead":
        values = first_lead(target)
    elif definition == "any_lead":
        target = np.asarray(target, dtype=float)
        if target.ndim == 1:
            values = target
        else:
            cold = (target < thresholds["lower"]).any(axis=1)
            warm = (target > thresholds["upper"]).any(axis=1)
            return {
                "cold": cold,
                "warm": warm,
                "either": cold | warm,
                "complement": ~(cold | warm),
            }
    else:
        raise KeyError(f"Unknown extreme definition {definition!r}")
    cold = values < thresholds["lower"]
    warm = values > thresholds["upper"]
    either = cold | warm
    return {
        "cold": cold,
        "warm": warm,
        "either": either,
        "complement": ~either,
    }


def summarize_extreme_subset(target, prediction, mask, *, min_origins=MIN_ORIGINS):
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if target.shape != prediction.shape or target.shape[0] != len(mask):
        raise ValueError("target, prediction and mask must align on forecast origins")
    count = int(mask.sum())
    payload = {
        "n_origins": count,
        "fraction": float(mask.mean()) if len(mask) else 0.0,
        "underpowered": count < int(min_origins),
        "rmse": None,
        "mae": None,
    }
    if count == 0:
        return payload
    payload["rmse"] = rmse(target[mask], prediction[mask])
    payload["mae"] = mae(target[mask], prediction[mask])
    return payload


def evaluate_extremes(
    train_target,
    test_target,
    test_prediction,
    *,
    lower_quantile=DEFAULT_LOWER_QUANTILE,
    upper_quantile=DEFAULT_UPPER_QUANTILE,
    definition="first_lead",
    min_origins=MIN_ORIGINS,
):
    thresholds = training_extreme_thresholds(
        train_target,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    masks = extreme_masks(test_target, thresholds, definition=definition)
    subsets = {
        name: summarize_extreme_subset(
            test_target, test_prediction, mask, min_origins=min_origins
        )
        for name, mask in masks.items()
    }
    return {
        "thresholds": thresholds,
        "definition": definition,
        "min_origins": int(min_origins),
        "n_test_origins": int(len(np.asarray(test_target))),
        "subsets": subsets,
    }
