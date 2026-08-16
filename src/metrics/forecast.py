"""Forecasting and selective-prediction metrics."""
from __future__ import annotations

import numpy as np


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def risk_coverage_curve(y, yhat, reliability):
    """Return coverage and RMSE after retaining most reliable samples first."""
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    reliability = np.asarray(reliability).reshape(-1)
    if not (len(y) == len(yhat) == len(reliability)):
        raise ValueError("Inputs must have equal length")
    order = np.argsort(-reliability)
    errors2 = (y[order] - yhat[order]) ** 2
    cumulative_risk = np.sqrt(np.cumsum(errors2) / np.arange(1, len(y) + 1))
    coverage = np.arange(1, len(y) + 1) / len(y)
    return coverage, cumulative_risk


def aurc(coverage, risk):
    return float(np.trapz(np.asarray(risk), np.asarray(coverage)))
