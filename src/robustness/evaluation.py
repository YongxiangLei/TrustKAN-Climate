"""Score forecasts on the pre-registered corruption grid."""
from __future__ import annotations

import numpy as np

from src.metrics.forecast import mae, rmse
from src.robustness.corruption import apply_corruption, expand_corruption_grid


def score_forecast(target, prediction):
    return {"rmse": rmse(target, prediction), "mae": mae(target, prediction)}


def evaluate_corruption_grid(
    history,
    target,
    predict_fn,
    config: dict,
    *,
    frequency: str,
    seed: int = 11,
    fill: float = 0.0,
):
    """Corrupt histories, call a frozen predictor, and score physical-unit targets."""
    history = np.asarray(history, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(history) != len(target):
        raise ValueError("history and target must contain the same number of origins")
    rows = []
    arrays = {}
    for spec in expand_corruption_grid(config, frequency=frequency, seed=seed):
        corrupted = (
            history.copy()
            if spec.kind == "clean"
            else apply_corruption(history, spec, fill=fill)
        )
        prediction = np.asarray(predict_fn(corrupted), dtype=float)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Predictor returned shape {prediction.shape} rather than {target.shape}"
            )
        metrics = score_forecast(target, prediction)
        key = f"{spec.kind}_{spec.level:g}"
        rows.append(
            {
                "kind": spec.kind,
                "level": spec.level,
                "seed": spec.seed,
                "n_origins": int(len(target)),
                **metrics,
            }
        )
        arrays[f"{key}_prediction"] = prediction
        arrays[f"{key}_history"] = corrupted
    return rows, arrays
