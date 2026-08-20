"""Train-only input corruptions for robustness evaluation.

Corruptions are applied to forecast histories after train-only standardization.
They never modify targets, never update the scaler, and must not be used to
select hyperparameters or reliability thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _validate_history(history):
    history = np.asarray(history, dtype=float)
    if history.ndim != 3:
        raise ValueError("history must have shape [n_samples, n_timesteps, n_features]")
    if not np.isfinite(history).all():
        raise ValueError("history contains non-finite values")
    return history


def add_gaussian_noise(history, scale: float, rng):
    """Add N(0, scale^2) noise in standardized units."""
    history = _validate_history(history)
    if scale < 0:
        raise ValueError("noise scale must be non-negative")
    if scale == 0:
        return history.copy()
    noise = rng.normal(loc=0.0, scale=scale, size=history.shape)
    return history + noise


def mask_random_timesteps(history, rate: float, rng, fill=0.0):
    """Replace a random fraction of history timesteps with the training mean."""
    history = _validate_history(history)
    if not 0 <= rate <= 1:
        raise ValueError("missingness rate must lie in [0, 1]")
    out = history.copy()
    if rate == 0:
        return out
    n_samples, n_steps, _ = out.shape
    n_drop = int(np.floor(rate * n_steps))
    if n_drop == 0:
        return out
    for index in range(n_samples):
        chosen = rng.choice(n_steps, size=n_drop, replace=False)
        out[index, chosen, :] = fill
    return out


def mask_recent_block(history, length: int, fill=0.0):
    """Replace the most recent contiguous history block with the training mean."""
    history = _validate_history(history)
    if length < 0:
        raise ValueError("block length must be non-negative")
    out = history.copy()
    if length == 0:
        return out
    if length > out.shape[1]:
        raise ValueError("block length cannot exceed the history length")
    out[:, -length:, :] = fill
    return out


@dataclass(frozen=True)
class CorruptionSpec:
    kind: str
    level: float
    seed: int


def apply_corruption(history, spec: CorruptionSpec, *, fill=0.0):
    rng = np.random.default_rng(spec.seed)
    if spec.kind == "noise":
        return add_gaussian_noise(history, spec.level, rng)
    if spec.kind == "random_missing":
        return mask_random_timesteps(history, spec.level, rng, fill=fill)
    if spec.kind == "block_missing":
        return mask_recent_block(history, int(spec.level), fill=fill)
    raise KeyError(f"Unknown corruption kind {spec.kind!r}")


def expand_corruption_grid(config: dict, *, frequency: str, seed: int = 11) -> list[CorruptionSpec]:
    """Materialize the pre-registered robustness grid without looking at metrics."""
    specs = [CorruptionSpec("clean", 0.0, seed)]
    for scale in config.get("noise_scales", []):
        specs.append(CorruptionSpec("noise", float(scale), seed))
    for rate in config.get("random_missing_rates", []):
        specs.append(CorruptionSpec("random_missing", float(rate), seed))
    lengths = config.get("block_missing_lengths", {})
    key = "hourly" if str(frequency).lower() in {"1h", "hourly"} else "daily"
    for length in lengths.get(key, lengths.get(frequency, [])):
        specs.append(CorruptionSpec("block_missing", float(length), seed))
    return specs
