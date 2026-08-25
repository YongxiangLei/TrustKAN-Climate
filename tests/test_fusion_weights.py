"""Guard the property that calibration-selected fusion cannot lose to itself.

The frozen equal weighting was beaten by its own width-only component at every
horizon, because the other component was anti-correlated with error yet still
carried half the score. Selecting weights on a grid whose endpoints are the
single-component scores removes that failure mode by construction, and these
tests pin that guarantee rather than the particular weights it produces.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.reliability.fusion import (
    choose_fusion_weights_on_calibration,
    fuse_reliability,
)


def synthetic(n=400, seed=0):
    """A useful component, an actively misleading one, and matching errors."""
    rng = np.random.default_rng(seed)
    difficulty = rng.uniform(size=n)
    target = rng.normal(scale=1.0 + 3.0 * difficulty)[:, None]
    prediction = np.zeros_like(target)
    useful = 1.0 - difficulty
    # Ranks the opposite way, which is what the embedding-shift term did.
    misleading = difficulty
    return target, prediction, useful, misleading


def test_selection_beats_or_matches_the_best_single_component():
    target, prediction, useful, misleading = synthetic()
    result = choose_fusion_weights_on_calibration(
        target, prediction, [useful, misleading]
    )
    endpoints = result["endpoint_aurc"]
    assert result["calibration_aurc"] <= min(endpoints.values()) + 1e-12


def test_selection_withholds_mass_from_a_misleading_component():
    target, prediction, useful, misleading = synthetic()
    result = choose_fusion_weights_on_calibration(
        target, prediction, [useful, misleading]
    )
    # The useful component is first, so it should take most of the mass.
    assert result["weights"][0] > result["weights"][1]


def test_selection_splits_mass_when_both_components_inform():
    """Two noisy views of the same difficulty should both earn weight."""
    rng = np.random.default_rng(3)
    n = 600
    difficulty = rng.uniform(size=n)
    target = rng.normal(scale=1.0 + 3.0 * difficulty)[:, None]
    prediction = np.zeros_like(target)
    first = np.clip(1.0 - difficulty + rng.normal(scale=0.30, size=n), 1e-6, 1.0)
    second = np.clip(1.0 - difficulty + rng.normal(scale=0.30, size=n), 1e-6, 1.0)
    result = choose_fusion_weights_on_calibration(target, prediction, [first, second])
    assert min(result["weights"]) > 0.0


def test_endpoints_reproduce_the_single_component_scores():
    target, prediction, useful, misleading = synthetic()
    result = choose_fusion_weights_on_calibration(
        target, prediction, [useful, misleading], grid=3
    )
    searched = {tuple(round(w, 6) for w in item["weights"]): item for item in result["searched"]}
    assert (1.0, 0.0) in searched and (0.0, 1.0) in searched


def test_all_mass_on_one_component_reproduces_that_component():
    _, _, useful, misleading = synthetic()
    fused = fuse_reliability(useful, misleading, weights=(1.0, 0.0))
    assert np.allclose(fused, np.clip(useful, 1e-8, 1.0))


def test_selection_rejects_unusable_inputs():
    target, prediction, useful, misleading = synthetic(n=20)
    with pytest.raises(ValueError, match="two components"):
        choose_fusion_weights_on_calibration(target, prediction, [useful])
    with pytest.raises(ValueError, match="equal length"):
        choose_fusion_weights_on_calibration(
            target, prediction, [useful, misleading[:-1]]
        )
    with pytest.raises(ValueError, match="grid"):
        choose_fusion_weights_on_calibration(
            target, prediction, [useful, misleading], grid=2
        )
