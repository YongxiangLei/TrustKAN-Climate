from __future__ import annotations

import numpy as np
import pytest

from src.extremes.subsets import (
    evaluate_extremes,
    extreme_masks,
    training_extreme_thresholds,
)
from src.models.baselines import PersistenceForecaster
from src.robustness.corruption import (
    CorruptionSpec,
    add_gaussian_noise,
    apply_corruption,
    expand_corruption_grid,
    mask_random_timesteps,
    mask_recent_block,
)
from src.robustness.evaluation import evaluate_corruption_grid


def test_noise_is_reproducible_and_zero_at_clean_level():
    history = np.ones((4, 8, 2))
    rng = np.random.default_rng(11)
    noisy = add_gaussian_noise(history, 0.1, rng)
    again = add_gaussian_noise(history, 0.1, np.random.default_rng(11))
    assert np.allclose(noisy, again)
    assert np.allclose(add_gaussian_noise(history, 0.0, np.random.default_rng(0)), history)


def test_random_missingness_uses_training_fill_and_exact_rate():
    history = np.arange(40, dtype=float).reshape(2, 10, 2)
    out = mask_random_timesteps(history, 0.4, np.random.default_rng(0), fill=0.0)
    dropped = np.all(out == 0.0, axis=2)
    assert dropped.sum(axis=1).tolist() == [4, 4]
    assert not np.array_equal(out, history)


def test_block_missingness_only_removes_recent_history():
    history = np.ones((3, 7, 1))
    out = mask_recent_block(history, 3, fill=0.0)
    assert np.all(out[:, :4] == 1.0)
    assert np.all(out[:, -3:] == 0.0)
    with pytest.raises(ValueError, match="cannot exceed"):
        mask_recent_block(history, 8)


def test_corruption_grid_is_pre_registered_and_frequency_specific():
    config = {
        "noise_scales": [0.05, 0.1],
        "random_missing_rates": [0.2],
        "block_missing_lengths": {"daily": [1, 7], "hourly": [1, 24]},
    }
    daily = expand_corruption_grid(config, frequency="1D", seed=11)
    hourly = expand_corruption_grid(config, frequency="hourly", seed=11)
    assert [spec.kind for spec in daily][0] == "clean"
    assert [spec.level for spec in daily if spec.kind == "block_missing"] == [1.0, 7.0]
    assert [spec.level for spec in hourly if spec.kind == "block_missing"] == [1.0, 24.0]


def test_robustness_scoring_does_not_touch_targets():
    history = np.zeros((6, 5, 1))
    history[:, -1, 0] = np.arange(6)
    target = np.full((6, 2), 9.0)
    original = target.copy()
    model = PersistenceForecaster(2)
    rows, _ = evaluate_corruption_grid(
        history,
        target,
        lambda item: model.predict(item).numpy(),
        {
            "noise_scales": [0.0],
            "random_missing_rates": [],
            "block_missing_lengths": {"daily": [1]},
        },
        frequency="daily",
        seed=11,
    )
    assert np.array_equal(target, original)
    kinds = {row["kind"] for row in rows}
    assert kinds == {"clean", "noise", "block_missing"}
    blocked = next(row for row in rows if row["kind"] == "block_missing")
    clean = next(row for row in rows if row["kind"] == "clean")
    assert blocked["rmse"] > clean["rmse"]


def test_apply_corruption_rejects_unknown_kind():
    with pytest.raises(KeyError):
        apply_corruption(np.zeros((1, 2, 1)), CorruptionSpec("other", 1.0, 0))


def test_extreme_thresholds_ignore_test_targets():
    train = np.linspace(0, 10, 101)
    test = np.array([1000.0, -1000.0, 5.0])
    thresholds = training_extreme_thresholds(train)
    assert thresholds["lower"] == pytest.approx(0.5)
    assert thresholds["upper"] == pytest.approx(9.5)
    masks = extreme_masks(test, thresholds)
    assert masks["warm"].tolist() == [True, False, False]
    assert masks["cold"].tolist() == [False, True, False]


def test_extreme_evaluation_marks_small_subsets_underpowered():
    train = np.arange(100, dtype=float)
    test = np.array([[0.0], [50.0], [99.0]])
    pred = test + 1.0
    result = evaluate_extremes(train, test, pred, min_origins=30)
    assert result["subsets"]["cold"]["underpowered"] is True
    assert result["subsets"]["complement"]["underpowered"] is True
    assert result["thresholds"]["n_train"] == 100


def test_any_lead_diagnostic_is_available_but_not_required():
    thresholds = {"lower": 0.0, "upper": 10.0}
    target = np.array([[1.0, -1.0], [5.0, 5.0], [12.0, 8.0]])
    masks = extreme_masks(target, thresholds, definition="any_lead")
    assert masks["cold"].tolist() == [True, False, False]
    assert masks["warm"].tolist() == [False, False, True]
