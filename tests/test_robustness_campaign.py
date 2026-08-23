from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from scripts.run_robustness_campaign import MODELS, RMSE_TOLERANCE, make_predictor
from src.data.timeseries import TrainOnlyStandardizer
from src.models.trustkan import TrustKAN
from src.robustness.corruption import expand_corruption_grid
from src.robustness.evaluation import evaluate_corruption_grid

POLICY = yaml.safe_load(open("configs/robustness.yaml", encoding="utf-8"))


def fitted_scaler() -> TrainOnlyStandardizer:
    return TrainOnlyStandardizer().fit(np.linspace(-5.0, 25.0, 400))


def test_predictor_returns_physical_units_in_target_shape():
    torch.manual_seed(0)
    model = TrustKAN(1, horizon=3, hidden_dim=8, grid_size=4).eval()
    predict_fn = make_predictor(model, fitted_scaler(), model.horizon, 16, torch.device("cpu"))
    out = predict_fn(np.zeros((7, 12, 1)))
    assert out.shape == (7, 3)
    assert np.isfinite(out).all()


def test_predictor_works_for_baselines_that_carry_no_horizon_attribute():
    # Only the proposed architecture stores its horizon on the module, so
    # reading it off the model silently excluded both comparators from the
    # sweep.
    from src.models.kan_baseline import StandardKANForecaster

    torch.manual_seed(0)
    model = StandardKANForecaster(12, 1, 3, hidden_dim=8, grid_size=4).eval()
    assert not hasattr(model, "horizon")
    predict_fn = make_predictor(model, fitted_scaler(), 3, 8, torch.device("cpu"))
    assert predict_fn(np.zeros((5, 12, 1))).shape == (5, 3)


def test_predictor_is_deterministic_across_calls():
    # The corruption grid calls the predictor once per level and compares the
    # results, so any nondeterminism would be read as an effect of corruption.
    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4).eval()
    predict_fn = make_predictor(model, fitted_scaler(), model.horizon, 4, torch.device("cpu"))
    history = np.random.default_rng(0).normal(size=(9, 10, 1))
    assert np.array_equal(predict_fn(history), predict_fn(history))


def test_clean_grid_entry_matches_an_uncorrupted_forecast():
    # The clean cell is what the runner checks against the benchmark ledger, so
    # it has to be the untouched forecast rather than a zero-level corruption
    # that happens to look like one.
    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4).eval()
    scaler = fitted_scaler()
    predict_fn = make_predictor(model, scaler, model.horizon, 8, torch.device("cpu"))
    history = np.random.default_rng(1).normal(size=(12, 10, 1))
    target = predict_fn(history)
    rows, arrays = evaluate_corruption_grid(
        history, target, predict_fn, POLICY, frequency="1D", seed=11
    )
    clean = next(row for row in rows if row["kind"] == "clean")
    assert clean["rmse"] == pytest.approx(0.0, abs=1e-9)
    assert np.array_equal(arrays["clean_0_history"], history)


def test_grid_covers_every_pre_registered_level():
    specs = expand_corruption_grid(POLICY, frequency="1D", seed=11)
    kinds = {}
    for spec in specs:
        kinds.setdefault(spec.kind, []).append(spec.level)
    assert kinds["clean"] == [0.0]
    assert kinds["noise"] == [0.05, 0.10, 0.20]
    assert kinds["random_missing"] == [0.10, 0.20, 0.40]
    assert kinds["block_missing"] == [1.0, 3.0, 7.0]


def test_daily_and_hourly_block_lengths_do_not_mix():
    daily = [s.level for s in expand_corruption_grid(POLICY, frequency="1D") if s.kind == "block_missing"]
    hourly = [s.level for s in expand_corruption_grid(POLICY, frequency="1h") if s.kind == "block_missing"]
    assert daily == [1.0, 3.0, 7.0]
    assert hourly == [1.0, 6.0, 24.0]


def test_scope_is_the_pre_specified_comparison_family():
    # Corrupting every baseline would cost more without bearing on any
    # pre-registered claim, so the scope is fixed to the family Section IV names.
    assert MODELS == ("trustkan", "kan", "transformer")


def test_reproduction_tolerance_is_tight_enough_to_reject_another_model():
    assert RMSE_TOLERANCE < 1e-4


def test_last_state_readout_sees_only_three_timesteps():
    # Two kernel-3 convolutions give the final encoder state a receptive field
    # of three steps, so a 365-day history reaches the readout as three days.
    # This is why block corruption saturates at length three in the sweep, and
    # a change to the stem that widened it would invalidate that reading.
    from scripts.analyze_robustness import receptive_field

    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4)
    assert receptive_field(model, history=40) == 3


def test_the_comparators_consume_their_whole_window():
    # The deficit is attributed to how much history reaches the readout, so it
    # matters that both pre-specified comparators are not similarly limited.
    from scripts.analyze_robustness import receptive_field
    from src.models.baselines import TransformerForecaster
    from src.models.kan_baseline import StandardKANForecaster

    torch.manual_seed(0)
    assert receptive_field(StandardKANForecaster(24, 1, 2, hidden_dim=8, grid_size=4), 24) == 24
    assert receptive_field(TransformerForecaster(1, 2), 24) == 24


def test_the_mean_pooled_defect_saw_the_whole_window():
    # The superseded readout averaged every position, so it reached all of the
    # history and lost recency instead; neither configuration gives the model
    # what the comparators have.
    from scripts.analyze_robustness import receptive_field

    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4, readout="mean")
    assert receptive_field(model, history=24) == 24


def test_block_corruption_saturates_at_the_receptive_field():
    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4).eval()
    predict_fn = make_predictor(model, fitted_scaler(), model.horizon, 8, torch.device("cpu"))
    history = np.random.default_rng(2).normal(size=(6, 20, 1))
    masked = {}
    for length in (1, 3, 7):
        corrupted = history.copy()
        corrupted[:, -length:, :] = 0.0
        masked[length] = predict_fn(corrupted)
    assert not np.allclose(masked[1], masked[3])
    assert np.allclose(masked[3], masked[7])
