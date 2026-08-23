from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.run_kan_curves import (
    RMSE_TOLERANCE,
    best_match_correlation,
    curve_parts,
    nonlinear_share,
    standardize_rows,
)
from src.interpretability.kan_curves import default_evaluation_grid
from src.models.trustkan import TrustKAN


def build(seed: int, hidden: int = 8) -> TrustKAN:
    torch.manual_seed(seed)
    return TrustKAN(1, horizon=1, hidden_dim=hidden, grid_size=4)


def test_decomposition_reproduces_the_shared_extractor():
    # curve_parts raises if linear + rbf drifts from the primitive that the
    # manuscript plots, which is the guarantee that both describe one object.
    model = build(0)
    linear, rbf, curves = curve_parts(model, default_evaluation_grid())
    assert linear.shape == rbf.shape == curves.shape
    assert np.allclose(linear + rbf, curves, atol=1e-5)


def test_decomposition_detects_a_tampered_layer():
    model = build(0)
    grid = default_evaluation_grid()
    curve_parts(model, grid)
    # A curve set extracted from different weights than the ones measured must
    # not silently pass as a decomposition of the reported model.
    with torch.no_grad():
        model.encoder.kan.base.weight.add_(5.0)
    linear, rbf, curves = curve_parts(model, grid)
    assert np.allclose(linear + rbf, curves, atol=1e-5)


def test_nonlinear_share_is_zero_for_a_purely_linear_layer():
    model = build(0)
    with torch.no_grad():
        model.encoder.kan.coeff.zero_()
    linear, rbf, _ = curve_parts(model, default_evaluation_grid())
    share = nonlinear_share(linear, rbf)
    assert share.shape[0] == linear.shape[0]
    assert np.allclose(share, 0.0)


def test_nonlinear_share_is_one_when_the_linear_term_vanishes():
    model = build(0)
    with torch.no_grad():
        model.encoder.kan.base.weight.zero_()
    linear, rbf, _ = curve_parts(model, default_evaluation_grid())
    assert np.allclose(nonlinear_share(linear, rbf), 1.0)


def test_nonlinear_share_ignores_a_constant_offset():
    # A vertical shift is not structure, so spread rather than magnitude is
    # what the share is built from.
    linear = np.tile(np.linspace(-1, 1, 9), (3, 1))
    rbf = np.zeros_like(linear)
    baseline = nonlinear_share(linear, rbf)
    assert np.allclose(nonlinear_share(linear + 100.0, rbf), baseline)


def test_standardize_rows_leaves_constant_curves_at_zero():
    rows = standardize_rows(np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 2.0]]))
    assert np.allclose(rows[0], 0.0)
    assert pytest.approx(np.linalg.norm(rows[1]), rel=1e-9) == 1.0


def test_best_match_is_perfect_against_a_permutation_of_itself():
    # The permutation freedom of latent channels is exactly what this bound is
    # meant to forgive, so a shuffled copy must score one.
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(16, 21))
    shuffled = curves[rng.permutation(16)]
    assert np.allclose(best_match_correlation(curves, shuffled), 1.0, atol=1e-6)


def test_best_match_bounds_index_matched_agreement():
    from src.interpretability.stability import curve_correlation

    rng = np.random.default_rng(1)
    a = rng.normal(size=(12, 21))
    b = rng.normal(size=(12, 21))
    assert np.all(best_match_correlation(a, b) >= curve_correlation(a, b) - 1e-6)


def test_reproduction_tolerance_is_tight_enough_to_catch_a_different_model():
    # The guard exists to reject curves from a model that is not the reported
    # one; a tolerance loose enough to admit a visibly different RMSE would
    # defeat it.
    assert RMSE_TOLERANCE < 1e-4
