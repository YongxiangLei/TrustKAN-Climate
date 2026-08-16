from __future__ import annotations

import numpy as np

from src.models.classical import make_random_forest, make_svr


def toy_data(horizon):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(24, 5, 1))
    y = rng.normal(size=(24, horizon))
    return x, y


def test_random_forest_preserves_single_output_axis():
    x, y = toy_data(1)
    model = make_random_forest(n_estimators=5, random_state=3, n_jobs=1).fit(x, y)
    assert model.predict(x[:4]).shape == (4, 1)


def test_random_forest_preserves_multi_output_axis():
    x, y = toy_data(3)
    model = make_random_forest(n_estimators=5, random_state=3, n_jobs=1).fit(x, y)
    assert model.predict(x[:4]).shape == (4, 3)


def test_svr_preserves_single_output_axis():
    x, y = toy_data(1)
    model = make_svr(c=1.0).fit(x, y)
    assert model.predict(x[:4]).shape == (4, 1)
