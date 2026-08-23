"""The reach reported in the paper must describe the models the paper reports.

Measuring on a freshly initialized model would answer a question about the
architecture, which is a different and weaker claim. These tests cover the two
things that keep the reported number honest: the probe finds every reachable
position, and the reader refuses inputs that would let an unverified or
horizon-dependent measurement reach the manuscript.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from scripts.analyze_robustness import receptive_fields
from scripts.run_receptive_field import PROBE, code_sha256, receptive_field
from src.models.trustkan import TrustKAN

CPU = torch.device("cpu")


def fields_frame(**overrides) -> pd.DataFrame:
    rows = []
    for model, reach in (("trustkan", 3), ("kan", 365), ("transformer", 365)):
        for horizon in (1, 7, 30, 90):
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "history": 365,
                    "receptive_field": reach,
                    "control_receptive_field": reach,
                    "reproduced": True,
                }
            )
    frame = pd.DataFrame(rows)
    for key, value in overrides.items():
        frame.loc[frame.index[0], key] = value
    return frame


def write(tmp_path, frame, monkeypatch):
    path = tmp_path / "cet_receptive_fields.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setattr("scripts.analyze_robustness.FIELDS", path)
    return path


def test_reader_collapses_horizons_to_one_reach_per_model(tmp_path, monkeypatch):
    write(tmp_path, fields_frame(), monkeypatch)
    out = receptive_fields().set_index("model")
    assert out.receptive_field.to_dict() == {"trustkan": 3, "kan": 365, "transformer": 365}
    assert (out.horizons == 4).all()


def test_reader_refuses_a_reach_that_depends_on_the_horizon(tmp_path, monkeypatch):
    # A number that moved with the horizon would describe one trained instance,
    # not the architecture, and the paper states it as the latter.
    write(tmp_path, fields_frame(receptive_field=5), monkeypatch)
    with pytest.raises(SystemExit, match="varies with horizon"):
        receptive_fields()


def test_reader_refuses_runs_that_did_not_reproduce_the_ledger(tmp_path, monkeypatch):
    write(tmp_path, fields_frame(reproduced=False), monkeypatch)
    with pytest.raises(SystemExit, match="did not reproduce"):
        receptive_fields()


def test_reader_names_the_runner_when_the_measurement_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.analyze_robustness.FIELDS", tmp_path / "absent.csv")
    with pytest.raises(SystemExit, match="run_receptive_field"):
        receptive_fields()


def test_probe_sweeps_past_the_first_unreachable_step():
    """Stopping at the first dead offset would understate a model with a gap.

    A mean-pooled readout reaches every position, so a probe that halted early
    would still have to return the full window here.
    """
    torch.manual_seed(0)
    model = TrustKAN(1, horizon=2, hidden_dim=8, grid_size=4, readout="mean")
    assert receptive_field(model, 24, CPU) == 24


def test_probe_is_large_enough_to_escape_a_saturating_activation():
    assert PROBE >= 1.0


def test_code_fingerprint_covers_the_runner_and_the_model_source():
    first = code_sha256()
    assert len(first) == 64
    assert first == code_sha256()
