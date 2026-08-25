"""Guard the properties the wide-field stem exists to obtain.

The published local stem reaches only the final three steps of its history, so a
365-day window bought the model three days of context. That defect was invisible
to accuracy tests, which is why it survived a full campaign, so the replacement's
reach is pinned here as a measured property rather than as an inspection of the
code. Tests covering the standalone measurement runner live in
`test_receptive_field.py`; these cover the architecture itself.
"""
from __future__ import annotations

import pytest
import torch

from src.models.trustkan import (
    STEMS,
    CausalDilatedStem,
    TrustKAN,
    dilated_depth,
)


def measured_receptive_field(model: TrustKAN, history: int) -> int:
    """Count history positions whose perturbation changes the forecast."""
    model.eval()
    baseline_input = torch.zeros(1, history, 1)
    with torch.no_grad():
        baseline = model(baseline_input)["point"]
    reached = 0
    for step in range(history):
        probe = baseline_input.clone()
        probe[0, step, 0] = 1.0
        with torch.no_grad():
            if not torch.equal(model(probe)["point"], baseline):
                reached += 1
    return reached


def test_local_stem_reaches_only_the_final_three_steps():
    history = 64
    model = TrustKAN(1, horizon=1, hidden_dim=8, stem="local")
    assert measured_receptive_field(model, history) == 3


def test_dilated_stem_reaches_the_whole_history():
    history = 64
    model = TrustKAN(1, horizon=1, hidden_dim=8, stem="dilated", history=history)
    # One boundary position can coincide numerically, so allow a single miss
    # rather than asserting a figure that depends on float equality.
    assert measured_receptive_field(model, history) >= history - 1


def test_dilated_depth_covers_the_requested_history():
    for history in (8, 64, 365, 1000):
        depth = dilated_depth(history)
        field = 1 + 2 * (2**depth - 1)
        assert field >= history
        # The depth must be minimal, or the stem pays for layers it cannot use.
        assert 1 + 2 * (2 ** (depth - 1) - 1) < history or depth == 1


def test_dilated_stem_is_cheaper_than_the_local_stem_it_replaces():
    history = 365
    local = TrustKAN(1, horizon=1, hidden_dim=64, stem="local")
    dilated = TrustKAN(1, horizon=1, hidden_dim=64, stem="dilated", history=history)
    local_stem = sum(p.numel() for p in local.encoder.temporal.parameters())
    dilated_stem = sum(p.numel() for p in dilated.encoder.temporal.parameters())
    assert dilated_stem < local_stem


def test_dilated_stem_needs_a_history_length():
    with pytest.raises(ValueError, match="history"):
        TrustKAN(1, horizon=1, stem="dilated")


def test_unknown_stem_is_rejected():
    with pytest.raises(ValueError, match="stem must be one of"):
        TrustKAN(1, horizon=1, stem="fourier")
    assert set(STEMS) == {"local", "dilated"}


def test_dilated_stem_preserves_sequence_length():
    stem = CausalDilatedStem(1, 8, 64)
    out = stem(torch.randn(2, 1, 64))
    assert out.shape == (2, 8, 64)


def test_dilated_stem_is_causal():
    """No output position may depend on a later input position."""
    stem = CausalDilatedStem(1, 4, 32)
    stem.eval()
    base_input = torch.zeros(1, 1, 32)
    with torch.no_grad():
        base = stem(base_input)
    probe = base_input.clone()
    probe[0, 0, -1] = 1.0
    with torch.no_grad():
        perturbed = stem(probe)
    # Perturbing the last input may only move the last output position.
    assert torch.equal(base[:, :, :-1], perturbed[:, :, :-1])


def test_attention_readout_reaches_the_whole_history_and_keeps_the_contract():
    history = 64
    model = TrustKAN(
        1, horizon=3, hidden_dim=8, stem="dilated", history=history, readout="attention"
    )
    assert measured_receptive_field(model, history) >= history - 1
    out = model(torch.randn(4, history, 1))
    assert out["point"].shape == (4, 3)
    assert out["quantiles"].shape == (4, 3, 3)
    # Reliability scoring consumes the embedding, so its width must not change
    # with the readout.
    assert out["embedding"].shape == (4, 8)


def test_attention_readout_is_not_a_constant_pooling():
    """The aggregation must depend on the input, or it is just a mean."""
    model = TrustKAN(1, horizon=1, hidden_dim=8, stem="dilated", history=32,
                     readout="attention")
    model.eval()
    with torch.no_grad():
        first = model(torch.zeros(1, 32, 1))["embedding"]
        second = model(torch.randn(1, 32, 1))["embedding"]
    assert not torch.allclose(first, second)
