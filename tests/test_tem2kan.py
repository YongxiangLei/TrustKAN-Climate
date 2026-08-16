from __future__ import annotations

import pytest
import torch

from src.models.tem2kan import Tem2KANReference


def test_strict_tem2kan_rejects_non_reference_io_before_optional_import():
    with pytest.raises(ValueError, match="history=300"):
        Tem2KANReference(history=365, n_features=1, horizon=20)


def test_tem2kan_reference_shape_without_checkpoint_side_effect(tmp_path, monkeypatch):
    pytest.importorskip("kan")
    monkeypatch.chdir(tmp_path)
    model = Tem2KANReference(history=300, n_features=1, horizon=20, seed=11)
    output = model(torch.randn(2, 300, 1))
    assert output.shape == (2, 20)
    assert model.reproduction_spec["width"] == [300, 32, 64, 32, 20]
    assert model.reproduction_spec["k"] == 10
    assert model.reproduction_spec["grid"] == 10
    assert not (tmp_path / "model").exists()
