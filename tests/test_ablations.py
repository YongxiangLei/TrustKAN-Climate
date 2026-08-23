from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from scripts.run_ablations import (
    evaluation_side_ablations,
    validate_ablation_config,
    validate_execution_filters,
)
from src.models.trustkan import TrustKAN, budget_matched_width, kan_layer_parameters


ROOT = Path(__file__).resolve().parents[1]


def test_a1_encoder_is_budget_matched_against_the_kan_layer():
    for hidden, grid in ((64, 8), (32, 8), (64, 4)):
        kan = TrustKAN(1, horizon=1, hidden_dim=hidden, grid_size=grid, encoder="kan")
        mlp = TrustKAN(1, horizon=1, hidden_dim=hidden, grid_size=grid, encoder="mlp")
        kan_total = sum(p.numel() for p in kan.parameters())
        mlp_total = sum(p.numel() for p in mlp.parameters())
        assert abs(kan_total - mlp_total) / kan_total < 0.01
        assert budget_matched_width(hidden, grid) >= 1
        assert kan_layer_parameters(hidden, hidden, grid) > 0


def test_a2_drops_quantile_head_and_degenerates_to_residual_intervals():
    model = TrustKAN(1, horizon=3, hidden_dim=16, quantile_head=False)
    assert model.quantile_head is None
    out = model(torch.randn(4, 12, 1))
    assert out["quantiles"].shape == (4, 3, 3)
    # Degenerate quantiles make split conformal equivalent to symmetric
    # absolute-residual intervals around the point forecast.
    assert torch.allclose(out["quantiles"][..., 0], out["quantiles"][..., -1])
    assert torch.allclose(out["quantiles"][..., 0], out["point"])


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="encoder must be one of"):
        TrustKAN(1, horizon=1, encoder="wavelet")


def test_unknown_readout_is_rejected():
    with pytest.raises(ValueError, match="readout must be one of"):
        TrustKAN(1, horizon=1, readout="attention")


def test_default_readout_preserves_recency():
    """Averaging over history discarded recency and lost to persistence.

    The default must stay the last encoder state: a constant-history batch that
    ends on a distinctive value has to change the prediction, which mean
    pooling would wash out.
    """
    model = TrustKAN(1, horizon=1, hidden_dim=16, readout="last")
    assert model.readout_name == "last"
    base = torch.zeros(1, 40, 1)
    spiked = base.clone()
    spiked[0, -1, 0] = 5.0
    earlier = base.clone()
    earlier[0, 0, 0] = 5.0
    with torch.no_grad():
        p_base = model(base)["point"]
        p_spiked = model(spiked)["point"]
        p_earlier = model(earlier)["point"]
    # A change at the final step must move the forecast...
    assert not torch.allclose(p_base, p_spiked, atol=1e-6)
    # ...more than the same change far in the past.
    assert (p_spiked - p_base).abs().item() > (p_earlier - p_base).abs().item()


def test_mean_readout_remains_available_for_the_ablation():
    model = TrustKAN(1, horizon=2, hidden_dim=16, readout="mean")
    assert model.readout_name == "mean"
    out = model(torch.randn(3, 20, 1))
    assert out["point"].shape == (3, 2)


def test_evaluation_side_ablations_report_signed_gains():
    metrics = {
        "point": {"rmse": 2.0, "mae": 1.5},
        "raw_interval": {"marginal_coverage": 0.80, "mean_width": 3.0},
        "horizonwise_conformal": {
            "test_marginal_coverage": 0.90,
            "test_mean_width": 4.0,
        },
        "selective": {
            "fused": {"aurc": 1.0, "test_rmse": 1.5, "test_coverage": 0.8},
            "width_only": {"aurc": 1.2},
            "shift_only": {"aurc": 1.1},
        },
    }
    result = evaluation_side_ablations(metrics)
    assert result["A3_no_conformal"]["coverage_gain"] == pytest.approx(0.10)
    assert result["A3_no_conformal"]["width_cost"] == pytest.approx(1.0)
    assert result["A4_no_embedding_shift"]["aurc_gain_from_shift"] == pytest.approx(0.2)
    assert result["A5_no_interval_width"]["aurc_gain_from_width"] == pytest.approx(0.1)
    # Fusion is only credited against the better of the two components.
    assert result["A6_no_fusion"]["best_component_aurc"] == pytest.approx(1.1)
    assert result["A6_no_fusion"]["aurc_gain_from_fusion"] == pytest.approx(0.1)
    assert result["A7_no_abstention"]["rmse_reduction"] == pytest.approx(0.5)


def test_publication_ablation_config_declares_required_variants():
    path = ROOT / "configs" / "ablations.yaml"
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert validate_ablation_config(cfg, path) == "ablations_cet_full"
    assert cfg["dataset"].get("max_observations") is None
    assert len(set(cfg["training"]["seeds"])) == 5
    ids = [item["id"] for item in cfg["variants"]]
    assert ids[0] == "A0"
    assert {"A0", "A1", "A2", "A9"}.issubset(ids)
    # A9 preserves the mean-pooling defect as a reproducible comparison.
    a9 = next(item for item in cfg["variants"] if item["id"] == "A9")
    assert a9["readout"] == "mean"
    # A2 must not receive pinball gradients through its degenerate quantiles.
    a2 = next(item for item in cfg["variants"] if item["id"] == "A2")
    assert a2["quantile_head"] is False
    assert a2["quantile_weight"] == 0.0
    validate_execution_filters(cfg, horizons=[1], seeds=[11], variants=["A0"])
    with pytest.raises(ValueError, match="variant"):
        validate_execution_filters(cfg, variants=["A99"])


def test_ablation_config_rejects_missing_reference_variant():
    cfg = {
        "experiment": {"name": "bad"},
        "split": {"train": 0.6, "validation": 0.15, "calibration": 0.10, "test": 0.15},
        "training": {"deterministic_algorithms": True, "deterministic_warn_only": False},
        "model": {"quantiles": [0.05, 0.5, 0.95]},
        "conformal": {"alpha": 0.10},
        "variants": [
            {"id": "A1", "encoder": "mlp", "quantile_head": True},
            {"id": "A2", "encoder": "kan", "quantile_head": False},
            {"id": "A9", "encoder": "kan", "quantile_head": True, "readout": "mean"},
        ],
    }
    with pytest.raises(ValueError, match="A0"):
        validate_ablation_config(cfg, Path("configs/bad.yaml"))
