"""The corruption sweep must describe the models the rest of the paper reports.

The benchmark stored predictions rather than weights, so the sweep retrains. That
makes provenance a claim rather than a given, and the claim is checkable: if the
retrained model is the reported model, its clean forecast is not merely close to
the stored one but identical to it. These tests assert that on the real
artifacts, so a future change that silently trains a different model is caught
here rather than in the manuscript.

They skip when the artifacts are absent, since a fresh clone has no campaign.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "results" / "aggregated" / "cet_full_runs.csv"
BENCH_RAW = ROOT / "results" / "raw" / "cet_full"
ROB_RAW = ROOT / "results" / "robustness" / "raw" / "cet_robustness"
ROB_RUNS = ROOT / "results" / "robustness" / "runs" / "cet_robustness"

pytestmark = pytest.mark.skipif(
    not (LEDGER.exists() and ROB_RAW.exists() and ROB_RUNS.exists()),
    reason="robustness campaign artifacts are not present in this checkout",
)


def ledger() -> pd.DataFrame:
    frame = pd.read_csv(LEDGER)
    return frame[frame.status.eq("ok")]


def records() -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(ROB_RUNS.glob("*.json"))]


def test_every_sweep_run_succeeded():
    assert {rec["status"] for rec in records()} == {"ok"}


def test_clean_forecasts_are_identical_to_the_benchmark_artifacts():
    """Not "within tolerance": identical. Different weights cannot do this."""
    checked = 0
    for path in sorted(ROB_RAW.glob("*.npz")):
        model, horizon, seed = path.stem.replace("robustness_", "").rsplit("_", 2)
        bench = BENCH_RAW / f"cet_{model}_{horizon}_{seed}.npz"
        if not bench.exists():
            continue
        with np.load(path, allow_pickle=True) as source:
            sweep = np.asarray(source["clean_0_prediction"], dtype=np.float64)
        with np.load(bench, allow_pickle=True) as source:
            reported = np.asarray(source["prediction"], dtype=np.float64)
        assert sweep.shape == reported.shape
        assert np.array_equal(sweep, reported), f"{path.name} is not the reported model"
        checked += 1
    assert checked == 60


def test_the_sweep_shares_the_benchmark_config_and_dataset_fingerprints():
    frame = ledger()
    for rec in records():
        assert rec["config_sha256"] in set(frame.config_sha256)
        assert rec["dataset_sha256"] in {str(v).lower() for v in frame.dataset_sha256}


def test_the_sweep_ran_on_the_same_device_class_as_the_reported_models():
    # A CPU rerun of a CUDA-trained model does not reproduce it, so a sweep that
    # had silently fallen back to CPU would be a different set of models.
    family = ledger()
    family = family[family.model.isin(["trustkan", "kan", "transformer"])]
    assert set(family.device) == {"cuda:0"}
    assert {rec["device"] for rec in records()} == {"cuda:0"}
    assert {rec["torch"] for rec in records()} == set(family.torch)


def test_the_grid_covers_the_pre_specified_family_at_every_horizon_and_seed():
    grid = pd.read_csv(
        ROOT / "results" / "robustness" / "aggregated" / "cet_robustness_grid.csv"
    )
    assert grid.groupby(["model", "horizon", "model_seed"]).ngroups == 60
    assert set(grid.model) == {"trustkan", "kan", "transformer"}
    assert grid.groupby("model").horizon.nunique().eq(4).all()
    assert grid.groupby(["model", "horizon"]).model_seed.nunique().eq(5).all()
