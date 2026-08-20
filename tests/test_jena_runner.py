from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.prepare_jena import code_sha256 as preparation_code_sha256
from scripts.run_cet_benchmark import build_model
from scripts.run_jena_benchmark import (
    DATASET_NAME,
    load_prepared_jena,
    main,
    resumable_record,
    validate_config,
)
from src.data.jena import REQUIRED_COLUMNS
from src.data.provenance import file_sha256


@contextmanager
def scratch_dir():
    root = Path(__file__).resolve().parent / "_scratch"
    root.mkdir(exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix="jena_run_", dir=root))
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _hourly_features(n=360):
    hours = pd.date_range("2010-01-01", periods=n, freq="h").to_numpy(dtype="datetime64[ns]")
    temperature = 8.0 + 4.0 * np.sin(np.arange(n) / 24.0)
    features = np.column_stack(
        [
            temperature,
            np.full(n, 1012.0),
            np.full(n, 70.0),
            np.full(n, 2.0),
        ]
    )
    return hours, features


def _write_prepared(path, panel_path, dates, features, code_hash=None, config_hash=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dates=dates,
        features=features,
        target=features[:, 0],
        feature_names=np.asarray(REQUIRED_COLUMNS),
        target_name=np.asarray("T (degC)"),
        config_sha256=np.asarray(config_hash or file_sha256(panel_path)),
        code_sha256=np.asarray(code_hash or preparation_code_sha256()),
    )


def test_build_model_honors_multivariate_width():
    model = build_model("trustkan", history=12, horizon=2, seed=11, n_features=4)
    import torch

    out = model(torch.zeros(3, 12, 4))
    assert out["point"].shape == (3, 2)
    with pytest.raises(ValueError, match="univariate"):
        build_model("tem2kan", history=300, horizon=20, n_features=4)


def test_full_jena_config_keeps_publication_gates():
    path = Path(__file__).resolve().parents[1] / "configs" / "jena.yaml"
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    assert validate_config(cfg, path) == "jena_full"
    assert "smoke" not in cfg["experiment"]["name"]
    assert cfg["dataset"].get("max_observations") is None
    assert cfg["window"] == {"history": 168, "horizons": [1, 6, 24]}
    assert len(set(cfg["training"]["seeds"])) == 5
    assert {"persistence", "svr", "mlp", "trustkan"}.issubset(cfg["models"])


def test_load_prepared_jena_rejects_wrong_columns():
    root = Path(__file__).resolve().parents[1]
    panel = root / "configs" / "datasets" / "jena_frozen.yaml"
    dates, features = _hourly_features(48)
    with scratch_dir() as path:
        artifact = path / "jena_hourly.npz"
        bad = features.copy()
        _write_prepared(artifact, panel, dates, bad)
        packed = dict(np.load(artifact, allow_pickle=False))
        packed["feature_names"] = np.asarray(["T (degC)", "p (mbar)", "rh (%)", "rain"])
        np.savez_compressed(artifact, **packed)
        cfg = {
            "dataset": {
                "panel_config": str(panel),
                "prepared_dir": str(path),
            }
        }
        with pytest.raises(ValueError, match="feature_names"):
            load_prepared_jena(cfg)


def test_jena_runner_persistence_is_multivariate_and_resumable():
    root = Path(__file__).resolve().parents[1]
    panel = root / "configs" / "datasets" / "jena_frozen.yaml"
    dates, features = _hourly_features(360)
    with scratch_dir() as path:
        prepared_dir = path / "prepared"
        _write_prepared(prepared_dir / "jena_hourly.npz", panel, dates, features)
        config_path = path / "jena_smoke.yaml"
        config = {
            "experiment": {"name": "jena_unit"},
            "dataset": {
                "panel_config": str(panel),
                "prepared_dir": str(prepared_dir),
                "frequency": "1h",
            },
            "window": {"history": 24, "horizons": [1]},
            "split": {
                "train": 0.60,
                "validation": 0.15,
                "calibration": 0.10,
                "test": 0.15,
            },
            "training": {
                "seeds": [11],
                "batch_size": 32,
                "epochs": 1,
                "patience": 1,
                "learning_rate": 0.001,
                "optimizer": "adamw",
                "weight_decay": 0.01,
                "deterministic_algorithms": True,
                "deterministic_warn_only": False,
            },
            "models": ["persistence"],
        }
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        rows = main(
            config_path,
            models=["persistence"],
            horizons=[1],
            seeds=[-1],
            results_root=path / "results",
        )
        assert len(rows) == 1
        assert rows[0]["status"] == "ok"
        assert rows[0]["dataset"] == DATASET_NAME
        assert rows[0]["n_features"] == 4
        artifact = Path(rows[0]["artifact_path"])
        packed = np.load(artifact, allow_pickle=False)
        assert packed["prediction"].shape == packed["target"].shape
        assert Path(path / "results" / "splits" / "jena_unit_train_target.npz").is_file()
        again = main(
            config_path,
            resume=True,
            models=["persistence"],
            horizons=[1],
            seeds=[-1],
            results_root=path / "results",
        )
        assert again[0]["artifact_sha256"] == rows[0]["artifact_sha256"]
        packed["prediction"]
        artifact.write_bytes(b"stale")
        assert (
            resumable_record(
                rows[0]["record_path"],
                artifact,
                rows[0]["config_sha256"],
                rows[0]["code_sha256"],
                rows[0]["dataset_sha256"],
            )
            is None
        )
