"""Run a first leakage-safe CET benchmark.

This script intentionally starts with a compact model set. Additional published
baselines can be added after the benchmark protocol is frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.timeseries import chronological_split, sliding_windows, TrainOnlyStandardizer, assign_windows_by_target_origin
from src.metrics.forecast import mae, rmse
from src.models.baselines import MLPForecaster, RNNForecaster, TransformerForecaster, PersistenceForecaster
from src.models.kan_baseline import StandardKANForecaster
from src.models.trustkan import TrustKAN
from src.training.engine import set_seed, train_regressor, predict


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_data(cfg, cache_dir=Path("data/raw")):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "cet_mean_station_series.csv"
    if not dest.exists():
        urllib.request.urlretrieve(cfg["dataset"]["source"], dest)
    return dest


def build_model(name, history, horizon):
    if name == "mlp": return MLPForecaster(history, 1, horizon)
    if name == "lstm": return RNNForecaster("lstm", 1, horizon)
    if name == "gru": return RNNForecaster("gru", 1, horizon)
    if name == "transformer": return TransformerForecaster(1, horizon)
    if name == "kan": return StandardKANForecaster(history, 1, horizon)
    if name == "trustkan": return TrustKAN(1, horizon=horizon, hidden_dim=64, grid_size=8)
    raise KeyError(name)


def loader(x, y, batch, shuffle=False):
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def inverse_target(scaler, a):
    a = np.asarray(a)
    shape = a.shape
    return scaler.scaler.inverse_transform(a.reshape(-1, 1)).reshape(shape)


def main(config):
    cfg = load_config(config)
    path = ensure_data(cfg)
    d = pd.read_csv(path, parse_dates=[cfg["dataset"]["date_column"]])
    s = d[cfg["dataset"]["target_column"]].dropna()
    s = s[s >= cfg["dataset"]["min_valid_temperature"]].astype(float)
    values = s.values

    split = chronological_split(
        len(values),
        cfg["split"]["train"],
        cfg["split"]["validation"],
        cfg["split"]["calibration"],
    )
    scaler = TrainOnlyStandardizer().fit(values[split.train])
    z = scaler.transform(values)
    rows = []
    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    for horizon in cfg["window"]["horizons"]:
        X, y, origins = sliding_windows(z, cfg["window"]["history"], horizon)
        masks = assign_windows_by_target_origin(origins, split)
        sets = {k: (X[m], y[m]) for k, m in masks.items()}

        # Ensure every target timestamp remains inside its declared region.
        for k, sl in [("train", split.train), ("val", split.val), ("calibration", split.calibration), ("test", split.test)]:
            valid = origins[masks[k]] + horizon <= sl.stop
            sets[k] = (sets[k][0][valid], sets[k][1][valid])

        batch = cfg["training"]["batch_size"]
        test_x, test_y = sets["test"]
        p_std = PersistenceForecaster(horizon).predict(test_x).numpy()
        p = inverse_target(scaler, p_std)
        test_y_raw = inverse_target(scaler, test_y)
        rows.append({
            "model": "persistence", "horizon": horizon, "seed": -1,
            "rmse": rmse(test_y_raw, p), "mae": mae(test_y_raw, p),
        })

        for name in [m for m in cfg["models"] if m != "persistence"]:
            for seed in cfg["training"]["seeds"]:
                set_seed(seed)
                model = build_model(name, cfg["window"]["history"], horizon)
                tr = loader(*sets["train"], batch, True)
                va = loader(*sets["val"], batch)
                te = loader(*sets["test"], batch)
                model, hist, seconds = train_regressor(
                    model, tr, va,
                    epochs=cfg["training"]["epochs"],
                    lr=cfg["training"]["learning_rate"],
                    patience=cfg["training"]["patience"],
                )
                pred_std, true_std = predict(model, te)
                pred = inverse_target(scaler, pred_std)
                true = inverse_target(scaler, true_std)
                np.savez_compressed(
                    outdir / f"cet_{name}_h{horizon}_s{seed}.npz",
                    prediction=pred, target=true,
                )
                rows.append({
                    "model": name, "horizon": horizon, "seed": seed,
                    "rmse": rmse(true, pred), "mae": mae(true, pred),
                    "train_seconds": seconds,
                    "parameters": sum(p.numel() for p in model.parameters()),
                })

    result = pd.DataFrame(rows)
    Path("results/aggregated").mkdir(parents=True, exist_ok=True)
    result.to_csv("results/aggregated/cet_runs.csv", index=False)
    summary = result.groupby(["model", "horizon"], as_index=False).agg(
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
    )
    summary.to_csv("results/aggregated/cet_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cet.yaml")
    main(ap.parse_args().config)
