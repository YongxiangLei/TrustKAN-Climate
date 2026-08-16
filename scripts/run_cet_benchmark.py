"""Leakage-safe CET benchmark with auditable per-run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution
from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)
from src.metrics.forecast import mae, rmse
from src.models.advanced_baselines import MambaForecaster, TCNForecaster
from src.models.baselines import (
    MLPForecaster,
    PersistenceForecaster,
    RNNForecaster,
    TransformerForecaster,
)
from src.models.classical import make_random_forest, make_svr, make_xgboost
from src.models.kan_baseline import StandardKANForecaster
from src.models.tem2kan import Tem2KANReference
from src.models.trustkan import TrustKAN
from src.training.engine import predict, set_seed, train_regressor


NEURAL = {"mlp", "lstm", "gru", "transformer", "tcn", "mamba", "kan", "tem2kan", "trustkan"}
CLASSICAL = {"svr", "random_forest", "xgboost"}
DETERMINISTIC = {"persistence", "svr"}
RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def config_sha256(cfg):
    canonical = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root / "src").rglob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_data(cfg, cache_dir=Path("data/raw")):
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "cet_mean_station_series.csv"
    if not dest.exists():
        urllib.request.urlretrieve(cfg["dataset"]["source"], dest)
    expected = cfg["dataset"].get("sha256")
    if expected:
        observed = file_sha256(dest)
        if observed.lower() != str(expected).lower():
            raise ValueError(
                f"Dataset checksum mismatch for {dest}: expected {expected}, observed {observed}"
            )
    return dest


def validate_config(cfg, config_path):
    run_name = cfg.get("experiment", {}).get("name", Path(config_path).stem)
    if not RUN_NAME.fullmatch(run_name):
        raise ValueError("experiment.name may contain only letters, numbers, '.', '_' and '-'")
    fractions = cfg["split"]
    implied_test = 1.0 - fractions["train"] - fractions["validation"] - fractions["calibration"]
    if "test" in fractions and not np.isclose(fractions["test"], implied_test):
        raise ValueError("split.test does not match the remaining chronological fraction")
    return run_name


def prepare_series(cfg, path):
    date_col = cfg["dataset"]["date_column"]
    target_col = cfg["dataset"]["target_column"]
    frame = pd.read_csv(path, usecols=[date_col, target_col])
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce", utc=True)
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    valid = frame[date_col].notna() & frame[target_col].notna()
    valid &= frame[target_col] >= cfg["dataset"]["min_valid_temperature"]
    frame = frame.loc[valid, [date_col, target_col]].sort_values(date_col)
    if frame[date_col].duplicated().any():
        raise ValueError("CET dataset contains duplicate valid timestamps")
    max_observations = cfg["dataset"].get("max_observations")
    if max_observations is not None:
        if not isinstance(max_observations, int) or max_observations <= 0:
            raise ValueError("dataset.max_observations must be a positive integer")
        frame = frame.tail(max_observations)
    return frame[target_col].to_numpy(float), frame[date_col].to_numpy(dtype="datetime64[ns]")


def build_model(name, history, horizon, seed=None, options=None):
    options = options or {}
    if name == "mlp":
        return MLPForecaster(history, 1, horizon)
    if name == "lstm":
        return RNNForecaster("lstm", 1, horizon)
    if name == "gru":
        return RNNForecaster("gru", 1, horizon)
    if name == "transformer":
        return TransformerForecaster(1, horizon)
    if name == "tcn":
        return TCNForecaster(1, horizon)
    if name == "mamba":
        return MambaForecaster(1, horizon)
    if name == "kan":
        return StandardKANForecaster(history, 1, horizon)
    if name == "tem2kan":
        return Tem2KANReference(history, 1, horizon, seed=1 if seed is None else seed)
    if name == "trustkan":
        return TrustKAN(1, horizon=horizon, hidden_dim=64, grid_size=8)
    if name == "svr":
        return make_svr(**options)
    if name == "random_forest":
        return make_random_forest(random_state=0 if seed is None else seed, **options)
    if name == "xgboost":
        return make_xgboost(random_state=0 if seed is None else seed, **options)
    raise KeyError(f"Unknown model: {name}")


def loader(x, y, batch, shuffle=False):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle)


def inverse_target(scaler, values):
    values = np.asarray(values)
    shape = values.shape
    return scaler.scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def environment():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def timed_prediction(function, n_samples):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    prediction = function()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return prediction, elapsed * 1000.0 / max(1, n_samples)


def select_classical_model(name, history, horizon, seed, train_set, val_set, candidates):
    if not isinstance(candidates, list) or not candidates or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError(f"model_search.{name} must be a non-empty list of mappings")
    search = []
    selected = None
    for index, options in enumerate(candidates):
        model = build_model(name, history, horizon, seed, options)
        start = time.perf_counter()
        model.fit(*train_set)
        fit_seconds = time.perf_counter() - start
        val_prediction = model.predict(val_set[0])
        val_rmse = rmse(val_set[1], val_prediction)
        search.append(
            {
                "candidate": index,
                "hyperparameters": options,
                "validation_rmse": val_rmse,
                "fit_seconds": fit_seconds,
            }
        )
        if selected is None or (val_rmse, index) < (selected[0], selected[1]):
            selected = (val_rmse, index, model, options, fit_seconds)
    validation_rmse, _, model, options, fit_seconds = selected
    return model, options, search, validation_rmse, fit_seconds


def atomic_savez(path, **arrays):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({key: json_safe(value) for key, value in payload.items()}, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resumable_record(path, artifact, cfg_hash, source_hash):
    path = Path(path)
    if not path.is_file() or not Path(artifact).is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        row = json.load(handle)
    if (
        row.get("status") != "ok"
        or row.get("config_sha256") != cfg_hash
        or row.get("code_sha256") != source_hash
    ):
        return None
    return row


def main(config, resume=False):
    cfg = load_config(config)
    run_name = validate_config(cfg, config)
    cfg_hash = config_sha256(cfg)
    source_hash = code_sha256()
    path = ensure_data(cfg)
    values, dates = prepare_series(cfg, path)
    split = chronological_split(
        len(values),
        cfg["split"]["train"],
        cfg["split"]["validation"],
        cfg["split"]["calibration"],
    )
    scaler = TrainOnlyStandardizer().fit(values[split.train])
    standardized = scaler.transform(values)

    raw_dir = Path("results/raw") / run_name
    record_dir = Path("results/runs") / run_name
    aggregated_dir = Path("results/aggregated")
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    with open(aggregated_dir / f"{run_name}_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    rows = []
    expected_step = pd.to_timedelta(cfg["dataset"]["frequency"]).to_timedelta64()
    for horizon in cfg["window"]["horizons"]:
        x, y, origins = sliding_windows(
            standardized,
            cfg["window"]["history"],
            horizon,
            timestamps=dates,
            expected_step=expected_step,
        )
        masks = assign_windows_by_target_origin(origins, split, horizon)
        sets = {name: (x[mask], y[mask]) for name, mask in masks.items()}
        test_x, test_y = sets["test"]
        test_y_raw = inverse_target(scaler, test_y)
        test_origins = origins[masks["test"]]
        target_times = np.stack([dates[origin : origin + horizon] for origin in test_origins])

        for name in cfg["models"]:
            seeds = [-1] if name in DETERMINISTIC else cfg["training"]["seeds"]
            for seed in seeds:
                artifact = raw_dir / f"cet_{name}_h{horizon}_s{seed}.npz"
                record = record_dir / f"cet_{name}_h{horizon}_s{seed}.json"
                if resume:
                    completed = resumable_record(record, artifact, cfg_hash, source_hash)
                    if completed is not None:
                        rows.append(completed)
                        print(f"RESUME {name} h={horizon} seed={seed}")
                        continue
                row = {
                    "dataset": cfg["dataset"]["name"],
                    "model": name,
                    "horizon": horizon,
                    "seed": seed,
                    "split": "test",
                    "status": "ok",
                    "rmse": np.nan,
                    "mae": np.nan,
                    "parameters": np.nan,
                    "train_seconds": np.nan,
                    "inference_ms": np.nan,
                    "validation_rmse": np.nan,
                    "search_seconds": 0.0,
                    "selected_hyperparameters": "{}",
                    "n_test": len(test_x),
                    "config_sha256": cfg_hash,
                    "code_sha256": source_hash,
                    "artifact_path": artifact.as_posix(),
                    "record_path": record.as_posix(),
                    **environment(),
                }
                history = []
                model_selection = []
                selected_hyperparameters = {}
                validation_rmse = np.nan
                search_seconds = 0.0
                try:
                    start = time.perf_counter()
                    if name == "persistence":
                        model = PersistenceForecaster(horizon)
                        pred_std, latency = timed_prediction(
                            lambda: model.predict(test_x).numpy(), len(test_x)
                        )
                        seconds = 0.0
                        params = 0
                    elif name in CLASSICAL:
                        if seed >= 0:
                            set_seed(seed)
                        candidates = cfg.get("model_search", {}).get(name, [{}])
                        model, selected_hyperparameters, model_selection, validation_rmse, seconds = (
                            select_classical_model(
                                name,
                                cfg["window"]["history"],
                                horizon,
                                seed,
                                sets["train"],
                                sets["val"],
                                candidates,
                            )
                        )
                        search_seconds = time.perf_counter() - start
                        pred_std, latency = timed_prediction(lambda: model.predict(test_x), len(test_x))
                        params = np.nan
                    elif name in NEURAL:
                        set_seed(seed)
                        model = build_model(name, cfg["window"]["history"], horizon, seed)
                        train_loader = loader(*sets["train"], cfg["training"]["batch_size"], True)
                        val_loader = loader(*sets["val"], cfg["training"]["batch_size"])
                        test_loader = loader(*sets["test"], cfg["training"]["batch_size"])
                        model, history, seconds = train_regressor(
                            model,
                            train_loader,
                            val_loader,
                            epochs=cfg["training"]["epochs"],
                            lr=cfg["training"]["learning_rate"],
                            patience=cfg["training"]["patience"],
                            optimizer_name=cfg["training"].get("optimizer", "adamw"),
                            weight_decay=cfg["training"].get("weight_decay"),
                        )
                        predicted, latency = timed_prediction(lambda: predict(model, test_loader), len(test_x))
                        pred_std, _ = predicted
                        params = sum(parameter.numel() for parameter in model.parameters())
                    else:
                        raise KeyError(name)

                    prediction = inverse_target(scaler, pred_std)
                    atomic_savez(
                        artifact,
                        prediction=prediction,
                        target=test_y_raw,
                        target_time=target_times,
                        target_origin=test_origins,
                        dataset=np.asarray(cfg["dataset"]["name"]),
                        model=np.asarray(name),
                        horizon=np.asarray(horizon),
                        seed=np.asarray(seed),
                        split=np.asarray("test"),
                        config_sha256=np.asarray(cfg_hash),
                        code_sha256=np.asarray(source_hash),
                        training_history_json=np.asarray(json.dumps(history)),
                        model_selection_json=np.asarray(json.dumps(model_selection)),
                        selected_hyperparameters_json=np.asarray(json.dumps(selected_hyperparameters)),
                    )
                    row.update(
                        rmse=rmse(test_y_raw, prediction),
                        mae=mae(test_y_raw, prediction),
                        train_seconds=seconds,
                        inference_ms=latency,
                        parameters=params,
                        validation_rmse=validation_rmse,
                        search_seconds=search_seconds,
                        selected_hyperparameters=json.dumps(selected_hyperparameters, sort_keys=True),
                    )
                except Exception as exc:  # preserve failures in the run ledger
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {name} h={horizon} seed={seed}: {exc}")
                atomic_write_json(record, row)
                rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(aggregated_dir / f"{run_name}_runs.csv", index=False)
    ok = result[result.status == "ok"]
    summary = ok.groupby(["dataset", "model", "horizon"], as_index=False).agg(
        n=("seed", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_ms_mean=("inference_ms", "mean"),
    )
    summary.to_csv(aggregated_dir / f"{run_name}_summary.csv", index=False)
    print(summary.to_string(index=False))
    with open(aggregated_dir / f"{run_name}_environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment(), handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cet.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful records only when their config hash and artifact match.",
    )
    args = parser.parse_args()
    main(args.config, args.resume)
