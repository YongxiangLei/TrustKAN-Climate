"""Leakage-safe hourly Jena benchmark with auditable per-run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from scripts.prepare_jena import code_sha256 as preparation_code_sha256
from scripts.run_cet_benchmark import (
    CLASSICAL,
    DETERMINISTIC,
    NEURAL,
    atomic_savez,
    atomic_write_json,
    build_model,
    environment,
    loader,
    select_classical_model,
    timed_prediction,
)
from src.data.jena import REQUIRED_COLUMNS, TARGET_COLUMN
from src.data.provenance import file_sha256
from src.experiments.jena import build_jena_windows
from src.metrics.forecast import mae, rmse
from src.models.baselines import PersistenceForecaster
from src.training.engine import predict, resolve_device, set_seed, train_regressor


RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
DATASET_NAME = "jena_beutenberg"
N_FEATURES = len(REQUIRED_COLUMNS)


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def combined_config_sha256(config_path, panel_path):
    digest = hashlib.sha256()
    for path in (Path(config_path), Path(panel_path)):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "run_cet_benchmark.py",
        root / "scripts" / "prepare_jena.py",
        *sorted((root / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_config(cfg, config_path):
    run_name = cfg.get("experiment", {}).get("name", Path(config_path).stem)
    if not RUN_NAME.fullmatch(run_name):
        raise ValueError("experiment.name may contain only letters, numbers, '.', '_' and '-'")
    fractions = cfg["split"]
    implied_test = 1.0 - fractions["train"] - fractions["validation"] - fractions["calibration"]
    if "test" in fractions and not np.isclose(fractions["test"], implied_test):
        raise ValueError("split.test does not match the remaining chronological fraction")
    for field in ("deterministic_algorithms", "deterministic_warn_only"):
        if not isinstance(cfg["training"].get(field), bool):
            raise ValueError(f"training.{field} must be explicitly true or false")
    return run_name


def validate_execution_filters(cfg, *, horizons=None, models=None, seeds=None):
    available = {
        "horizon": set(cfg["window"]["horizons"]),
        "model": set(cfg["models"]),
        "seed": set(cfg["training"]["seeds"]) | {-1},
    }
    requested = {"horizon": horizons, "model": models, "seed": seeds}
    for name, values in requested.items():
        if values is None:
            continue
        unknown = set(values) - available[name]
        if unknown:
            raise ValueError(f"Unknown {name} execution filters: {sorted(unknown)}")


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_yaml(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def collect_current_records(record_dir, config_hash, source_hash):
    rows = []
    for path in sorted(Path(record_dir).glob("*.json")):
        with open(path, "r", encoding="utf-8") as handle:
            row = json.load(handle)
        if row.get("config_sha256") == config_hash and row.get("code_sha256") == source_hash:
            rows.append(row)
    return rows


def _scalar(artifact, name):
    value = artifact[name]
    if getattr(value, "shape", ()) != ():
        if np.asarray(value).ndim == 0:
            return str(value)
        raise ValueError(f"Prepared metadata field {name} must be scalar")
    return str(value.item() if hasattr(value, "item") else value)


def load_prepared_jena(cfg):
    panel_path = Path(cfg["dataset"]["panel_config"])
    panel = load_config(panel_path)
    artifact = Path(cfg["dataset"]["prepared_dir"]) / "jena_hourly.npz"
    if not artifact.is_file():
        raise FileNotFoundError(
            f"Prepared Jena artifact not found: {artifact}. "
            "Run python scripts/prepare_jena.py --require-eligible first."
        )
    expected = {
        "config_sha256": file_sha256(panel_path),
        "code_sha256": preparation_code_sha256(),
    }
    with np.load(artifact, allow_pickle=False) as packed:
        required = {"dates", "features", "target", "feature_names", "config_sha256", "code_sha256"}
        missing = required - set(packed.files)
        if missing:
            raise ValueError(f"Prepared Jena artifact is missing {sorted(missing)}")
        for field, expected_value in expected.items():
            observed = _scalar(packed, field)
            if observed != expected_value:
                raise ValueError(
                    f"Prepared {field} mismatch for {artifact}: "
                    f"expected {expected_value}, got {observed}"
                )
        names = [str(item) for item in packed["feature_names"].tolist()]
        if names != list(REQUIRED_COLUMNS):
            raise ValueError(f"Prepared feature_names {names} != {list(REQUIRED_COLUMNS)}")
        dates = packed["dates"].copy()
        features = packed["features"].astype(float)
        target = packed["target"].astype(float)
    if features.ndim != 2 or features.shape[1] != N_FEATURES:
        raise ValueError(f"Prepared features must have width {N_FEATURES}")
    if not (len(dates) == len(features) == len(target)):
        raise ValueError("Prepared Jena arrays are misaligned")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("Prepared Jena arrays contain non-finite values")
    if not np.allclose(features[:, 0], target, atol=1e-6):
        raise ValueError("Prepared target must equal the first feature column T (degC)")
    return panel_path, panel, artifact, dates, features


def dataset_sha256(artifact_hash, panel_hash):
    digest = hashlib.sha256()
    digest.update(str(panel_hash).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(artifact_hash).encode("utf-8"))
    return digest.hexdigest()


def resumable_record(path, artifact, config_hash, source_hash, data_hash):
    path = Path(path)
    if not path.is_file() or not Path(artifact).is_file():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        row = json.load(handle)
    if (
        row.get("status") != "ok"
        or row.get("config_sha256") != config_hash
        or row.get("code_sha256") != source_hash
        or row.get("dataset_sha256") != data_hash
        or row.get("artifact_sha256") != file_sha256(artifact)
    ):
        return None
    return row


def run_horizon(
    cfg,
    bundle,
    horizon,
    paths,
    hashes,
    resume,
    *,
    selected_models=None,
    selected_seeds=None,
    device=None,
):
    raw_dir, record_dir = paths
    cfg_hash, source_hash, data_hash = hashes
    rows = []
    test_x, _ = bundle.sets["test"]
    for name in cfg["models"]:
        if selected_models is not None and name not in selected_models:
            continue
        seeds = [-1] if name in DETERMINISTIC else cfg["training"]["seeds"]
        execution_device = device if name in NEURAL else resolve_device("cpu")
        for seed in seeds:
            if selected_seeds is not None and seed not in selected_seeds:
                continue
            set_seed(
                seed if seed >= 0 else 0,
                deterministic=cfg["training"]["deterministic_algorithms"],
                warn_only=cfg["training"]["deterministic_warn_only"],
            )
            run_environment = environment(execution_device)
            stem = f"jena_{name}_h{horizon}_s{seed}"
            artifact = raw_dir / f"{stem}.npz"
            record = record_dir / f"{stem}.json"
            if resume:
                completed = resumable_record(
                    record, artifact, cfg_hash, source_hash, data_hash
                )
                if completed is not None:
                    rows.append(completed)
                    print(f"RESUME {stem}")
                    continue
            row = {
                "dataset": DATASET_NAME,
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
                "n_train": len(bundle.sets["train"][0]),
                "n_validation": len(bundle.sets["val"][0]),
                "n_test": len(test_x),
                "n_features": N_FEATURES,
                "target": TARGET_COLUMN,
                "config_sha256": cfg_hash,
                "code_sha256": source_hash,
                "dataset_sha256": data_hash,
                "artifact_sha256": np.nan,
                "artifact_path": artifact.as_posix(),
                "record_path": record.as_posix(),
                "requested_device": str(device),
                **run_environment,
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
                        lambda: model.predict(test_x).numpy(),
                        len(test_x),
                        execution_device,
                    )
                    seconds = 0.0
                    params = 0
                elif name in CLASSICAL:
                    (
                        model,
                        selected_hyperparameters,
                        model_selection,
                        validation_rmse,
                        seconds,
                    ) = select_classical_model(
                        name,
                        cfg["window"]["history"],
                        horizon,
                        seed,
                        bundle.sets["train"],
                        bundle.sets["val"],
                        cfg.get("model_search", {}).get(name, [{}]),
                        n_features=N_FEATURES,
                    )
                    search_seconds = time.perf_counter() - start
                    pred_std, latency = timed_prediction(
                        lambda: model.predict(test_x), len(test_x), execution_device
                    )
                    params = np.nan
                elif name in NEURAL:
                    model = build_model(
                        name,
                        cfg["window"]["history"],
                        horizon,
                        seed,
                        n_features=N_FEATURES,
                    )
                    train_loader = loader(*bundle.sets["train"], cfg["training"]["batch_size"], True)
                    val_loader = loader(*bundle.sets["val"], cfg["training"]["batch_size"])
                    test_loader = loader(*bundle.sets["test"], cfg["training"]["batch_size"])
                    model, history, seconds = train_regressor(
                        model,
                        train_loader,
                        val_loader,
                        epochs=cfg["training"]["epochs"],
                        lr=cfg["training"]["learning_rate"],
                        patience=cfg["training"]["patience"],
                        optimizer_name=cfg["training"].get("optimizer", "adamw"),
                        weight_decay=cfg["training"].get("weight_decay"),
                        device=execution_device,
                    )
                    predicted, latency = timed_prediction(
                        lambda: predict(model, test_loader, execution_device),
                        len(test_x),
                        execution_device,
                    )
                    pred_std, _ = predicted
                    params = sum(parameter.numel() for parameter in model.parameters())
                else:
                    raise KeyError(name)

                prediction = bundle.scaler.inverse_column(pred_std, 0)
                atomic_savez(
                    artifact,
                    prediction=prediction,
                    target=bundle.test_target_raw,
                    target_time=bundle.test_target_times,
                    target_origin=bundle.test_origins,
                    dataset=np.asarray(DATASET_NAME),
                    model=np.asarray(name),
                    horizon=np.asarray(horizon),
                    seed=np.asarray(seed),
                    split=np.asarray("test"),
                    n_features=np.asarray(N_FEATURES),
                    config_sha256=np.asarray(cfg_hash),
                    code_sha256=np.asarray(source_hash),
                    dataset_sha256=np.asarray(data_hash),
                    requested_device=np.asarray(str(device)),
                    environment_json=np.asarray(json.dumps(run_environment, sort_keys=True)),
                    training_history_json=np.asarray(json.dumps(history)),
                    model_selection_json=np.asarray(json.dumps(model_selection)),
                    selected_hyperparameters_json=np.asarray(
                        json.dumps(selected_hyperparameters)
                    ),
                )
                row.update(
                    rmse=rmse(bundle.test_target_raw, prediction),
                    mae=mae(bundle.test_target_raw, prediction),
                    train_seconds=seconds,
                    inference_ms=latency,
                    parameters=params,
                    validation_rmse=validation_rmse,
                    search_seconds=search_seconds,
                    selected_hyperparameters=json.dumps(
                        selected_hyperparameters, sort_keys=True
                    ),
                    artifact_sha256=file_sha256(artifact),
                )
                print(f"OK {name} h={horizon} seed={seed} RMSE={row['rmse']:.4f}")
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED {name} h={horizon} seed={seed}: {exc}")
            atomic_write_json(record, row)
            rows.append(row)
    return rows


def write_split_export(bundle, split_dir, horizon, run_name):
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        split_dir / f"{run_name}_h{horizon}_splits.npz",
        x_train=bundle.sets["train"][0],
        y_train=bundle.sets["train"][1],
        x_val=bundle.sets["val"][0],
        y_val=bundle.sets["val"][1],
        x_cal=bundle.sets["calibration"][0],
        y_cal=bundle.sets["calibration"][1],
        x_test=bundle.sets["test"][0],
        y_test=bundle.sets["test"][1],
        train_target=bundle.train_target_raw,
        test_target=bundle.test_target_raw,
        test_origin=bundle.test_origins,
    )
    np.savez_compressed(
        split_dir / f"{run_name}_train_target.npz",
        target=bundle.train_target_raw,
        train_target=bundle.train_target_raw,
    )


def main(
    config,
    resume=False,
    *,
    horizons=None,
    models=None,
    seeds=None,
    collect_only=False,
    defer_collection=False,
    device=None,
    results_root="results",
):
    if collect_only and defer_collection:
        raise ValueError("--collect-only and --defer-collection are mutually exclusive")
    cfg = load_config(config)
    run_name = validate_config(cfg, config)
    panel_path, panel, artifact, dates, features = load_prepared_jena(cfg)
    validate_execution_filters(cfg, horizons=horizons, models=models, seeds=seeds)
    resolved_device = resolve_device(device)
    cfg_hash = combined_config_sha256(config, panel_path)
    source_hash = code_sha256()
    data_hash = dataset_sha256(file_sha256(artifact), file_sha256(panel_path))
    root = Path(results_root)
    raw_dir = root / "raw" / run_name
    record_dir = root / "runs" / run_name
    aggregated_dir = root / "aggregated"
    split_dir = root / "splits"
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    atomic_yaml(aggregated_dir / f"{run_name}_config.yaml", cfg)
    atomic_yaml(aggregated_dir / f"{run_name}_panel.yaml", panel)

    selected_rows = []
    for horizon in ([] if collect_only else cfg["window"]["horizons"]):
        if horizons is not None and horizon not in horizons:
            continue
        bundle = build_jena_windows(
            dates,
            features,
            cfg["split"],
            cfg["window"]["history"],
            int(horizon),
            expected_step=cfg["dataset"].get("frequency", "1h"),
            max_observations=cfg["dataset"].get("max_observations"),
        )
        write_split_export(bundle, split_dir, horizon, run_name)
        selected_rows.extend(
            run_horizon(
                cfg,
                bundle,
                int(horizon),
                (raw_dir, record_dir),
                (cfg_hash, source_hash, data_hash),
                resume,
                selected_models=models,
                selected_seeds=seeds,
                device=resolved_device,
            )
        )
    if not collect_only:
        if not selected_rows:
            raise RuntimeError("No Jena jobs matched the execution filters")
        failed = [row for row in selected_rows if row["status"] != "ok"]
        if failed:
            raise RuntimeError(f"{len(failed)} Jena runs failed; inspect their run records")
        if defer_collection:
            return selected_rows
    rows = collect_current_records(record_dir, cfg_hash, source_hash)
    if not rows:
        raise RuntimeError("No current Jena records are available to collect")
    result = pd.DataFrame(rows)
    atomic_csv(result, aggregated_dir / f"{run_name}_runs.csv")
    ok = result[result.status.eq("ok")]
    summary = ok.groupby(["dataset", "model", "horizon"], as_index=False).agg(
        n=("seed", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_ms_mean=("inference_ms", "mean"),
    )
    atomic_csv(summary, aggregated_dir / f"{run_name}_summary.csv")
    print(summary.to_string(index=False))
    with open(aggregated_dir / f"{run_name}_environment.json", "w", encoding="utf-8") as handle:
        json.dump(environment(resolved_device), handle, indent=2)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jena_smoke.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--defer-collection", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.resume,
        horizons=args.horizons,
        models=args.models,
        seeds=args.seeds,
        collect_only=args.collect_only,
        defer_collection=args.defer_collection,
        device=args.device,
    )
