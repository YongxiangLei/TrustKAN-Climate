"""Run auditable within-station and leave-one-region-out GHCN benchmarks."""
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
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution
from scripts.prepare_ghcn_panel import code_sha256 as preparation_code_sha256
from scripts.run_cet_benchmark import (
    CLASSICAL,
    DETERMINISTIC,
    NEURAL,
    atomic_savez,
    atomic_write_json,
    build_model,
    environment,
    inverse_target,
    loader,
    select_classical_model,
    timed_prediction,
)
from src.data.provenance import file_sha256
from src.experiments.ghcn import (
    StationSeries,
    build_evaluation_specs,
    build_station_windows,
)
from src.metrics.forecast import mae, rmse
from src.models.baselines import PersistenceForecaster
from src.training.engine import predict, set_seed, train_regressor


RUN_NAME = re.compile(r"^[A-Za-z0-9._-]+$")


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
    protocols = cfg.get("protocols", [])
    if not protocols:
        raise ValueError("At least one GHCN evaluation protocol is required")
    fractions = cfg["split"]
    implied_test = 1.0 - fractions["train"] - fractions["validation"] - fractions["calibration"]
    if "test" in fractions and not np.isclose(fractions["test"], implied_test):
        raise ValueError("split.test does not match the remaining chronological fraction")
    return run_name


def _scalar(artifact, name):
    value = artifact[name]
    if value.shape != ():
        raise ValueError(f"Prepared metadata field {name} must be scalar")
    return str(value)


def load_station_panel(cfg):
    panel_path = Path(cfg["dataset"]["panel_config"])
    panel = load_config(panel_path)
    prepared_dir = Path(cfg["dataset"]["prepared_dir"])
    expected_config_hash = file_sha256(panel_path)
    expected_preparation_hash = preparation_code_sha256()
    selected_regions = cfg["dataset"].get("station_regions")
    if selected_regions is not None:
        unknown = set(selected_regions) - {item["region"] for item in panel["stations"]}
        if unknown:
            raise ValueError(f"Unknown station_regions: {sorted(unknown)}")
    series = []
    for expected in panel["stations"]:
        if selected_regions is not None and expected["region"] not in selected_regions:
            continue
        artifact_path = prepared_dir / f'{expected["region"]}_{expected["station_id"]}.npz'
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Missing prepared station artifact {artifact_path}; run "
                "python scripts/prepare_ghcn_panel.py first"
            )
        with np.load(artifact_path, allow_pickle=False) as artifact:
            required = {
                "date",
                "tmax",
                "tmin",
                "target",
                "region",
                "station_id",
                "raw_sha256",
                "frozen_config_sha256",
                "preparation_code_sha256",
            }
            missing = required - set(artifact.files)
            if missing:
                raise ValueError(
                    f"Prepared station artifact {artifact_path} is missing {sorted(missing)}"
                )
            metadata = {
                "region": expected["region"],
                "station_id": expected["station_id"],
                "raw_sha256": expected["raw_sha256"],
                "frozen_config_sha256": expected_config_hash,
                "preparation_code_sha256": expected_preparation_hash,
            }
            for field, expected_value in metadata.items():
                observed = _scalar(artifact, field)
                if observed != str(expected_value):
                    raise ValueError(
                        f"Prepared {field} mismatch for {artifact_path}: "
                        f"expected {expected_value}, got {observed}"
                    )
            dates = artifact["date"].copy()
            tmax = artifact["tmax"].astype(float)
            tmin = artifact["tmin"].astype(float)
            target = artifact["target"].astype(float)
        if not (len(dates) == len(tmax) == len(tmin) == len(target)):
            raise ValueError(f"Prepared arrays are misaligned in {artifact_path}")
        if not np.isfinite(tmax).all() or not np.isfinite(tmin).all():
            raise ValueError(f"Prepared extrema contain non-finite values in {artifact_path}")
        if np.any(tmax < tmin) or not np.allclose(target, (tmax + tmin) / 2.0, atol=1e-5):
            raise ValueError(f"Prepared target identity failed for {artifact_path}")
        series.append(
            StationSeries(
                expected["region"],
                expected["station_id"],
                dates,
                target,
                expected["raw_sha256"],
            ).validate()
        )
    if not series:
        raise ValueError("No GHCN stations selected")
    return panel_path, panel, series


def dataset_sha256(spec, panel_hash):
    digest = hashlib.sha256()
    for value in (
        panel_hash,
        spec.protocol,
        spec.target_region,
        spec.target_station,
        *spec.source_regions,
        *spec.source_stations,
        spec.source_pooling,
        *spec.raw_hashes,
    ):
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
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


def run_spec(cfg, spec, horizon, paths, hashes, resume):
    raw_dir, record_dir = paths
    cfg_hash, source_hash, panel_hash = hashes
    data_hash = dataset_sha256(spec, panel_hash)
    rows = []
    train_set = spec.train_set
    val_set = spec.validation_set
    test_set = spec.test_set
    test_x = test_set[0]
    source_regions_json = json.dumps(spec.source_regions)
    source_stations_json = json.dumps(spec.source_stations)
    for name in cfg["models"]:
        seeds = [-1] if name in DETERMINISTIC else cfg["training"]["seeds"]
        for seed in seeds:
            stem = f"{spec.dataset}_{name}_h{horizon}_s{seed}"
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
                "dataset": spec.dataset,
                "protocol": spec.protocol,
                "target_region": spec.target_region,
                "target_station": spec.target_station,
                "source_regions": source_regions_json,
                "source_stations": source_stations_json,
                "source_pooling": spec.source_pooling,
                "normalization": "per_station_training_period",
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
                "n_train": len(train_set[0]),
                "n_validation": len(val_set[0]),
                "n_test": len(test_x),
                "config_sha256": cfg_hash,
                "code_sha256": source_hash,
                "dataset_sha256": data_hash,
                "artifact_sha256": np.nan,
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
                        train_set,
                        val_set,
                        candidates,
                    )
                    search_seconds = time.perf_counter() - start
                    pred_std, latency = timed_prediction(
                        lambda: model.predict(test_x), len(test_x)
                    )
                    params = np.nan
                elif name in NEURAL:
                    set_seed(seed)
                    model = build_model(
                        name, cfg["window"]["history"], horizon, seed
                    )
                    train_loader = loader(
                        *train_set, cfg["training"]["batch_size"], True
                    )
                    val_loader = loader(*val_set, cfg["training"]["batch_size"])
                    test_loader = loader(*test_set, cfg["training"]["batch_size"])
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
                    predicted, latency = timed_prediction(
                        lambda: predict(model, test_loader), len(test_x)
                    )
                    pred_std, _ = predicted
                    params = sum(parameter.numel() for parameter in model.parameters())
                else:
                    raise KeyError(name)

                prediction = inverse_target(spec.target_scaler, pred_std)
                atomic_savez(
                    artifact,
                    prediction=prediction,
                    target=spec.test_target_raw,
                    target_time=spec.test_target_times,
                    target_origin=spec.test_origins,
                    dataset=np.asarray(spec.dataset),
                    model=np.asarray(name),
                    horizon=np.asarray(horizon),
                    seed=np.asarray(seed),
                    split=np.asarray("test"),
                    protocol=np.asarray(spec.protocol),
                    target_region=np.asarray(spec.target_region),
                    target_station=np.asarray(spec.target_station),
                    source_regions_json=np.asarray(source_regions_json),
                    source_stations_json=np.asarray(source_stations_json),
                    source_pooling=np.asarray(spec.source_pooling),
                    normalization=np.asarray("per_station_training_period"),
                    config_sha256=np.asarray(cfg_hash),
                    code_sha256=np.asarray(source_hash),
                    dataset_sha256=np.asarray(data_hash),
                    training_history_json=np.asarray(json.dumps(history)),
                    model_selection_json=np.asarray(json.dumps(model_selection)),
                    selected_hyperparameters_json=np.asarray(
                        json.dumps(selected_hyperparameters)
                    ),
                )
                row.update(
                    rmse=rmse(spec.test_target_raw, prediction),
                    mae=mae(spec.test_target_raw, prediction),
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
                print(
                    f"OK {spec.protocol} target={spec.target_region} {name} "
                    f"h={horizon} seed={seed} RMSE={row['rmse']:.4f}"
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(
                    f"FAILED {spec.protocol} target={spec.target_region} {name} "
                    f"h={horizon} seed={seed}: {exc}"
                )
            atomic_write_json(record, row)
            rows.append(row)
    return rows


def main(config, resume=False):
    cfg = load_config(config)
    run_name = validate_config(cfg, config)
    panel_path, panel, station_series = load_station_panel(cfg)
    cfg_hash = combined_config_sha256(config, panel_path)
    source_hash = code_sha256()
    panel_hash = file_sha256(panel_path)
    raw_dir = Path("results/raw") / run_name
    record_dir = Path("results/runs") / run_name
    aggregated_dir = Path("results/aggregated")
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)
    with open(aggregated_dir / f"{run_name}_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with open(aggregated_dir / f"{run_name}_panel.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(panel, handle, sort_keys=False)

    rows = []
    for horizon in cfg["window"]["horizons"]:
        bundles = [
            build_station_windows(
                series,
                cfg["split"],
                cfg["window"]["history"],
                horizon,
                expected_step=cfg["dataset"].get("frequency", "1D"),
                max_observations=cfg["dataset"].get("max_observations"),
            )
            for series in station_series
        ]
        specs = build_evaluation_specs(bundles, cfg["protocols"])
        for spec in specs:
            rows.extend(
                run_spec(
                    cfg,
                    spec,
                    horizon,
                    (raw_dir, record_dir),
                    (cfg_hash, source_hash, panel_hash),
                    resume,
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(aggregated_dir / f"{run_name}_runs.csv", index=False)
    ok = result[result.status.eq("ok")]
    summary = ok.groupby(
        ["protocol", "dataset", "target_region", "model", "horizon"],
        as_index=False,
    ).agg(
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
    parser.add_argument("--config", default="configs/ghcn_smoke.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only checksum-matched successful records and artifacts.",
    )
    args = parser.parse_args()
    main(args.config, args.resume)
