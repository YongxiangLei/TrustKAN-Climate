"""Target-calibrated TrustKAN reliability experiments on the CET series."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401
from scripts.run_cet_benchmark import (
    atomic_savez,
    collect_current_records,
    config_sha256,
    ensure_data,
    environment,
    inverse_target,
    load_config,
    loader,
    prepare_series,
    timed_prediction,
    validate_config,
)
from scripts.run_ghcn_reliability import (
    atomic_csv,
    atomic_json,
    atomic_yaml,
    record_from_metrics,
    reference_subset,
    resumable_record,
    _json_safe,
)
from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)
from src.models.trustkan import TrustKAN
from src.reliability.evaluation import evaluate_calibrated_reliability, inverse_standardized
from src.training.engine import resolve_device, set_seed
from src.training.trust_engine import predict_trustkan, train_trustkan
from src.data.provenance import file_sha256


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


def validate_reliability_config(cfg, config_path):
    run_name = validate_config(cfg, config_path)
    levels = cfg["model"]["quantiles"]
    if (
        len(levels) < 2
        or not all(0.0 < float(level) < 1.0 for level in levels)
        or any(right <= left for left, right in zip(levels, levels[1:]))
    ):
        raise ValueError("model.quantiles must be strictly increasing")
    alpha = float(cfg["conformal"]["alpha"])
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal.alpha must lie strictly between zero and one")
    if not np.isclose(levels[0], alpha / 2.0) or not np.isclose(
        levels[-1], 1.0 - alpha / 2.0
    ):
        raise ValueError("Outer model quantiles must match alpha/2 and 1-alpha/2")
    weights = cfg["reliability"]["fusion_weights"]
    if (
        len(weights) != 2
        or not np.isfinite(weights).all()
        or min(weights) < 0
        or sum(weights) <= 0
    ):
        raise ValueError("reliability.fusion_weights must define width and shift weights")
    return run_name


def validate_execution_filters(cfg, *, horizons=None, seeds=None):
    available = {
        "horizon": set(cfg["window"]["horizons"]),
        "seed": set(cfg["training"]["seeds"]),
    }
    requested = {"horizon": horizons, "seed": seeds}
    for name, values in requested.items():
        if values is None:
            continue
        unknown = set(values) - available[name]
        if unknown:
            raise ValueError(f"Unknown {name} execution filters: {sorted(unknown)}")


def build_sets(values, dates, cfg, horizon):
    split = chronological_split(
        len(values),
        cfg["split"]["train"],
        cfg["split"]["validation"],
        cfg["split"]["calibration"],
    )
    scaler = TrainOnlyStandardizer().fit(values[split.train])
    standardized = scaler.transform(values)
    expected_step = pd.to_timedelta(cfg["dataset"]["frequency"]).to_timedelta64()
    x, y, origins = sliding_windows(
        standardized,
        cfg["window"]["history"],
        horizon,
        timestamps=dates,
        expected_step=expected_step,
    )
    masks = assign_windows_by_target_origin(origins, split, horizon)
    sets = {name: (x[mask], y[mask]) for name, mask in masks.items()}
    for name, arrays in sets.items():
        if len(arrays[0]) == 0:
            raise ValueError(f"CET reliability split {name!r} is empty at horizon {horizon}")
    return {
        "scaler": scaler,
        "sets": sets,
        "calibration_target_raw": inverse_target(scaler, sets["calibration"][1]),
        "calibration_origins": origins[masks["calibration"]],
        "calibration_times": np.stack(
            [dates[origin : origin + horizon] for origin in origins[masks["calibration"]]]
        ),
        "test_target_raw": inverse_target(scaler, sets["test"][1]),
        "test_origins": origins[masks["test"]],
        "test_times": np.stack(
            [dates[origin : origin + horizon] for origin in origins[masks["test"]]]
        ),
    }


def main(
    config,
    resume=False,
    *,
    horizons=None,
    seeds=None,
    collect_only=False,
    defer_collection=False,
    device=None,
):
    if collect_only and defer_collection:
        raise ValueError("--collect-only and --defer-collection are mutually exclusive")
    cfg = load_config(config)
    run_name = validate_reliability_config(cfg, config)
    validate_execution_filters(cfg, horizons=horizons, seeds=seeds)
    resolved_device = resolve_device(device)
    cfg_hash = config_sha256(cfg)
    source_hash = code_sha256()
    data_hash = str(cfg["dataset"].get("sha256", "")).lower()
    path = ensure_data(cfg)
    values, dates = prepare_series(cfg, path)
    raw_dir = Path("results/reliability/raw") / run_name
    record_dir = Path("results/reliability/runs") / run_name
    aggregate_dir = Path("results/reliability/aggregated")
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    atomic_yaml(aggregate_dir / f"{run_name}_config.yaml", cfg)

    selected_rows = []
    quantiles = tuple(float(value) for value in cfg["model"]["quantiles"])
    for horizon in ([] if collect_only else cfg["window"]["horizons"]):
        if horizons is not None and horizon not in horizons:
            continue
        bundle = build_sets(values, dates, cfg, int(horizon))
        for seed in cfg["training"]["seeds"]:
            if seeds is not None and seed not in seeds:
                continue
            stem = f"cet_trustkan_reliability_h{horizon}_s{seed}"
            artifact = raw_dir / f"{stem}.npz"
            record = record_dir / f"{stem}.json"
            set_seed(
                seed,
                deterministic=cfg["training"]["deterministic_algorithms"],
                warn_only=cfg["training"]["deterministic_warn_only"],
            )
            run_environment = environment(resolved_device)
            if resume:
                completed = resumable_record(
                    record, artifact, cfg_hash, source_hash, data_hash
                )
                if completed is not None:
                    selected_rows.append(completed)
                    print(f"RESUME {stem}")
                    continue
            row = {
                "dataset": cfg["dataset"]["name"],
                "model": "trustkan",
                "horizon": int(horizon),
                "seed": int(seed),
                "split": "test",
                "status": "ok",
                "nominal_coverage": 1.0 - float(cfg["conformal"]["alpha"]),
                "n_train": int(len(bundle["sets"]["train"][0])),
                "n_validation": int(len(bundle["sets"]["val"][0])),
                "n_calibration": int(len(bundle["sets"]["calibration"][0])),
                "n_test": int(len(bundle["sets"]["test"][0])),
                "parameters": np.nan,
                "train_seconds": np.nan,
                "inference_ms": np.nan,
                "config_sha256": cfg_hash,
                "code_sha256": source_hash,
                "dataset_sha256": data_hash,
                "artifact_sha256": None,
                "artifact_path": artifact.as_posix(),
                "record_path": record.as_posix(),
                "requested_device": str(device),
                **run_environment,
            }
            try:
                train_loader = loader(*bundle["sets"]["train"], cfg["training"]["batch_size"], True)
                val_loader = loader(*bundle["sets"]["val"], cfg["training"]["batch_size"])
                cal_loader = loader(*bundle["sets"]["calibration"], cfg["training"]["batch_size"])
                test_loader = loader(*bundle["sets"]["test"], cfg["training"]["batch_size"])
                reference = reference_subset(
                    bundle["sets"]["train"], cfg["reliability"]["embedding_reference_max"]
                )
                reference_loader = loader(*reference, cfg["training"]["batch_size"])
                model = TrustKAN(
                    1,
                    horizon=int(horizon),
                    hidden_dim=cfg["model"]["hidden_dim"],
                    grid_size=cfg["model"]["grid_size"],
                    quantiles=quantiles,
                    stem=cfg["model"].get("stem", "local"),
                    readout=cfg["model"].get("readout", "last"),
                    history=int(cfg["window"]["history"]),
                )
                model, history, train_seconds = train_trustkan(
                    model,
                    train_loader,
                    val_loader,
                    epochs=cfg["training"]["epochs"],
                    lr=cfg["training"]["learning_rate"],
                    patience=cfg["training"]["patience"],
                    point_weight=cfg["training"]["point_weight"],
                    quantile_weight=cfg["training"]["quantile_weight"],
                    weight_decay=cfg["training"]["weight_decay"],
                    device=resolved_device,
                )
                reference_prediction = predict_trustkan(model, reference_loader, resolved_device)
                calibration_prediction = predict_trustkan(model, cal_loader, resolved_device)
                test_prediction, inference_ms = timed_prediction(
                    lambda: predict_trustkan(model, test_loader, resolved_device),
                    len(bundle["sets"]["test"][0]),
                    resolved_device,
                )
                if not np.allclose(
                    inverse_standardized(bundle["scaler"], calibration_prediction["target"]),
                    bundle["calibration_target_raw"],
                    rtol=1e-6,
                    atol=5e-6,
                ):
                    raise ValueError("Calibration target inversion does not match CET windows")
                metrics, arrays, calibration_state = evaluate_calibrated_reliability(
                    calibration_prediction,
                    test_prediction,
                    reference_prediction["embedding"],
                    bundle["scaler"],
                    quantile_levels=quantiles,
                    alpha=cfg["conformal"]["alpha"],
                    reliability_weights=cfg["reliability"]["fusion_weights"],
                    weight_selection=cfg["reliability"].get(
                        "weight_selection", "frozen"
                    ),
                    min_coverage=cfg["reliability"]["min_calibration_coverage"],
                    error_quantile=cfg["reliability"]["error_quantile"],
                )
                atomic_savez(
                    artifact,
                    **arrays,
                    target=arrays["test_target"],
                    prediction=arrays["test_prediction"],
                    target_time=bundle["test_times"],
                    target_origin=bundle["test_origins"],
                    calibration_target_time=bundle["calibration_times"],
                    calibration_target_origin=bundle["calibration_origins"],
                    scaler_mean=np.asarray(bundle["scaler"].scaler.mean_),
                    scaler_scale=np.asarray(bundle["scaler"].scaler.scale_),
                    quantile_levels=np.asarray(quantiles),
                    metrics_json=np.asarray(json.dumps(_json_safe(metrics), sort_keys=True)),
                    dataset=np.asarray(cfg["dataset"]["name"]),
                    model=np.asarray("trustkan"),
                    horizon=np.asarray(horizon),
                    seed=np.asarray(seed),
                    split=np.asarray("test"),
                    config_sha256=np.asarray(cfg_hash),
                    code_sha256=np.asarray(source_hash),
                    dataset_sha256=np.asarray(data_hash),
                    training_history_json=np.asarray(json.dumps(_json_safe(history))),
                    environment_json=np.asarray(json.dumps(run_environment, sort_keys=True)),
                )
                row.update(
                    parameters=sum(parameter.numel() for parameter in model.parameters()),
                    train_seconds=train_seconds,
                    inference_ms=inference_ms,
                )
                record_from_metrics(row, metrics, calibration_state)
                row["artifact_sha256"] = file_sha256(artifact)
                print(
                    f"OK CET reliability h={horizon} seed={seed} "
                    f"coverage={row['conformal_marginal_coverage']:.3f}"
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED CET reliability h={horizon} seed={seed}: {exc}")
            atomic_json(record, row)
            selected_rows.append(_json_safe(row))

    if not collect_only:
        if not selected_rows:
            raise RuntimeError("No CET reliability jobs matched the execution filters")
        failed = [row for row in selected_rows if row["status"] != "ok"]
        if failed:
            raise RuntimeError(f"{len(failed)} CET reliability runs failed")
        if defer_collection:
            return selected_rows
    rows = collect_current_records(record_dir, cfg_hash, source_hash)
    if not rows:
        raise RuntimeError("No current CET reliability records are available to collect")
    result = pd.DataFrame(rows)
    atomic_csv(result, aggregate_dir / f"{run_name}_runs.csv")
    ok = result[result.status.eq("ok")]
    if not ok.empty:
        summary = ok.groupby(["dataset", "horizon"], as_index=False).agg(
            n=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            marginal_coverage_mean=("conformal_marginal_coverage", "mean"),
            fused_aurc_mean=("fused_aurc", "mean"),
        )
        atomic_csv(summary, aggregate_dir / f"{run_name}_summary.csv")
        print(summary.to_string(index=False))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cet_reliability_smoke.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--defer-collection", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.resume,
        horizons=args.horizons,
        seeds=args.seeds,
        collect_only=args.collect_only,
        defer_collection=args.defer_collection,
        device=args.device,
    )
