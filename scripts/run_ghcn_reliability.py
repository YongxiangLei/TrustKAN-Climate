"""Run target-calibrated TrustKAN reliability experiments on frozen GHCN tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import atomic_savez, environment, loader, timed_prediction
from scripts.run_ghcn_benchmark import (
    combined_config_sha256,
    dataset_sha256,
    load_config,
    load_station_panel,
    validate_config,
)
from src.data.provenance import file_sha256
from src.experiments.ghcn import build_evaluation_specs, build_station_windows
from src.models.trustkan import TrustKAN
from src.reliability.evaluation import (
    evaluate_calibrated_reliability,
    inverse_standardized,
)
from src.training.engine import resolve_device, set_seed
from src.training.trust_engine import predict_trustkan, train_trustkan


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "run_ghcn_benchmark.py",
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
    min_coverage = float(cfg["reliability"]["min_calibration_coverage"])
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("reliability.min_calibration_coverage must lie in (0, 1]")
    reference_max = cfg["reliability"]["embedding_reference_max"]
    if not isinstance(reference_max, int) or reference_max < 2:
        raise ValueError("reliability.embedding_reference_max must be at least two")
    error_quantile = float(cfg["reliability"]["error_quantile"])
    if not 0.0 < error_quantile < 1.0:
        raise ValueError("reliability.error_quantile must lie strictly between zero and one")
    seeds = cfg["training"]["seeds"]
    if not seeds or len(set(seeds)) != len(seeds) or any(
        not isinstance(seed, int) for seed in seeds
    ):
        raise ValueError("training.seeds must contain unique integer seeds")
    return run_name


def validate_execution_filters(
    cfg, station_series, *, protocols=None, regions=None, horizons=None, seeds=None
):
    available = {
        "protocol": set(cfg["protocols"]),
        "region": {series.region for series in station_series},
        "horizon": set(cfg["window"]["horizons"]),
        "seed": set(cfg["training"]["seeds"]),
    }
    requested = {
        "protocol": protocols,
        "region": regions,
        "horizon": horizons,
        "seed": seeds,
    }
    for name, values in requested.items():
        if values is None:
            continue
        unknown = set(values) - available[name]
        if unknown:
            raise ValueError(f"Unknown {name} execution filters: {sorted(unknown)}")


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_yaml(path, payload):
    path = Path(path)
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
        if (
            row.get("config_sha256") == config_hash
            and row.get("code_sha256") == source_hash
        ):
            rows.append(row)
    return rows


def reference_subset(train_set, maximum):
    x, y = train_set
    maximum = int(maximum)
    if maximum < 2:
        raise ValueError("embedding_reference_max must be at least two")
    count = min(len(x), maximum)
    indices = np.rint(np.linspace(0, len(x) - 1, count)).astype(int)
    if len(np.unique(indices)) != count:
        raise RuntimeError("Reference embedding sampling produced duplicate indices")
    return x[indices], y[indices]


def resumable_record(path, artifact, config_hash, source_hash, data_hash):
    path = Path(path)
    artifact = Path(artifact)
    if not path.is_file() or not artifact.is_file():
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


def record_from_metrics(row, metrics, calibration_state):
    horizonwise = metrics["horizonwise_conformal"]
    simultaneous = metrics["simultaneous_conformal"]
    fused = metrics["selective"]["fused"]
    diagnostics = metrics["reliability_diagnostics"]["fused"]
    # The fusion weights decide whether the score is a real combination or one
    # component wearing a different name, so they belong in the ledger rather
    # than only inside the artifact's nested state.
    fusion = metrics.get("fusion", {})
    weights = fusion.get("weights") or []
    row.update(
        fusion_weight_selection=fusion.get("weight_selection"),
        fusion_weight_width=weights[0] if len(weights) > 0 else None,
        fusion_weight_shift=weights[1] if len(weights) > 1 else None,
    )
    row.update(
        rmse=metrics["point"]["rmse"],
        mae=metrics["point"]["mae"],
        raw_marginal_coverage=metrics["raw_interval"]["marginal_coverage"],
        conformal_marginal_coverage=horizonwise["test_marginal_coverage"],
        conformal_joint_coverage=horizonwise["test_joint_coverage"],
        conformal_mean_width=horizonwise["test_mean_width"],
        conformal_interval_score=horizonwise["test_mean_interval_score"],
        simultaneous_joint_coverage=simultaneous["test_joint_coverage"],
        simultaneous_mean_width=simultaneous["test_mean_width"],
        simultaneous_interval_score=simultaneous["test_mean_interval_score"],
        fused_aurc=fused["aurc"],
        width_only_aurc=metrics["selective"]["width_only"]["aurc"],
        shift_only_aurc=metrics["selective"]["shift_only"]["aurc"],
        fused_selected_coverage=fused["test_coverage"],
        fused_selected_rmse=fused["test_rmse"],
        reliability_spearman=diagnostics["association"]["spearman_rho"],
        error_detection_auroc=diagnostics["top_error_detection"]["auroc"],
        error_detection_auprc=diagnostics["top_error_detection"]["auprc"],
        horizonwise_coverage_json=json.dumps(
            horizonwise["test_horizonwise_coverage"]
        ),
        calibration_state_json=json.dumps(_json_safe(calibration_state), sort_keys=True),
    )


def run_spec(
    cfg, spec, horizon, paths, hashes, resume, *, selected_seeds=None, device=None
):
    raw_dir, record_dir = paths
    cfg_hash, source_hash, panel_hash = hashes
    data_hash = dataset_sha256(spec, panel_hash)
    rows = []
    quantiles = tuple(float(value) for value in cfg["model"]["quantiles"])
    for seed in cfg["training"]["seeds"]:
        if selected_seeds is not None and seed not in selected_seeds:
            continue
        stem = f"{spec.dataset}_trustkan_reliability_h{horizon}_s{seed}"
        artifact = raw_dir / f"{stem}.npz"
        record = record_dir / f"{stem}.json"
        set_seed(
            seed,
            deterministic=cfg["training"]["deterministic_algorithms"],
            warn_only=cfg["training"]["deterministic_warn_only"],
        )
        run_environment = environment(device)
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
            "source_regions": json.dumps(spec.source_regions),
            "source_stations": json.dumps(spec.source_stations),
            "source_pooling": spec.source_pooling,
            "normalization": "per_station_training_period",
            "model": "trustkan",
            "horizon": int(horizon),
            "seed": int(seed),
            "split": "test",
            "status": "ok",
            "nominal_coverage": 1.0 - float(cfg["conformal"]["alpha"]),
            "n_train": int(len(spec.train_set[0])),
            "n_validation": int(len(spec.validation_set[0])),
            "n_calibration": int(len(spec.calibration_set[0])),
            "n_test": int(len(spec.test_set[0])),
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
            train_loader = loader(
                *spec.train_set, cfg["training"]["batch_size"], True
            )
            validation_loader = loader(
                *spec.validation_set, cfg["training"]["batch_size"]
            )
            calibration_loader = loader(
                *spec.calibration_set, cfg["training"]["batch_size"]
            )
            test_loader = loader(*spec.test_set, cfg["training"]["batch_size"])
            reference = reference_subset(
                spec.train_set, cfg["reliability"]["embedding_reference_max"]
            )
            reference_loader = loader(
                *reference, cfg["training"]["batch_size"]
            )
            model = TrustKAN(
                1,
                horizon=horizon,
                hidden_dim=cfg["model"]["hidden_dim"],
                grid_size=cfg["model"]["grid_size"],
                quantiles=quantiles,
            )
            model, history, train_seconds = train_trustkan(
                model,
                train_loader,
                validation_loader,
                epochs=cfg["training"]["epochs"],
                lr=cfg["training"]["learning_rate"],
                patience=cfg["training"]["patience"],
                point_weight=cfg["training"]["point_weight"],
                quantile_weight=cfg["training"]["quantile_weight"],
                weight_decay=cfg["training"]["weight_decay"],
                device=device,
            )
            reference_prediction = predict_trustkan(model, reference_loader, device)
            calibration_prediction = predict_trustkan(model, calibration_loader, device)
            test_prediction, inference_ms = timed_prediction(
                lambda: predict_trustkan(model, test_loader, device),
                len(spec.test_set[0]),
                device,
            )
            if not np.allclose(
                inverse_standardized(spec.target_scaler, calibration_prediction["target"]),
                spec.calibration_target_raw,
                rtol=1e-6,
                atol=5e-6,
            ):
                raise ValueError("Calibration target inversion does not match frozen windows")
            if not np.allclose(
                inverse_standardized(spec.target_scaler, test_prediction["target"]),
                spec.test_target_raw,
                rtol=1e-6,
                atol=5e-6,
            ):
                raise ValueError("Test target inversion does not match frozen windows")
            metrics, arrays, calibration_state = evaluate_calibrated_reliability(
                calibration_prediction,
                test_prediction,
                reference_prediction["embedding"],
                spec.target_scaler,
                quantile_levels=quantiles,
                alpha=cfg["conformal"]["alpha"],
                reliability_weights=cfg["reliability"]["fusion_weights"],
                min_coverage=cfg["reliability"]["min_calibration_coverage"],
                error_quantile=cfg["reliability"]["error_quantile"],
            )
            atomic_savez(
                artifact,
                **arrays,
                target=arrays["test_target"],
                prediction=arrays["test_prediction"],
                target_time=spec.test_target_times,
                target_origin=spec.test_origins,
                calibration_quantiles_standardized=calibration_prediction["quantiles"],
                calibration_embedding=calibration_prediction["embedding"],
                reference_embedding=reference_prediction["embedding"],
                calibration_target_time=spec.calibration_target_times,
                calibration_target_origin=spec.calibration_origins,
                test_target_time=spec.test_target_times,
                test_target_origin=spec.test_origins,
                scaler_mean=np.asarray(spec.target_scaler.scaler.mean_),
                scaler_scale=np.asarray(spec.target_scaler.scaler.scale_),
                quantile_levels=np.asarray(quantiles),
                conformal_alpha=np.asarray(cfg["conformal"]["alpha"]),
                reliability_weights=np.asarray(cfg["reliability"]["fusion_weights"]),
                metrics_json=np.asarray(json.dumps(_json_safe(metrics), sort_keys=True)),
                calibration_state_json=np.asarray(
                    json.dumps(_json_safe(calibration_state), sort_keys=True)
                ),
                training_history_json=np.asarray(json.dumps(_json_safe(history))),
                dataset=np.asarray(spec.dataset),
                model=np.asarray("trustkan"),
                horizon=np.asarray(horizon),
                seed=np.asarray(seed),
                split=np.asarray("test"),
                protocol=np.asarray(spec.protocol),
                target_region=np.asarray(spec.target_region),
                target_station=np.asarray(spec.target_station),
                source_regions_json=np.asarray(row["source_regions"]),
                source_stations_json=np.asarray(row["source_stations"]),
                source_pooling=np.asarray(spec.source_pooling),
                normalization=np.asarray(row["normalization"]),
                config_sha256=np.asarray(cfg_hash),
                code_sha256=np.asarray(source_hash),
                dataset_sha256=np.asarray(data_hash),
                requested_device=np.asarray(str(device)),
                environment_json=np.asarray(
                    json.dumps(run_environment, sort_keys=True)
                ),
            )
            row.update(
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                train_seconds=train_seconds,
                inference_ms=inference_ms,
            )
            record_from_metrics(row, metrics, calibration_state)
            row["artifact_sha256"] = file_sha256(artifact)
            print(
                f"OK {spec.protocol} target={spec.target_region} h={horizon} "
                f"seed={seed} coverage={row['conformal_marginal_coverage']:.3f} "
                f"AURC={row['fused_aurc']:.4f}"
            )
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(
                f"FAILED {spec.protocol} target={spec.target_region} h={horizon} "
                f"seed={seed}: {exc}"
            )
        atomic_json(record, row)
        rows.append(_json_safe(row))
    return rows


def main(
    config,
    resume=False,
    *,
    protocols=None,
    regions=None,
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
    panel_path, panel, station_series = load_station_panel(cfg)
    resolved_device = resolve_device(device)
    validate_execution_filters(
        cfg,
        station_series,
        protocols=protocols,
        regions=regions,
        horizons=horizons,
        seeds=seeds,
    )
    cfg_hash = combined_config_sha256(config, panel_path)
    source_hash = code_sha256()
    panel_hash = file_sha256(panel_path)
    raw_dir = Path("results/reliability/raw") / run_name
    record_dir = Path("results/reliability/runs") / run_name
    aggregate_dir = Path("results/reliability/aggregated")
    raw_dir.mkdir(parents=True, exist_ok=True)
    record_dir.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    atomic_yaml(aggregate_dir / f"{run_name}_config.yaml", cfg)
    atomic_yaml(aggregate_dir / f"{run_name}_panel.yaml", panel)

    selected_rows = []
    for horizon in ([] if collect_only else cfg["window"]["horizons"]):
        if horizons is not None and horizon not in horizons:
            continue
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
        selected_protocols = (
            cfg["protocols"] if protocols is None else protocols
        )
        for spec in build_evaluation_specs(bundles, selected_protocols):
            if regions is not None and spec.target_region not in regions:
                continue
            selected_rows.extend(
                run_spec(
                    cfg,
                    spec,
                    horizon,
                    (raw_dir, record_dir),
                    (cfg_hash, source_hash, panel_hash),
                    resume,
                    selected_seeds=seeds,
                    device=resolved_device,
                )
            )
    if not collect_only:
        if not selected_rows:
            raise RuntimeError("No reliability jobs matched the execution filters")
        failed_count = sum(row["status"] != "ok" for row in selected_rows)
        if failed_count:
            raise RuntimeError(
                f"{failed_count} reliability runs failed; inspect their run records"
            )
        if defer_collection:
            return
    rows = collect_current_records(record_dir, cfg_hash, source_hash)
    if not rows:
        raise RuntimeError("No current reliability records are available to collect")
    result = pd.DataFrame(rows)
    atomic_csv(result, aggregate_dir / f"{run_name}_runs.csv")
    ok = result[result.status.eq("ok")]
    if not ok.empty:
        group = ["protocol", "dataset", "target_region", "horizon"]
        summary = ok.groupby(group, as_index=False).agg(
            n=("seed", "nunique"),
            rmse_mean=("rmse", "mean"),
            marginal_coverage_mean=("conformal_marginal_coverage", "mean"),
            joint_coverage_mean=("simultaneous_joint_coverage", "mean"),
            interval_width_mean=("conformal_mean_width", "mean"),
            fused_aurc_mean=("fused_aurc", "mean"),
        )
        atomic_csv(summary, aggregate_dir / f"{run_name}_summary.csv")
        print(summary.to_string(index=False))
    failed_count = sum(row["status"] != "ok" for row in rows)
    if failed_count:
        raise RuntimeError(
            f"{failed_count} reliability runs failed; inspect the run ledger"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ghcn_reliability_smoke.yaml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only checksum-matched successful records and artifacts.",
    )
    parser.add_argument("--protocol", action="append", dest="protocols")
    parser.add_argument("--region", action="append", dest="regions")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Rebuild ledgers from checksum-matched records without training.",
    )
    parser.add_argument(
        "--defer-collection",
        action="store_true",
        help="Write atomic run records only; use --collect-only after all shards.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device for TrustKAN, e.g. cpu, cuda, or cuda:0.",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.resume,
        protocols=args.protocols,
        regions=args.regions,
        horizons=args.horizons,
        seeds=args.seeds,
        collect_only=args.collect_only,
        defer_collection=args.defer_collection,
        device=args.device,
    )
