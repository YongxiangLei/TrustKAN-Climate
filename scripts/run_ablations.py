"""Falsifiable A0-A8 TrustKAN ablations on a frozen dataset.

A0-A2 are trained variants. A3-A7 are evaluation-side removals recomputed from
the A0 artifact, so they consume no extra training budget. A8 replays the A0
predictions through sequential adaptive conformal.
"""
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
    config_sha256,
    ensure_data,
    environment,
    load_config,
    loader,
    prepare_series,
    timed_prediction,
    validate_config,
)
from scripts.run_cet_reliability import build_sets
from scripts.run_ghcn_reliability import (
    atomic_csv,
    atomic_json,
    atomic_yaml,
    record_from_metrics,
    reference_subset,
    resumable_record,
    _json_safe,
)
from src.data.provenance import file_sha256
from src.models.trustkan import TrustKAN
from src.reliability.evaluation import evaluate_calibrated_reliability, inverse_standardized
from src.training.engine import resolve_device, set_seed
from src.training.trust_engine import predict_trustkan, train_trustkan
from src.uncertainty.adaptive import adaptive_conformal, conformity_score, rolling_coverage
from src.uncertainty.conformal import interval_coverage, mean_interval_width


REQUIRED_VARIANTS = {"A0", "A1", "A2", "A9"}


def code_sha256():
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "scripts" / "run_cet_benchmark.py",
        root / "scripts" / "run_cet_reliability.py",
        *sorted((root / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_ablation_config(cfg, config_path):
    run_name = validate_config(cfg, config_path)
    variants = cfg["variants"]
    ids = [str(item["id"]) for item in variants]
    if len(ids) != len(set(ids)):
        raise ValueError("Ablation variant ids must be unique")
    missing = REQUIRED_VARIANTS - set(ids)
    if missing:
        raise ValueError(f"Ablation config is missing trained variants {sorted(missing)}")
    if ids[0] != "A0":
        raise ValueError("A0 must be declared first because it is the reference")
    for item in variants:
        if item["encoder"] not in {"kan", "mlp"}:
            raise ValueError(f"Unknown encoder for {item['id']}")
        if not isinstance(item["quantile_head"], bool):
            raise ValueError(f"quantile_head must be boolean for {item['id']}")
    levels = cfg["model"]["quantiles"]
    alpha = float(cfg["conformal"]["alpha"])
    if not np.isclose(levels[0], alpha / 2.0) or not np.isclose(levels[-1], 1.0 - alpha / 2.0):
        raise ValueError("Outer model quantiles must match alpha/2 and 1-alpha/2")
    return run_name


def validate_execution_filters(cfg, *, horizons=None, seeds=None, variants=None):
    available = {
        "horizon": set(cfg["window"]["horizons"]),
        "seed": set(cfg["training"]["seeds"]),
        "variant": {str(item["id"]) for item in cfg["variants"]},
    }
    requested = {"horizon": horizons, "seed": seeds, "variant": variants}
    for name, values in requested.items():
        if values is None:
            continue
        unknown = set(values) - available[name]
        if unknown:
            raise ValueError(f"Unknown {name} execution filters: {sorted(unknown)}")


def evaluation_side_ablations(metrics):
    """A3-A7 read directly off the A0 reliability metrics."""
    raw = metrics["raw_interval"]
    conformal = metrics["horizonwise_conformal"]
    selective = metrics["selective"]
    fused = selective["fused"]
    width_only = selective["width_only"]
    shift_only = selective["shift_only"]
    best_component = min(width_only["aurc"], shift_only["aurc"])
    return {
        "A3_no_conformal": {
            "hypothesis": "conformal correction improves empirical coverage",
            "raw_marginal_coverage": raw["marginal_coverage"],
            "conformal_marginal_coverage": conformal["test_marginal_coverage"],
            "raw_mean_width": raw["mean_width"],
            "conformal_mean_width": conformal["test_mean_width"],
            "coverage_gain": conformal["test_marginal_coverage"] - raw["marginal_coverage"],
            "width_cost": conformal["test_mean_width"] - raw["mean_width"],
        },
        "A4_no_embedding_shift": {
            "hypothesis": "representation shift helps identify unreliable samples",
            "fused_aurc": fused["aurc"],
            "width_only_aurc": width_only["aurc"],
            "aurc_gain_from_shift": width_only["aurc"] - fused["aurc"],
        },
        "A5_no_interval_width": {
            "hypothesis": "interval sharpness contributes useful trust evidence",
            "fused_aurc": fused["aurc"],
            "shift_only_aurc": shift_only["aurc"],
            "aurc_gain_from_width": shift_only["aurc"] - fused["aurc"],
        },
        "A6_no_fusion": {
            "hypothesis": "fusion outperforms the best single component",
            "fused_aurc": fused["aurc"],
            "best_component_aurc": best_component,
            "aurc_gain_from_fusion": best_component - fused["aurc"],
        },
        "A7_no_abstention": {
            "hypothesis": "selective prediction reduces retained-set risk",
            "full_test_rmse": metrics["point"]["rmse"],
            "retained_rmse": fused["test_rmse"],
            "retained_coverage": fused["test_coverage"],
            "rmse_reduction": (
                None
                if fused["test_rmse"] is None
                else metrics["point"]["rmse"] - fused["test_rmse"]
            ),
        },
    }


def adaptive_conformal_ablation(cfg, arrays, scaler, prediction, calibration):
    """A8: static split conformal versus sequential adaptive conformal."""
    alpha = float(cfg["conformal"]["alpha"])
    settings = cfg.get("adaptive", {})
    cal_lower = calibration["quantiles"][..., 0]
    cal_upper = calibration["quantiles"][..., -1]
    test_lower = prediction["quantiles"][..., 0]
    test_upper = prediction["quantiles"][..., -1]
    initial = conformity_score(calibration["target"], cal_lower, cal_upper).reshape(-1)
    adaptive_lower_std, adaptive_upper_std, diagnostics = adaptive_conformal(
        prediction["target"],
        test_lower,
        test_upper,
        initial,
        alpha=alpha,
        gamma=float(settings.get("gamma", 0.01)),
        window=int(settings.get("window", 256)),
    )
    adaptive_lower = inverse_standardized(scaler, adaptive_lower_std)
    adaptive_upper = inverse_standardized(scaler, adaptive_upper_std)
    target = arrays["test_target"]
    static_lower = arrays["test_lower_horizonwise"]
    static_upper = arrays["test_upper_horizonwise"]
    coverage_window = int(settings.get("coverage_window", 100))
    static_rolling = rolling_coverage(target, static_lower, static_upper, coverage_window)
    adaptive_rolling = rolling_coverage(target, adaptive_lower, adaptive_upper, coverage_window)
    nominal = 1.0 - alpha

    def deviation(values):
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return None
        return float(np.abs(finite - nominal).mean())

    summary = {
        "hypothesis": "adaptation improves coverage stability under temporal shift",
        "nominal_coverage": nominal,
        "static_marginal_coverage": interval_coverage(target, static_lower, static_upper),
        "adaptive_marginal_coverage": interval_coverage(target, adaptive_lower, adaptive_upper),
        "static_mean_width": mean_interval_width(static_lower, static_upper),
        "adaptive_mean_width": mean_interval_width(adaptive_lower, adaptive_upper),
        "static_rolling_deviation": deviation(static_rolling),
        "adaptive_rolling_deviation": deviation(adaptive_rolling),
        "coverage_window": coverage_window,
    }
    static_dev = summary["static_rolling_deviation"]
    adaptive_dev = summary["adaptive_rolling_deviation"]
    summary["rolling_deviation_improvement"] = (
        None if static_dev is None or adaptive_dev is None else static_dev - adaptive_dev
    )
    payload = {
        "adaptive_lower": adaptive_lower,
        "adaptive_upper": adaptive_upper,
        "adaptive_radius_standardized": diagnostics["radius"],
        "adaptive_alpha_path": diagnostics["alpha"],
        "adaptive_miss": diagnostics["miss"],
        "static_rolling_coverage": static_rolling,
        "adaptive_rolling_coverage": adaptive_rolling,
    }
    return summary, payload


def main(
    config,
    resume=False,
    *,
    horizons=None,
    seeds=None,
    variants=None,
    collect_only=False,
    defer_collection=False,
    device=None,
    results_root="results",
):
    if collect_only and defer_collection:
        raise ValueError("--collect-only and --defer-collection are mutually exclusive")
    cfg = load_config(config)
    run_name = validate_ablation_config(cfg, config)
    validate_execution_filters(cfg, horizons=horizons, seeds=seeds, variants=variants)
    resolved_device = resolve_device(device)
    cfg_hash = config_sha256(cfg)
    source_hash = code_sha256()
    data_hash = str(cfg["dataset"].get("sha256", "")).lower()
    path = ensure_data(cfg)
    values, dates = prepare_series(cfg, path)
    root = Path(results_root)
    raw_dir = root / "ablations" / "raw" / run_name
    record_dir = root / "ablations" / "runs" / run_name
    aggregate_dir = root / "ablations" / "aggregated"
    for directory in (raw_dir, record_dir, aggregate_dir):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_yaml(aggregate_dir / f"{run_name}_config.yaml", cfg)

    quantiles = tuple(float(value) for value in cfg["model"]["quantiles"])
    selected_rows = []
    for horizon in ([] if collect_only else cfg["window"]["horizons"]):
        if horizons is not None and horizon not in horizons:
            continue
        bundle = build_sets(values, dates, cfg, int(horizon))
        for variant in cfg["variants"]:
            variant_id = str(variant["id"])
            if variants is not None and variant_id not in variants:
                continue
            for seed in cfg["training"]["seeds"]:
                if seeds is not None and seed not in seeds:
                    continue
                stem = f"ablation_{variant_id}_h{horizon}_s{seed}"
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
                    "model": f"trustkan_{variant_id}",
                    "ablation_id": variant_id,
                    "ablation_label": variant.get("label", variant_id),
                    "encoder": variant["encoder"],
                    "quantile_head": bool(variant["quantile_head"]),
                    "readout": variant.get("readout", "last"),
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "split": "test",
                    "status": "ok",
                    "nominal_coverage": 1.0 - float(cfg["conformal"]["alpha"]),
                    "n_train": int(len(bundle["sets"]["train"][0])),
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
                    train_loader = loader(
                        *bundle["sets"]["train"], cfg["training"]["batch_size"], True
                    )
                    val_loader = loader(*bundle["sets"]["val"], cfg["training"]["batch_size"])
                    cal_loader = loader(
                        *bundle["sets"]["calibration"], cfg["training"]["batch_size"]
                    )
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
                        encoder=variant["encoder"],
                        quantile_head=bool(variant["quantile_head"]),
                        readout=variant.get("readout", "last"),
                    )
                    model, history, train_seconds = train_trustkan(
                        model,
                        train_loader,
                        val_loader,
                        epochs=cfg["training"]["epochs"],
                        lr=cfg["training"]["learning_rate"],
                        patience=cfg["training"]["patience"],
                        point_weight=variant.get(
                            "point_weight", cfg["training"]["point_weight"]
                        ),
                        quantile_weight=variant.get(
                            "quantile_weight", cfg["training"]["quantile_weight"]
                        ),
                        weight_decay=cfg["training"]["weight_decay"],
                        device=resolved_device,
                    )
                    reference_prediction = predict_trustkan(
                        model, reference_loader, resolved_device
                    )
                    calibration_prediction = predict_trustkan(model, cal_loader, resolved_device)
                    test_prediction, inference_ms = timed_prediction(
                        lambda: predict_trustkan(model, test_loader, resolved_device),
                        len(bundle["sets"]["test"][0]),
                        resolved_device,
                    )
                    metrics, arrays, calibration_state = evaluate_calibrated_reliability(
                        calibration_prediction,
                        test_prediction,
                        reference_prediction["embedding"],
                        bundle["scaler"],
                        quantile_levels=quantiles,
                        alpha=cfg["conformal"]["alpha"],
                        reliability_weights=cfg["reliability"]["fusion_weights"],
                        min_coverage=cfg["reliability"]["min_calibration_coverage"],
                        error_quantile=cfg["reliability"]["error_quantile"],
                    )
                    ablations = evaluation_side_ablations(metrics)
                    adaptive_summary, adaptive_arrays = adaptive_conformal_ablation(
                        cfg,
                        arrays,
                        bundle["scaler"],
                        test_prediction,
                        calibration_prediction,
                    )
                    ablations["A8_static_vs_adaptive_conformal"] = adaptive_summary
                    atomic_savez(
                        artifact,
                        **arrays,
                        **adaptive_arrays,
                        target=arrays["test_target"],
                        prediction=arrays["test_prediction"],
                        target_origin=bundle["test_origins"],
                        quantile_levels=np.asarray(quantiles),
                        metrics_json=np.asarray(
                            json.dumps(_json_safe(metrics), sort_keys=True)
                        ),
                        ablation_json=np.asarray(
                            json.dumps(_json_safe(ablations), sort_keys=True)
                        ),
                        dataset=np.asarray(cfg["dataset"]["name"]),
                        model=np.asarray(f"trustkan_{variant_id}"),
                        ablation_id=np.asarray(variant_id),
                        horizon=np.asarray(horizon),
                        seed=np.asarray(seed),
                        split=np.asarray("test"),
                        config_sha256=np.asarray(cfg_hash),
                        code_sha256=np.asarray(source_hash),
                        dataset_sha256=np.asarray(data_hash),
                        training_history_json=np.asarray(json.dumps(_json_safe(history))),
                        environment_json=np.asarray(
                            json.dumps(run_environment, sort_keys=True)
                        ),
                    )
                    row.update(
                        parameters=sum(p.numel() for p in model.parameters()),
                        train_seconds=train_seconds,
                        inference_ms=inference_ms,
                    )
                    record_from_metrics(row, metrics, calibration_state)
                    row["ablation_json"] = json.dumps(_json_safe(ablations), sort_keys=True)
                    row["adaptive_rolling_deviation"] = adaptive_summary[
                        "adaptive_rolling_deviation"
                    ]
                    row["static_rolling_deviation"] = adaptive_summary[
                        "static_rolling_deviation"
                    ]
                    row["artifact_sha256"] = file_sha256(artifact)
                    print(
                        f"OK {variant_id} h={horizon} seed={seed} "
                        f"rmse={row['rmse']:.4f} aurc={row['fused_aurc']:.4f}"
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {variant_id} h={horizon} seed={seed}: {exc}")
                atomic_json(record, row)
                selected_rows.append(_json_safe(row))

    if not collect_only:
        if not selected_rows:
            raise RuntimeError("No ablation jobs matched the execution filters")
        failed = [row for row in selected_rows if row["status"] != "ok"]
        if failed:
            raise RuntimeError(f"{len(failed)} ablation runs failed")
        if defer_collection:
            return selected_rows

    rows = []
    for path_item in sorted(Path(record_dir).glob("*.json")):
        with open(path_item, "r", encoding="utf-8") as handle:
            item = json.load(handle)
        if (
            item.get("config_sha256") == cfg_hash
            and item.get("code_sha256") == source_hash
        ):
            rows.append(item)
    if not rows:
        raise RuntimeError("No current ablation records are available to collect")
    result = pd.DataFrame(rows)
    atomic_csv(result, aggregate_dir / f"{run_name}_runs.csv")
    ok = result[result.status.eq("ok")]
    if not ok.empty:
        summary = ok.groupby(["ablation_id", "horizon"], as_index=False).agg(
            n=("seed", "nunique"),
            parameters=("parameters", "max"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            coverage_mean=("conformal_marginal_coverage", "mean"),
            width_mean=("conformal_mean_width", "mean"),
            fused_aurc_mean=("fused_aurc", "mean"),
        )
        atomic_csv(summary, aggregate_dir / f"{run_name}_summary.csv")
        print(summary.to_string(index=False))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ablations_smoke.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--defer-collection", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.resume,
        horizons=args.horizons,
        seeds=args.seeds,
        variants=args.variants,
        collect_only=args.collect_only,
        defer_collection=args.defer_collection,
        device=args.device,
    )
