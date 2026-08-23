"""Score the pre-registered corruption grid on CET.

The corruptions apply to test histories only, so a model needs to be trained
once and then queried on each perturbed copy. The benchmark runner stored
predictions rather than weights, which is why this sweep could not be scored
from the completed campaign and why each run is retrained here under the frozen
protocol with its original seed.

The grid validates itself. Its first entry is the clean history, so the clean
RMSE must equal the value the benchmark ledger already holds for that run; a run
whose clean score does not reproduce is refused rather than reported, because
its corrupted scores would describe some other model.

Scope is the pre-specified comparison family of Section~IV: the proposed model
and the two comparators fixed in advance against it. Corrupting every baseline
would multiply the cost without bearing on any pre-registered claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import (
    atomic_savez,
    atomic_write_json,
    build_model,
    config_sha256,
    ensure_data,
    environment,
    inverse_target,
    load_config,
    loader,
    prepare_series,
)
from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)
from src.robustness.evaluation import evaluate_corruption_grid
from src.training.engine import predict, resolve_device, set_seed, train_regressor

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LEDGER = ROOT / "results" / "aggregated" / "cet_full_runs.csv"
RUN_NAME = "cet_robustness"
# The pre-specified comparison family: the proposed model and the two
# comparators fixed against it before any result was seen.
MODELS = ("trustkan", "kan", "transformer")
# Deterministic training should land on the ledger value exactly; the tolerance
# only absorbs float accumulation in a different summation order.
RMSE_TOLERANCE = 1e-6


def code_sha256() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_cet_benchmark.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def make_predictor(model, scaler, horizon: int, batch_size: int, device):
    """Physical-unit forecasts from standardized histories.

    The corruption grid hands back arrays shaped like the stored test
    histories, so the predictor owns batching and inverse standardization and
    the grid stays ignorant of both. The horizon is passed in rather than read
    off the model, because only the proposed architecture carries it.
    """

    def predict_fn(history):
        values = np.asarray(history, dtype=np.float32)
        placeholder = np.zeros((len(values), horizon), dtype=np.float32)
        batches = loader(values, placeholder, batch_size)
        standardized, _ = predict(model, batches, device)
        return inverse_target(scaler, standardized)

    return predict_fn


def main(config, robustness_config, *, horizons=None, models=None, seeds=None, device=None, resume=False):
    cfg = load_config(config)
    policy = load_config(robustness_config)
    cfg_hash = config_sha256(cfg)
    policy_hash = config_sha256(policy)
    source_hash = code_sha256()
    data_hash = str(cfg["dataset"].get("sha256", "")).lower()
    resolved_device = resolve_device(device)

    if not BENCHMARK_LEDGER.exists():
        raise SystemExit(f"missing {BENCHMARK_LEDGER.relative_to(ROOT)}")
    ledger = pd.read_csv(BENCHMARK_LEDGER)
    ledger = ledger[ledger.status.eq("ok")]
    expected = {
        (str(row.model), int(row.horizon), int(row.seed)): float(row.rmse)
        for row in ledger.itertuples()
    }

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
    expected_step = pd.to_timedelta(cfg["dataset"]["frequency"]).to_timedelta64()
    frequency = str(cfg["dataset"]["frequency"])

    raw_dir = ROOT / "results" / "robustness" / "raw" / RUN_NAME
    record_dir = ROOT / "results" / "robustness" / "runs" / RUN_NAME
    aggregated_dir = ROOT / "results" / "robustness" / "aggregated"
    for directory in (raw_dir, record_dir, aggregated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_horizons = horizons or cfg["window"]["horizons"]
    run_models = models or MODELS
    run_seeds = seeds or cfg["training"]["seeds"]
    rows = []
    for horizon in run_horizons:
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
        for name in run_models:
            for seed in run_seeds:
                artifact = raw_dir / f"robustness_{name}_h{horizon}_s{seed}.npz"
                record = record_dir / f"robustness_{name}_h{horizon}_s{seed}.json"
                if resume and artifact.is_file() and record.is_file():
                    stored = json.loads(record.read_text(encoding="utf-8"))
                    if (
                        stored.get("status") == "ok"
                        and stored.get("code_sha256") == source_hash
                        and stored.get("config_sha256") == cfg_hash
                        and stored.get("policy_sha256") == policy_hash
                    ):
                        rows.extend(stored["grid"])
                        print(f"RESUME {name} h={horizon} seed={seed}")
                        continue

                # Match the benchmark's call order exactly: seeding, model
                # construction, and loader creation all draw on the global RNG.
                set_seed(
                    seed,
                    deterministic=cfg["training"]["deterministic_algorithms"],
                    warn_only=cfg["training"]["deterministic_warn_only"],
                )
                row = {
                    "dataset": cfg["dataset"]["name"],
                    "model": name,
                    "horizon": int(horizon),
                    "seed": int(seed),
                    "status": "ok",
                    "config_sha256": cfg_hash,
                    "policy_sha256": policy_hash,
                    "code_sha256": source_hash,
                    "dataset_sha256": data_hash,
                    **environment(resolved_device),
                }
                try:
                    model = build_model(name, cfg["window"]["history"], horizon, seed)
                    train_loader = loader(*sets["train"], cfg["training"]["batch_size"], True)
                    val_loader = loader(*sets["val"], cfg["training"]["batch_size"])
                    start = time.perf_counter()
                    model, history, seconds = train_regressor(
                        model,
                        train_loader,
                        val_loader,
                        epochs=cfg["training"]["epochs"],
                        lr=cfg["training"]["learning_rate"],
                        patience=cfg["training"]["patience"],
                        optimizer_name=cfg["training"].get("optimizer", "adamw"),
                        weight_decay=cfg["training"].get("weight_decay"),
                        device=resolved_device,
                    )
                    grid, arrays = evaluate_corruption_grid(
                        test_x,
                        test_y_raw,
                        make_predictor(
                            model,
                            scaler,
                            horizon,
                            cfg["training"]["batch_size"],
                            resolved_device,
                        ),
                        policy,
                        frequency=frequency,
                        seed=int(policy["corruption"]["seed"]),
                        fill=float(policy["corruption"]["fill_value"]),
                    )
                    clean = next(entry for entry in grid if entry["kind"] == "clean")
                    target = expected.get((name, int(horizon), int(seed)))
                    if target is None or abs(clean["rmse"] - target) > RMSE_TOLERANCE:
                        # The clean cell of the grid is the benchmark condition,
                        # so a mismatch means the corrupted cells belong to a
                        # model the paper never reported.
                        raise RuntimeError(
                            f"clean RMSE {clean['rmse']:.9f} does not reproduce the "
                            f"ledger value {target}"
                        )
                    for entry in grid:
                        entry.update(
                            dataset=cfg["dataset"]["name"],
                            model=name,
                            horizon=int(horizon),
                            model_seed=int(seed),
                            clean_rmse=float(clean["rmse"]),
                            rmse_increase=float(entry["rmse"] - clean["rmse"]),
                            relative_increase=float(
                                (entry["rmse"] - clean["rmse"]) / clean["rmse"]
                            ),
                            config_sha256=cfg_hash,
                            policy_sha256=policy_hash,
                            code_sha256=source_hash,
                        )
                    row["grid"] = grid
                    row["train_seconds"] = float(seconds)
                    row["wall_seconds"] = float(time.perf_counter() - start)
                    atomic_savez(
                        artifact,
                        target=test_y_raw,
                        target_origin=origins[masks["test"]],
                        grid_json=np.asarray(json.dumps(grid)),
                        training_history_json=np.asarray(json.dumps(history)),
                        config_sha256=np.asarray(cfg_hash),
                        policy_sha256=np.asarray(policy_hash),
                        code_sha256=np.asarray(source_hash),
                        dataset_sha256=np.asarray(data_hash),
                        **{
                            key: value
                            for key, value in arrays.items()
                            if key.endswith("_prediction")
                        },
                    )
                    rows.extend(grid)
                    worst = max(grid, key=lambda entry: entry["rmse"])
                    print(
                        f"OK {name} h={horizon} seed={seed} clean={clean['rmse']:.4f} "
                        f"worst={worst['kind']}@{worst['level']:g} {worst['rmse']:.4f} "
                        f"(+{100 * worst['relative_increase']:.1f}%)"
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {name} h={horizon} seed={seed}: {exc}")
                atomic_write_json(
                    record, {**row, "grid": json.dumps(row.get("grid", []))}
                )

    if not rows:
        raise SystemExit("no robustness rows were produced")
    frame = pd.DataFrame(rows)
    frame.to_csv(aggregated_dir / f"{RUN_NAME}_grid.csv", index=False)
    summary = frame.groupby(["model", "horizon", "kind", "level"], as_index=False).agg(
        n=("model_seed", "count"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        relative_increase_mean=("relative_increase", "mean"),
    )
    summary.to_csv(aggregated_dir / f"{RUN_NAME}_summary.csv", index=False)
    print(f"  wrote {(aggregated_dir / f'{RUN_NAME}_summary.csv').relative_to(ROOT)}")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cet.yaml")
    parser.add_argument("--robustness-config", default="configs/robustness.yaml")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(
        args.config,
        args.robustness_config,
        horizons=args.horizons,
        models=args.models,
        seeds=args.seeds,
        device=args.device,
        resume=args.resume,
    )
