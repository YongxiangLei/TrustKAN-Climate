"""Measure how much of its history window each model actually reads.

The corruption sweep found that masking the most recent days costs TrustKAN far
more than any other corruption, and that the cost stops growing once the block
passes three days: past that point the error is not merely similar but
bit-identical across block lengths, which is only possible if the extra masked
days never reach the readout. This module measures that reach directly.

The measurement perturbs one timestep at a time and records the furthest
position back that still moves the forecast. Doing that on a randomly
initialized model would answer a question about the architecture; the paper
needs the answer for the models it actually reports, so each model here is
retrained under the frozen protocol with its original seed and is accepted only
when its test RMSE reproduces the benchmark ledger. A randomly initialized
control is measured alongside, which is what distinguishes a structural limit
from an accident of one particular set of weights.
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

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import (
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
from scripts.run_robustness_campaign import MODELS, RMSE_TOLERANCE
from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)
from src.metrics.forecast import rmse
from src.training.engine import predict, resolve_device, set_seed, train_regressor

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LEDGER = ROOT / "results" / "aggregated" / "cet_full_runs.csv"
RUN_NAME = "cet_receptive_field"
# Large enough to clear any activation's dead zone, so a reachable timestep
# cannot be missed for want of signal.
PROBE = 10.0
PROBE_TOLERANCE = 1e-6


def code_sha256() -> str:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts" / "run_cet_benchmark.py",
        ROOT / "scripts" / "run_robustness_campaign.py",
        *sorted((ROOT / "src").rglob("*.py")),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def forecast(model, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    return out["point"] if isinstance(out, dict) else out


def receptive_field(model, history: int, device) -> int:
    """Trailing timesteps that can still change the forecast.

    Sweeping every offset rather than stopping at the first unreachable one
    matters: a model could in principle skip a position and read an earlier
    one, and only the furthest reachable offset bounds what the model sees.
    """
    model = model.eval()
    base = torch.zeros(1, history, 1, device=device)
    with torch.no_grad():
        reference = forecast(model, base)
        reach = 0
        for offset in range(history):
            probe = base.clone()
            probe[0, history - 1 - offset, 0] = PROBE
            if not torch.allclose(forecast(model, probe), reference, atol=PROBE_TOLERANCE):
                reach = offset + 1
    return int(reach)


def main(config, *, horizons=None, models=None, seed=None, device=None):
    cfg = load_config(config)
    cfg_hash = config_sha256(cfg)
    source_hash = code_sha256()
    data_hash = str(cfg["dataset"].get("sha256", "")).lower()
    resolved_device = resolve_device(device)

    if not BENCHMARK_LEDGER.exists():
        raise SystemExit(f"missing {BENCHMARK_LEDGER.relative_to(ROOT)}")
    ledger = pd.read_csv(BENCHMARK_LEDGER)
    ledger = ledger[ledger.status.eq("ok")]
    expected = {
        (str(r.model), int(r.horizon), int(r.seed)): float(r.rmse) for r in ledger.itertuples()
    }
    benchmark_code = sorted({str(c) for c in ledger.code_sha256.unique()})
    if len(benchmark_code) != 1:
        raise SystemExit(f"ledger mixes code fingerprints: {benchmark_code}")

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
    history = int(cfg["window"]["history"])

    record_dir = ROOT / "results" / "robustness" / "runs" / RUN_NAME
    aggregated_dir = ROOT / "results" / "robustness" / "aggregated"
    for directory in (record_dir, aggregated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_horizons = horizons or cfg["window"]["horizons"]
    run_models = models or MODELS
    run_seed = int(seed if seed is not None else cfg["training"]["seeds"][0])
    rows = []
    for name in run_models:
        # The control shares the architecture but not the training, so a
        # difference between the two would mean the reach depends on weights.
        set_seed(0, deterministic=False, warn_only=True)
        control = build_model(name, history, run_horizons[0], seed=0).to(resolved_device)
        control_reach = receptive_field(control, history, resolved_device)
        for horizon in run_horizons:
            x, y, origins = sliding_windows(
                standardized,
                history,
                horizon,
                timestamps=dates,
                expected_step=expected_step,
            )
            masks = assign_windows_by_target_origin(origins, split, horizon)
            sets = {key: (x[mask], y[mask]) for key, mask in masks.items()}
            test_x, test_y = sets["test"]
            test_y_raw = inverse_target(scaler, test_y)

            # Match the benchmark's call order exactly: seeding, model
            # construction, and loader creation all draw on the global RNG.
            set_seed(
                run_seed,
                deterministic=cfg["training"]["deterministic_algorithms"],
                warn_only=cfg["training"]["deterministic_warn_only"],
            )
            row = {
                "dataset": cfg["dataset"]["name"],
                "model": name,
                "horizon": int(horizon),
                "seed": run_seed,
                "history": history,
                "status": "ok",
                "config_sha256": cfg_hash,
                "code_sha256": source_hash,
                "dataset_sha256": data_hash,
                "benchmark_code_sha256": benchmark_code[0],
                **environment(resolved_device),
            }
            try:
                model = build_model(name, history, horizon, run_seed)
                start = time.perf_counter()
                model, _, seconds = train_regressor(
                    model,
                    loader(*sets["train"], cfg["training"]["batch_size"], True),
                    loader(*sets["val"], cfg["training"]["batch_size"]),
                    epochs=cfg["training"]["epochs"],
                    lr=cfg["training"]["learning_rate"],
                    patience=cfg["training"]["patience"],
                    optimizer_name=cfg["training"].get("optimizer", "adamw"),
                    weight_decay=cfg["training"].get("weight_decay"),
                    device=resolved_device,
                )
                standardized_prediction, _ = predict(
                    model,
                    loader(test_x, test_y, cfg["training"]["batch_size"]),
                    resolved_device,
                )
                achieved = float(
                    rmse(test_y_raw, inverse_target(scaler, standardized_prediction))
                )
                target = expected.get((name, int(horizon), run_seed))
                if target is None or abs(achieved - target) > RMSE_TOLERANCE:
                    # Without this the reach would describe some other model,
                    # which is exactly the claim the paper cannot afford to make.
                    raise RuntimeError(
                        f"test RMSE {achieved:.9f} does not reproduce the ledger "
                        f"value {target}"
                    )
                reach = receptive_field(model, history, resolved_device)
                row.update(
                    rmse=achieved,
                    benchmark_rmse=target,
                    rmse_gap=abs(achieved - target),
                    reproduced=True,
                    receptive_field=reach,
                    history_used=reach / history,
                    control_receptive_field=control_reach,
                    train_seconds=float(seconds),
                    wall_seconds=float(time.perf_counter() - start),
                )
                rows.append(row)
                print(
                    f"OK {name} h={horizon} seed={run_seed} rmse={achieved:.6f} "
                    f"sees {reach}/{history} (untrained control {control_reach})"
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED {name} h={horizon} seed={run_seed}: {exc}")
            atomic_write_json(record_dir / f"{name}_h{horizon}_s{run_seed}.json", row)

    ok = [row for row in rows if row["status"] == "ok"]
    if not ok:
        raise SystemExit("no receptive-field rows were produced")
    frame = pd.DataFrame(ok)
    out = aggregated_dir / "cet_receptive_fields.csv"
    frame.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(ROOT)}")
    spread = frame.groupby("model").receptive_field.nunique()
    if (spread > 1).any():
        print(f"  note: reach varies with horizon for {list(spread[spread > 1].index)}")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cet.yaml")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(
        args.config,
        horizons=args.horizons,
        models=args.models,
        seed=args.seed,
        device=args.device,
    )
