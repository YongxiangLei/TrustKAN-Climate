"""Extract and audit TrustKAN's univariate KAN curves on CET.

The benchmark runner stored predictions rather than weights, so the curves that
the interpretability claim rests on cannot be read off the completed campaign.
Training is deterministic under the frozen protocol, so this runner retrains
each TrustKAN run with the identical seed, model, and loader sequence and then
*verifies* that it reproduced the run: a curve set is only written if the
retrained model's test RMSE matches the benchmark ledger. That check is what
licenses treating these curves as the curves of the reported models.

Two properties are measured. The first is how much of each learned univariate
map is actually nonlinear, since a KAN layer is a linear map plus an RBF
expansion by construction and a near-linear curve carries no explanation the
linear term did not already carry. The second is whether curves are stable
across seeds, both index-matched (what a reader comparing "curve k" across runs
would see) and under a best-match upper bound that forgives the permutation
freedom of the latent channels.
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
from src.interpretability.kan_curves import (
    default_evaluation_grid,
    extract_trustkan_curves,
)
from src.interpretability.stability import curve_correlation, normalized_curve_distance
from src.metrics.forecast import rmse
from src.models.trustkan import TrustKAN
from src.training.engine import predict, resolve_device, set_seed, train_regressor

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_LEDGER = ROOT / "results" / "aggregated" / "cet_full_runs.csv"
RUN_NAME = "cet_kan_curves"
# The retrained model must land on the reported RMSE. Deterministic training
# makes this exact in principle; the tolerance only absorbs the last bits of
# float accumulation in a different summation order.
RMSE_TOLERANCE = 1e-6


def code_sha256() -> str:
    # The benchmark runner supplies the data preparation and training path, so
    # a change there changes these curves and must move the fingerprint.
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


def curve_parts(model: TrustKAN, grid):
    """Split each univariate map into its linear and RBF contributions.

    `src.interpretability.kan_curves` returns the sum, which is the object the
    manuscript plots. Decomposing it here (rather than editing `src/`, which
    would invalidate every completed run's code fingerprint) lets us ask how
    much of the curve the RBF expansion is responsible for. The two parts are
    checked against the shared primitive so this stays the same object.
    """
    layer = model.encoder.kan
    x = torch.as_tensor(grid, dtype=layer.coeff.dtype, device=layer.coeff.device)
    scale = layer.log_scale.exp().clamp_min(1e-4)
    basis = torch.exp(-0.5 * ((x[:, None, None] - layer.grid) / scale) ** 2)
    rbf = torch.einsum("xig,oig->oix", basis, layer.coeff)
    linear = layer.base.weight[:, :, None] * x[None, None, :]
    shape = (layer.out_features * layer.in_features, -1)
    linear = linear.detach().cpu().numpy().reshape(shape)
    rbf = rbf.detach().cpu().numpy().reshape(shape)
    reference = extract_trustkan_curves(model, x_grid=grid)["curves"]
    if not np.allclose(linear + rbf, reference, atol=1e-5):
        raise RuntimeError("curve decomposition does not reproduce the shared extractor")
    return linear, rbf, reference


def nonlinear_share(linear, rbf):
    """Fraction of each curve's variation contributed by the RBF expansion.

    Both parts are measured by their spread over the evaluation grid, so a
    constant offset does not count as structure.
    """
    linear_spread = linear.std(axis=1)
    rbf_spread = rbf.std(axis=1)
    total = linear_spread + rbf_spread
    return np.divide(rbf_spread, total, out=np.zeros_like(total), where=total > 0)


def standardize_rows(curves):
    """Z-score each curve over the grid so correlations become a dot product."""
    centred = curves - curves.mean(axis=1, keepdims=True)
    spread = np.linalg.norm(centred, axis=1, keepdims=True)
    return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 0)


def best_match_correlation(curves_a, curves_b):
    """For each curve in A, the largest correlation with any curve in B.

    This is an upper bound on what any post-hoc alignment could achieve, and it
    does not require the match to be one-to-one. If index-matched stability is
    poor while this is high, the curves are reproducible but not identifiable:
    the same functions reappear without a stable name.
    """
    similarity = standardize_rows(curves_a).astype(np.float32) @ standardize_rows(
        curves_b
    ).astype(np.float32).T
    return similarity.max(axis=1)


def training_sets(cfg, standardized, dates, split, horizon, expected_step):
    x, y, origins = sliding_windows(
        standardized,
        cfg["window"]["history"],
        horizon,
        timestamps=dates,
        expected_step=expected_step,
    )
    masks = assign_windows_by_target_origin(origins, split, horizon)
    return {name: (x[mask], y[mask]) for name, mask in masks.items()}


def main(config, *, horizons=None, seeds=None, device=None, resume=False):
    cfg = load_config(config)
    cfg_hash = config_sha256(cfg)
    source_hash = code_sha256()
    data_hash = str(cfg["dataset"].get("sha256", "")).lower()
    resolved_device = resolve_device(device)

    if not BENCHMARK_LEDGER.exists():
        raise SystemExit(f"missing {BENCHMARK_LEDGER.relative_to(ROOT)}")
    ledger = pd.read_csv(BENCHMARK_LEDGER)
    reference = ledger[ledger.model.eq("trustkan") & ledger.status.eq("ok")]
    if reference.empty:
        raise SystemExit("no successful TrustKAN runs in the benchmark ledger")
    expected_rmse = {
        (int(row.horizon), int(row.seed)): float(row.rmse)
        for row in reference.itertuples()
    }
    benchmark_fingerprint = str(reference.code_sha256.iloc[0])

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
    grid = default_evaluation_grid()

    raw_dir = ROOT / "results" / "interpretability" / "raw" / RUN_NAME
    record_dir = ROOT / "results" / "interpretability" / "runs" / RUN_NAME
    aggregated_dir = ROOT / "results" / "interpretability" / "aggregated"
    for directory in (raw_dir, record_dir, aggregated_dir):
        directory.mkdir(parents=True, exist_ok=True)

    run_horizons = horizons or cfg["window"]["horizons"]
    run_seeds = seeds or cfg["training"]["seeds"]
    rows = []
    for horizon in run_horizons:
        sets = training_sets(cfg, standardized, dates, split, horizon, expected_step)
        test_y_raw = inverse_target(scaler, sets["test"][1])
        for seed in run_seeds:
            artifact = raw_dir / f"kan_curves_h{horizon}_s{seed}.npz"
            record = record_dir / f"kan_curves_h{horizon}_s{seed}.json"
            if resume and artifact.is_file() and record.is_file():
                stored = json.loads(record.read_text(encoding="utf-8"))
                if (
                    stored.get("status") == "ok"
                    and stored.get("code_sha256") == source_hash
                    and stored.get("config_sha256") == cfg_hash
                ):
                    rows.append(stored)
                    print(f"RESUME h={horizon} seed={seed}")
                    continue

            # Reproduce the benchmark's call sequence exactly: seeding, model
            # construction, and loader creation all draw on the global RNG, so
            # their order is part of the protocol.
            set_seed(
                seed,
                deterministic=cfg["training"]["deterministic_algorithms"],
                warn_only=cfg["training"]["deterministic_warn_only"],
            )
            row = {
                "dataset": cfg["dataset"]["name"],
                "model": "trustkan",
                "horizon": int(horizon),
                "seed": int(seed),
                "status": "ok",
                "config_sha256": cfg_hash,
                "code_sha256": source_hash,
                "dataset_sha256": data_hash,
                "benchmark_code_sha256": benchmark_fingerprint,
                "artifact_path": artifact.relative_to(ROOT).as_posix(),
                **environment(resolved_device),
            }
            try:
                model = build_model("trustkan", cfg["window"]["history"], horizon, seed)
                train_loader = loader(*sets["train"], cfg["training"]["batch_size"], True)
                val_loader = loader(*sets["val"], cfg["training"]["batch_size"])
                test_loader = loader(*sets["test"], cfg["training"]["batch_size"])
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
                pred_std, _ = predict(model, test_loader, resolved_device)
                achieved = rmse(test_y_raw, inverse_target(scaler, pred_std))
                target = expected_rmse.get((int(horizon), int(seed)))
                row["rmse"] = float(achieved)
                row["benchmark_rmse"] = target
                row["rmse_gap"] = None if target is None else float(abs(achieved - target))
                row["reproduced"] = bool(
                    target is not None and abs(achieved - target) <= RMSE_TOLERANCE
                )
                row["train_seconds"] = float(seconds)
                row["wall_seconds"] = float(time.perf_counter() - start)
                if not row["reproduced"]:
                    # Curves from a model that is not the reported model would
                    # silently misattribute the interpretation, so refuse them.
                    raise RuntimeError(
                        f"retrained RMSE {achieved:.9f} does not reproduce the "
                        f"ledger value {target}"
                    )

                linear, rbf, curves = curve_parts(model, grid)
                share = nonlinear_share(linear, rbf)
                row.update(
                    n_curves=int(curves.shape[0]),
                    grid_points=int(curves.shape[1]),
                    nonlinear_share_mean=float(share.mean()),
                    nonlinear_share_median=float(np.median(share)),
                    nonlinear_share_p90=float(np.quantile(share, 0.90)),
                    curve_spread_mean=float(curves.std(axis=1).mean()),
                )
                atomic_savez(
                    artifact,
                    x_grid=grid,
                    curves=curves.astype(np.float32),
                    linear_part=linear.astype(np.float32),
                    rbf_part=rbf.astype(np.float32),
                    nonlinear_share=share,
                    horizon=np.asarray(horizon),
                    seed=np.asarray(seed),
                    rmse=np.asarray(achieved),
                    benchmark_rmse=np.asarray(target),
                    config_sha256=np.asarray(cfg_hash),
                    code_sha256=np.asarray(source_hash),
                    dataset_sha256=np.asarray(data_hash),
                    training_history_json=np.asarray(json.dumps(history)),
                )
                print(
                    f"OK h={horizon} seed={seed} rmse={achieved:.6f} "
                    f"(ledger {target:.6f}) nonlinear share={share.mean():.3f}"
                )
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                print(f"FAILED h={horizon} seed={seed}: {exc}")
            atomic_write_json(record, row)
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(aggregated_dir / f"{RUN_NAME}_runs.csv", index=False)
    failed = frame[frame.status.ne("ok")]
    if not failed.empty:
        raise SystemExit(f"{len(failed)} curve extraction run(s) failed; see the ledger")
    stability(raw_dir, aggregated_dir, run_horizons, run_seeds)
    return rows


def stability(raw_dir: Path, aggregated_dir: Path, horizons, seeds) -> None:
    """Cross-seed agreement of the learned curves, index-matched and best-match."""
    records = []
    for horizon in horizons:
        loaded = {}
        for seed in seeds:
            path = raw_dir / f"kan_curves_h{horizon}_s{seed}.npz"
            if path.is_file():
                with np.load(path, allow_pickle=False) as source:
                    loaded[seed] = np.asarray(source["curves"], dtype=float)
        for i, first in enumerate(sorted(loaded)):
            for second in sorted(loaded)[i + 1 :]:
                a, b = loaded[first], loaded[second]
                records.append(
                    {
                        "horizon": int(horizon),
                        "seed_a": int(first),
                        "seed_b": int(second),
                        "matched_correlation_mean": float(
                            np.nanmean(curve_correlation(a, b))
                        ),
                        "matched_distance_mean": float(
                            np.nanmean(normalized_curve_distance(a, b))
                        ),
                        "best_match_correlation_mean": float(
                            np.nanmean(best_match_correlation(a, b))
                        ),
                    }
                )
                print(
                    f"  h={horizon} seeds {first}/{second}: "
                    f"matched r={records[-1]['matched_correlation_mean']:.3f} "
                    f"best-match r={records[-1]['best_match_correlation_mean']:.3f}"
                )
    if not records:
        print("  no curve pairs available for stability analysis")
        return
    frame = pd.DataFrame(records)
    frame.to_csv(aggregated_dir / f"{RUN_NAME}_stability.csv", index=False)
    print(f"  wrote {(aggregated_dir / f'{RUN_NAME}_stability.csv').relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cet.yaml")
    parser.add_argument("--horizon", action="append", type=int, dest="horizons")
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--stability-only",
        action="store_true",
        help="Recompute cross-seed stability from existing curve artifacts.",
    )
    args = parser.parse_args()
    if args.stability_only:
        cfg = load_config(args.config)
        stability(
            ROOT / "results" / "interpretability" / "raw" / RUN_NAME,
            ROOT / "results" / "interpretability" / "aggregated",
            args.horizons or cfg["window"]["horizons"],
            args.seeds or cfg["training"]["seeds"],
        )
    else:
        main(
            args.config,
            horizons=args.horizons,
            seeds=args.seeds,
            device=args.device,
            resume=args.resume,
        )
