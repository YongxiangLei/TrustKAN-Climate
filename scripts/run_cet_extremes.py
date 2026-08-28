"""Evaluate the pre-registered extreme subsets on the CET benchmark artifacts.

Thresholds come from training-period targets only, as the frozen policy in
`configs/robustness.yaml` requires, so the split is rebuilt here with the same
primitives the benchmark runner uses rather than reconstructed by hand. Nothing
is retrained: every model's stored test predictions are re-scored on the cold,
warm, and complement subsets.

Test predictions are compared against the same origins for every model, so the
subset assignment is shared and differences between models are attributable to
the models.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

import _bootstrap  # noqa: F401  # repository-root import setup
from run_cet_benchmark import inverse_target, prepare_series
from src.data.timeseries import (
    TrainOnlyStandardizer,
    assign_windows_by_target_origin,
    chronological_split,
    sliding_windows,
)
from src.extremes.subsets import evaluate_extremes

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "cet_full"

import pandas as pd


def training_targets(cfg, horizon):
    """Training-region targets in physical units for one horizon."""
    path = ROOT / "data" / "raw" / Path(cfg["dataset"]["source"]).name
    if not path.exists():
        from run_cet_benchmark import ensure_data

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
    _, y, origins = sliding_windows(
        standardized,
        cfg["window"]["history"],
        horizon,
        timestamps=dates,
        expected_step=expected_step,
    )
    masks = assign_windows_by_target_origin(origins, split, horizon)
    return inverse_target(scaler, y[masks["train"]])


def display_path(path: Path) -> str:
    """Repository-relative when possible, so a relative argument cannot abort a run."""
    resolved = path.resolve()
    if resolved.is_relative_to(ROOT):
        return resolved.relative_to(ROOT).as_posix()
    return resolved.as_posix()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def resolve_artifact(raws: list[Path], name: str) -> Path | None:
    """First matching artifact across raw directories, so a later campaign can
    quote models it did not re-run without mixing two copies of the same file.
    """
    for raw in raws:
        candidate = raw / name
        if candidate.exists():
            return candidate
    return None


def discover_models(raws: list[Path]) -> list[str]:
    """Model names present in any raw directory, parsed from the artifact stem.

    A later campaign's config may omit models it quotes from an earlier one
    (the classical shard) or add models the earlier config never named (the
    corrected architecture). Scoring only `cfg["models"]` would silently drop
    one of those sets.
    """
    found = set()
    for raw in raws:
        for path in raw.glob("cet_*_h*_s*.npz"):
            stem = path.stem
            body = stem[len("cet_") :]
            marker = body.rfind("_h")
            if marker <= 0:
                continue
            found.add(body[:marker])
    return sorted(found)


def main(config: Path, policy_path: Path, outdir: Path, raws=None) -> None:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))["extreme"]
    raws = [Path(item) for item in (raws or [RAW])]
    models = sorted(set(cfg["models"]) | set(discover_models(raws)))
    print(f"scoring {len(models)} models: {', '.join(models)}")
    rows = []
    for horizon in cfg["window"]["horizons"]:
        train_target = training_targets(cfg, horizon)
        print(f"\nhorizon {horizon}: {train_target.shape[0]} training origins")
        for model in models:
            seeds = [-1] if model in {"persistence", "svr"} else cfg["training"]["seeds"]
            for seed in seeds:
                artifact = resolve_artifact(raws, f"cet_{model}_h{horizon}_s{seed}.npz")
                if artifact is None:
                    continue
                with np.load(artifact, allow_pickle=False) as source:
                    target = source["target"]
                    prediction = source["prediction"]
                    fingerprint = str(source["code_sha256"].item())
                result = evaluate_extremes(
                    train_target,
                    target,
                    prediction,
                    lower_quantile=policy["lower_quantile"],
                    upper_quantile=policy["upper_quantile"],
                    definition=policy["definition"],
                    min_origins=policy["min_origins"],
                )
                subsets = result["subsets"]
                rows.append(
                    {
                        "model": model,
                        "horizon": horizon,
                        "seed": seed,
                        "code_sha256": fingerprint,
                        "thresholds_from": policy["thresholds_from"],
                        "lower_threshold": result["thresholds"]["lower"],
                        "upper_threshold": result["thresholds"]["upper"],
                        "n_test_origins": result["n_test_origins"],
                        **{
                            f"{name}_{field}": subsets[name][field]
                            for name in ("cold", "warm", "either", "complement")
                            for field in ("n_origins", "rmse", "underpowered")
                        },
                    }
                )
        done = [r for r in rows if r["horizon"] == horizon]
        if done:
            example = done[0]
            print(
                f"  thresholds {example['lower_threshold']:.2f} / "
                f"{example['upper_threshold']:.2f} degC; "
                f"cold {example['cold_n_origins']}, warm {example['warm_n_origins']}, "
                f"complement {example['complement_n_origins']} origins"
            )
    frame = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(outdir / "cet_extremes_runs.csv", index=False)
    atomic_json(
        outdir / "cet_extremes_policy.json",
        {"policy": policy, "config": display_path(config)},
    )
    summary = frame.groupby(["model", "horizon"])[
        ["cold_rmse", "warm_rmse", "either_rmse", "complement_rmse"]
    ].mean()
    summary.to_csv(outdir / "cet_extremes_summary.csv")
    print("\nMean RMSE by subset (degC):")
    print(summary.round(3).to_string())
    print(f"\nwrote {display_path(outdir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "cet.yaml"))
    parser.add_argument("--policy", default=str(ROOT / "configs" / "robustness.yaml"))
    parser.add_argument("--outdir", default=str(ROOT / "results" / "extremes"))
    parser.add_argument(
        "--raw",
        action="append",
        dest="raws",
        help=(
            "Directory of stored test predictions to re-score; no model is retrained. "
            "Pass more than once to search later directories for models the first "
            "campaign did not produce; the first match wins."
        ),
    )
    args = parser.parse_args()
    main(
        Path(args.config),
        Path(args.policy),
        Path(args.outdir),
        [Path(item) for item in (args.raws or [str(RAW)])],
    )
