"""Validate immutable per-run artifacts and build publication summaries."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED = {
    "dataset",
    "model",
    "horizon",
    "seed",
    "split",
    "status",
    "rmse",
    "mae",
    "parameters",
    "train_seconds",
    "inference_ms",
    "validation_rmse",
    "search_seconds",
    "selected_hyperparameters",
    "config_sha256",
    "code_sha256",
    "artifact_path",
}
KEY = ["dataset", "model", "horizon", "seed", "split"]
ARTIFACT_REQUIRED = {
    "prediction",
    "target",
    "target_time",
    "target_origin",
    "dataset",
    "model",
    "horizon",
    "seed",
    "split",
    "config_sha256",
    "code_sha256",
    "model_selection_json",
    "selected_hyperparameters_json",
}
DETERMINISTIC = {"persistence", "svr"}


def validate_artifact(row):
    path = Path(row.artifact_path)
    if not path.is_file():
        raise ValueError(f"Missing artifact for successful run: {path}")
    with np.load(path, allow_pickle=False) as artifact:
        missing = ARTIFACT_REQUIRED - set(artifact.files)
        if missing:
            raise ValueError(f"Artifact {path} is missing fields: {sorted(missing)}")
        prediction = artifact["prediction"]
        target = artifact["target"]
        target_time = artifact["target_time"]
        if prediction.shape != target.shape or target_time.shape != target.shape:
            raise ValueError(f"Artifact {path} has inconsistent prediction/target/time shapes")
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f"Artifact {path} contains non-finite predictions or targets")
        metadata = {
            "dataset": str(artifact["dataset"]),
            "model": str(artifact["model"]),
            "horizon": int(artifact["horizon"]),
            "seed": int(artifact["seed"]),
            "split": str(artifact["split"]),
            "config_sha256": str(artifact["config_sha256"]),
            "code_sha256": str(artifact["code_sha256"]),
        }
        for field, observed in metadata.items():
            expected = getattr(row, field)
            if field in {"horizon", "seed"}:
                expected = int(expected)
            else:
                expected = str(expected)
            if observed != expected:
                raise ValueError(
                    f"Artifact {path} metadata mismatch for {field}: {observed!r} != {expected!r}"
                )


def validate_ledger(df, min_seeds=1):
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    if df.duplicated(KEY).any():
        duplicates = df.loc[df.duplicated(KEY, keep=False), KEY]
        raise ValueError(f"Duplicate run keys detected:\n{duplicates.to_string(index=False)}")
    ok = df[df.status.eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful runs to aggregate")
    if not ok.split.eq("test").all():
        raise ValueError("Benchmark aggregation accepts final-test rows only")
    for row in ok.itertuples(index=False):
        validate_artifact(row)
    if min_seeds > 1:
        stochastic = ok[~ok.model.isin(DETERMINISTIC)]
        counts = stochastic.groupby(["dataset", "model", "horizon", "split"]).seed.nunique()
        insufficient = counts[counts < min_seeds]
        if not insufficient.empty:
            raise ValueError(
                f"Runs do not meet the required {min_seeds} unique seeds:\n{insufficient.to_string()}"
            )
    return ok


def main(path, outdir, min_seeds=1):
    df = pd.read_csv(path)
    ok = validate_ledger(df, min_seeds=min_seeds)
    group = ["dataset", "model", "horizon"]
    summary = ok.groupby(group, as_index=False).agg(
        n=("seed", "nunique"),
        rmse_mean=("rmse", "mean"),
        rmse_sd=("rmse", "std"),
        mae_mean=("mae", "mean"),
        mae_sd=("mae", "std"),
        parameters=("parameters", "mean"),
        train_seconds_mean=("train_seconds", "mean"),
        inference_ms_mean=("inference_ms", "mean"),
        search_seconds_mean=("search_seconds", "mean"),
    )
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "benchmark_summary.csv", index=False)
    latex = summary.copy()
    latex["RMSE"] = latex.apply(
        lambda row: f"{row.rmse_mean:.4f} ± {row.rmse_sd:.4f}"
        if pd.notna(row.rmse_sd)
        else f"{row.rmse_mean:.4f}",
        axis=1,
    )
    latex["MAE"] = latex.apply(
        lambda row: f"{row.mae_mean:.4f} ± {row.mae_sd:.4f}"
        if pd.notna(row.mae_sd)
        else f"{row.mae_mean:.4f}",
        axis=1,
    )
    latex[["dataset", "model", "horizon", "n", "RMSE", "MAE"]].to_latex(
        out / "benchmark_summary.tex", index=False, escape=True
    )
    df[~df.status.eq("ok")].to_csv(out / "failed_runs.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/aggregated/cet_full_runs.csv")
    parser.add_argument("--outdir", default="results/tables")
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Minimum unique seeds per non-persistence model (use 5 for paper tables).",
    )
    args = parser.parse_args()
    main(args.input, args.outdir, args.min_seeds)
