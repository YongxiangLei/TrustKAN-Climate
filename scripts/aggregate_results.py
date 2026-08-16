"""Validate immutable per-run artifacts and build publication summaries."""
from __future__ import annotations

import argparse
import hashlib
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
OPTIONAL_ARTIFACT_METADATA = {
    "protocol": "protocol",
    "target_region": "target_region",
    "target_station": "target_station",
    "source_regions": "source_regions_json",
    "source_stations": "source_stations_json",
    "source_pooling": "source_pooling",
    "normalization": "normalization",
    "dataset_sha256": "dataset_sha256",
}


def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact(row):
    path = Path(row.artifact_path)
    if not path.is_file():
        raise ValueError(f"Missing artifact for successful run: {path}")
    if "artifact_sha256" in row._fields and pd.notna(row.artifact_sha256):
        observed_hash = file_sha256(path)
        if observed_hash != str(row.artifact_sha256):
            raise ValueError(
                f"Artifact checksum mismatch for {path}: "
                f"{observed_hash!r} != {str(row.artifact_sha256)!r}"
            )
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
        row_fields = set(row._fields)
        for ledger_field, artifact_field in OPTIONAL_ARTIFACT_METADATA.items():
            if ledger_field not in row_fields or pd.isna(getattr(row, ledger_field)):
                continue
            if artifact_field not in artifact.files:
                raise ValueError(f"Artifact {path} is missing metadata field {artifact_field}")
            observed = str(artifact[artifact_field])
            expected = str(getattr(row, ledger_field))
            if observed != expected:
                raise ValueError(
                    f"Artifact {path} metadata mismatch for {ledger_field}: "
                    f"{observed!r} != {expected!r}"
                )


def validate_region_coverage(all_runs, min_regions):
    if min_regions is None:
        return
    required = {"protocol", "target_region", "model", "horizon"}
    if not required.issubset(all_runs.columns):
        raise ValueError("Regional coverage enforcement requires GHCN protocol metadata")
    ok = (
        all_runs[all_runs.status.eq("ok")]
        if "status" in all_runs.columns
        else all_runs
    )
    expected = pd.MultiIndex.from_frame(
        all_runs[["protocol", "model", "horizon"]].drop_duplicates()
    )
    counts = ok.groupby(["protocol", "model", "horizon"]).target_region.nunique()
    counts = counts.reindex(expected, fill_value=0)
    insufficient = counts[counts < min_regions]
    if not insufficient.empty:
        raise ValueError(
            f"Runs do not meet the required {min_regions} target regions:\n"
            f"{insufficient.to_string()}"
        )


def validate_ledger(df, min_seeds=1, min_regions=None):
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
        stochastic_all = df[~df.model.isin(DETERMINISTIC)]
        stochastic = ok[~ok.model.isin(DETERMINISTIC)]
        group = ["dataset", "model", "horizon", "split"]
        expected = pd.MultiIndex.from_frame(stochastic_all[group].drop_duplicates())
        counts = stochastic.groupby(group).seed.nunique().reindex(expected, fill_value=0)
        insufficient = counts[counts < min_seeds]
        if not insufficient.empty:
            raise ValueError(
                f"Runs do not meet the required {min_seeds} unique seeds:\n{insufficient.to_string()}"
            )
    validate_region_coverage(df, min_regions)
    return ok


def regional_macro_summary(ok):
    required = {"protocol", "target_region", "model", "horizon", "seed", "rmse", "mae"}
    if not required.issubset(ok.columns):
        return None
    region_level = ok.groupby(
        ["protocol", "target_region", "model", "horizon"], as_index=False
    ).agg(
        n_seeds=("seed", "nunique"),
        region_rmse=("rmse", "mean"),
        region_mae=("mae", "mean"),
    )
    return region_level.groupby(
        ["protocol", "model", "horizon"], as_index=False
    ).agg(
        n_regions=("target_region", "nunique"),
        min_seeds_per_region=("n_seeds", "min"),
        max_seeds_per_region=("n_seeds", "max"),
        rmse_macro_mean=("region_rmse", "mean"),
        rmse_between_region_sd=("region_rmse", "std"),
        mae_macro_mean=("region_mae", "mean"),
        mae_between_region_sd=("region_mae", "std"),
    )


def main(path, outdir, min_seeds=1, min_regions=None):
    df = pd.read_csv(path)
    ok = validate_ledger(df, min_seeds=min_seeds, min_regions=min_regions)
    group = [
        column
        for column in ("protocol", "dataset", "target_region", "model", "horizon")
        if column in ok.columns and not ok[column].isna().all()
    ]
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
    macro = regional_macro_summary(ok)
    if macro is not None:
        macro.to_csv(out / "benchmark_macro_summary.csv", index=False)
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
    identity = [
        column
        for column in ("protocol", "dataset", "target_region", "model", "horizon")
        if column in latex.columns
    ]
    latex[[*identity, "n", "RMSE", "MAE"]].to_latex(
        out / "benchmark_summary.tex", index=False, escape=True
    )
    df[~df.status.eq("ok")].to_csv(out / "failed_runs.csv", index=False)
    print(summary.to_string(index=False))
    if macro is not None:
        print("\nEqual-region macro summary")
        print(macro.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/aggregated/cet_full_runs.csv")
    parser.add_argument("--outdir", default="results/tables")
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=1,
        help="Minimum unique seeds per stochastic model (use 5 for paper tables).",
    )
    parser.add_argument(
        "--min-regions",
        type=int,
        default=None,
        help="Required target-region count for every protocol/model/horizon group.",
    )
    args = parser.parse_args()
    main(args.input, args.outdir, args.min_seeds, args.min_regions)
