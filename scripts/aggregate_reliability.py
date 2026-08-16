"""Validate reliability artifacts and build equal-region publication summaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from src.data.provenance import file_sha256
from src.metrics.forecast import aurc, mae, rmse, sample_risk_coverage_curve
from src.uncertainty.conformal import (
    interval_coverage,
    joint_interval_coverage,
    mean_interval_score,
    mean_interval_width,
)


KEY = ["dataset", "horizon", "seed", "split"]
REQUIRED = {
    "dataset",
    "protocol",
    "target_region",
    "target_station",
    "model",
    "horizon",
    "seed",
    "split",
    "status",
    "nominal_coverage",
    "n_calibration",
    "n_test",
    "rmse",
    "mae",
    "conformal_marginal_coverage",
    "conformal_joint_coverage",
    "conformal_mean_width",
    "conformal_interval_score",
    "simultaneous_joint_coverage",
    "simultaneous_mean_width",
    "simultaneous_interval_score",
    "fused_aurc",
    "fused_selected_coverage",
    "fused_selected_rmse",
    "config_sha256",
    "code_sha256",
    "dataset_sha256",
    "artifact_sha256",
    "artifact_path",
}
ARTIFACT_REQUIRED = {
    "target",
    "prediction",
    "target_time",
    "target_origin",
    "calibration_target",
    "calibration_target_time",
    "calibration_target_origin",
    "test_lower_horizonwise",
    "test_upper_horizonwise",
    "test_lower_simultaneous",
    "test_upper_simultaneous",
    "fused_reliability",
    "fused_mask",
    "fused_coverage",
    "fused_risk",
    "conformal_alpha",
    "calibration_state_json",
    "dataset",
    "protocol",
    "target_region",
    "target_station",
    "model",
    "horizon",
    "seed",
    "split",
    "source_regions_json",
    "source_stations_json",
    "source_pooling",
    "normalization",
    "config_sha256",
    "code_sha256",
    "dataset_sha256",
}


def _assert_close(path, field, observed, expected, atol=1e-10):
    if expected is None or pd.isna(expected):
        if observed is not None:
            raise ValueError(f"Artifact {path} unexpectedly defines {field}")
        return
    if not np.isclose(float(observed), float(expected), rtol=1e-9, atol=atol):
        raise ValueError(
            f"Artifact {path} metric mismatch for {field}: {observed} != {expected}"
        )


def validate_artifact(row):
    path = Path(row.artifact_path)
    if not path.is_file():
        raise ValueError(f"Missing reliability artifact: {path}")
    observed_hash = file_sha256(path)
    if observed_hash != str(row.artifact_sha256):
        raise ValueError(f"Reliability artifact checksum mismatch for {path}")
    with np.load(path, allow_pickle=False) as artifact:
        missing = ARTIFACT_REQUIRED - set(artifact.files)
        if missing:
            raise ValueError(f"Artifact {path} is missing fields: {sorted(missing)}")
        target = artifact["target"]
        prediction = artifact["prediction"]
        target_time = artifact["target_time"]
        target_origin = artifact["target_origin"]
        cal_target = artifact["calibration_target"]
        cal_time = artifact["calibration_target_time"]
        cal_origin = artifact["calibration_target_origin"]
        lower_h = artifact["test_lower_horizonwise"]
        upper_h = artifact["test_upper_horizonwise"]
        lower_s = artifact["test_lower_simultaneous"]
        upper_s = artifact["test_upper_simultaneous"]
        horizon = int(artifact["horizon"])
        if horizon <= 0 or len(target) == 0 or len(cal_target) == 0:
            raise ValueError(f"Artifact {path} has empty or invalid evaluation dimensions")
        expected_shape = (len(target), horizon)
        if target.shape != expected_shape or prediction.shape != expected_shape:
            raise ValueError(f"Artifact {path} has inconsistent target dimensions")
        for name, values in {
            "target_time": target_time,
            "test_lower_horizonwise": lower_h,
            "test_upper_horizonwise": upper_h,
            "test_lower_simultaneous": lower_s,
            "test_upper_simultaneous": upper_s,
        }.items():
            if values.shape != expected_shape:
                raise ValueError(f"Artifact {path} has inconsistent {name} dimensions")
        if (
            cal_target.ndim != 2
            or cal_target.shape != cal_time.shape
            or cal_target.shape[1] != horizon
        ):
            raise ValueError(f"Artifact {path} has inconsistent calibration dimensions")
        if len(target_origin) != len(target) or len(cal_origin) != len(cal_target):
            raise ValueError(f"Artifact {path} has misaligned forecast origins")
        if not np.issubdtype(target_time.dtype, np.datetime64) or not np.issubdtype(
            cal_time.dtype, np.datetime64
        ):
            raise ValueError(f"Artifact {path} timestamps are not datetime arrays")
        if not np.issubdtype(target_origin.dtype, np.number) or not np.issubdtype(
            cal_origin.dtype, np.number
        ):
            raise ValueError(f"Artifact {path} origins are not numeric arrays")
        if not np.isfinite(target_origin).all() or not np.isfinite(cal_origin).all():
            raise ValueError(f"Artifact {path} contains non-finite forecast origins")
        if np.any(np.diff(target_origin) <= 0) or np.any(np.diff(cal_origin) <= 0):
            raise ValueError(f"Artifact {path} forecast origins are not strictly increasing")
        if np.max(cal_time) >= np.min(target_time):
            raise ValueError(f"Artifact {path} calibration timestamps overlap test timestamps")
        if np.isnat(target_time).any() or np.isnat(cal_time).any():
            raise ValueError(f"Artifact {path} contains missing evaluation timestamps")
        reliability = artifact["fused_reliability"]
        stored_mask = artifact["fused_mask"]
        stored_coverage = artifact["fused_coverage"]
        stored_risk = artifact["fused_risk"]
        for name, values in {
            "fused_reliability": reliability,
            "fused_mask": stored_mask,
            "fused_coverage": stored_coverage,
            "fused_risk": stored_risk,
        }.items():
            if values.shape != (len(target),):
                raise ValueError(f"Artifact {path} has inconsistent {name} dimensions")
        if (
            not np.isfinite(reliability).all()
            or not np.isfinite(stored_coverage).all()
            or not np.isfinite(stored_risk).all()
            or np.any((reliability < 0.0) | (reliability > 1.0))
        ):
            raise ValueError(f"Artifact {path} contains invalid selective arrays")
        if not all(
            np.isfinite(values).all()
            for values in (
                target,
                prediction,
                cal_target,
                lower_h,
                upper_h,
                lower_s,
                upper_s,
            )
        ):
            raise ValueError(f"Artifact {path} contains non-finite evaluation arrays")
        metadata = {
            "dataset": str(artifact["dataset"]),
            "protocol": str(artifact["protocol"]),
            "target_region": str(artifact["target_region"]),
            "target_station": str(artifact["target_station"]),
            "model": str(artifact["model"]),
            "horizon": horizon,
            "seed": int(artifact["seed"]),
            "split": str(artifact["split"]),
            "config_sha256": str(artifact["config_sha256"]),
            "code_sha256": str(artifact["code_sha256"]),
            "dataset_sha256": str(artifact["dataset_sha256"]),
        }
        for field, observed in metadata.items():
            expected = getattr(row, field)
            expected = int(expected) if field in {"horizon", "seed"} else str(expected)
            if observed != expected:
                raise ValueError(f"Artifact {path} metadata mismatch for {field}")
        for ledger_field, artifact_field in {
            "source_regions": "source_regions_json",
            "source_stations": "source_stations_json",
            "source_pooling": "source_pooling",
            "normalization": "normalization",
        }.items():
            if str(artifact[artifact_field]) != str(getattr(row, ledger_field)):
                raise ValueError(
                    f"Artifact {path} metadata mismatch for {ledger_field}"
                )
        if int(row.n_test) != len(target) or int(row.n_calibration) != len(cal_target):
            raise ValueError(f"Artifact {path} sample counts do not match the ledger")
        alpha = float(artifact["conformal_alpha"])
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"Artifact {path} has an invalid conformal alpha")
        try:
            state = json.loads(str(artifact["calibration_state_json"]))
            threshold = float(
                state["thresholds"]["fused"]["threshold"]
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Artifact {path} has invalid fused calibration state"
            ) from error
        expected_mask = reliability >= threshold
        if not np.array_equal(stored_mask.astype(bool), expected_mask):
            raise ValueError(f"Artifact {path} fused mask disagrees with its threshold")
        expected_coverage, expected_risk = sample_risk_coverage_curve(
            target, prediction, reliability
        )
        if not np.allclose(stored_coverage, expected_coverage, rtol=0.0, atol=1e-12):
            raise ValueError(f"Artifact {path} has a stale fused coverage curve")
        if not np.allclose(stored_risk, expected_risk, rtol=1e-9, atol=1e-10):
            raise ValueError(f"Artifact {path} has a stale fused risk curve")
        _assert_close(path, "nominal_coverage", 1.0 - alpha, row.nominal_coverage)
        recomputed = {
            "rmse": rmse(target, prediction),
            "mae": mae(target, prediction),
            "conformal_marginal_coverage": interval_coverage(target, lower_h, upper_h),
            "conformal_joint_coverage": joint_interval_coverage(target, lower_h, upper_h),
            "conformal_mean_width": mean_interval_width(lower_h, upper_h),
            "conformal_interval_score": mean_interval_score(target, lower_h, upper_h, alpha),
            "simultaneous_joint_coverage": joint_interval_coverage(target, lower_s, upper_s),
            "simultaneous_mean_width": mean_interval_width(lower_s, upper_s),
            "simultaneous_interval_score": mean_interval_score(target, lower_s, upper_s, alpha),
            "fused_aurc": aurc(expected_coverage, expected_risk),
            "fused_selected_coverage": float(np.mean(expected_mask)),
        }
        selected_rmse = (
            rmse(target[expected_mask], prediction[expected_mask])
            if expected_mask.any()
            else None
        )
        for field, observed in recomputed.items():
            _assert_close(path, field, observed, getattr(row, field))
        _assert_close(path, "fused_selected_rmse", selected_rmse, row.fused_selected_rmse)


def validate_ledger(frame, min_seeds=1, min_regions=None):
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing reliability columns: {sorted(missing)}")
    if frame.duplicated(KEY).any():
        raise ValueError("Duplicate reliability run keys detected")
    ok = frame[frame.status.eq("ok")].copy()
    if ok.empty:
        raise ValueError("No successful reliability runs")
    if not ok.model.eq("trustkan").all() or not ok.split.eq("test").all():
        raise ValueError("Reliability aggregation accepts TrustKAN final-test rows only")
    for row in ok.itertuples(index=False):
        validate_artifact(row)
    if min_seeds > 1:
        group = ["dataset", "horizon", "split"]
        expected = pd.MultiIndex.from_frame(frame[group].drop_duplicates())
        counts = ok.groupby(group).seed.nunique().reindex(expected, fill_value=0)
        insufficient = counts[counts < min_seeds]
        if not insufficient.empty:
            raise ValueError(
                f"Reliability runs do not meet {min_seeds} seeds:\n{insufficient}"
            )
    if min_regions is not None:
        group = ["protocol", "horizon"]
        expected = pd.MultiIndex.from_frame(frame[group].drop_duplicates())
        counts = (
            ok.groupby(group).target_region.nunique().reindex(expected, fill_value=0)
        )
        insufficient = counts[counts < min_regions]
        if not insufficient.empty:
            raise ValueError(
                f"Reliability runs do not meet {min_regions} regions:\n{insufficient}"
            )
    return ok


def regional_macro_summary(ok):
    metrics = [
        "rmse",
        "conformal_marginal_coverage",
        "conformal_mean_width",
        "conformal_interval_score",
        "simultaneous_joint_coverage",
        "simultaneous_mean_width",
        "fused_aurc",
        "fused_selected_coverage",
        "fused_selected_rmse",
        "reliability_spearman",
        "error_detection_auroc",
        "error_detection_auprc",
    ]
    region_level = ok.groupby(
        ["protocol", "target_region", "horizon"], as_index=False
    )[metrics].mean()
    aggregations = {"n_regions": ("target_region", "nunique")}
    aggregations.update({f"{metric}_macro_mean": (metric, "mean") for metric in metrics})
    aggregations.update(
        {f"{metric}_between_region_sd": (metric, "std") for metric in metrics}
    )
    return region_level.groupby(["protocol", "horizon"], as_index=False).agg(
        **aggregations
    )


def main(path, outdir, min_seeds=1, min_regions=None):
    frame = pd.read_csv(path)
    ok = validate_ledger(frame, min_seeds=min_seeds, min_regions=min_regions)
    metrics = [
        "rmse",
        "mae",
        "conformal_marginal_coverage",
        "conformal_joint_coverage",
        "conformal_mean_width",
        "conformal_interval_score",
        "simultaneous_joint_coverage",
        "simultaneous_mean_width",
        "simultaneous_interval_score",
        "fused_aurc",
        "width_only_aurc",
        "shift_only_aurc",
        "fused_selected_coverage",
        "fused_selected_rmse",
        "reliability_spearman",
        "error_detection_auroc",
        "error_detection_auprc",
    ]
    group = ["protocol", "dataset", "target_region", "horizon"]
    named = {"n_seeds": ("seed", "nunique")}
    named.update({f"{metric}_mean": (metric, "mean") for metric in metrics})
    named.update({f"{metric}_sd": (metric, "std") for metric in metrics})
    summary = ok.groupby(group, as_index=False).agg(**named)
    macro = regional_macro_summary(ok)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outdir / "reliability_summary.csv", index=False)
    macro.to_csv(outdir / "reliability_macro_summary.csv", index=False)
    latex_columns = [
        "protocol",
        "target_region",
        "horizon",
        "n_seeds",
        "rmse_mean",
        "conformal_marginal_coverage_mean",
        "simultaneous_joint_coverage_mean",
        "conformal_mean_width_mean",
        "fused_aurc_mean",
    ]
    summary[latex_columns].to_latex(
        outdir / "reliability_summary.tex", index=False, escape=True
    )
    frame[~frame.status.eq("ok")].to_csv(outdir / "failed_runs.csv", index=False)
    print(summary.to_string(index=False))
    print("\nEqual-region reliability macro summary")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/reliability/aggregated/ghcn_reliability_full_runs.csv",
    )
    parser.add_argument("--outdir", default="results/tables/ghcn_reliability_full")
    parser.add_argument("--min-seeds", type=int, default=1)
    parser.add_argument("--min-regions", type=int, default=None)
    args = parser.parse_args()
    main(args.input, args.outdir, args.min_seeds, args.min_regions)
