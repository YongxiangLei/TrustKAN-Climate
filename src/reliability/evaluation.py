"""Calibration-only uncertainty and selective-forecast evaluation."""
from __future__ import annotations

import numpy as np

from src.drift.scores import mahalanobis_shift, percentile_to_reliability
from src.metrics.forecast import aurc, mae, rmse, sample_risk_coverage_curve
from src.reliability.calibration import (
    reliability_error_association,
    sample_rmse,
    top_error_detection,
)
from src.reliability.fusion import (
    choose_threshold_on_calibration,
    fuse_reliability,
    normalize_interval_width,
    selective_mask,
)
from src.uncertainty.conformal import (
    apply_conformal,
    horizonwise_conformal_radii,
    horizonwise_interval_coverage,
    interval_coverage,
    joint_interval_coverage,
    mean_interval_score,
    mean_interval_width,
    simultaneous_conformal_radius,
)


def inverse_standardized(scaler, values):
    values = np.asarray(values, dtype=float)
    shape = values.shape
    return scaler.scaler.inverse_transform(values.reshape(-1, 1)).reshape(shape)


def _validate_prediction_bundle(bundle, name):
    required = {"target", "point", "quantiles", "embedding"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"{name} predictions are missing {sorted(missing)}")
    target = np.asarray(bundle["target"])
    point = np.asarray(bundle["point"])
    quantiles = np.asarray(bundle["quantiles"])
    embedding = np.asarray(bundle["embedding"])
    if target.shape != point.shape or quantiles.shape[:2] != target.shape:
        raise ValueError(f"{name} prediction shapes are inconsistent")
    if embedding.ndim != 2 or len(embedding) != len(target):
        raise ValueError(f"{name} embeddings must align with forecast origins")
    if not all(np.isfinite(item).all() for item in (target, point, quantiles, embedding)):
        raise ValueError(f"{name} predictions contain non-finite values")


def evaluate_calibrated_reliability(
    calibration,
    test,
    reference_embeddings,
    scaler,
    *,
    quantile_levels,
    alpha=0.1,
    reliability_weights=(0.5, 0.5),
    min_coverage=0.5,
    error_quantile=0.9,
):
    """Evaluate intervals and reliability without selecting anything on test labels."""
    _validate_prediction_bundle(calibration, "calibration")
    _validate_prediction_bundle(test, "test")
    reference_embeddings = np.asarray(reference_embeddings, dtype=float)
    if reference_embeddings.ndim != 2 or len(reference_embeddings) < 2:
        raise ValueError("At least two reference embeddings are required")
    levels = np.asarray(quantile_levels, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or np.any(np.diff(levels) <= 0):
        raise ValueError("quantile_levels must be strictly increasing")
    if calibration["quantiles"].shape[-1] != len(levels):
        raise ValueError("Quantile level count does not match model output")

    cal_lower_std = calibration["quantiles"][..., 0]
    cal_upper_std = calibration["quantiles"][..., -1]
    test_lower_std = test["quantiles"][..., 0]
    test_upper_std = test["quantiles"][..., -1]
    marginal_radii_std = horizonwise_conformal_radii(
        calibration["target"], cal_lower_std, cal_upper_std, alpha
    )
    simultaneous_radius_std = simultaneous_conformal_radius(
        calibration["target"], cal_lower_std, cal_upper_std, alpha
    )
    cal_lower_m_std, cal_upper_m_std = apply_conformal(
        cal_lower_std, cal_upper_std, marginal_radii_std
    )
    test_lower_m_std, test_upper_m_std = apply_conformal(
        test_lower_std, test_upper_std, marginal_radii_std
    )
    cal_lower_s_std, cal_upper_s_std = apply_conformal(
        cal_lower_std, cal_upper_std, simultaneous_radius_std
    )
    test_lower_s_std, test_upper_s_std = apply_conformal(
        test_lower_std, test_upper_std, simultaneous_radius_std
    )

    cal_target = inverse_standardized(scaler, calibration["target"])
    cal_point = inverse_standardized(scaler, calibration["point"])
    test_target = inverse_standardized(scaler, test["target"])
    test_point = inverse_standardized(scaler, test["point"])
    cal_lower_raw = inverse_standardized(scaler, cal_lower_std)
    cal_upper_raw = inverse_standardized(scaler, cal_upper_std)
    test_lower_raw = inverse_standardized(scaler, test_lower_std)
    test_upper_raw = inverse_standardized(scaler, test_upper_std)
    cal_lower_m = inverse_standardized(scaler, cal_lower_m_std)
    cal_upper_m = inverse_standardized(scaler, cal_upper_m_std)
    test_lower_m = inverse_standardized(scaler, test_lower_m_std)
    test_upper_m = inverse_standardized(scaler, test_upper_m_std)
    cal_lower_s = inverse_standardized(scaler, cal_lower_s_std)
    cal_upper_s = inverse_standardized(scaler, cal_upper_s_std)
    test_lower_s = inverse_standardized(scaler, test_lower_s_std)
    test_upper_s = inverse_standardized(scaler, test_upper_s_std)

    cal_width = (cal_upper_m_std - cal_lower_m_std).mean(axis=1)
    test_width = (test_upper_m_std - test_lower_m_std).mean(axis=1)
    width_rel_cal = normalize_interval_width(cal_width, cal_width)
    width_rel_test = normalize_interval_width(test_width, cal_width)
    cal_shift = mahalanobis_shift(reference_embeddings, calibration["embedding"])
    test_shift = mahalanobis_shift(reference_embeddings, test["embedding"])
    shift_rel_cal = percentile_to_reliability(cal_shift, cal_shift)
    shift_rel_test = percentile_to_reliability(test_shift, cal_shift)
    fused_cal = fuse_reliability(
        width_rel_cal, shift_rel_cal, weights=reliability_weights
    )
    fused_test = fuse_reliability(
        width_rel_test, shift_rel_test, weights=reliability_weights
    )

    component_cal = {
        "fused": fused_cal,
        "width_only": width_rel_cal,
        "shift_only": shift_rel_cal,
    }
    component_test = {
        "fused": fused_test,
        "width_only": width_rel_test,
        "shift_only": shift_rel_test,
    }
    selective = {}
    curves = {}
    associations = {}
    test_error = sample_rmse(test_target, test_point)
    for name, reliability_cal in component_cal.items():
        selected = choose_threshold_on_calibration(
            cal_target,
            cal_point,
            reliability_cal,
            min_coverage=min_coverage,
        )
        if selected is None:
            raise RuntimeError(f"No calibration threshold available for {name}")
        reliability_test = component_test[name]
        mask = selective_mask(reliability_test, selected["threshold"])
        coverage, risk = sample_risk_coverage_curve(
            test_target, test_point, reliability_test
        )
        selective[name] = {
            "threshold_from_calibration": selected,
            "test_coverage": float(mask.mean()),
            "test_rmse": rmse(test_target[mask], test_point[mask])
            if mask.any()
            else None,
            "aurc": aurc(coverage, risk),
        }
        curves[name] = {"coverage": coverage, "risk": risk, "mask": mask}
        associations[name] = {
            "association": reliability_error_association(
                reliability_test, test_error
            ),
            "top_error_detection": top_error_detection(
                reliability_test, test_error, error_quantile
            ),
        }

    metrics = {
        "point": {
            "rmse": rmse(test_target, test_point),
            "mae": mae(test_target, test_point),
        },
        "raw_interval": {
            "marginal_coverage": interval_coverage(
                test_target, test_lower_raw, test_upper_raw
            ),
            "joint_coverage": joint_interval_coverage(
                test_target, test_lower_raw, test_upper_raw
            ),
            "mean_width": mean_interval_width(test_lower_raw, test_upper_raw),
            "mean_interval_score": mean_interval_score(
                test_target, test_lower_raw, test_upper_raw, alpha
            ),
        },
        "horizonwise_conformal": {
            "calibration_marginal_coverage": interval_coverage(
                cal_target, cal_lower_m, cal_upper_m
            ),
            "test_marginal_coverage": interval_coverage(
                test_target, test_lower_m, test_upper_m
            ),
            "test_horizonwise_coverage": horizonwise_interval_coverage(
                test_target, test_lower_m, test_upper_m
            ).tolist(),
            "test_joint_coverage": joint_interval_coverage(
                test_target, test_lower_m, test_upper_m
            ),
            "test_mean_width": mean_interval_width(test_lower_m, test_upper_m),
            "test_mean_interval_score": mean_interval_score(
                test_target, test_lower_m, test_upper_m, alpha
            ),
        },
        "simultaneous_conformal": {
            "calibration_joint_coverage": joint_interval_coverage(
                cal_target, cal_lower_s, cal_upper_s
            ),
            "test_joint_coverage": joint_interval_coverage(
                test_target, test_lower_s, test_upper_s
            ),
            "test_marginal_coverage": interval_coverage(
                test_target, test_lower_s, test_upper_s
            ),
            "test_mean_width": mean_interval_width(test_lower_s, test_upper_s),
            "test_mean_interval_score": mean_interval_score(
                test_target, test_lower_s, test_upper_s, alpha
            ),
        },
        "selective": selective,
        "reliability_diagnostics": associations,
    }
    arrays = {
        "calibration_target": cal_target,
        "calibration_prediction": cal_point,
        "test_target": test_target,
        "test_prediction": test_point,
        "test_lower_raw": test_lower_raw,
        "test_upper_raw": test_upper_raw,
        "test_lower_horizonwise": test_lower_m,
        "test_upper_horizonwise": test_upper_m,
        "test_lower_simultaneous": test_lower_s,
        "test_upper_simultaneous": test_upper_s,
        "marginal_radii_standardized": marginal_radii_std,
        "simultaneous_radius_standardized": np.asarray(simultaneous_radius_std),
        "fused_reliability": fused_test,
        "width_reliability": width_rel_test,
        "shift_reliability": shift_rel_test,
        "test_embedding": np.asarray(test["embedding"]),
        "test_error": test_error,
        **{
            f"{name}_{field}": value
            for name, curve in curves.items()
            for field, value in curve.items()
        },
    }
    calibration_state = {
        "marginal_radii_standardized": marginal_radii_std.tolist(),
        "simultaneous_radius_standardized": simultaneous_radius_std,
        "thresholds": {
            name: values["threshold_from_calibration"]
            for name, values in selective.items()
        },
    }
    return metrics, arrays, calibration_state
