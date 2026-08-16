"""Run TrustKAN uncertainty and selective forecasting on a prepared split.

This dataset-agnostic entrypoint assumes that targets are already expressed in
the reporting units. The frozen GHCN experiment uses ``run_ghcn_reliability``
instead because it additionally verifies station provenance and inverse scaling.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

from scripts.run_cet_benchmark import atomic_savez
from src.drift.scores import mahalanobis_shift, percentile_to_reliability
from src.metrics.forecast import aurc, rmse, sample_risk_coverage_curve
from src.models.trustkan import TrustKAN
from src.reliability.fusion import (
    choose_threshold_on_calibration,
    fuse_reliability,
    normalize_interval_width,
    selective_mask,
)
from src.training.engine import set_seed
from src.training.trust_engine import predict_trustkan, train_trustkan
from src.uncertainty.conformal import (
    apply_conformal,
    horizonwise_conformal_radii,
    interval_coverage,
    joint_interval_coverage,
    mean_interval_width,
    simultaneous_conformal_radius,
)


REQUIRED_SPLIT_KEYS = {
    "x_train",
    "y_train",
    "x_val",
    "y_val",
    "x_cal",
    "y_cal",
    "x_test",
    "y_test",
}


def load_split(path):
    with np.load(path) as archive:
        missing = REQUIRED_SPLIT_KEYS - set(archive.files)
        if missing:
            raise ValueError(f"Split archive is missing {sorted(missing)}")
        data = {key: archive[key] for key in REQUIRED_SPLIT_KEYS}
    if any(not np.isfinite(value).all() for value in data.values()):
        raise ValueError("Split archive contains non-finite values")
    input_shape = data["x_train"].shape[1:]
    horizon = data["y_train"].shape[1:] if data["y_train"].ndim == 2 else None
    for name in ("train", "val", "cal", "test"):
        x = data[f"x_{name}"]
        y = data[f"y_{name}"]
        if x.ndim != 3 or y.ndim != 2 or len(x) == 0 or len(x) != len(y):
            raise ValueError(
                f"{name} split must contain aligned, non-empty [N,T,F] and [N,H] arrays"
            )
        if x.shape[1:] != input_shape or y.shape[1:] != horizon:
            raise ValueError("All split arrays must share input and horizon dimensions")
    if len(data["x_train"]) < 2:
        raise ValueError("Training split needs at least two embedding references")
    return data


def loader(x, y, batch=64, shuffle=False):
    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        ),
        batch_size=batch,
        shuffle=shuffle,
    )


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(args):
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie strictly between zero and one")
    set_seed(args.seed)
    data = load_split(args.split_file)
    train_loader = loader(data["x_train"], data["y_train"], args.batch, True)
    train_reference_loader = loader(
        data["x_train"], data["y_train"], args.batch
    )
    validation_loader = loader(data["x_val"], data["y_val"], args.batch)
    calibration_loader = loader(data["x_cal"], data["y_cal"], args.batch)
    test_loader = loader(data["x_test"], data["y_test"], args.batch)
    quantile_levels = (args.alpha / 2.0, 0.5, 1.0 - args.alpha / 2.0)
    model = TrustKAN(
        data["x_train"].shape[-1],
        horizon=data["y_train"].shape[-1],
        hidden_dim=args.hidden,
        grid_size=args.grid,
        quantiles=quantile_levels,
    )
    model, history, seconds = train_trustkan(
        model,
        train_loader,
        validation_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weight_decay=args.weight_decay,
    )
    reference = predict_trustkan(model, train_reference_loader)
    calibration = predict_trustkan(model, calibration_loader)
    test = predict_trustkan(model, test_loader)

    calibration_lower = calibration["quantiles"][..., 0]
    calibration_upper = calibration["quantiles"][..., -1]
    test_lower = test["quantiles"][..., 0]
    test_upper = test["quantiles"][..., -1]
    marginal_radii = horizonwise_conformal_radii(
        calibration["target"], calibration_lower, calibration_upper, args.alpha
    )
    simultaneous_radius = simultaneous_conformal_radius(
        calibration["target"], calibration_lower, calibration_upper, args.alpha
    )
    calibration_lower_m, calibration_upper_m = apply_conformal(
        calibration_lower, calibration_upper, marginal_radii
    )
    test_lower_m, test_upper_m = apply_conformal(
        test_lower, test_upper, marginal_radii
    )
    test_lower_s, test_upper_s = apply_conformal(
        test_lower, test_upper, simultaneous_radius
    )

    calibration_width = (calibration_upper_m - calibration_lower_m).mean(axis=1)
    test_width = (test_upper_m - test_lower_m).mean(axis=1)
    width_reliability_calibration = normalize_interval_width(
        calibration_width, calibration_width
    )
    width_reliability_test = normalize_interval_width(
        test_width, calibration_width
    )
    calibration_shift = mahalanobis_shift(
        reference["embedding"], calibration["embedding"]
    )
    test_shift = mahalanobis_shift(reference["embedding"], test["embedding"])
    shift_reliability_calibration = percentile_to_reliability(
        calibration_shift, calibration_shift
    )
    shift_reliability_test = percentile_to_reliability(
        test_shift, calibration_shift
    )
    reliability_calibration = fuse_reliability(
        width_reliability_calibration, shift_reliability_calibration
    )
    reliability_test = fuse_reliability(
        width_reliability_test, shift_reliability_test
    )
    selected = choose_threshold_on_calibration(
        calibration["target"],
        calibration["point"],
        reliability_calibration,
        min_coverage=args.min_coverage,
    )
    if selected is None:
        raise RuntimeError("No valid reliability threshold found on calibration data")
    mask = selective_mask(reliability_test, selected["threshold"])
    coverage_curve, risk_curve = sample_risk_coverage_curve(
        test["target"], test["point"], reliability_test
    )
    result = {
        "seed": args.seed,
        "train_seconds": seconds,
        "epochs_completed": len(history),
        "quantile_levels": list(quantile_levels),
        "conformal_alpha": args.alpha,
        "marginal_radii": marginal_radii.tolist(),
        "simultaneous_radius": simultaneous_radius,
        "calibration_marginal_coverage": interval_coverage(
            calibration["target"], calibration_lower_m, calibration_upper_m
        ),
        "test_marginal_coverage": interval_coverage(
            test["target"], test_lower_m, test_upper_m
        ),
        "test_marginal_joint_coverage": joint_interval_coverage(
            test["target"], test_lower_m, test_upper_m
        ),
        "test_simultaneous_joint_coverage": joint_interval_coverage(
            test["target"], test_lower_s, test_upper_s
        ),
        "test_mean_interval_width": mean_interval_width(
            test_lower_m, test_upper_m
        ),
        "test_rmse_all": rmse(test["target"], test["point"]),
        "threshold_from_calibration": selected,
        "test_sample_coverage_after_abstention": float(mask.mean()),
        "test_rmse_selected": (
            rmse(test["target"][mask], test["point"][mask])
            if mask.any()
            else None
        ),
        "aurc": aurc(coverage_curve, risk_curve),
    }
    out = Path(args.out)
    atomic_json(out, result)
    stem = out.with_suffix("")
    atomic_savez(
        str(stem) + "_reliability.npz",
        target=test["target"],
        prediction=test["point"],
        reliability=reliability_test,
        width_reliability=width_reliability_test,
        shift_reliability=shift_reliability_test,
        selected_mask=mask,
        embedding=test["embedding"],
    )
    atomic_savez(
        str(stem) + "_conformal_input.npz",
        y_cal=calibration["target"],
        cal_lower=calibration_lower,
        cal_upper=calibration_upper,
        y_test=test["target"],
        test_lower=test_lower,
        test_upper=test_upper,
        marginal_radii=marginal_radii,
        simultaneous_radius=np.asarray(simultaneous_radius),
    )
    atomic_savez(
        str(stem) + "_risk_coverage.npz",
        coverage=coverage_curve,
        risk=risk_curve,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--out", default="results/reliability/trustkan.json")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--grid", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
