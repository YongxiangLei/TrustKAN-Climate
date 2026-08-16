"""Dependence-aware comparison of two paired raw prediction artifacts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  # repository-root import setup
from src.statistics.paired_tests import (
    paired_block_bootstrap_difference,
    paired_cohens_d,
    wilcoxon_paired,
)


REQUIRED = {
    "target",
    "prediction",
    "target_time",
    "target_origin",
    "dataset",
    "model",
    "horizon",
    "seed",
    "split",
    "config_sha256",
    "code_sha256",
}


def load_artifact(path):
    with np.load(path, allow_pickle=False) as source:
        missing = REQUIRED - set(source.files)
        if missing:
            raise ValueError(f"Artifact {path} is missing fields: {sorted(missing)}")
        return {key: np.array(source[key], copy=True) for key in REQUIRED}


def scalar(artifact, key):
    return artifact[key].item()


def validate_pair(a, b):
    if a["target"].shape != b["target"].shape or not np.allclose(a["target"], b["target"]):
        raise ValueError("Artifacts do not share identical targets; paired comparison is invalid")
    if not np.array_equal(a["target_time"], b["target_time"]):
        raise ValueError("Artifacts do not share identical target timestamps")
    if not np.array_equal(a["target_origin"], b["target_origin"]):
        raise ValueError("Artifacts do not share identical target origins")
    for field in ("dataset", "horizon", "split", "config_sha256", "code_sha256"):
        if scalar(a, field) != scalar(b, field):
            raise ValueError(f"Artifact metadata mismatch for {field}")
    seed_a, seed_b = int(scalar(a, "seed")), int(scalar(b, "seed"))
    if seed_a != seed_b and seed_a >= 0 and seed_b >= 0:
        raise ValueError("Stochastic model artifacts must use the same seed")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(a_path, b_path, out, *, n_boot=5000, confidence=0.95, block_length=None, seed=0):
    a = load_artifact(a_path)
    b = load_artifact(b_path)
    validate_pair(a, b)
    model_a, model_b = str(scalar(a, "model")), str(scalar(b, "model"))
    result = {
        "provenance": {
            "artifact_a": str(a_path),
            "artifact_b": str(b_path),
            "model_a": model_a,
            "model_b": model_b,
            "dataset": str(scalar(a, "dataset")),
            "horizon": int(scalar(a, "horizon")),
            "seed_a": int(scalar(a, "seed")),
            "seed_b": int(scalar(b, "seed")),
            "config_sha256": str(scalar(a, "config_sha256")),
            "code_sha256": str(scalar(a, "code_sha256")),
        },
        "primary_block_bootstrap": {
            "mae_difference_a_minus_b": paired_block_bootstrap_difference(
                a["target"],
                a["prediction"],
                b["prediction"],
                metric="mae",
                n_boot=n_boot,
                confidence=confidence,
                block_length=block_length,
                seed=seed,
            ),
            "rmse_difference_a_minus_b": paired_block_bootstrap_difference(
                a["target"],
                a["prediction"],
                b["prediction"],
                metric="rmse",
                n_boot=n_boot,
                confidence=confidence,
                block_length=block_length,
                seed=seed + 1,
            ),
        },
        "sensitivity_analysis": {
            "wilcoxon_origin_mae": wilcoxon_paired(
                a["target"], a["prediction"], b["prediction"]
            ),
            "paired_cohens_d_origin_mae": paired_cohens_d(
                a["target"], a["prediction"], b["prediction"]
            ),
            "warning": (
                "Wilcoxon and Cohen's d do not model serial dependence; use the block-bootstrap "
                "intervals as primary evidence."
            ),
        },
        "interpretation": (
            "Negative differences favor model A. Practical effect magnitude and the pre-specified "
            "family of comparisons must be considered alongside interval exclusion of zero."
        ),
    }
    atomic_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--out", default="results/statistical_tests/comparison.json")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--block-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(
        args.a,
        args.b,
        args.out,
        n_boot=args.n_boot,
        confidence=args.confidence,
        block_length=args.block_length,
        seed=args.seed,
    )
