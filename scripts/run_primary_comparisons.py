"""Run the pre-specified paired comparisons behind the manuscript's claims.

Each comparison pairs two models on identical forecast origins and an identical
seed, then applies the circular moving-block bootstrap from
`src.statistics.paired_tests`, which is the project's primary test because it
respects serial dependence. One JSON artifact is written per pair so that every
interval quoted in the paper can be re-derived.

The comparison family is fixed here rather than chosen after inspecting
results: TrustKAN against the plain KAN isolates the temporal KAN module, and
TrustKAN against the strongest neural baseline tests the accuracy claim.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  # repository-root import setup
from compare_models import load_artifact, validate_pair
from src.statistics.paired_tests import (
    circular_block_indices,
    metric_difference,
    paired_arrays,
    paired_block_bootstrap_difference,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw" / "cet_full"
HORIZONS = (1, 7, 30, 90)
SEEDS = (11, 22, 33, 44, 55)
# Pre-specified family: (model A, model B, what the pair is meant to isolate).
FAMILY = (
    ("trustkan", "kan", "temporal KAN module beyond a plain KAN"),
    ("trustkan", "transformer", "accuracy against the strongest neural baseline"),
)
# Second study, declared before its comparisons were run. It is a separate
# family set rather than an extension of the one above: appending pairs to a
# family that has already been read would inflate the multiplicity the Holm
# correction is there to control. Each family is corrected within itself.
V2_FAMILY = (
    ("trustkan_v2", "transformer", "accuracy against the strongest neural baseline"),
    ("trustkan_v2", "kan", "temporal representation beyond a plain KAN"),
    ("trustkan_v2", "trustkan_dilated", "global attention readout beyond a wide stem"),
    ("trustkan_v2", "trustkan", "the corrected architecture against the published one"),
)
FAMILY_SETS = {"primary": FAMILY, "v2": V2_FAMILY}
# Variants trained by the stem pilots rather than by the frozen campaign. The
# protocol is identical, so pairs remain matched on origin and seed, but the
# artifacts live under their own experiment names.
PILOT_RAW = {
    "trustkan_v2": ROOT / "results" / "raw" / "cet_stem_pilot_v2",
    "trustkan_dilated": ROOT / "results" / "raw" / "cet_stem_pilot",
}


def artifact(model: str, horizon: int, seed: int) -> Path:
    return PILOT_RAW.get(model, RAW) / f"cet_{model}_h{horizon}_s{seed}.npz"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def bootstrap_p_value(a, b, metric, n_boot, block_length, seed) -> float:
    """Two-sided achieved significance level from the same block bootstrap.

    The protocol calls for Holm correction across each comparison family, which
    needs p-values rather than intervals. These draws use the same resampling
    scheme and block-length rule as the interval, so the two agree by
    construction.
    """
    y, pa, pb = paired_arrays(a["target"], a["prediction"], b["prediction"])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for index in range(n_boot):
        sampled = circular_block_indices(len(y), block_length, rng)
        draws[index] = metric_difference(y[sampled], pa[sampled], pb[sampled], metric)
    tail = min((draws >= 0).mean(), (draws <= 0).mean())
    return float(min(1.0, max(2 * tail, 1.0 / n_boot)))


def holm(p_values: list[float], alpha: float) -> list[bool]:
    """Holm step-down: returns which hypotheses are rejected at family level alpha."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] <= alpha / (len(p_values) - rank):
            rejected[index] = True
        else:
            break
    return rejected


def verdict(ci_low: float, ci_high: float, a: str, b: str) -> str:
    if ci_high < 0:
        return f"{a} better"
    if ci_low > 0:
        return f"{b} better"
    return "inconclusive"


def compare_pair(a_model, b_model, horizon, seed, n_boot, confidence):
    a = load_artifact(artifact(a_model, horizon, seed))
    b = load_artifact(artifact(b_model, horizon, seed))
    validate_pair(a, b)
    result = {}
    for offset, metric in enumerate(("rmse", "mae")):
        result[metric] = paired_block_bootstrap_difference(
            a["target"],
            a["prediction"],
            b["prediction"],
            metric=metric,
            n_boot=n_boot,
            confidence=confidence,
            seed=seed + offset,
        )
        result[metric]["p_value"] = bootstrap_p_value(
            a,
            b,
            metric,
            n_boot,
            result[metric]["block_length"],
            seed + offset,
        )
    return result, str(a["code_sha256"].item())


def main(outdir: Path, n_boot: int, confidence: float, families=FAMILY) -> None:
    alpha = 1 - confidence
    index = []
    for a_model, b_model, isolates in families:
        print(f"\n{a_model} vs {b_model}: {isolates}")
        # The family spans every horizon and seed for this comparator, so Holm
        # is applied once across all of them rather than per horizon.
        pairs = []
        for horizon in HORIZONS:
            for seed in SEEDS:
                if not artifact(a_model, horizon, seed).exists():
                    continue
                if not artifact(b_model, horizon, seed).exists():
                    continue
                result, fingerprint = compare_pair(
                    a_model, b_model, horizon, seed, n_boot, confidence
                )
                pairs.append((horizon, seed, result, fingerprint))
        if not pairs:
            continue
        rejected = holm([r["rmse"]["p_value"] for _, _, r, _ in pairs], alpha)
        print(
            f"Holm at family level {alpha:.2f} over {len(pairs)} comparisons: "
            f"{sum(rejected)} rejected"
        )
        print(
            f"{'h':>4} {'seed':>5} {'dRMSE':>9} {'ci_low':>9} {'ci_high':>9} "
            f"{'p':>8}  verdict"
        )
        by_horizon = {}
        for (horizon, seed, result, fingerprint), reject in zip(pairs, rejected):
            rmse = result["rmse"]
            call = verdict(rmse["ci_low"], rmse["ci_high"], a_model, b_model)
            holm_call = (
                "inconclusive"
                if not reject
                else (f"{a_model} better" if rmse["mean_difference"] < 0 else f"{b_model} better")
            )
            payload = {
                "model_a": a_model,
                "model_b": b_model,
                "isolates": isolates,
                "dataset": "CET_Pershore_College",
                "horizon": horizon,
                "seed": seed,
                "n_boot": n_boot,
                "confidence": confidence,
                "code_sha256": fingerprint,
                "verdict_rmse": call,
                "family_size": len(pairs),
                "holm_alpha": alpha,
                "holm_rejected": bool(reject),
                "verdict_rmse_holm": holm_call,
                "interpretation": (
                    "Negative differences favour model A. Intervals are circular "
                    "moving-block bootstrap over forecast origins. Holm correction "
                    "is applied across the whole comparator family."
                ),
                **result,
            }
            name = f"cet_{a_model}_vs_{b_model}_h{horizon}_s{seed}.json"
            atomic_json(outdir / name, payload)
            print(
                f"{horizon:>4} {seed:>5} {rmse['mean_difference']:>9.4f} "
                f"{rmse['ci_low']:>9.4f} {rmse['ci_high']:>9.4f} "
                f"{rmse['p_value']:>8.4f}  {holm_call}"
            )
            entry = by_horizon.setdefault(horizon, {"diffs": [], "calls": []})
            entry["diffs"].append(rmse["mean_difference"])
            entry["calls"].append(holm_call)
        for horizon, entry in sorted(by_horizon.items()):
            index.append(
                {
                    "model_a": a_model,
                    "model_b": b_model,
                    "isolates": isolates,
                    "horizon": horizon,
                    "n_pairs": len(entry["diffs"]),
                    "mean_rmse_difference": float(np.mean(entry["diffs"])),
                    "a_better": entry["calls"].count(f"{a_model} better"),
                    "b_better": entry["calls"].count(f"{b_model} better"),
                    "inconclusive": entry["calls"].count("inconclusive"),
                    "multiplicity": "holm",
                    "family_size": len(pairs),
                }
            )
    atomic_json(outdir / "primary_comparisons_index.json", index)
    print("\nFamily summary (negative mean favours model A):")
    for row in index:
        print(
            f"  {row['model_a']} vs {row['model_b']} h{row['horizon']:<3} "
            f"mean={row['mean_rmse_difference']:+.4f}  "
            f"A better {row['a_better']}/{row['n_pairs']}, "
            f"B better {row['b_better']}/{row['n_pairs']}, "
            f"inconclusive {row['inconclusive']}/{row['n_pairs']}"
        )
    print(f"\nwrote artifacts to {outdir.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir", default=str(ROOT / "results" / "statistical_tests" / "cet_primary")
    )
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--family", choices=sorted(FAMILY_SETS), default="primary")
    args = parser.parse_args()
    main(
        Path(args.outdir),
        args.n_boot,
        args.confidence,
        families=FAMILY_SETS[args.family],
    )
