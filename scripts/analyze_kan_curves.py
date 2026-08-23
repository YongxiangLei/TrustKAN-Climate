"""Audit the extracted TrustKAN curves for what they can actually support.

`run_kan_curves.py` retrains each reported model and stores its univariate
curves. This module derives the interpretability measurements from those stored
curves, so it can be re-run and extended without retraining and without moving
the runner's code fingerprint.

The measurement that needs care is cross-seed agreement. Index-matched
correlation asks whether "curve k" denotes the same function in another run.
Best-match correlation asks the weaker question of whether the same shape
reappears anywhere, and on its own it is easy to over-read: with thousands of
curves drawn from a low-dimensional space of shapes, a near-perfect partner
exists for almost any curve. This module therefore always reports the
within-seed control alongside it -- the same best-match statistic computed
against a run's own curves -- so the cross-seed number can be read against the
value that carries no cross-seed information at all.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401  # file-path execution
except ModuleNotFoundError:
    from scripts import _bootstrap  # noqa: F401  # module execution

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "interpretability" / "raw" / "cet_kan_curves"
AGGREGATED = ROOT / "results" / "interpretability" / "aggregated"
HORIZONS = (1, 7, 30, 90)
SEEDS = (11, 22, 33, 44, 55)
VARIANCE_TARGET = 0.95


def standardize_rows(curves: np.ndarray) -> np.ndarray:
    centred = curves - curves.mean(axis=1, keepdims=True)
    spread = np.linalg.norm(centred, axis=1, keepdims=True)
    return np.divide(centred, spread, out=np.zeros_like(centred), where=spread > 0)


def load(horizon: int, seed: int):
    path = RAW / f"kan_curves_h{horizon}_s{seed}.npz"
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as source:
        return {
            "curves": np.asarray(source["curves"], dtype=float),
            "grid": np.asarray(source["x_grid"], dtype=float),
            "nonlinear_share": np.asarray(source["nonlinear_share"], dtype=float),
        }


def effective_shape_rank(unit: np.ndarray) -> int:
    """Principal components needed to span the learned shapes.

    A KAN layer is presented as a library of distinct univariate functions. If
    a handful of components span thousands of curves, the library is far
    smaller than the curve count suggests, and any "a near-identical curve
    exists" statement is close to vacuous.
    """
    singular = np.linalg.svd(unit - unit.mean(axis=0), compute_uv=False)
    explained = np.cumsum(singular**2) / np.sum(singular**2)
    return int(np.searchsorted(explained, VARIANCE_TARGET) + 1)


def linear_fraction(curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Share of each curve's variance captured by a straight line.

    The layer is a linear map plus an RBF expansion, so a curve that a line
    already explains adds no structure the linear term did not carry.
    """
    design = np.vstack([grid, np.ones_like(grid)]).T
    coefficients, *_ = np.linalg.lstsq(design, curves.T, rcond=None)
    residual = curves - (design @ coefficients).T
    total = curves.var(axis=1)
    return np.divide(
        1.0 - residual.var(axis=1) / np.where(total > 0, total, 1.0),
        1.0,
        out=np.zeros_like(total),
        where=total > 0,
    )


def best_match(unit_a: np.ndarray, unit_b: np.ndarray, *, exclude_self: bool) -> np.ndarray:
    similarity = unit_a.astype(np.float32) @ unit_b.astype(np.float32).T
    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)
    return similarity.max(axis=1).astype(float)


def index_matched(unit_a: np.ndarray, unit_b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", unit_a, unit_b)


def shuffled_control(unit_a: np.ndarray, unit_b: np.ndarray, seed: int = 0) -> np.ndarray:
    """Index-matched correlation after destroying the index correspondence.

    Curves that share an index correlate strongly in magnitude, but so would
    any two curves drawn from a shape space this narrow. Pairing each curve
    with an unrelated one from the other run gives the value that a matched
    statistic has to beat before it means anything.
    """
    rng = np.random.default_rng(seed)
    return index_matched(unit_a, unit_b[rng.permutation(len(unit_b))])


def main(outdir: Path) -> None:
    if not RAW.is_dir():
        raise SystemExit(f"missing {RAW.relative_to(ROOT)}; run scripts/run_kan_curves.py")
    per_run, per_pair = [], []
    for horizon in HORIZONS:
        loaded, units = {}, {}
        for seed in SEEDS:
            entry = load(horizon, seed)
            if entry is None:
                continue
            loaded[seed] = entry
            units[seed] = standardize_rows(entry["curves"])
        if len(loaded) < 2:
            print(f"  h={horizon}: fewer than two seeds available, skipping")
            continue
        for seed, entry in loaded.items():
            unit = units[seed]
            linear = linear_fraction(entry["curves"], entry["grid"])
            per_run.append(
                {
                    "horizon": horizon,
                    "seed": seed,
                    "n_curves": int(entry["curves"].shape[0]),
                    "nonlinear_share_mean": float(entry["nonlinear_share"].mean()),
                    "linear_r2_mean": float(linear.mean()),
                    "linear_r2_median": float(np.median(linear)),
                    "nearly_linear_fraction": float((linear >= 0.9).mean()),
                    "effective_shape_rank": effective_shape_rank(unit),
                    "within_seed_best_match": float(
                        best_match(unit, unit, exclude_self=True).mean()
                    ),
                }
            )
        for first, second in combinations(sorted(loaded), 2):
            matched = index_matched(units[first], units[second])
            shuffled = shuffled_control(units[first], units[second], seed=first * 100 + second)
            across = best_match(units[first], units[second], exclude_self=False)
            control = best_match(units[first], units[first], exclude_self=True)
            per_pair.append(
                {
                    "horizon": horizon,
                    "seed_a": first,
                    "seed_b": second,
                    "matched_correlation_mean": float(matched.mean()),
                    "matched_correlation_abs_mean": float(np.abs(matched).mean()),
                    "shuffled_correlation_abs_mean": float(np.abs(shuffled).mean()),
                    "matched_abs_excess_over_shuffled": float(
                        np.abs(matched).mean() - np.abs(shuffled).mean()
                    ),
                    "best_match_correlation_mean": float(across.mean()),
                    "within_seed_best_match_mean": float(control.mean()),
                    "best_match_excess_over_control": float(across.mean() - control.mean()),
                }
            )
        summary = per_pair[-1]
        print(
            f"  h={horizon}: matched r={summary['matched_correlation_mean']:+.3f}, "
            f"|r|={summary['matched_correlation_abs_mean']:.3f} "
            f"vs shuffled {summary['shuffled_correlation_abs_mean']:.3f}; "
            f"best-match {summary['best_match_correlation_mean']:.4f} "
            f"vs within-seed {summary['within_seed_best_match_mean']:.4f}"
        )
    if not per_pair:
        raise SystemExit("no curve pairs available; run scripts/run_kan_curves.py first")
    outdir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(per_run)
    pairs = pd.DataFrame(per_pair)
    runs.to_csv(outdir / "cet_kan_curves_shapes.csv", index=False)
    pairs.to_csv(outdir / "cet_kan_curves_stability.csv", index=False)
    print(f"  wrote {(outdir / 'cet_kan_curves_shapes.csv').relative_to(ROOT)}")
    print(f"  wrote {(outdir / 'cet_kan_curves_stability.csv').relative_to(ROOT)}")
    (outdir / "cet_kan_curves_analysis.json").write_text(
        json.dumps(
            {
                "variance_target": VARIANCE_TARGET,
                "horizons": list(HORIZONS),
                "seeds": list(SEEDS),
                "runs": len(runs),
                "pairs": len(pairs),
                "matched_correlation_abs_mean": float(pairs.matched_correlation_abs_mean.mean()),
                "shuffled_correlation_abs_mean": float(pairs.shuffled_correlation_abs_mean.mean()),
                "matched_abs_excess_over_shuffled": float(
                    pairs.matched_abs_excess_over_shuffled.mean()
                ),
                "best_match_excess_over_control": float(
                    pairs.best_match_excess_over_control.mean()
                ),
                "effective_shape_rank_max": int(runs.effective_shape_rank.max()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(AGGREGATED))
    args = parser.parse_args()
    main(Path(args.outdir))
