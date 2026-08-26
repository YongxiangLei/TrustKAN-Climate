"""Prove the v2 source edits did not disturb the models they did not touch.

Changing `src/models/trustkan.py` changes `code_sha256` for every runner, so the
frozen ledger and the v2 ledger cannot be compared on fingerprints alone. The
edits were additive -- a stem option, a readout option, two model names -- and
should therefore leave every other architecture bit-for-bit identical, but that
has to be demonstrated rather than asserted: a paired test across the two
studies is only sound if the comparator really is the same model.

Equality of stored test predictions is the strong form of that evidence.
Matching RMSE alone could be a coincidence between two nearby models; an entire
prediction vector agreeing to the last bit, for every horizon and seed, cannot.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  # repository-root import setup

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "raw" / "cet_full"
HORIZONS = (1, 7, 30, 90)
SEEDS = (11, 22, 33, 44, 55)
# Architectures present in both studies. The v2 variants have no frozen
# counterpart, and the classical baselines were not re-run.
SHARED = ("mlp", "lstm", "gru", "tcn", "transformer", "kan", "trustkan")


def compare(new_raw: Path) -> int:
    failures = 0
    print(f"{'model':<14}{'pairs':>7}{'max |Δprediction|':>20}{'max |ΔRMSE|':>14}")
    for model in SHARED:
        deltas, rmse_deltas, pairs = [], [], 0
        for horizon in HORIZONS:
            for seed in SEEDS:
                name = f"cet_{model}_h{horizon}_s{seed}.npz"
                old_path, new_path = FROZEN / name, new_raw / name
                if not (old_path.exists() and new_path.exists()):
                    continue
                with np.load(old_path, allow_pickle=False) as old, np.load(
                    new_path, allow_pickle=False
                ) as new:
                    if not np.array_equal(old["target"], new["target"]):
                        print(f"  {name}: evaluation targets differ; pairing is invalid")
                        failures += 1
                        continue
                    a = np.asarray(old["prediction"], dtype=float)
                    b = np.asarray(new["prediction"], dtype=float)
                    if a.shape != b.shape:
                        print(f"  {name}: prediction shapes differ {a.shape} vs {b.shape}")
                        failures += 1
                        continue
                    deltas.append(float(np.abs(a - b).max()))
                    truth = np.asarray(old["target"], dtype=float)
                    rmse_deltas.append(
                        abs(
                            float(np.sqrt(((truth - a) ** 2).mean()))
                            - float(np.sqrt(((truth - b) ** 2).mean()))
                        )
                    )
                    pairs += 1
        if not pairs:
            print(f"{model:<14}{0:>7}{'no shared runs':>20}")
            continue
        worst, worst_rmse = max(deltas), max(rmse_deltas)
        print(f"{model:<14}{pairs:>7}{worst:>20.3e}{worst_rmse:>14.3e}")
        if worst != 0.0:
            failures += 1
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(ROOT / "results" / "raw" / "cet_v2_neural"))
    args = parser.parse_args()
    bad = compare(Path(args.raw))
    if bad:
        raise SystemExit(
            f"{bad} architecture(s) changed under the v2 edits; the cross-study "
            "comparison is not sound until this is explained"
        )
    print("\nEvery shared architecture reproduces its frozen predictions exactly.")
