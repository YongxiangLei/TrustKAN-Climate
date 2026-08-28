"""Prove the v2 source edits did not disturb the models they did not touch.

Changing `src/models/trustkan.py` changes `code_sha256` for every runner, so the
frozen ledger and the v2 ledger cannot be compared on fingerprints alone. The
edits were additive -- a stem option, a readout option, two model names -- and
should therefore leave every other architecture bit-for-bit identical, but that
has to be demonstrated rather than asserted: a paired test across the two
studies is only sound if the comparator really is the same model, and the
classical rows are only quotable from the frozen campaign if the edit cannot
reach them.

Equality of stored test predictions is the strong form of that evidence, and
equality is checked on the raw bytes rather than on a difference. Matching RMSE
alone could be a coincidence between two nearby models; a maximum absolute
difference of zero could hide a dtype or shape change; an entire prediction
buffer agreeing byte for byte, for every horizon and seed, cannot.

The check refuses to pass if the two campaigns share a code fingerprint, since
then it would be comparing a campaign with itself and would prove nothing.
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
# counterpart. The classical baselines carry the deterministic seed -1 for SVR
# and are only spot-checked, because their full shard costs about twenty-five
# hours; see configs/cet_v2_classical_probe.yaml.
NEURAL = ("mlp", "lstm", "gru", "tcn", "transformer", "kan", "trustkan")
CLASSICAL = ("svr", "random_forest")
CLASSICAL_SEEDS = (-1, 11)


def fingerprints(directory: Path) -> set[str]:
    seen = set()
    for path in sorted(directory.glob("cet_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            if "code_sha256" in data:
                seen.add(str(data["code_sha256"].item()))
    return seen


def compare(new_raw: Path, models, seeds, horizons) -> int:
    failures = 0
    print(f"{'model':<16}{'pairs':>7}{'byte-identical':>17}{'max |ΔRMSE|':>14}")
    for model in models:
        rmse_deltas, pairs, identical = [], 0, 0
        for horizon in horizons:
            for seed in seeds:
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
                    a, b = old["prediction"], new["prediction"]
                    pairs += 1
                    if a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes():
                        identical += 1
                    else:
                        print(
                            f"  {name}: predictions differ "
                            f"({a.dtype}{a.shape} vs {b.dtype}{b.shape})"
                        )
                    truth = np.asarray(old["target"], dtype=float)
                    rmse_deltas.append(
                        abs(
                            float(np.sqrt(((truth - np.asarray(a, dtype=float)) ** 2).mean()))
                            - float(np.sqrt(((truth - np.asarray(b, dtype=float)) ** 2).mean()))
                        )
                    )
        if not pairs:
            print(f"{model:<16}{0:>7}{'no shared runs':>17}")
            continue
        print(f"{model:<16}{pairs:>7}{f'{identical}/{pairs}':>17}{max(rmse_deltas):>14.3e}")
        if identical != pairs:
            failures += 1
    return failures


def main(neural_raw: Path, classical_raw: Path | None) -> int:
    frozen_codes = fingerprints(FROZEN)
    if len(frozen_codes) != 1:
        raise SystemExit(f"the frozen campaign mixes fingerprints: {sorted(frozen_codes)}")
    frozen_code = next(iter(frozen_codes))
    failures = 0
    for label, raw, models, seeds, horizons in (
        ("neural", neural_raw, NEURAL, SEEDS, HORIZONS),
        ("classical", classical_raw, CLASSICAL, CLASSICAL_SEEDS, (1,)),
    ):
        if raw is None or not raw.exists():
            print(f"\n{label}: no campaign at {raw}; skipped")
            continue
        codes = fingerprints(raw)
        print(f"\n{label} campaign {raw.name}: code_sha256 {sorted(c[:12] for c in codes)}")
        if frozen_code in codes:
            raise SystemExit(
                f"{raw.name} shares the frozen fingerprint {frozen_code[:12]}; "
                "comparing a campaign with itself would prove nothing"
            )
        failures += compare(raw, models, seeds, horizons)
    print(f"\nfrozen fingerprint {frozen_code[:12]}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(ROOT / "results" / "raw" / "cet_v2_neural"))
    parser.add_argument(
        "--classical-raw",
        default=str(ROOT / "results" / "raw" / "cet_v2_classical_probe"),
    )
    args = parser.parse_args()
    bad = main(Path(args.raw), Path(args.classical_raw) if args.classical_raw else None)
    if bad:
        raise SystemExit(
            f"{bad} architecture(s) changed under the v2 edits; the cross-study "
            "comparison is not sound until this is explained"
        )
    print("\nEvery shared architecture reproduces its frozen predictions byte for byte.")
