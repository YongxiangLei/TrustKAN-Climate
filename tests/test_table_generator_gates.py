"""The table generators are the last gate between a stale run and a printed number.

`scripts/aggregate_results.py` is tested for the ledger it builds, but the
manuscript does not read a ledger; it reads the tables the generators emit from
one. So the generator's own refusal to publish a superseded run is load-bearing
for the claim that every number in the paper comes from current code, and it is
tested here against the ways a ledger can go wrong: entirely stale, partly
stale, holding a failure, or holding no status column at all.

The second half checks the property that the gate exists to protect. A
generated file committed to `paper/` is a claim that the generators produce it,
and that claim decays silently whenever a generator changes and its output is
not rebuilt: the file still parses, still looks generated, and still carries a
provenance header. Re-running the generators and comparing bytes is the only
way to notice.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
make_paper_tables = importlib.import_module("make_paper_tables")

# Hex digests that cannot be read back as integers, which an all-digit stand-in
# would be, hiding the string comparison the gate actually performs.
LIVE = "ab" * 32
SUPERSEDED = "cd" * 32


def ledger(tmp_path: Path, rows: list[dict], name: str = "runs.csv") -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_row(**overrides) -> dict:
    row = {
        "model": "trustkan",
        "horizon": 1,
        "seed": 11,
        "status": "ok",
        "rmse": 2.0,
        "code_sha256": LIVE,
    }
    row.update(overrides)
    return row


def test_current_runs_pass_the_gate(tmp_path):
    path = ledger(tmp_path, [run_row()])
    assert len(make_paper_tables.load_ok(path, LIVE)) == 1


def test_wholly_stale_ledger_is_refused_rather_than_silently_emptied(tmp_path):
    """The failure mode the gate exists for: a stale ledger looks unanimous."""
    path = ledger(tmp_path, [run_row(code_sha256=SUPERSEDED, seed=seed) for seed in (11, 22)])
    with pytest.raises(SystemExit) as error:
        make_paper_tables.load_ok(path, LIVE)
    message = str(error.value)
    assert "holds no run matching the current code fingerprint" in message
    # The operator has to be told which fingerprints were found and what to run,
    # or the error is indistinguishable from a missing file.
    assert LIVE[:12] in message and SUPERSEDED[:12] in message
    assert "--collect-only" in message


def test_superseded_rows_are_dropped_beside_current_ones(tmp_path):
    path = ledger(
        tmp_path,
        [run_row(seed=11), run_row(seed=22, code_sha256=SUPERSEDED)],
    )
    kept = make_paper_tables.load_ok(path, LIVE)
    assert list(kept.seed) == [11]


def test_failed_runs_never_reach_a_table(tmp_path):
    path = ledger(tmp_path, [run_row(status="failed")])
    with pytest.raises(ValueError, match="No successful runs"):
        make_paper_tables.load_ok(path, LIVE)


def test_rescored_ledger_without_status_still_faces_the_fingerprint_gate(tmp_path):
    """The extremes ledger re-scores stored predictions and carries no status.

    Waiving the status filter for it must not waive the fingerprint check too,
    which is the mistake that would let the extremes table quote a model set the
    accuracy table no longer contains.
    """
    row = run_row(code_sha256=SUPERSEDED)
    del row["status"]
    path = ledger(tmp_path, [row])
    with pytest.raises(SystemExit, match="holds no run matching"):
        make_paper_tables.load_ok(path, LIVE, has_status=False)


def test_status_bearing_ledger_is_not_read_as_a_rescored_one(tmp_path):
    row = run_row(status="failed")
    del row["status"]
    path = ledger(tmp_path, [row])
    assert len(make_paper_tables.load_ok(path, LIVE, has_status=False)) == 1
    with pytest.raises(AttributeError):
        make_paper_tables.load_ok(path, LIVE)


def test_missing_ledger_is_an_error_not_an_empty_table(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_paper_tables.load_ok(tmp_path / "absent.csv", LIVE)


GENERATORS = (
    ("make_paper_tables.py", "tables"),
    ("make_paper_figures.py", "figures"),
    ("analyze_robustness.py", "tables"),
)
LEDGERS = (
    ROOT / "results" / "aggregated" / "cet_full_runs.csv",
    ROOT / "results" / "reliability" / "aggregated" / "cet_reliability_full_runs.csv",
    ROOT / "results" / "robustness" / "aggregated" / "cet_robustness_grid.csv",
    ROOT / "results" / "robustness" / "aggregated" / "cet_receptive_fields.csv",
)


@pytest.mark.skipif(
    not all(path.exists() for path in LEDGERS),
    reason="run ledgers are not present in this checkout",
)
def test_committed_tables_are_byte_identical_to_a_fresh_generation(tmp_path):
    """A generated file in paper/ must be what today's generators emit.

    The scratch directory lives under the repository because the generators log
    paths relative to it, and because a bundle build already works this way.
    Only .tex output is compared: the figure PDFs embed a creation timestamp,
    so they differ byte-wise on every run while their data does not.
    """
    scratch = ROOT / "dist" / f".generator_check.{tmp_path.name}"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        for script, subdir in GENERATORS:
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts" / script), "--outdir", str(scratch / subdir)],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"{script} failed:\n{result.stdout}{result.stderr}"
        fresh = sorted((scratch / "tables").glob("*.tex"))
        assert fresh, "the generators emitted no LaTeX at all"
        stale = [
            path.name
            for path in fresh
            if (ROOT / "paper" / "tables" / path.name).read_bytes() != path.read_bytes()
        ]
        assert not stale, (
            "these committed tables are not what the generators now produce, so "
            "the manuscript is quoting output no generator would emit today; "
            "re-run them and commit the result: " + ", ".join(stale)
        )
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)
