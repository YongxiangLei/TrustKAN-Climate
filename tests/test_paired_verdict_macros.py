"""The comparison verdict must come from the evidence, not from the prose.

A sentence that says the proposed model beats a baseline can outlive the
evidence for it, which is the failure the paired tests exist to catch. The
verdict is therefore emitted as a macro derived from the Holm-corrected counts,
and these tests pin the mapping from counts to wording.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_paper_tables as generator  # noqa: E402


def index_rows(a_better, b_better, n_pairs=5, comparator="transformer"):
    return [
        {
            "model_a": "trustkan_v2",
            "model_b": comparator,
            "horizon": horizon,
            "n_pairs": n_pairs,
            "mean_rmse_difference": -0.01,
            "a_better": a,
            "b_better": b,
            "inconclusive": n_pairs - a - b,
        }
        for horizon, a, b in zip((1, 7, 30, 90), a_better, b_better)
    ]


def macros_for(tmp_path, rows, monkeypatch):
    comparisons = tmp_path / "primary_comparisons_index.json"
    comparisons.write_text(json.dumps(rows), encoding="utf-8")
    study = generator.STUDIES["v2"]
    monkeypatch.setattr(
        generator,
        "STUDY",
        generator.Study(
            benchmark=study.benchmark,
            reliability=study.reliability,
            comparisons=comparisons,
            ablations=study.ablations,
            ablation_runs=study.ablation_runs,
            extremes=study.extremes,
            proposed="trustkan_v2",
        ),
    )
    bench = pd.DataFrame(
        {
            "model": ["trustkan_v2"] * 4 + ["transformer"] * 4,
            "horizon": [1, 7, 30, 90] * 2,
            "rmse": [2.0, 2.7, 2.9, 2.95, 2.0, 2.7, 2.9, 2.96],
        }
    )
    rel = pd.DataFrame(
        {
            "horizon": [1, 7, 30, 90],
            "conformal_marginal_coverage": [0.9] * 4,
            "conformal_joint_coverage": [0.8] * 4,
            "simultaneous_joint_coverage": [0.9] * 4,
            "error_detection_auroc": [0.6] * 4,
            "reliability_spearman": [-0.3] * 4,
            "nominal_coverage": [0.9] * 4,
        }
    )
    out = tmp_path / "prose.tex"
    generator.prose_macros(bench, rel, None, None, None, out)
    text = out.read_text(encoding="utf-8")
    return {
        line.split("}{", 1)[0].replace("\\newcommand{\\", ""): line.split("}{", 1)[1].rstrip("}\n")
        for line in text.splitlines()
        if line.startswith("\\newcommand")
    }


def test_no_significant_pair_reads_as_indistinguishable(tmp_path, monkeypatch):
    macros = macros_for(tmp_path, index_rows([0, 0, 0, 0], [0, 0, 0, 0]), monkeypatch)
    assert macros["pairedTransformerVerdict"] == "statistically indistinguishable from"
    assert macros["pairedTransformerWins"] == "0"
    assert macros["pairedTransformerTotal"] == "20"


def test_a_clean_sweep_reads_as_better(tmp_path, monkeypatch):
    macros = macros_for(tmp_path, index_rows([5, 5, 5, 5], [0, 0, 0, 0]), monkeypatch)
    assert macros["pairedTransformerVerdict"] == "better than"


def test_a_partial_win_is_not_reported_as_a_sweep(tmp_path, monkeypatch):
    macros = macros_for(tmp_path, index_rows([5, 0, 0, 0], [0, 0, 0, 0]), monkeypatch)
    assert macros["pairedTransformerVerdict"] == "better than or level with"


def test_losing_pairs_are_never_described_as_a_win(tmp_path, monkeypatch):
    macros = macros_for(tmp_path, index_rows([0, 0, 0, 0], [5, 5, 5, 5]), monkeypatch)
    assert macros["pairedTransformerVerdict"] == "worse than"
    mixed = macros_for(tmp_path, index_rows([5, 0, 0, 0], [0, 0, 0, 5]), monkeypatch)
    assert mixed["pairedTransformerVerdict"] == "mixed against"


@pytest.mark.parametrize("wins,losses", [([5, 5, 5, 5], [0, 0, 0, 0]), ([0] * 4, [0] * 4)])
def test_counts_are_summed_across_the_whole_family(tmp_path, monkeypatch, wins, losses):
    macros = macros_for(tmp_path, index_rows(wins, losses), monkeypatch)
    assert int(macros["pairedTransformerWins"]) == sum(wins)
    assert int(macros["pairedTransformerLosses"]) == sum(losses)
