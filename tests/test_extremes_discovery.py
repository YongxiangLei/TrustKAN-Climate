"""The extremes runner must score every model a raw directory actually holds."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cet_extremes import discover_models, resolve_artifact


def test_discover_models_reads_underscore_names(tmp_path):
    first = tmp_path / "v2"
    second = tmp_path / "frozen"
    first.mkdir()
    second.mkdir()
    (first / "cet_trustkan_v2_h90_s11.npz").write_bytes(b"x")
    (first / "cet_trustkan_dilated_h90_s11.npz").write_bytes(b"x")
    (second / "cet_random_forest_h90_s11.npz").write_bytes(b"x")
    assert discover_models([first, second]) == [
        "random_forest",
        "trustkan_dilated",
        "trustkan_v2",
    ]


def test_resolve_artifact_prefers_the_first_raw_directory(tmp_path):
    first = tmp_path / "v2"
    second = tmp_path / "frozen"
    first.mkdir()
    second.mkdir()
    newer = first / "cet_kan_h1_s11.npz"
    older = second / "cet_kan_h1_s11.npz"
    newer.write_bytes(b"new")
    older.write_bytes(b"old")
    assert resolve_artifact([first, second], "cet_kan_h1_s11.npz") == newer
    assert resolve_artifact([second], "cet_kan_h1_s11.npz") == older
    assert resolve_artifact([first], "cet_svr_h1_s-1.npz") is None
