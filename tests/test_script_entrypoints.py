from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cet_benchmark_file_entrypoint_can_import_project():
    result = subprocess.run(
        [sys.executable, "scripts/run_cet_benchmark.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--resume" in result.stdout


def test_ghcn_preparation_file_entrypoint_can_import_project():
    result = subprocess.run(
        [sys.executable, "scripts/prepare_ghcn_panel.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--manifest" in result.stdout


def test_ghcn_benchmark_file_entrypoint_can_import_project():
    result = subprocess.run(
        [sys.executable, "scripts/run_ghcn_benchmark.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--resume" in result.stdout


def test_ghcn_window_audit_file_entrypoint_can_import_project():
    result = subprocess.run(
        [sys.executable, "scripts/audit_ghcn_windows.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--out" in result.stdout


def test_code_fingerprint_is_independent_of_working_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"
    command = [
        sys.executable,
        "-c",
        (
            f"import sys; sys.path.insert(0, {str(scripts)!r}); "
            "import run_cet_benchmark as benchmark; print(benchmark.code_sha256())"
        ),
    ]
    from_root = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    from_tmp = subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    assert from_root.stdout.strip() == from_tmp.stdout.strip()
