from __future__ import annotations

import subprocess
import sys


def test_cet_benchmark_file_entrypoint_can_import_project():
    result = subprocess.run(
        [sys.executable, "scripts/run_cet_benchmark.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
