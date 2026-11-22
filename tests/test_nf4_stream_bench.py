"""Exercise the public CPU entry point used before the #842 GPU measurements."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_cpu_self_test(tmp_path):
    pytest.importorskip("torch")
    script = Path(__file__).parents[1] / "benchmarks" / "harness" / "nf4_stream_bench.py"
    result = tmp_path / "result.json"
    subprocess.run(
        [sys.executable, str(script), "--self-test", "--json", str(result)], check=True, timeout=90
    )
    assert json.loads(result.read_text())["self_test"] == "PASS"
