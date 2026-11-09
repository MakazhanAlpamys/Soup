"""GGUF progress CLI tests must ignore the developer's cached toolchains."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import strip_ansi


@pytest.mark.parametrize("source", ["home", "environment"])
def test_export_progress_ignores_external_llama_cpp(tmp_path: Path, source: str) -> None:
    external_home = tmp_path / "external-home"
    external_home.mkdir()
    checkout = (
        external_home / ".soup" / "llama.cpp"
        if source == "home"
        else tmp_path / "external-toolchain"
    )
    checkout.mkdir(parents=True)
    converter = checkout / "convert_hf_to_gguf.py"
    converter.write_text("# external converter must not be probed\n", encoding="utf-8")
    env = os.environ.copy()
    env["HOME"] = str(external_home)
    env["USERPROFILE"] = str(external_home)
    env.pop("LLAMA_CPP_PATH", None)
    if source == "environment":
        env["LLAMA_CPP_PATH"] = str(checkout)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_issue1760_gguf_progress.py::"
            "test_no_install_deps_present_never_runs_pip_and_reports_phases",
            "-q",
            "--no-cov",
            "--tb=short",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    output = strip_ansi(result.stdout)
    assert "1 passed" in output
    assert converter.read_text(encoding="utf-8") == "# external converter must not be probed\n"
