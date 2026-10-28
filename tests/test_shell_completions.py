"""Generated completion scripts must speak the installed Typer's protocol."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app


@pytest.fixture
def completion_process(tmp_path, monkeypatch):
    entrypoint = tmp_path / "soup"
    entrypoint.write_text(
        f"#!{sys.executable}\nfrom soup_cli.cli import run\nrun()\n", encoding="utf-8"
    )
    entrypoint.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ.get("PATH", ""))
    source = str(Path(__file__).parents[1] / "src")
    monkeypatch.setenv("PYTHONPATH", source + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_NO_TELEMETRY", "1")
    monkeypatch.delenv("_SOUP_COMPLETE", raising=False)
    return entrypoint


def _script(shell: str) -> str:
    result = CliRunner().invoke(app, ["completions", shell])
    assert result.exit_code == 0, (result.output, repr(result.exception))
    return result.stdout


@pytest.mark.parametrize("shell", ["bash", "zsh", "fish"])
@pytest.mark.parametrize(
    ("words", "cword", "expected", "excluded"),
    [("soup rec", "1", "recipes", "version"), ("soup train --res", "2", "--resume", "--config")],
)
def test_generated_completion_instruction_reaches_live_command_tree(
    completion_process, shell, words, cword, expected, excluded
):
    # Take the instruction from the emitted script, not a hardcoded working value.
    instruction = re.search(r"_SOUP_COMPLETE=([a-z_]+)", _script(shell))
    assert instruction is not None
    result = subprocess.run(
        [sys.executable, str(completion_process)],
        env={
            **os.environ,
            "_SOUP_COMPLETE": instruction.group(1),
            "COMP_WORDS": words,
            "COMP_CWORD": cword,
            "_TYPER_COMPLETE_ARGS": words,
            "_TYPER_COMPLETE_FISH_ACTION": "get-args",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    if shell == "zsh":
        assert f'"{expected}"' in result.stdout
        assert f'"{excluded}"' not in result.stdout
    else:
        candidates = {line.split("\t", 1)[0] for line in result.stdout.splitlines()}
        assert expected in candidates
        assert excluded not in candidates


@pytest.mark.skipif(os.name == "nt", reason="executes a POSIX shell entrypoint")
@pytest.mark.parametrize(
    ("words", "cword", "expected", "excluded"),
    [("soup rec", 1, "recipes", "version"), ("soup train --res", 2, "--resume", "--config")],
)
def test_sourced_bash_script_returns_completion_values(
    completion_process, tmp_path, words, cword, expected, excluded
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not installed")
    script = _script("bash")
    function = re.search(r"-F\s+(\w+)\s+soup", script)
    assert function is not None
    script_path = tmp_path / "completion.bash"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [
            bash, "--noprofile", "--norc", "-c",
            'source "$1"\n'
            f"COMP_WORDS=({words})\nCOMP_CWORD={cword}\n"
            f"{function.group(1)} soup\n"
            'printf "%s\\n" "${COMPREPLY[@]}"\n',
            "completion-test", str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stderr == ""
    candidates = result.stdout.splitlines()
    assert expected in candidates
    assert excluded not in candidates


@pytest.mark.skipif(os.name == "nt", reason="executes a POSIX shell entrypoint")
@pytest.mark.parametrize(
    ("words", "current", "expected", "excluded"),
    [("soup rec", 2, "recipes", "version"), ("soup train --res", 3, "--resume", "--config")],
)
def test_sourced_zsh_script_passes_live_candidates_to_completion_widgets(
    completion_process, tmp_path, words, current, expected, excluded
):
    zsh = shutil.which("zsh")
    if zsh is None:
        pytest.skip("zsh is not installed")
    script_path = tmp_path / "completion.zsh"
    script_path.write_text(_script("zsh"), encoding="utf-8")
    result = subprocess.run(
        [
            zsh, "-f", "-c",
            # Capture the widget boundary; the source, child command and response
            # evaluation still run in a real zsh without requiring an active ZLE.
            'compdef() { completion_function="$1"; }\n'
            '_arguments() { print -rl -- "$@"; }\n'
            '_files() { print -r -- files-fallback; }\n'
            '_describe() { print -rl -- "${completions[@]}"; }\n'
            'source "$1"\n'
            f"words=({words})\nCURRENT={current}\n"
            '"$completion_function"\n',
            "completion-test", str(script_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stderr == ""
    assert f'"{expected}"' in result.stdout
    assert f'"{excluded}"' not in result.stdout
