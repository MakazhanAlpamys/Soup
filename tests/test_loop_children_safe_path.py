"""The loop's ``python -m`` / ``python -c`` children start with ``-P``.

Every child of the daemon is ``python -m soup_cli.cli ...`` or ``python -c <script>``, and both put
the current directory first on ``sys.path``: a file named ``torch.py``, ``yaml.py`` or
``datasets.py`` in the project directory would be imported by the child, with the daemon's
environment. ``-P`` (Python 3.11, the declared floor) keeps the current directory off
``sys.path``. All children are built by ``loop_spawn.python_child_argv``, and a scan checks that no
loop module builds its own ``[sys.executable, ...]``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

import soup_cli
from soup_cli.utils.loop_spawn import python_child_argv

SRC = Path(soup_cli.__file__).resolve().parent


def test_the_argv_has_the_interpreter_then_dash_p_then_the_rest():
    assert python_child_argv("-m", "soup_cli.cli", "train") == [
        sys.executable, "-P", "-m", "soup_cli.cli", "train",
    ]
    assert python_child_argv("-c", "print(1)")[:3] == [sys.executable, "-P", "-c"]


def _run(argv, cwd):
    return subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, timeout=60, check=False,
    )


def test_a_library_named_file_in_the_current_directory_is_not_imported(tmp_path):
    (tmp_path / "json.py").write_text("SHADOW = True\n", encoding="utf-8")
    probe = "import json; print(hasattr(json, 'SHADOW'))"

    plain = _run([sys.executable, "-c", probe], tmp_path)
    safe = _run(python_child_argv("-c", probe), tmp_path)

    assert plain.stdout.strip() == "True", (plain.stdout, plain.stderr)  # the setup is real
    assert safe.stdout.strip() == "False", (safe.stdout, safe.stderr)


def test_a_module_run_with_dash_m_does_not_find_the_current_directory_either(tmp_path):
    (tmp_path / "mylocal.py").write_text("print('shadow')\n", encoding="utf-8")

    plain = _run([sys.executable, "-m", "mylocal"], tmp_path)
    safe = _run(python_child_argv("-m", "mylocal"), tmp_path)

    assert plain.stdout.strip() == "shadow", (plain.stdout, plain.stderr)
    assert safe.returncode != 0 and "No module named" in safe.stderr


def _uses_sys_executable(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Attribute) and node.attr == "executable"
        and isinstance(node.value, ast.Name) and node.value.id == "sys"
        for node in ast.walk(tree)
    )


def test_no_loop_module_builds_its_own_interpreter_argv():
    loop_modules = [SRC / "commands" / "loop.py", *sorted((SRC / "utils").glob("loop_*.py"))]
    offenders = [
        path.name for path in loop_modules
        if path.name != "loop_spawn.py" and _uses_sys_executable(path)
    ]

    assert offenders == [], f"build the child argv with loop_spawn.python_child_argv: {offenders}"


def test_the_detached_daemon_is_started_with_dash_p(tmp_path, monkeypatch):
    from unittest import mock

    from soup_cli.cli import app

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_DETACHED", raising=False)
    runner = CliRunner()
    init = runner.invoke(app, ["loop", "init", "m", "--eval", "e.json", "--baseline", "b"])
    assert init.exit_code == 0, (init.output, repr(init.exception))

    with mock.patch("soup_cli.commands.loop.subprocess.Popen") as popen:
        popen.return_value.pid = 123
        result = runner.invoke(app, ["loop", "watch", "--detach"])

    assert result.exit_code == 0, (result.output, repr(result.exception))
    argv = popen.call_args.args[0]
    assert argv[:4] == [sys.executable, "-P", "-m", "soup_cli.cli"]
    assert argv[4:7] == ["loop", "watch", "--foreground"]


def test_the_tokenizer_child_is_started_with_dash_p(monkeypatch):
    import soup_cli.utils.loop_max_length as loop_max_length

    monkeypatch.undo()  # the autouse conftest stub of the measurement: this test wants the real one
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(Exception):  # noqa: B017 - a failed child is an error; the argv is the point
        loop_max_length._measure_in_child("org/base", "pairs.jsonl", "sft")

    assert seen[0][:3] == [sys.executable, "-P", "-c"]


def test_the_memory_probe_child_is_started_with_dash_p(monkeypatch):
    import soup_cli.utils.loop_quantization as loop_quantization

    monkeypatch.undo()  # the autouse conftest stub of the probe: this test wants the real one
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert loop_quantization.free_vram_bytes() is None  # the fake child failed

    assert seen[0][:3] == [sys.executable, "-P", "-c"]
