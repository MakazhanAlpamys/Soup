"""#1811 — ``soup serve --backend vllm|sglang|mii`` refuses the flags only transformers applies.

The only consumers of ``--structured-output`` / ``--json-schema`` /
``--regex-pattern``, ``--reasoning-parser``, ``--trace-log`` and
``--record-thumbs`` are in the transformers branch of ``serve``. On vLLM and
SGLang the app was built with no constraint, parser, trace writer or thumbs
log, and MII returned before the structured-output checks ran at all: exit 0, a
missing ``--json-schema`` file never opened, ``--trace-log`` never created. A
client relying on the schema got unconstrained output.

They are now refused with exit 2 before any engine is built, naming each flag
and the backend.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

pytest.importorskip("fastapi")  # serve exits before the backend check without it

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(text: str) -> str:
    return " ".join(_ANSI_RE.sub("", text).split())


@pytest.fixture()
def model_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "model"
    directory.mkdir()
    return directory


def _serve(model_dir, backend, *extra):
    return CliRunner().invoke(
        app, ["serve", "--model", str(model_dir), "--backend", backend, *extra]
    )


_FLAGS = [
    pytest.param(("--structured-output", "json", "--json-schema", "missing.json"),
                 ["--structured-output", "--json-schema"], id="json-schema"),
    pytest.param(("--structured-output", "regex", "--regex-pattern", "[a-z]+"),
                 ["--structured-output", "--regex-pattern"], id="regex"),
    pytest.param(("--reasoning-parser", "deepseek-r1"), ["--reasoning-parser"], id="reasoning"),
    pytest.param(("--trace-log", "t.jsonl"), ["--trace-log"], id="trace-log"),
    pytest.param(("--record-thumbs", "thumbs.db"), ["--record-thumbs"], id="record-thumbs"),
]


@pytest.mark.parametrize("backend", ["vllm", "sglang", "mii"])
@pytest.mark.parametrize("args,named", _FLAGS)
def test_a_transformers_only_flag_is_refused_on_another_backend(
    model_dir, tmp_path, backend, args, named
):
    result = _serve(model_dir, backend, *args)
    out = _plain(result.output)
    assert result.exit_code == 2, (result.exit_code, out)
    for flag in named:
        assert flag in out
    assert f"--backend {backend}" in out
    assert "--backend transformers" in out
    assert not (tmp_path / "t.jsonl").exists()
    assert not (tmp_path / "thumbs.db").exists()


def test_several_flags_are_named_in_one_message(model_dir):
    result = _serve(model_dir, "vllm", "--reasoning-parser", "qwen3", "--trace-log", "t.jsonl")
    out = _plain(result.output)
    assert result.exit_code == 2
    assert out.count("--backend transformers") == 1
    assert "--reasoning-parser" in out and "--trace-log" in out


@pytest.mark.parametrize("backend", ["vllm", "sglang", "mii"])
def test_control_without_those_flags_the_backend_path_is_reached(model_dir, backend):
    # No engine is installed here, so the run stops at the backend's own check,
    # which is past the new refusal.
    result = _serve(model_dir, backend)
    out = _plain(result.output)
    assert "applied on --backend transformers only" not in out


def test_control_structured_output_off_is_not_a_flag_to_refuse(model_dir):
    result = _serve(model_dir, "vllm", "--structured-output", "off")
    assert "applied on --backend transformers only" not in _plain(result.output)
