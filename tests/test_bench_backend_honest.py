"""#1811 — ``soup bench infer --backend vllm|sglang|mlx`` no longer relabels a transformers run.

``soup bench infer`` loads and generates through ``infer._load_model`` /
``infer._generate`` whatever ``--backend`` says, so ``--backend vllm`` (or
``sglang``, ``mlx``, or a directory auto-detected as ``mlx``) ran the
transformers engine and only changed the table row to ``Vllm`` / ``Sglang`` /
``Mlx``. A user comparing engines compared transformers with itself.

An explicit non-transformers backend is now refused before the model loads; an
auto-detected one is named and the run is measured, and labelled, as
transformers.
"""

from __future__ import annotations

import re
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BOX_RE = re.compile(r"[─-╿]")


def _plain(text: str) -> str:
    return " ".join(_BOX_RE.sub(" ", _ANSI_RE.sub("", text)).split())


@pytest.fixture()
def bench_env(monkeypatch, tmp_path):
    from soup_cli.commands import infer

    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda: ("cpu", None))
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    load = MagicMock(return_value=(object(), object()))
    monkeypatch.setattr(infer, "_load_model", load)
    monkeypatch.setattr(infer, "_count_prompt_tokens", lambda tokenizer, text: len(text.split()))
    monkeypatch.setattr(infer, "_generate", MagicMock(return_value=("answer", 2)))
    return model_dir, load


def _bench(model_dir, *extra):
    return CliRunner().invoke(
        app, ["bench", "infer", str(model_dir), "--num-prompts", "1", "--max-tokens", "8", *extra]
    )


@pytest.mark.parametrize("backend", ["vllm", "sglang", "mlx"])
def test_an_explicit_other_backend_is_refused_before_loading(bench_env, backend):
    model_dir, load = bench_env
    result = _bench(model_dir, "--backend", backend)
    out = _plain(result.output)
    assert result.exit_code == 2, (result.exit_code, out)
    assert f"--backend {backend}" in out
    assert "measures the transformers engine" in out
    load.assert_not_called()


def test_an_auto_detected_mlx_dir_is_benchmarked_and_labelled_as_transformers(bench_env):
    model_dir, load = bench_env
    (model_dir / "config.json").unlink()
    (model_dir / "weights.npz").write_bytes(b"")  # what detect_backend reads as mlx
    result = _bench(model_dir)
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "detected mlx" in out
    assert "Transformers" in out
    assert "Mlx" not in out
    load.assert_called_once()


@pytest.mark.parametrize("extra", [(), ("--backend", "transformers")])
def test_control_transformers_is_unchanged(bench_env, extra):
    model_dir, load = bench_env
    result = _bench(model_dir, *extra)
    out = _plain(result.output)
    assert result.exit_code == 0, (result.exit_code, out)
    assert "Transformers" in out
    load.assert_called_once()


def test_control_an_unknown_backend_keeps_its_message(bench_env):
    model_dir, _ = bench_env
    result = _bench(model_dir, "--backend", "bogus")
    assert result.exit_code == 2
    assert "Unknown --backend" in _plain(result.output)


def test_the_help_lists_what_bench_measures():
    result = CliRunner().invoke(app, ["bench", "infer", "--help"], env={"COLUMNS": "200"})
    out = _plain(result.output)
    assert "auto (default) | transformers" in out
    assert "| mlx" not in out
