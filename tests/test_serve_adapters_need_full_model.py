"""#2729: ``soup serve --adapters`` refuses a LoRA adapter as ``--model``.

Named adapters are served on top of a plain model, and a request that selects
none runs the base with every adapter disabled. With an adapter as ``--model``
that answered every such request with the bare base instead of the adapter the
operator served, and PEFT's ``default`` adapter could not be selected at all.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import serve as serve_cmd
from tests.conftest import strip_ansi

runner = CliRunner()


class _LoadReachedError(RuntimeError):
    pass


def _adapter_dir(path):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "org/base"}),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COLUMNS", "200")

    def _never_load(*_args, **_kwargs):
        raise _LoadReachedError("model load reached")

    monkeypatch.setattr(serve_cmd, "_load_model", _never_load)
    _adapter_dir(tmp_path / "adA")
    _adapter_dir(tmp_path / "adB")
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path


def test_an_adapter_model_with_named_adapters_is_refused_before_loading(project):
    result = runner.invoke(app, ["serve", "--model", "adA", "--adapters", "b=adB"])
    out = " ".join(strip_ansi(result.output).split())
    assert result.exit_code == 2, (out, repr(result.exception))
    assert "--adapters needs a full model as --model" in out
    assert "adA" in out
    assert not isinstance(result.exception, _LoadReachedError)


def test_a_full_model_with_named_adapters_still_reaches_the_load(project):
    result = runner.invoke(app, ["serve", "--model", "full", "--adapters", "b=adB",
                                 "--device", "cpu"])
    out = strip_ansi(result.output)
    assert "--adapters needs a full model" not in out
    assert isinstance(result.exception, _LoadReachedError), (out, repr(result.exception))


def test_an_adapter_model_alone_still_reaches_the_load(project):
    result = runner.invoke(app, ["serve", "--model", "adA", "--device", "cpu"])
    out = strip_ansi(result.output)
    assert "--adapters needs a full model" not in out
    assert isinstance(result.exception, _LoadReachedError), (out, repr(result.exception))
