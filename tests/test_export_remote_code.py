"""Local export trust checks exercise the paths passed to each loader."""

from __future__ import annotations

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.commands import export as export_cmd
from tests.conftest import strip_ansi

_CUSTOM_CONFIG = {"model_type": "soup_test_custom", "auto_map": {"AutoConfig": "c.C"}}
_CUSTOM_TOKENIZER = {"auto_map": {"AutoTokenizer": ["tok.Custom", None]}}
_OPT_IN = "Re-run with --trust-remote-code if you trust the source."


@pytest.fixture
def sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    return base, adapter


@pytest.fixture
def loaders(monkeypatch):
    saved = Mock()
    merged = SimpleNamespace(save_pretrained=saved)
    model = Mock()
    adapter = Mock(return_value=SimpleNamespace(merge_and_unload=lambda: merged))
    tokenizer = Mock(return_value=merged)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float16=object()))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=adapter),
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model),
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
    ))
    return SimpleNamespace(model=model, adapter=adapter, tokenizer=tokenizer, saved=saved)


def test_export_merge_refuses_custom_base_config_before_loading(sources, loaders):
    base, adapter = sources
    (base / "config.json").write_text(json.dumps(_CUSTOM_CONFIG), encoding="utf-8")

    with pytest.raises(ValueError) as exc:
        export_cmd._merge_adapter(str(adapter), str(base), "out")

    assert str(exc.value) == (
        f"Model {base} requires trust_remote_code=True (custom modeling code via auto_map). "
        f"{_OPT_IN} This change in v0.36.0 makes the opt-in explicit; "
        "it was previously enabled by default."
    )
    loaders.model.assert_not_called()


def test_export_merge_refuses_custom_adapter_tokenizer_before_loading(sources, loaders):
    base, adapter = sources
    (adapter / "tokenizer_config.json").write_text(
        json.dumps(_CUSTOM_TOKENIZER), encoding="utf-8"
    )

    with pytest.raises(ValueError) as exc:
        export_cmd._merge_adapter(str(adapter), str(base), "out")

    assert str(exc.value) == (
        f"Tokenizer {adapter} requires trust_remote_code=True (custom tokenizer code via "
        f"auto_map). {_OPT_IN}"
    )
    loaders.model.assert_not_called()
    loaders.tokenizer.assert_not_called()
    loaders.saved.assert_not_called()


def test_export_merge_uses_metadata_from_the_paths_it_loads(sources, loaders):
    base, adapter = sources
    # These metadata files belong to loaders that export does not call on these paths.
    (base / "tokenizer_config.json").write_text(json.dumps(_CUSTOM_TOKENIZER), encoding="utf-8")
    (adapter / "config.json").write_text(json.dumps(_CUSTOM_CONFIG), encoding="utf-8")

    export_cmd._merge_adapter(str(adapter), str(base), "out")

    assert loaders.model.call_args.args == (str(base),)
    assert loaders.model.call_args.kwargs["trust_remote_code"] is False
    loaders.tokenizer.assert_called_once_with(str(adapter), trust_remote_code=False)
    assert loaders.saved.call_count == 2


def test_export_merge_explicit_opt_in_reaches_both_loaders(sources, loaders):
    base, adapter = sources
    (base / "config.json").write_text(json.dumps(_CUSTOM_CONFIG), encoding="utf-8")
    (adapter / "tokenizer_config.json").write_text(
        json.dumps(_CUSTOM_TOKENIZER), encoding="utf-8"
    )

    export_cmd._merge_adapter(str(adapter), str(base), "out", trust_remote_code=True)

    assert loaders.model.call_args.kwargs["trust_remote_code"] is True
    loaders.tokenizer.assert_called_once_with(str(adapter), trust_remote_code=True)


def test_export_torchao_refuses_custom_tokenizer_on_a_native_model(sources, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.utils import save_formats

    base, _ = sources
    (base / "config.json").write_text(
        '{"model_type": "deepseek_v3", "auto_map": {"AutoConfig": "c.C"}}', encoding="utf-8"
    )
    (base / "tokenizer_config.json").write_text(json.dumps(_CUSTOM_TOKENIZER), encoding="utf-8")
    (base.parent / "q.yaml").write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
    pytest.importorskip("transformers")
    export_torchao = Mock()
    monkeypatch.setattr(save_formats, "export_torchao", export_torchao)
    buffer = StringIO()
    monkeypatch.setattr(export_cmd, "console", Console(file=buffer, width=200))

    result = CliRunner().invoke(
        app, ["export", "--model", "base", "--format", "torchao", "--quant-config", "q.yaml"]
    )

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert (
        "Tokenizer base requires trust_remote_code=True (custom tokenizer code via auto_map). "
        f"{_OPT_IN}"
    ) in strip_ansi(buffer.getvalue())
    export_torchao.assert_not_called()


def test_export_torchao_checks_path_containment_before_trust_metadata(sources, monkeypatch):
    from soup_cli.cli import app
    from soup_cli.utils import trust_remote

    base, adapter = sources
    monkeypatch.chdir(adapter)
    (adapter / "q.yaml").write_text("scheme: Int4WeightOnly\n", encoding="utf-8")
    probe = Mock(side_effect=AssertionError("outside model metadata was probed"))
    monkeypatch.setattr(trust_remote, "model_requires_trust_remote_code", probe)
    buffer = StringIO()
    monkeypatch.setattr(export_cmd, "console", Console(file=buffer, width=200))

    result = CliRunner().invoke(
        app, ["export", "--model", str(base), "--format", "torchao", "--quant-config", "q.yaml"]
    )

    assert result.exit_code == 2, (result.output, repr(result.exception))
    assert f"model_dir '{base}' must stay under cwd" in strip_ansi(buffer.getvalue())
    probe.assert_not_called()
