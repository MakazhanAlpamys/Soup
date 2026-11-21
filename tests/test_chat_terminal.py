"""Chat displays literal outside text and retains original loading/history values."""

import copy
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands import chat as command
from tests._windows_filenames import skip_colon_on_windows
from tests.conftest import strip_ansi

VALUES = [
    pytest.param("ordinary", "ordinary", id="plain-control"),
    pytest.param(":rocket:", ":rocket:", id="emoji-alias"),
    pytest.param("x[red]name", "x[red]name", id="markup"),
    pytest.param(r"x\[unfinished", r"x\[unfinished", id="backslash"),
    pytest.param("x\x1b\x07\x9b\u202e[end]", "x[end]", id="controls"),
]
FILE_VALUES = [
    pytest.param(*case.values, id=case.id,
                 marks=skip_colon_on_windows if ":" in case.values[0] else ())
    for case in VALUES
]


def _plain(text):
    text = strip_ansi(text)
    assert not any(c in text for c in ("\x1b", "\x07", "\x9b", "\u202e")), repr(text)
    return " ".join(text.split())


def _assert_style(output: str, expected: str, code: str) -> None:
    spans = re.findall(r"\x1b\[([\d;]+)m([^\x1b]*)", output)
    assert any(code in codes.split(";") and text.strip() == expected
               for codes, text in spans), (expected, output)


def _filename(value):
    return "model-" + value.replace("\x1b", "").replace("\x07", "")


@pytest.fixture
def printed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stream = io.StringIO()
    monkeypatch.setattr(command, "console", Console(
        file=stream, width=500, height=150, force_terminal=True,
        color_system="standard", no_color=False,
    ))
    return stream


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("kind", ["native-option", "native-load", "hub-value", "hub-import"])
def test_chat_refusals_preserve_literal_messages_and_exit_codes(
    printed, monkeypatch, value, shown, kind
):
    from soup_cli.utils import hubs, mlx_generation

    Path("model").mkdir()
    args = ["chat", "--model", "model"]

    def fail(*args, **kwargs):
        error = ImportError if kind == "hub-import" else ValueError
        raise error(f"Cannot use {value}")

    if kind == "native-option":
        monkeypatch.setattr(mlx_generation, "reject_native_adapter", fail)
    elif kind == "native-load":
        args += ["--backend", "mlx"]
        monkeypatch.setattr(mlx_generation, "validate_native_options", lambda **kwargs: None)
        monkeypatch.setattr(mlx_generation, "load_native_model", fail)
    else:
        args += ["--hub", "modelscope"]
        monkeypatch.setattr(hubs, "apply_hub_to_cli_model", fail)
    result = CliRunner().invoke(app, args)
    assert result.exit_code == (2 if kind == "hub-value" else 1)
    assert _plain(printed.getvalue()) == f"Cannot use {shown}"
    assert "\x1b[31m" in printed.getvalue()


@pytest.mark.parametrize("value,shown", FILE_VALUES)
def test_chat_missing_model_path_is_literal(printed, value, shown):
    path = Path(_filename(value))
    result = CliRunner().invoke(app, ["chat", "--model", str(path)])
    assert result.exit_code == 1
    assert _plain(printed.getvalue()) == f"Model path not found: {Path(_filename(shown))}"


@pytest.mark.parametrize("value,shown", FILE_VALUES)
@pytest.mark.parametrize("device,shown_device", VALUES)
def test_chat_loading_panel_keeps_the_literal_existing_model_path(
    printed, monkeypatch, value, shown, device, shown_device
):
    from soup_cli.utils import trust_remote

    path = Path(_filename(value))
    path.mkdir(parents=True)
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return object(), object()

    monkeypatch.setattr(command, "_load_model", load)
    monkeypatch.setattr(command.console, "input", lambda *args: "/quit")
    monkeypatch.setattr(trust_remote, "resolve_trust_remote_code", lambda *a, **k: False)
    monkeypatch.setattr(trust_remote, "model_requires_trust_remote_code", lambda *a: False)
    result = CliRunner().invoke(app, ["chat", "--model", str(path), "--device", device])
    assert result.exit_code == 0, (result.output, result.exception)
    assert f"Model: {Path(_filename(shown))}" in _plain(printed.getvalue())
    assert "Type: Full model" in _plain(printed.getvalue())
    assert f"Device: {shown_device}" in _plain(printed.getvalue())
    _assert_style(printed.getvalue(), str(Path(_filename(shown))), "1")
    _assert_style(printed.getvalue(), shown_device, "1")
    _assert_style(printed.getvalue(), "Full model", "1")
    assert calls == [{"model_path": str(path), "base_model": None, "is_adapter": False,
                      "device": device, "trust_remote_code": False}]


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("backend", ["transformers", "mlx"])
def test_chat_history_keeps_original_text_while_system_and_replies_are_literal(
    printed, monkeypatch, value, shown, backend
):
    from soup_cli.utils import mlx_generation, trust_remote

    Path("model").mkdir()
    metadata = Path("model/adapter_config.json")
    metadata.write_text(json.dumps({"base_model_name_or_path": value}), encoding="utf-8")
    before = metadata.read_bytes()
    inputs = iter([f"/system set {value}", f"ask {value}", "/clear", f"again {value}", "/quit"])
    monkeypatch.setattr(command.console, "input", lambda *args: next(inputs))
    loads, generations = [], []
    pair = (object(), object())

    def load(**kwargs):
        loads.append(kwargs)
        return pair

    def generate(model, tokenizer, history, **kwargs):
        assert (model, tokenizer) == pair
        generations.append((copy.deepcopy(history), kwargs))
        return f"reply {value}"

    monkeypatch.setattr(trust_remote, "resolve_trust_remote_code", lambda *a, **k: True)
    monkeypatch.setattr(trust_remote, "model_requires_trust_remote_code", lambda *a: False)
    if backend == "transformers":
        monkeypatch.setattr(command, "_load_model", load)
        monkeypatch.setattr(command, "_generate", generate)
    else:
        def native_load(path, base, **kwargs):
            loads.append({"path": path, "base": base, **kwargs})
            return pair

        monkeypatch.setattr(mlx_generation, "validate_native_options", lambda **kwargs: None)
        monkeypatch.setattr(mlx_generation, "load_native_model", native_load)
        monkeypatch.setattr(mlx_generation, "generate_native",
                            lambda *a, **k: (generate(*a, **k), 1))
    result = CliRunner().invoke(app, [
        "chat", "--model", "model", "--base", value, "--backend", backend,
        "--device", "cpu", "--max-tokens", "9", "--temperature", "0",
        "--system", f"initial {value}", "--trust-remote-code",
    ])
    assert result.exit_code == 0, (result.output, result.exception)
    text = _plain(printed.getvalue())
    assert f"System prompt set: set {shown}" in text
    _assert_style(printed.getvalue(), f"System prompt set: set {shown}", "2")
    assert text.count(f"Assistant: reply {shown}") == 2
    assert "Conversation cleared." in text and "Goodbye!" in text
    expected = [
        [{"role": "system", "content": f"set {value}"},
         {"role": "user", "content": f"ask {value}"}],
        [{"role": "system", "content": f"initial {value}"},
         {"role": "user", "content": f"again {value}"}],
    ]
    assert generations == [(history, {"max_tokens": 9, "temperature": 0.0})
                           for history in expected]
    if backend == "transformers":
        assert loads == [{"model_path": "model", "base_model": value, "is_adapter": True,
                          "device": "cpu", "trust_remote_code": True}]
        assert f"Base: {shown}" in text and "Loading model" in text
        _assert_style(printed.getvalue(), shown, "1")
        _assert_style(printed.getvalue(), "LoRA adapter", "1")
    else:
        assert loads == [{"path": "model", "base": value, "trust_remote_code": True}]
    assert metadata.read_bytes() == before
    assert "\x1b[1;32mAssistant:" in printed.getvalue()


@pytest.mark.parametrize("value,shown", VALUES)
@pytest.mark.parametrize("adapter", [False, True])
def test_chat_loader_progress_is_literal_and_keeps_original_model_arguments(
    printed, monkeypatch, value, shown, adapter
):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = SimpleNamespace(pad_token=None, eos_token="eos")
    model = SimpleNamespace(eval=lambda: None)
    calls = []

    def token_load(path, **kwargs):
        calls.append(("tokenizer", path, kwargs))
        return tokenizer

    def model_load(path, **kwargs):
        calls.append(("model", path, kwargs))
        return model

    def adapter_load(base, path):
        assert base is model
        calls.append(("adapter", path, {}))
        return model

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", token_load)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", model_load)
    monkeypatch.setattr(PeftModel, "from_pretrained", adapter_load)
    result = command._load_model(f"model-{value}", f"base-{value}", adapter, "cpu", True)
    assert result == (model, tokenizer) and tokenizer.pad_token == "eos"
    text = _plain(printed.getvalue())
    assert "Loading tokenizer..." in text
    # Rich highlights the ellipsis separately while keeping the line dim.
    _assert_style(printed.getvalue(), "Loading tokenizer", "2")
    if adapter:
        assert f"Loading base model: base-{shown}..." in text
        assert f"Loading LoRA adapter: model-{shown}..." in text
        _assert_style(printed.getvalue(), f"Loading base model: base-{shown}...", "2")
        _assert_style(printed.getvalue(), f"Loading LoRA adapter: model-{shown}...", "2")
    else:
        assert f"Loading model: model-{shown}..." in text
        _assert_style(printed.getvalue(), f"Loading model: model-{shown}...", "2")
    assert [(kind, path) for kind, path, _ in calls] == (
        [("tokenizer", f"model-{value}"), ("model", f"base-{value}"),
         ("adapter", f"model-{value}")] if adapter else
        [("tokenizer", f"model-{value}"), ("model", f"model-{value}")]
    )
    assert calls[0][2] == {"trust_remote_code": True}
    assert calls[1][2]["trust_remote_code"] is True
