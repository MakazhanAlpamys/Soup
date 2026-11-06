"""Training gates must run on the selected path or be refused (#1811)."""

from __future__ import annotations

import importlib

import pytest
import yaml
from rich.console import Console

from soup_cli.cli import app
from soup_cli.config.loader import load_config, load_config_from_string
from soup_cli.config.schema import SoupConfig
from tests.conftest import make_cli_runner, strip_ansi

_MLX_REFUSAL = (
    "training.eval_gate is not supported for backend='mlx' "
    "because MLX does not attach a live training callback"
)
_TASK_REFUSALS = {
    "prm": "training.eval_gate is not supported for task='prm' "
    "because 'prm' does not attach a live training callback",
    "unlearn": "training.eval_gate is not supported for task='unlearn' "
    "because 'unlearn' does not attach a live training callback",
    "moe_lora_routing": "training.eval_gate is not supported for task='moe_lora_routing' "
    "because 'moe_lora_routing' does not attach a live training callback",
}
_UNSUPPORTED = [
    pytest.param(task, "mlx", _MLX_REFUSAL, id=f"mlx-{task}")
    for task in ("sft", "dpo", "orpo", "simpo", "kto", "grpo")
] + [
    pytest.param(task, "transformers", message, id=task)
    for task, message in _TASK_REFUSALS.items()
]
_SUPPORTED = [("sft", "transformers"), ("dpo", "transformers"),
              ("grpo", "transformers"), ("sft", "unsloth")]


def _config(task: str = "sft", backend: str = "transformers", gate: str = "enabled") -> dict:
    raw = {
        "base": "sshleifer/tiny-gpt2",
        "task": task,
        "backend": backend,
        "data": {"train": "train.jsonl", "format": "alpaca"},
        "training": {"eval_gate": {"enabled": True, "suite": "suite.yaml"}},
        "output": "out",
    }
    if task in ("dpo", "orpo", "simpo", "preference"):
        raw["data"]["format"] = "dpo"
    elif task == "kto":
        raw["data"]["format"] = "kto"
    elif task == "grpo":
        raw["data"]["format"] = "chatml"
    elif task == "prm":
        raw["data"]["format"] = "prm"
    elif task == "unlearn":
        raw["data"]["forget_set"] = "forget.jsonl"
        raw["training"]["unlearn_method"] = "npo"
    elif task == "moe_lora_routing":
        raw["training"]["mole_task_adapters"] = ["adapter_a", "adapter_b"]
    if gate == "omitted":
        raw["training"].pop("eval_gate")
    elif gate == "null":
        raw["training"]["eval_gate"] = None
    elif gate == "disabled":
        raw["training"]["eval_gate"] = {"enabled": False, "suite": "missing.yaml"}
    return raw


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
def test_string_loader_refuses_an_enabled_gate(task, backend, message):
    with pytest.raises(ValueError) as caught:
        load_config_from_string(yaml.safe_dump(_config(task, backend)))

    assert str(caught.value) == ": Value error, " + message


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
def test_schema_refuses_an_enabled_gate(task, backend, message):
    with pytest.raises(ValueError) as caught:
        SoupConfig(**_config(task, backend))

    assert [error["msg"] for error in caught.value.errors()] == ["Value error, " + message]


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
def test_file_loader_refuses_with_exit_one(task, backend, message, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("soup_cli.config.loader.console", Console(width=240, height=100))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(task, backend)), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        load_config(path)

    assert caught.value.code == 1
    assert message in [line.strip() for line in strip_ansi(capsys.readouterr().out).splitlines()]


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
@pytest.mark.parametrize("gate", ["omitted", "null", "disabled"])
def test_inactive_gates_still_load(task, backend, message, gate):
    cfg = load_config_from_string(yaml.safe_dump(_config(task, backend, gate)))

    assert cfg.training.eval_gate is None or cfg.training.eval_gate.enabled is False


@pytest.mark.parametrize(("task", "backend"), _SUPPORTED)
def test_supported_enabled_gates_still_load(task, backend):
    cfg = load_config_from_string(yaml.safe_dump(_config(task, backend)))

    assert cfg.training.eval_gate.enabled is True
    assert cfg.training.eval_gate.suite == "suite.yaml"


@pytest.mark.parametrize("loss", ["dpo", "orpo", "simpo"])
def test_mlx_preference_aliases_cannot_bypass_the_backend_refusal(loss):
    raw = _config("preference", "mlx")
    raw["training"]["preference_loss"] = loss
    with pytest.raises(ValueError) as caught:
        load_config_from_string(yaml.safe_dump(raw))

    assert str(caught.value) == ": Value error, " + _MLX_REFUSAL


class _HardwareBoundaryError(RuntimeError):
    """Stop valid controls before touching hardware or loading a model."""


@pytest.fixture(params=[False, True], ids=["plain", "forced-color"])
def train_cli(tmp_path, monkeypatch, request):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SOUP_NO_AUDIT_LOG", "1")
    monkeypatch.setenv("SOUP_TELEMETRY", "0")
    module = importlib.import_module("soup_cli.commands.train")
    console = Console(width=240, height=100, force_terminal=request.param, no_color=False)
    monkeypatch.setattr(module, "console", console)
    monkeypatch.setattr("soup_cli.config.loader.console", console)
    calls = []

    def hardware_boundary(*args, **kwargs):
        calls.append("hardware")
        raise _HardwareBoundaryError("hardware initialization reached")

    def unexpected_data_or_model(*args, **kwargs):
        calls.append("data/model")
        raise AssertionError("refusal must precede data/model construction")

    monkeypatch.setattr(module, "detect_device", hardware_boundary)
    monkeypatch.setattr(module, "load_dataset", unexpected_data_or_model)
    monkeypatch.setattr(
        "soup_cli.trainer.dispatch.build_trainer", unexpected_data_or_model
    )
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump({
        "suite": "regression",
        "tasks": [{"type": "custom", "name": "exact", "tasks": "eval.jsonl",
                   "scorer": "exact", "threshold": 0.8}],
    }), encoding="utf-8")

    def invoke(raw, *flags):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        result = make_cli_runner().invoke(
            app, ["train", "--config", str(path), "--dry-run", "--no-reexec", *flags]
        )
        return result, calls

    return invoke


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
@pytest.mark.parametrize("source", ["yaml", "shortcut", "shortcut-disabled"])
def test_train_refuses_before_hardware_data_and_model(task, backend, message, source, train_cli):
    gate = {"yaml": "enabled", "shortcut": "omitted", "shortcut-disabled": "disabled"}[source]
    flags = [] if source == "yaml" else ["--gate", "suite.yaml"]
    result, calls = train_cli(_config(task, backend, gate), *flags)

    assert calls == []
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert message in [line.strip() for line in strip_ansi(result.stdout).splitlines()]
    assert isinstance(result.exception, SystemExit)
    assert result.stderr == ""


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
@pytest.mark.parametrize("gate", ["omitted", "null", "disabled"])
def test_train_allows_inactive_gates_to_reach_hardware(task, backend, message, gate, train_cli):
    result, calls = train_cli(_config(task, backend, gate))

    assert calls == ["hardware"]
    assert isinstance(result.exception, _HardwareBoundaryError)
    assert "Eval gate enabled" not in strip_ansi(result.stdout)


@pytest.mark.parametrize(("task", "backend"), _SUPPORTED)
@pytest.mark.parametrize("source", ["yaml", "shortcut", "shortcut-disabled"])
def test_train_allows_supported_gates_to_reach_hardware(task, backend, source, train_cli):
    gate = {"yaml": "enabled", "shortcut": "omitted", "shortcut-disabled": "disabled"}[source]
    flags = [] if source == "yaml" else ["--gate", "suite.yaml"]
    result, calls = train_cli(_config(task, backend, gate), *flags)

    assert calls == ["hardware"]
    assert isinstance(result.exception, _HardwareBoundaryError)
    if source != "yaml":
        assert "Eval gate enabled with suite: suite.yaml" in strip_ansi(result.stdout).splitlines()


def test_supported_shortcut_keeps_invalid_suite_refusal(train_cli):
    result, calls = train_cli(_config(gate="omitted"), "--gate", "missing.yaml")

    assert calls == []
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Invalid --gate suite:" in strip_ansi(result.stdout)
    assert "training.eval_gate is not supported" not in result.stdout


def test_unsupported_shortcut_is_refused_before_suite_loading(train_cli):
    result, calls = train_cli(_config(backend="mlx", gate="omitted"), "--gate", "missing.yaml")

    assert calls == []
    assert result.exit_code == 1
    assert _MLX_REFUSAL in strip_ansi(result.stdout).splitlines()
    assert "Invalid --gate suite:" not in result.stdout


@pytest.mark.parametrize(("task", "backend", "message"), _UNSUPPORTED)
def test_doctor_refuses_the_same_config_with_exit_two(
    task, backend, message, tmp_path, monkeypatch,
):
    module = importlib.import_module("soup_cli.commands.doctor")
    console = Console(width=240, height=100, force_terminal=True, no_color=False)
    monkeypatch.setattr(module, "console", console)
    monkeypatch.setattr("soup_cli.config.loader.console", console)
    # These are system probes; the config loader and doctor command remain real.
    monkeypatch.setattr(module, "_check_gpu", lambda: None)
    monkeypatch.setattr(module, "_check_mlx", lambda: None)
    monkeypatch.setattr(module, "_check_resources", lambda **kwargs: None)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(_config(task, backend)), encoding="utf-8")
    result = make_cli_runner().invoke(app, ["doctor", "--config", str(path)])

    assert result.exit_code == 2, (result.output, repr(result.exception))
    lines = [line.strip() for line in strip_ansi(result.stdout).splitlines()]
    assert message in lines
    assert "Config could not be loaded (see above)." in lines
    assert "All checks passed!" not in result.stdout
    assert isinstance(result.exception, SystemExit)
