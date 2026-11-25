"""``soup train --gate`` refuses an unsupported gate with the lines of the config file (#3088, B-5).

An enabled ``training.eval_gate`` cannot run for ``backend: mlx`` or for ``task: prm``,
``unlearn`` and ``moe_lora_routing``. Set in ``soup.yaml``, the refusal is printed under the
loader's ``Config validation error:`` header, indented. Through ``--gate <suite>`` it was one
red line with no header. Both now print the same lines through one function of the loader,
and both exit 1 before hardware, data or a model is touched.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.cli import app
from tests.conftest import make_cli_runner, strip_ansi

_SUITE = {
    "suite": "regression",
    "tasks": [
        {"type": "custom", "name": "exact", "tasks": "eval.jsonl", "scorer": "exact",
         "threshold": 0.8},
    ],
}


def _config(task: str, backend: str, *, gate_in_file: bool) -> dict:
    raw = {
        "base": "sshleifer/tiny-gpt2",
        "task": task,
        "backend": backend,
        "data": {"train": "train.jsonl", "format": "alpaca"},
        "training": {},
        "output": "out",
    }
    if task == "prm":
        raw["data"]["format"] = "prm"
    elif task == "unlearn":
        raw["data"]["forget_set"] = "forget.jsonl"
        raw["training"]["unlearn_method"] = "npo"
    elif task == "moe_lora_routing":
        raw["training"]["mole_task_adapters"] = ["adapter_a", "adapter_b"]
    if gate_in_file:
        raw["training"]["eval_gate"] = {"enabled": True, "suite": "suite.yaml"}
    return raw


class _HardwareReachedError(RuntimeError):
    pass


@pytest.fixture
def train(tmp_path, monkeypatch):
    import soup_cli.commands.train as module

    def hardware(*_args, **_kwargs):
        raise _HardwareReachedError("the refusal must come before hardware detection")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "detect_device", hardware)
    (tmp_path / "suite.yaml").write_text(yaml.safe_dump(_SUITE), encoding="utf-8")

    def invoke(raw: dict, *flags: str):
        (tmp_path / "soup.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")
        return make_cli_runner().invoke(
            app, ["train", "--config", "soup.yaml", "--dry-run", "--no-reexec", *flags]
        )

    return invoke


def _collapsed(output: str) -> str:
    # Rich may fold a line anywhere, so each output is read with its whitespace collapsed.
    return " ".join(strip_ansi(output).split())


UNSUPPORTED = [
    pytest.param("sft", "mlx", id="mlx"),
    pytest.param("prm", "transformers", id="prm"),
    pytest.param("unlearn", "transformers", id="unlearn"),
    pytest.param("moe_lora_routing", "transformers", id="moe_lora_routing"),
]


@pytest.mark.parametrize(("task", "backend"), UNSUPPORTED)
def test_the_gate_flag_prints_what_the_config_file_prints(train, task, backend):
    from_file = train(_config(task, backend, gate_in_file=True))
    from_flag = train(_config(task, backend, gate_in_file=False), "--gate", "suite.yaml")

    assert from_file.exit_code == 1, strip_ansi(from_file.output)
    assert from_flag.exit_code == 1, strip_ansi(from_flag.output)
    assert not isinstance(from_flag.exception, _HardwareReachedError)
    assert "Config validation error:" in _collapsed(from_file.output)
    assert _collapsed(from_flag.output) == _collapsed(from_file.output)


@pytest.mark.parametrize(("task", "backend"), UNSUPPORTED)
def test_the_gate_flag_refusal_is_indented_under_the_header(train, task, backend):
    result = train(_config(task, backend, gate_in_file=False), "--gate", "suite.yaml")

    lines = strip_ansi(result.output).splitlines()
    header = lines.index("Config validation error:")
    assert lines[header + 1] == ""
    assert lines[header + 2].startswith("  training.eval_gate is not supported for ")
    assert result.exit_code == 1
