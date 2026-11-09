"""`soup sweep` resolves quantization for the detected device, as `soup train` does.

`soup train` downgrades ``4bit`` / ``8bit`` to ``none`` on a CPU, where bitsandbytes cannot
run, and prints the warning. A sweep arm used to skip that step and hand the configured
value straight to the trainer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from tests.conftest import strip_ansi

CPU_WARNING = "Warning: 4bit quantization is not supported on CPU. Switching to quantization: none."

_CONFIG = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: auto
training:
  quantization: 4bit
output: ./o
"""


def _plain(result) -> str:
    return " ".join(strip_ansi(result.output).split())


@pytest.fixture
def quant_seen(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run real `_run_single` arms against a stub trainer; record each arm's quantization."""
    seen: list[str] = []

    def build_trainer(cfg, device=None):
        seen.append(cfg.training.quantization)
        trainer = Mock()
        trainer.train.return_value = {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "total_steps": 1,
            "duration_secs": 0.1,
            "output_dir": "o",
        }
        return trainer

    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", build_trainer)
    monkeypatch.setattr("soup_cli.data.loader.load_dataset", lambda *a, **kw: {"train": [{}]})
    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", lambda *a, **kw: {})
    return seen


def _use_device(monkeypatch: pytest.MonkeyPatch, device: str) -> None:
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda *a, **kw: (device, device))


def _sweep(tmp_path: Path, *params: str):
    config = tmp_path / "soup.yaml"
    config.write_text(_CONFIG, encoding="utf-8")
    args = ["sweep", "--config", str(config), "--yes"]
    for param in params:
        args += ["--param", param]
    return CliRunner().invoke(app, args)


def test_cpu_sweep_runs_each_arm_unquantized_and_warns_per_arm(tmp_path, monkeypatch, quant_seen):
    _use_device(monkeypatch, "cpu")

    result = _sweep(tmp_path, "lr=1e-5,2e-5")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert quant_seen == ["none", "none"], quant_seen
    plain = _plain(result)
    assert plain.count(CPU_WARNING) == 2, plain
    # The warning belongs to the arm that was downgraded: it follows that arm's header.
    first, second = plain.index("Run 1/2"), plain.index("Run 2/2")
    assert first < plain.index(CPU_WARNING) < second < plain.rindex(CPU_WARNING), plain


def test_cuda_sweep_keeps_the_configured_quantization_and_stays_quiet(
    tmp_path, monkeypatch, quant_seen
):
    _use_device(monkeypatch, "cuda")

    result = _sweep(tmp_path, "lr=1e-5,2e-5")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert quant_seen == ["4bit", "4bit"], quant_seen
    assert "quantization is not supported on CPU" not in _plain(result)


def test_cpu_sweep_over_quantization_names_each_arms_own_value(tmp_path, monkeypatch, quant_seen):
    """`training.quantization` can itself be swept; each arm's warning names what it asked for."""
    _use_device(monkeypatch, "cpu")

    result = _sweep(tmp_path, "quantization=4bit,8bit")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert quant_seen == ["none", "none"], quant_seen
    plain = _plain(result)
    assert plain.count(CPU_WARNING) == 1, plain
    assert plain.count(CPU_WARNING.replace("4bit", "8bit")) == 1, plain
    assert plain.index(CPU_WARNING) < plain.index(CPU_WARNING.replace("4bit", "8bit")), plain


def test_cpu_sweep_records_the_resolved_quantization_on_the_run(
    tmp_path, monkeypatch, quant_seen
):
    """The tracker stores what trained, as `soup train` does (it resolves before start_run)."""
    import json

    from soup_cli.experiment.tracker import ExperimentTracker

    _use_device(monkeypatch, "cpu")

    result = _sweep(tmp_path, "lr=1e-5")

    assert result.exit_code == 0, (result.output, repr(result.exception))
    runs = ExperimentTracker().list_runs()
    assert len(runs) == 1, runs
    config = json.loads(runs[0]["config_json"])
    assert config["training"]["quantization"] == "none", config["training"]


def test_sweep_and_train_share_one_guard() -> None:
    """One helper owns the decision and the printed text; neither command spells it out."""
    import inspect

    from soup_cli.commands import sweep, train

    assert "apply_quantization_guard" in inspect.getsource(sweep._run_single)
    assert "apply_quantization_guard" in inspect.getsource(train)
    assert "resolve_quantization(" not in inspect.getsource(sweep)
    assert "resolve_quantization(" not in inspect.getsource(train)
