"""#3393 — MLX sweep arms overwrite each other's artifacts.

The MLX trainers and unlearning write directly to the configured output
(`training_output_dir` returns it flat), unlike the other backends, which
nest under `output/<experiment_name>`. A sweep sets each arm's
`experiment_name`, so on those other backends every arm already trained into
its own directory — but a `backend: mlx` sweep wrote every arm's
`adapters.safetensors` and `soup_run.json` into the one shared output, and a
later arm replaced the earlier arm's artifact and run metadata.

These tests pin the sweep-side fix: `_run_single` routes the flat backends
through `output/<run_name>`, so each arm's config carries its own output
directory and the tracker records a distinct `output_dir` per arm. The
transformers path is unchanged — its trainer already nests, and the sweep
must not double-nest it.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.sweep import _run_single
from tests.conftest import strip_ansi

_MLX_CONFIG = """\
base: mlx-community/Llama-3.1-8B-Instruct-4bit
task: sft
backend: mlx
data:
  train: ./t.jsonl
  format: alpaca
training:
  epochs: 1
output: shared-output
"""

_TRANSFORMERS_CONFIG = """\
base: hf/model
task: sft
data:
  train: ./t.jsonl
  format: alpaca
training:
  epochs: 1
output: shared-output
"""


@pytest.fixture
def arm_outputs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run real `_run_single` arms against a stub trainer; record each arm's output."""
    seen: list[str] = []

    def build_trainer(cfg, device=None):
        seen.append(cfg.output)
        trainer = Mock()
        trainer.train.return_value = {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "total_steps": 1,
            "duration_secs": 0.1,
            "output_dir": cfg.output,
        }
        return trainer

    monkeypatch.setattr("soup_cli.trainer.dispatch.build_trainer", build_trainer)
    monkeypatch.setattr(
        "soup_cli.data.loader.load_dataset", lambda *a, **kw: {"train": [{}]}
    )
    monkeypatch.setattr("soup_cli.utils.gpu.get_gpu_info", lambda *a, **kw: {})
    monkeypatch.setattr(
        "soup_cli.utils.gpu.detect_device", lambda *a, **kw: ("mps", "mps")
    )
    monkeypatch.setattr(
        "soup_cli.utils.run_provenance.write_run_provenance", lambda *a, **kw: None
    )
    return seen


def _write_config(config_text: str, tmp_path: Path) -> Path:
    config = tmp_path / "soup.yaml"
    config.write_text(config_text, encoding="utf-8")
    return config


def test_run_single_gives_each_flat_backend_arm_its_own_output(
    tmp_path, monkeypatch, arm_outputs
):
    from soup_cli.config.loader import load_config

    monkeypatch.chdir(tmp_path)
    config = _write_config(_MLX_CONFIG, tmp_path)
    cfg = load_config(config)

    for idx, lr in enumerate((0.04, 0.01), start=1):
        _run_single(cfg, {"training.lr": lr}, f"sweep_{idx}", config)

    assert arm_outputs == [
        str(Path("shared-output") / "sweep_1"),
        str(Path("shared-output") / "sweep_2"),
    ], arm_outputs


def test_run_single_leaves_the_nesting_backends_alone(
    tmp_path, monkeypatch, arm_outputs
):
    """The transformers trainer already nests under experiment_name; the sweep
    must hand it the configured output untouched, or arms would land in
    output/run_name/run_name."""
    from soup_cli.config.loader import load_config

    monkeypatch.chdir(tmp_path)
    config = _write_config(_TRANSFORMERS_CONFIG, tmp_path)
    cfg = load_config(config)

    _run_single(cfg, {"training.lr": 0.04}, "sweep_1", config)

    assert arm_outputs == ["shared-output"], arm_outputs


def test_a_mlx_sweep_records_a_distinct_output_dir_per_arm(
    tmp_path, monkeypatch, arm_outputs
):
    """End to end through the CLI: two MLX arms, one shared --output, and the
    trainer sees two different output directories — the exact overwrite of
    #3393 happens the moment they are the same string."""
    monkeypatch.setenv("COLUMNS", "500")
    monkeypatch.chdir(tmp_path)
    config = _write_config(_MLX_CONFIG, tmp_path)

    result = CliRunner().invoke(
        app,
        ["sweep", "--config", str(config), "--yes", "--param", "training.lr=0.04,0.01"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert len(arm_outputs) == 2, arm_outputs
    assert arm_outputs[0] != arm_outputs[1], arm_outputs
    plain = " ".join(strip_ansi(result.output).split())
    assert "failed" not in plain, plain
