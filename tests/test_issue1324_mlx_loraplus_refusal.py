"""Issue #1324 — ``training.loraplus_lr_ratio`` is refused on ``backend: mlx``.

On ``backend: mlx`` (``task: sft`` is the only reachable MLX LoRA case), the
config used to load, validate, pass ``soup train --dry-run`` and
``soup doctor --config``, and then train plain LoRA: the MLX optimizer is
built from a single scalar (``plan_optimizer`` in ``trainer/mlx_optim.py``
takes no parameter groups), so A and B ran at one learning rate with no
warning. Mirroring LoRA-FA's backend gate (#725), the field is now refused
at load on every backend except ``transformers`` and ``unsloth`` (unsloth
reaches the transformers wrappers' ``attach_loraplus_optimizer`` call, so it
stays allowed — regression guard for #1080's wiring).

The pre-existing ``training.use_lorafa`` row in the ``_MLX_SFT`` doctor
table stays: the mlx trainer itself warns about it at runtime, and the
#755 drift test requires the table to match that warning list.
"""

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.config.backend_support import _MLX_SFT
from soup_cli.config.loader import load_config_from_string

_MLX_LORAPLUS = """
base: mlx-community/Qwen2.5-0.5B-Instruct-4bit
task: sft
backend: mlx
data:
  train: ./data/train.jsonl
  format: alpaca
training:
  epochs: 1
  lr: 1.0e-4
  loraplus_lr_ratio: 16
  lora:
    r: 8
    alpha: 16
output: ./output
"""


def _transformers_variant(backend: str) -> str:
    return _MLX_LORAPLUS.replace("backend: mlx", f"backend: {backend}").replace(
        "mlx-community/Qwen2.5-0.5B-Instruct-4bit", "Qwen/Qwen2.5-0.5B-Instruct"
    )


def _plain(text: str) -> str:
    """Strip ANSI escapes and collapse whitespace (Rich colours and wraps on CI)."""
    return " ".join(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "").split())


def test_mlx_loraplus_is_refused_at_load_with_message():
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_MLX_LORAPLUS)
    msg = str(excinfo.value)
    assert "training.loraplus_lr_ratio is not implemented on backend='mlx'" in msg
    assert "Remove training.loraplus_lr_ratio, or use backend: transformers." in msg


def test_transformers_loraplus_still_loads():
    cfg = load_config_from_string(_transformers_variant("transformers"))
    assert cfg.training.loraplus_lr_ratio == pytest.approx(16.0)
    assert cfg.backend == "transformers"


def test_unsloth_loraplus_still_loads():
    cfg = load_config_from_string(_transformers_variant("unsloth"))
    assert cfg.training.loraplus_lr_ratio == pytest.approx(16.0)
    assert cfg.backend == "unsloth"


def test_mlx_loraplus_without_ratio_still_loads():
    # The gate is on the ratio field, not on the backend itself.
    cfg = load_config_from_string(
        _MLX_LORAPLUS.replace("  loraplus_lr_ratio: 16\n", "")
    )
    assert cfg.backend == "mlx"


def test_train_dry_run_exits_nonzero_on_mlx_loraplus(tmp_path: Path):
    cfg_file = tmp_path / "c1_mlx_loraplus.yaml"
    cfg_file.write_text(_MLX_LORAPLUS, encoding="utf-8")
    result = CliRunner().invoke(app, ["train", "--config", str(cfg_file), "--dry-run"])
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert "training.loraplus_lr_ratio is not implemented on backend='mlx'" in _plain(
        result.output
    )


@pytest.mark.parametrize(
    "task, data_format",
    [("dpo", "dpo"), ("kto", "kto"), ("pretrain", "plaintext"), ("embedding", "alpaca")],
)
def test_a_task_mlx_cannot_run_is_refused_for_that_first(task, data_format):
    """The MLX task gate runs first: switching backend is the whole fix there."""
    text = _MLX_LORAPLUS.replace("task: sft", f"task: {task}").replace(
        "format: alpaca", f"format: {data_format}"
    )
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(text)
    msg = str(excinfo.value)
    assert "MLX backend only ships SFT" in msg
    assert "is not implemented on backend=" not in msg


def test_a_task_with_no_lora_b_keeps_its_own_refusal_on_mlx():
    """moe_lora_routing refuses LoRA+ on every backend; "use backend: transformers"
    would lead straight into that refusal, so it must come first."""
    text = """
base: org/m
task: moe_lora_routing
backend: mlx
data:
  train: ./x.jsonl
training:
  loraplus_lr_ratio: 16
  mole_task_adapters: [./a, ./b]
"""
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(text)
    msg = str(excinfo.value)
    assert "trains only the routing gate" in msg
    assert "is not implemented on backend=" not in msg


def test_mlx_doctor_table_has_no_loraplus_row():
    # A refused-at-load field must not appear in the doctor table (#958).
    names = {entry.field for entry in _MLX_SFT}
    assert "training.loraplus_lr_ratio" not in names
