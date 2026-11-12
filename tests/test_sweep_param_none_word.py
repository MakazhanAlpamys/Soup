"""`soup sweep --param quantization=none` sweeps `quantization: none`.

`--param` reads `none` as null, and `training.quantization` refuses null: `none` is one
of its values. So the `none` arm of `quantization=4bit,none` failed schema validation
every time ("Input should be '4bit', '8bit', 'none', ..."), while the same sweep from
`--sweep-config`, where YAML keeps `none` a string, trained. A field that takes null
still gets null.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.commands.sweep import _parse_sweep_params
from tests.conftest import strip_ansi

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


@pytest.mark.parametrize(
    ("param", "expected"),
    [
        ("quantization=4bit,none", {"quantization": ["4bit", "none"]}),
        ("training.quantization=None", {"training.quantization": ["none"]}),
        # Optional fields keep null: it is how NEFTune and LoRA+ are switched off.
        ("neftune_alpha=none,5", {"neftune_alpha": [None, 5]}),
        ("loraplus_lr_ratio=none,16", {"loraplus_lr_ratio": [None, 16]}),
        # A key that names no field is left for the unknown-key refusal to name.
        ("no_such_field=none", {"no_such_field": [None]}),
    ],
)
def test_none_is_the_word_only_where_the_field_refuses_null(param, expected):
    assert _parse_sweep_params([param]) == expected


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
    monkeypatch.setattr("soup_cli.utils.gpu.detect_device", lambda *a, **kw: ("cuda", "cuda"))
    return seen


def test_the_none_arm_of_a_quantization_sweep_trains_unquantized(tmp_path, quant_seen):
    config = tmp_path / "soup.yaml"
    config.write_text(_CONFIG, encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["sweep", "--config", str(config), "--yes", "--param", "quantization=4bit,none"],
    )

    assert result.exit_code == 0, (result.output, repr(result.exception))
    assert quant_seen == ["4bit", "none"], quant_seen
    plain = " ".join(strip_ansi(result.output).split())
    assert "failed" not in plain, plain
    assert "quantization = none" in plain, plain
