"""Historical #1324 refusal now follows the bounded native #1975 support.

Keep unsupported combinations refused before loading data/models and preserve
Transformers/Unsloth wiring while the qualified MLX SFT slice loads.
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


def test_mlx_loraplus_now_loads_in_qualified_slice():
    cfg = load_config_from_string(_MLX_LORAPLUS)
    assert cfg.training.loraplus_lr_ratio == 16


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


def test_train_dry_run_accepts_qualified_mlx_loraplus(tmp_path: Path):
    cfg_file = _write_runnable_config(tmp_path)
    result = CliRunner().invoke(app, ["train", "--config", str(cfg_file), "--dry-run"])
    assert result.exit_code == 0, (result.output, repr(result.exception))


@pytest.mark.parametrize(
    "task, data_format",
    [("ipo", "dpo"), ("kto", "kto"), ("pretrain", "plaintext"), ("embedding", "alpaca")],
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


# --------------------------------------------------------------------------
# More paths to the same refusal, and the controls around it
# --------------------------------------------------------------------------

_REFUSAL = "MLX loraplus_lr_ratio cannot be combined with DoRA"


@pytest.mark.parametrize("ratio", ["16", "1.0", "0.5", "2"])
def test_supported_ratios_load_on_mlx(ratio):
    text = _MLX_LORAPLUS.replace("loraplus_lr_ratio: 16", f"loraplus_lr_ratio: {ratio}")
    assert load_config_from_string(text).training.loraplus_lr_ratio == float(ratio)


def test_direct_construction_enforces_new_boundary():
    from soup_cli.config.schema import SoupConfig

    with pytest.raises(ValueError, match="cannot be combined with DoRA"):
        SoupConfig(base="owner/base", backend="mlx", data={"train": "t.jsonl"},
                   training={"loraplus_lr_ratio": 16, "lora": {"use_dora": True}})


def test_an_explicit_null_ratio_still_loads_on_mlx():
    cfg = load_config_from_string(
        _MLX_LORAPLUS.replace("loraplus_lr_ratio: 16", "loraplus_lr_ratio: null")
    )
    assert cfg.backend == "mlx"
    assert cfg.training.loraplus_lr_ratio is None


@pytest.mark.parametrize("backend", ["transformers", "unsloth"])
@pytest.mark.parametrize("task, data_format", [("pretrain", "plaintext"), ("dpo", "dpo")])
def test_the_wired_backends_keep_the_ratio_beyond_sft(backend, task, data_format):
    """#1080 wired LoRA+ after the backend branch of these trainers too, so the
    refusal must stay on mlx and not spread to the other backends' tasks."""
    text = (
        _transformers_variant(backend)
        .replace("task: sft", f"task: {task}")
        .replace("format: alpaca", f"format: {data_format}")
    )
    cfg = load_config_from_string(text)
    assert cfg.backend == backend
    assert cfg.task == task
    assert cfg.training.loraplus_lr_ratio == pytest.approx(16.0)


def _write_runnable_config(tmp_path: Path) -> Path:
    train = tmp_path / "train.jsonl"
    train.write_text('{"instruction": "q", "output": "a"}\n' * 4, encoding="utf-8")
    path = tmp_path / "soup.yaml"
    path.write_text(
        _MLX_LORAPLUS.replace("./data/train.jsonl", train.as_posix()).replace(
            "./output", (tmp_path / "out").as_posix()
        ),
        encoding="utf-8",
    )
    return path


def test_plain_soup_train_stops_at_config_load(tmp_path, monkeypatch):
    """Not only ``--dry-run``: a real ``soup train`` must stop before the data or
    the model is touched."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.chdir(tmp_path)
    config = _write_runnable_config(tmp_path)
    config.write_text(config.read_text().replace("    r: 8", "    use_dora: true\n    r: 8"))

    result = CliRunner().invoke(app, ["train", "--config", str(config)])

    output = _plain(result.output)
    assert result.exit_code == 1, (result.output, repr(result.exception))
    assert _REFUSAL in output, output
    for later_line in ("Config valid.", "Ready to train", "Data OK", "Training Setup"):
        assert later_line not in output, (later_line, output)


def test_soup_doctor_config_reports_the_config_as_unloadable(tmp_path, capsys):
    """``soup doctor --config`` printed its all-clear for this config before
    #1324; an unloadable config is exit 2."""
    import typer

    from soup_cli.commands.doctor import doctor

    config = _write_runnable_config(tmp_path)
    config.write_text(config.read_text().replace("    r: 8", "    use_dora: true\n    r: 8"))

    with pytest.raises(typer.Exit) as excinfo:
        doctor(nccl=False, disk=False, config=str(config))

    assert excinfo.value.exit_code == 2
    assert _REFUSAL in _plain(capsys.readouterr().out)


def test_the_loraplus_docs_state_the_mlx_refusal_and_the_unsloth_wiring():
    root = Path(__file__).resolve().parents[1]
    text = (root / "docs" / "peft-and-efficiency.md").read_text(encoding="utf-8")
    section = text.split("## LoRA+ (Differentiated Learning Rates)", 1)[1]
    flat = " ".join(section.split("\n## ", 1)[0].split())
    assert "Native MLX text SFT" in flat, flat
    assert "`backend: unsloth` gets the same LoRA+ optimizer as `transformers`" in flat, flat
