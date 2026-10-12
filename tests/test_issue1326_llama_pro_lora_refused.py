"""Issue #1326: LLaMA Pro under LoRA does not train the appended blocks.

``apply_block_expansion_if_configured`` freezes everything but the appended
blocks, then ``get_peft_model`` freezes every non-adapter parameter again,
appended blocks included, and puts adapters on the original layers. The ruling
(route 1) refuses ``expand_layers`` together with a LoRA rank at config load and
points to ``lora.r: 0``, which ``task: pretrain`` does not have.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from soup_cli.config.loader import load_config_from_string

REPO_ROOT = Path(__file__).resolve().parent.parent
_DOCS = ("docs/peft-and-efficiency.md", "docs/training.md")
_FENCE = re.compile(r"```ya?ml\n(.*?)```", re.DOTALL)
_HEAD = "base: org/model\ntask: sft\ndata:\n  train: ./data.jsonl\n"


def _config(task: str, lora: str = "") -> str:
    return (
        "base: org/model\n"
        f"task: {task}\n"
        "data:\n  train: ./data.jsonl\n"
        "training:\n  quantization: none\n  expand_layers: 4\n  freeze_trainable_layers: 4\n"
        + lora
    )


@pytest.mark.parametrize("task", ["sft", "pretrain"])
@pytest.mark.parametrize(
    ("lora", "rank"),
    [("", 64), ("  lora:\n    r: 8\n", 8)],
    ids=["default_rank", "explicit_rank"],
)
def test_a_lora_rank_with_llama_pro_is_refused(task, lora, rank):
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_config(task, lora))
    message = " ".join(str(excinfo.value).split())
    assert "training.expand_layers: 4" in message
    assert "training.freeze_trainable_layers: 4" in message
    assert f"training.lora.r: {rank}" in message
    assert "freezes the appended blocks again" in message
    assert "lora.r: 0" in message
    assert "quantization: none" in message
    if task == "pretrain":
        assert "needs task: sft" in message
    else:
        assert "needs task: sft" not in message


@pytest.mark.parametrize(
    ("mode", "named"),
    [
        ("  stream_layers: true\n  batch_size: 1\n", "stream_layers"),
        ("  unfrozen_parameters:\n    - 'model.layers.0.*'\n", "unfrozen_parameters"),
        ("  lisa_enabled: true\n", "lisa_enabled"),
        ("  quantization_aware: quest\n", "quantization_aware"),
    ],
    ids=["stream_layers", "spectrum", "lisa", "quest"],
)
def test_modes_that_refuse_expand_layers_outright_keep_their_own_message(mode, named):
    # lora.r: 0 would not make these configs load, so the #1326 message must not win.
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(_config("sft", mode))
    message = " ".join(str(excinfo.value).split())
    assert named in message
    assert "freezes the appended blocks again" not in message


@pytest.mark.parametrize("freeze", [None, 0, 2], ids=["missing", "zero", "smaller"])
def test_a_wrong_freeze_value_keeps_the_1409_message(freeze):
    # The #1409 check runs first, so the #1326 message only ever shows matching values.
    config = _config("sft").replace("  freeze_trainable_layers: 4\n", "")
    if freeze is not None:
        config += f"  freeze_trainable_layers: {freeze}\n"
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(config)
    message = " ".join(str(excinfo.value).split())
    assert "Set freeze_trainable_layers: 4." in message
    assert "freezes the appended blocks again" not in message


def test_full_fine_tuning_of_the_new_blocks_loads_on_sft():
    cfg = load_config_from_string(_config("sft", "  lora:\n    r: 0\n"))
    assert cfg.training.expand_layers == 4
    assert cfg.training.freeze_trainable_layers == 4
    assert cfg.training.lora.r == 0


def test_lora_without_llama_pro_still_loads():
    cfg = load_config_from_string(_HEAD + "training:\n  quantization: none\n")
    assert cfg.training.expand_layers is None
    assert cfg.training.lora.r == 64


def _llama_pro_examples() -> list[tuple[str, str]]:
    examples = []
    for rel in _DOCS:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for block in _FENCE.findall(text):
            if re.search(r"^\s*expand_layers:", block, re.MULTILINE):
                examples.append((rel, block))
    return examples


def test_every_llama_pro_example_in_the_docs_loads():
    examples = _llama_pro_examples()
    assert {rel for rel, _ in examples} == set(_DOCS), examples
    for rel, block in examples:
        text = block if re.search(r"^base:", block, re.MULTILINE) else _HEAD + block
        cfg = load_config_from_string(text)
        assert cfg.training.expand_layers is not None, rel
        assert cfg.training.lora.r == 0, rel


def test_the_training_docs_put_the_expansion_before_get_peft_model():
    flat = " ".join((REPO_ROOT / "docs" / "training.md").read_text(encoding="utf-8").split())
    assert "`use_mod` / `expand_layers` attach AFTER" not in flat
    assert "`expand_layers` runs BEFORE it" in flat
    assert "#1326" in flat
