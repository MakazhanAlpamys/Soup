"""Issue #1409: with ``expand_layers`` set, ``freeze_trainable_layers`` must equal it.

The value was read only for its sign: 1, 2 and 4 all trained just the appended
blocks, and 0 or a negative value silently trained the whole expanded model.
The ruling on #1409 gives the field one meaning (freeze the original model,
train only the appended blocks) and refuses every value other than
``expand_layers`` at config load.
"""

from __future__ import annotations

import pytest

from soup_cli.config.loader import load_config_from_string


def _load(task, expand, freeze):
    lines = ["  quantization: none\n", f"  expand_layers: {expand}\n"]
    if freeze is not None:
        lines.append(f"  freeze_trainable_layers: {freeze}\n")
    return load_config_from_string(
        "base: org/model\n"
        f"task: {task}\n"
        "data:\n  train: ./data.jsonl\n"
        "training:\n" + "".join(lines)
    )


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_freeze_equal_to_expand_loads(task):
    cfg = _load(task, expand=4, freeze=4)
    assert cfg.training.expand_layers == 4
    assert cfg.training.freeze_trainable_layers == 4


@pytest.mark.parametrize("task", ["sft", "pretrain"])
@pytest.mark.parametrize(
    "freeze", [0, -1, -4, 2, 8], ids=["zero", "minus_one", "minus_n", "smaller", "larger"]
)
def test_any_other_value_is_refused(task, freeze):
    with pytest.raises(ValueError) as excinfo:
        _load(task, expand=4, freeze=freeze)
    message = str(excinfo.value)
    assert f"freeze_trainable_layers: {freeze}" in message
    assert "expand_layers: 4" in message
    assert "trains only the appended blocks" in message
    assert "top-N or bottom-N" in message
    assert "Set freeze_trainable_layers: 4." in message


@pytest.mark.parametrize("task", ["sft", "pretrain"])
def test_expand_without_freeze_names_the_expand_value(task):
    with pytest.raises(ValueError, match=r"Set freeze_trainable_layers: 3\."):
        _load(task, expand=3, freeze=None)


def test_freeze_without_expand_keeps_the_1399_message():
    with pytest.raises(
        ValueError,
        match="only applies together with expand_layers.*must equal expand_layers.*Remove it",
    ):
        load_config_from_string(
            "base: org/model\n"
            "task: sft\n"
            "data:\n  train: ./data.jsonl\n"
            "training:\n  quantization: none\n  freeze_trainable_layers: 2\n"
        )


def test_setup_trains_exactly_the_appended_blocks(tmp_path, monkeypatch):
    """Through the real ``SFTTrainerWrapper.setup()`` on a tiny Llama (#341 fixture)."""
    from tests.test_issue341_seed_and_fullft import _requires_train_extra, _wrapper

    _requires_train_extra()
    wrapper, dataset = _wrapper(
        tmp_path,
        monkeypatch,
        expand_layers=2,
        freeze_trainable_layers=2,
        lora={"r": 0},
    )
    wrapper.setup(dataset)

    model = wrapper.trainer.model
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    appended = {
        name
        for name, _ in model.named_parameters()
        if name.startswith(("model.layers.2.", "model.layers.3."))
    }
    assert len(model.model.layers) == 4
    assert appended
    assert trainable == appended
