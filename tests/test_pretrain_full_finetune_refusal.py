"""Issue #1813: ``task: pretrain`` with ``lora.r: 0`` is refused at config load.

``PretrainTrainerWrapper`` has no ``r == 0`` branch: the transformers path builds
a LoRA config and calls ``get_peft_model`` (peft: "`r` should be a positive
integer value"), and the unsloth path passes the rank to unsloth's
``get_peft_model``. Route (a) of the ruling refuses the config at load instead.
LISA is the exception: its pretrain path attaches no adapter and never reads the
rank, so ``lisa_enabled`` with ``lora.r: 0`` trains today and keeps loading.

The same change drops ``task: pretrain`` from the ``expand_layers`` scope: with a
rank the appended blocks are frozen again (#1326) and with ``lora.r: 0`` setup
fails, so no pretrain LLaMA Pro config trains.
"""

from __future__ import annotations

import pytest
import yaml

from soup_cli.config.loader import load_config_from_string

_RANK_ZERO_MESSAGE = "always attaches a LoRA adapter"
_EXPAND = {"expand_layers": 2, "freeze_trainable_layers": 2}


def _yaml(task="pretrain", backend="transformers", training=None, lora=None, modality=None):
    block = {"quantization": "none", **(training or {})}
    if lora is not None:
        block["lora"] = lora
    top = {
        "base": "org/model",
        "task": task,
        "backend": backend,
        "data": {"train": "train.jsonl"},
        "training": block,
    }
    if modality is not None:
        top["modality"] = modality
    return yaml.safe_dump(top)


def _refusal(text: str) -> str:
    with pytest.raises(ValueError) as excinfo:
        load_config_from_string(text)
    return " ".join(str(excinfo.value).split())


@pytest.mark.parametrize(
    ("backend", "training"),
    [("transformers", {}), ("transformers", {"quantization": "4bit"}), ("unsloth", {})],
    ids=["transformers", "transformers_4bit", "unsloth"],
)
def test_rank_zero_on_pretrain_is_refused_at_load(backend, training):
    # The same config with a rank loads, so the refusal can only come from lora.r: 0.
    load_config_from_string(_yaml(backend=backend, training=training, lora={"r": 4}))
    message = _refusal(_yaml(backend=backend, training=training, lora={"r": 0}))
    assert "training.lora.r=0" in message
    assert "task='pretrain'" in message
    assert _RANK_ZERO_MESSAGE in message
    assert "lora.r >= 1" in message
    assert "lisa_enabled: true" in message
    assert "task: sft with lora.r: 0" in message


def test_lisa_with_rank_zero_still_loads_on_pretrain():
    cfg = load_config_from_string(_yaml(training={"lisa_enabled": True}, lora={"r": 0}))
    assert cfg.training.lisa_enabled is True
    assert cfg.training.lora.r == 0


@pytest.mark.parametrize(
    ("over", "named"),
    [
        ({"backend": "mlx"}, "MLX backend only ships SFT"),
        (
            {"training": {"stream_layers": True, "batch_size": 1}},
            "training.stream_layers supports task",
        ),
    ],
    ids=["mlx", "stream_layers"],
)
def test_routes_that_refuse_pretrain_outright_keep_their_own_message(over, named):
    message = _refusal(_yaml(lora={"r": 0}, **over))
    assert named in message
    assert _RANK_ZERO_MESSAGE not in message


@pytest.mark.parametrize("task", ["sft", "embedding"])
def test_rank_zero_still_loads_where_it_means_full_fine_tuning(task):
    cfg = load_config_from_string(_yaml(task=task, lora={"r": 0}))
    assert cfg.training.lora.r == 0


def test_pretrain_with_a_rank_still_loads():
    cfg = load_config_from_string(_yaml(lora={"r": 4}))
    assert cfg.training.lora.r == 4


@pytest.mark.parametrize(
    ("training", "lora"),
    [({}, {"r": 0}), ({}, {"r": 8}), ({}, None), ({"lisa_enabled": True}, {"r": 0})],
    ids=["rank_zero", "explicit_rank", "default_rank", "lisa"],
)
def test_llama_pro_on_pretrain_gets_the_scope_refusal(training, lora):
    message = _refusal(_yaml(training={**_EXPAND, **training}, lora=lora))
    assert "training.expand_layers (LLaMA Pro)" in message
    assert "task='pretrain'" in message
    assert "no lora.r: 0 branch" in message
    assert "task: sft with lora.r: 0 and quantization: none" in message
    # One message per config: neither the #1326 refusal nor the rank-zero one.
    assert "cannot be combined with LoRA" not in message
    assert _RANK_ZERO_MESSAGE not in message


def test_llama_pro_scope_refusal_on_other_tasks_does_not_mention_pretrain():
    message = _refusal(_yaml(task="dpo", training=_EXPAND, lora={"r": 0}))
    assert "task='dpo'" in message
    assert "Remove expand_layers and freeze_trainable_layers." in message
    assert "pretrain" not in message


@pytest.mark.parametrize(
    ("backend", "modality"),
    [("unsloth", "text"), ("mlx", "text"), ("transformers", "vision")],
    ids=["unsloth", "mlx", "vision"],
)
def test_llama_pro_on_pretrain_elsewhere_gets_the_generic_scope_refusal(backend, modality):
    """Only pretrain on transformers text runs the expansion, so only there does the message
    describe it and name the sft recipe: the unsloth pretrain path never expands, and mlx or
    another modality has no such recipe."""
    message = _refusal(_yaml(backend=backend, training=_EXPAND, lora={"r": 0}, modality=modality))
    assert "training.expand_layers (LLaMA Pro)" in message
    assert f"backend='{backend}'" in message and f"modality='{modality}'" in message
    assert "Remove expand_layers and freeze_trainable_layers." in message
    assert "pretrain trainer runs the expansion" not in message
    assert "task: sft with lora.r: 0" not in message


def test_lisa_with_rank_zero_trains_without_an_adapter(tmp_path, monkeypatch):
    """Through the real ``PretrainTrainerWrapper.setup()`` and one epoch on the tiny
    Llama (#341 fixture): the LISA path never reaches peft, so the rank is unused."""
    from tests.test_issue341_seed_and_fullft import _requires_train_extra, _tiny_llama_dir

    _requires_train_extra()
    import torch

    from soup_cli.trainer.pretrain import PretrainTrainerWrapper

    base = _tiny_llama_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": base,
                "task": "pretrain",
                "data": {"train": "train.jsonl", "max_length": 64},
                "training": {
                    "quantization": "none",
                    "epochs": 1,
                    "batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "lr": 1e-3,
                    "warmup_ratio": 0.0,
                    "logging_steps": 100,
                    "save_steps": 10_000,
                    "lisa_enabled": True,
                    "lora": {"r": 0},
                },
                "output": str(tmp_path / "out"),
            }
        )
    )
    wrapper = PretrainTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": [{"text": "the cat sat on the mat"}] * 8})

    assert not hasattr(wrapper.model, "peft_config")
    before = {
        name: param.detach().clone()
        for name, param in wrapper.model.named_parameters()
        if param.requires_grad
    }
    assert before
    wrapper.trainer.train()
    changed = [
        name
        for name, param in wrapper.model.named_parameters()
        if name in before and not torch.equal(param.detach(), before[name])
    ]
    assert changed
