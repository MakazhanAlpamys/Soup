"""Resident SFT can leave an untied frozen input embedding on the CPU."""

from __future__ import annotations

import re

import pytest

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.embed_on_host import embedding_on_host
from soup_cli.utils.kbit_prepare import prepare_kbit_model
from tests.conftest import strip_ansi
from tests.test_kbit_prepare_keeps_dtype import _setup_trainer, _tiny_model, torch


def _config(**training):
    return {
        "base": "test-model",
        "data": {"train": "data.jsonl", "format": "alpaca"},
        "training": {"quantization": "4bit", "embed_on_host": True, **training},
    }


@pytest.mark.parametrize("quantization", ["4bit", "8bit"])
def test_config_allows_resident_kbit_sft(quantization):
    assert SoupConfig(**_config(quantization=quantization)).training.embed_on_host


@pytest.mark.parametrize(
    "top, training, message",
    [
        ({"task": "dpo"}, {}, "requires task='sft' and backend='transformers'"),
        ({"backend": "unsloth"}, {}, "requires task='sft' and backend='transformers'"),
        ({"modality": "vision"}, {}, "requires modality='text'"),
        ({}, {"quantization": "none"}, "requires quantization='4bit' or '8bit'"),
        ({}, {"stream_layers": True}, "requires resident SFT with training.stream_layers=false"),
        (
            {},
            {"unfrozen_parameters": ["embed_tokens"]},
            "requires a frozen input embedding. Remove training.unfrozen_parameters",
        ),
        ({}, {"packed_lora": {"enabled": True}}, "cannot be combined with training.packed_lora"),
    ],
)
def test_config_refuses_unsupported_routes(top, training, message):
    with pytest.raises(ValueError, match=re.escape(f"training.embed_on_host {message}")):
        SoupConfig(**{**_config(**training), **top})


def test_sft_wires_the_host_embedding_after_kbit_prep(monkeypatch, capsys):
    model = _setup_trainer(
        monkeypatch,
        task="sft",
        module="soup_cli.trainer.sft",
        cls_name="SFTTrainerWrapper",
        flags=(True, False),
        training={"embed_on_host": True},
    )
    assert "Input embedding stays on the CPU" in strip_ansi(capsys.readouterr().out)
    assert not model.get_input_embeddings().weight.requires_grad
    weight = model.get_input_embeddings().weight
    model.to(torch.float32)
    assert weight.dtype is torch.bfloat16


def test_sft_refuses_a_distributed_launch(monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper

    cfg = SoupConfig(**_config())
    wrapper = object.__new__(SFTTrainerWrapper)
    monkeypatch.setenv("WORLD_SIZE", "2")
    message = "training.embed_on_host requires single-process SFT without DeepSpeed or FSDP"
    with pytest.raises(ValueError, match=re.escape(message)):
        wrapper._setup_transformers(cfg, cfg.training)


def test_rows_logits_hooks_and_module_survive_host_placement():
    model = _tiny_model(torch.bfloat16)
    prepare_kbit_model(model, use_gradient_checkpointing=True, compute_dtype=torch.bfloat16)
    model.eval()
    embedding = model.get_input_embeddings()
    ids = torch.tensor([[1, 2, 3]])
    rows = embedding(ids).detach().clone()
    logits = model(input_ids=ids).logits.detach().clone()
    inputs = []

    def record_input(module, args):
        inputs.append(args[0])

    embedding.register_forward_pre_hook(record_input)
    weight = embedding.weight
    assert embedding_on_host(model) == 0
    assert embedding_on_host(model) == 0
    assert model.get_input_embeddings() is embedding
    assert embedding.weight is weight
    output = embedding(ids)
    assert inputs[0] is ids
    assert output.requires_grad
    assert torch.equal(output, rows)
    assert torch.equal(model(input_ids=ids).logits, logits)

    model.to(torch.float32)
    assert embedding.weight is weight
    assert weight.dtype is torch.bfloat16
    assert weight.device.type == "cpu"
    assert model.get_output_embeddings().weight.dtype is torch.float32


def test_custom_embedding_forward_is_preserved():
    class ScaledEmbedding(torch.nn.Embedding):
        def forward(self, input_ids):
            return super().forward(input_ids) * 2

    model = _tiny_model(torch.bfloat16)
    embedding = ScaledEmbedding(64, 16, dtype=torch.bfloat16)
    embedding.requires_grad_(False)
    model.set_input_embeddings(embedding)
    ids = torch.tensor([[1, 2, 3]])
    expected = embedding(input_ids=ids)
    embedding_on_host(model)
    assert torch.equal(embedding(input_ids=ids), expected)


@pytest.mark.parametrize("alias", [False, True])
def test_host_embedding_refuses_tied_output_weights(alias):
    model = _tiny_model(torch.bfloat16)
    weight = model.get_input_embeddings().weight
    model.get_output_embeddings().weight = torch.nn.Parameter(weight.detach()) if alias else weight
    model.requires_grad_(False)
    message = "training.embed_on_host requires an untied input embedding"
    with pytest.raises(ValueError, match=re.escape(message)):
        embedding_on_host(model)


def test_host_embedding_refuses_trainable_input_weights():
    message = "training.embed_on_host requires a frozen input embedding"
    with pytest.raises(ValueError, match=re.escape(message)):
        embedding_on_host(_tiny_model(torch.bfloat16))


def test_host_embedding_refuses_a_lora_adapter_on_the_input_embedding():
    from peft import LoraConfig, get_peft_model

    model = _tiny_model(torch.bfloat16)
    model = get_peft_model(
        model, LoraConfig(target_modules=["embed_tokens"], task_type="CAUSAL_LM")
    )
    message = "training.embed_on_host requires a frozen input embedding"
    with pytest.raises(ValueError, match=re.escape(message)):
        embedding_on_host(model)


def test_host_embedding_refuses_dispatched_embeddings():
    model = _tiny_model(torch.bfloat16)
    model.requires_grad_(False)
    model.get_input_embeddings()._hf_hook = object()
    message = "training.embed_on_host requires a resident embedding on a single device"
    with pytest.raises(ValueError, match=re.escape(message)):
        embedding_on_host(model)


@pytest.mark.gpu
def test_cuda_host_embedding_remains_pinned_after_model_to():
    model = _tiny_model(torch.bfloat16)
    model.is_loaded_in_4bit = False
    model.cuda().requires_grad_(False)
    ids = torch.tensor([[1, 2, 3]], device="cuda")
    expected = model.get_input_embeddings()(ids)
    assert embedding_on_host(model) == 64 * 16 * 2
    model.to("cuda")

    embedding = model.get_input_embeddings()
    assert embedding.weight.device.type == "cpu"
    assert embedding.weight.is_pinned()
    rows = embedding(ids)
    assert rows.device == ids.device
    assert torch.equal(rows, expected)


@pytest.mark.gpu
def test_resident_qlora_step_with_host_embedding(tmp_path, monkeypatch):
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from tests.test_issue691_sft_packing import _write_tiny_tokenizer

    bnb = pytest.importorskip("bitsandbytes")
    model = _tiny_model(torch.bfloat16)
    model.is_loaded_in_4bit = False
    model.save_pretrained(tmp_path)
    _write_tiny_tokenizer(str(tmp_path))
    cfg = SoupConfig(**{**_config(gradient_checkpointing=True), "base": str(tmp_path)})
    monkeypatch.setattr("soup_cli.trainer.sft.resolve_device_map", lambda device: {"": 0})
    wrapper = SFTTrainerWrapper(cfg, device="cuda")
    wrapper._setup_transformers(cfg, cfg.training)
    model = wrapper.model
    assert any(isinstance(layer, bnb.nn.Linear4bit) for layer in model.modules())

    model.train()
    ids = torch.tensor([[1, 2, 4, 5]], device="cuda")
    dtype = wrapper._kbit_compute_dtype(cfg.training, cfg.base)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    with torch.autocast("cuda", dtype=dtype):
        loss = model(input_ids=ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [param.grad for param in trainable if param.grad is not None]
    assert all(torch.isfinite(grad).all() for grad in gradients)
    assert any(torch.count_nonzero(grad) > 0 for grad in gradients)
    optimizer.step()
    embedding = model.get_input_embeddings()
    assert embedding.weight.is_pinned()
    assert embedding.weight.grad is None
