"""#796: SFT gradient-checkpointing tiers must produce different execution."""

from __future__ import annotations

import json
import math

import pytest


def _tiny_llama():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return transformers.LlamaForCausalLM(config)


def _write_tiny_sft_assets(directory) -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers

    model_dir = directory / "model"
    model_dir.mkdir()
    _tiny_llama().save_pretrained(model_dir)

    vocab = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "user": 4,
        "assistant": 5,
        "hello": 6,
        "world": 7,
    }
    tokenizer = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "pad_token": "<pad>",
                "model_max_length": 64,
                "clean_up_tokenization_spaces": False,
            }
        ),
        encoding="utf-8",
    )
    return str(model_dir)


def _setup_real_sft(
    tmp_path,
    monkeypatch,
    *,
    tier: str,
    quantization: str = "none",
    memory_gb: int = 40,
):
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    model_dir = _write_tiny_sft_assets(tmp_path)
    cfg = SoupConfig(
        base=model_dir,
        task="sft",
        data={
            "train": "train.jsonl",
            "max_length": 64,
            "chat_template": "chatml",
        },
        training={
            "batch_size": 1,
            "epochs": 1,
            "logging_steps": 1,
            "save_steps": 1000,
            "quantization": quantization,
            "gradient_checkpointing": tier,
            "lora": {
                "r": 4,
                "alpha": 8,
                "target_modules": ["q_proj", "v_proj"],
            },
        },
        output=str(tmp_path / "out"),
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "soup_cli.utils.gpu.get_gpu_info",
        lambda: {
            "memory_total_bytes": memory_gb * 1024**3,
            "name": "test-cpu",
        },
    )

    if quantization == "4bit":
        model = _tiny_llama()
        model.is_loaded_in_4bit = True
        monkeypatch.setattr(
            "soup_cli.utils.quant_menu.build_quantization_config_for_loader",
            lambda **_kwargs: None,
        )
        monkeypatch.setattr(
            "transformers.AutoModelForCausalLM.from_pretrained",
            lambda *_args, **_kwargs: model,
        )

    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    row = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    }
    wrapper.setup({"train": [row, row]})
    return wrapper


def test_medium_uses_transformers_every_n_layers_and_skips_half_the_blocks() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "medium", gpu_memory_gb=40)

    assert plan.granularity == "medium"
    assert plan.kwargs == {
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
            "every_n_layers": 2,
        },
    }

    model.gradient_checkpointing_enable(
        every_n_layers=plan.kwargs["gradient_checkpointing_kwargs"]["every_n_layers"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    active = [layer.gradient_checkpointing for layer in model.model.layers]
    assert active == [True, False, True, False]


def test_full_and_medium_are_observably_different_on_a_tiny_model() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    full_model = _tiny_llama()
    medium_model = _tiny_llama()
    full = plan_gradient_checkpointing(full_model, "full")
    medium = plan_gradient_checkpointing(medium_model, "medium")

    full_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=full.kwargs["gradient_checkpointing_kwargs"],
    )
    medium_kwargs = dict(medium.kwargs["gradient_checkpointing_kwargs"])
    every_n_layers = medium_kwargs.pop("every_n_layers")
    medium_model.gradient_checkpointing_enable(
        every_n_layers=every_n_layers,
        gradient_checkpointing_kwargs=medium_kwargs,
    )

    assert sum(layer.gradient_checkpointing for layer in full_model.model.layers) == 4
    assert sum(layer.gradient_checkpointing for layer in medium_model.model.layers) == 2


def test_selective_wraps_one_attention_module_per_block_without_hf_checkpointing() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "selective")

    assert plan.granularity == "selective"
    assert plan.kwargs == {"gradient_checkpointing": False}
    assert plan.hooked_modules == 4
    for layer in model.model.layers:
        assert layer.self_attn.forward.__name__ == "_checkpointed_forward"
        assert layer.self_attn.q_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.k_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.v_proj.forward.__name__ != "_checkpointed_forward"
        assert layer.self_attn.o_proj.forward.__name__ != "_checkpointed_forward"


def test_selective_planning_is_idempotent() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    first = plan_gradient_checkpointing(model, "selective")
    forwards = [layer.self_attn.forward for layer in model.model.layers]
    second = plan_gradient_checkpointing(model, "selective")

    assert first.hooked_modules == second.hooked_modules == 4
    assert second.granularity == "selective"
    assert [layer.self_attn.forward for layer in model.model.layers] == forwards


def test_selective_tiny_model_completes_forward_and_backward() -> None:
    torch = pytest.importorskip("torch")
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    model = _tiny_llama()
    plan = plan_gradient_checkpointing(model, "selective")
    input_ids = torch.tensor([[1, 2, 3, 4]])

    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()

    assert plan.hooked_modules == 4
    assert loss.isfinite().item() is True
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None


def test_selective_falls_back_truthfully_when_architecture_has_no_attention_child() -> None:
    from soup_cli.utils.gradient_ckpt import plan_gradient_checkpointing

    class Block:
        def named_children(self):
            return iter(())

    class Model:
        def named_modules(self):
            yield "model.layers.0", Block()

    plan = plan_gradient_checkpointing(Model(), "selective")

    assert plan.granularity == "full"
    assert plan.kwargs["gradient_checkpointing"] is True
    assert plan.hooked_modules == 0
    assert "full fallback" in plan.description


@pytest.mark.parametrize(("tier", "memory_gb"), [("medium", 40), ("auto", 40)])
def test_sft_setup_forwards_medium_plan_to_real_trainer(
    tmp_path, monkeypatch, tier: str, memory_gb: int
) -> None:
    wrapper = _setup_real_sft(
        tmp_path, monkeypatch, tier=tier, memory_gb=memory_gb
    )

    assert wrapper.trainer.args.gradient_checkpointing is True
    assert wrapper.trainer.args.gradient_checkpointing_kwargs == {
        "use_reentrant": False,
        "every_n_layers": 2,
    }

    wrapper.trainer.train()
    layers = wrapper.trainer.model.get_base_model().model.layers
    assert [layer.gradient_checkpointing for layer in layers] == [True, False, True, False]


def test_sft_selective_disables_kbit_full_checkpointing_and_wraps_attention(
    tmp_path, monkeypatch
) -> None:
    wrapper = _setup_real_sft(
        tmp_path,
        monkeypatch,
        tier="selective",
        quantization="4bit",
        memory_gb=120,
    )

    model = wrapper.trainer.model
    base = model.get_base_model()
    assert model.is_gradient_checkpointing is False
    assert wrapper.trainer.args.gradient_checkpointing is False
    assert [layer.gradient_checkpointing for layer in base.model.layers] == [False] * 4
    for layer in base.model.layers:
        assert layer.self_attn.forward.__name__ == "_checkpointed_forward"

    wrapper.trainer.train()
    grad_norms = [
        float(entry["grad_norm"])
        for entry in wrapper.trainer.state.log_history
        if "grad_norm" in entry
    ]
    assert grad_norms
    assert all(math.isfinite(value) for value in grad_norms)
    assert any(value > 0 for value in grad_norms)
