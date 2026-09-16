"""#1012: a LoRA target on lm_head/embed_tokens died at the first streamed forward.

``build_streamed_model`` runs ``get_peft_model()`` before ``install_streaming()``
wraps the input/output embeddings in ``StreamedLargeLayer`` (when the shard carries
embed_tokens and lm_head as separate, untied large-layer weights). A LoRA target on
either module therefore wraps a plain ``nn.Linear``/``nn.Embedding`` into peft's
``lora.Linear``/``lora.Embedding`` FIRST, moving the real (still-meta) base weight to
``base_layer.weight`` and leaving peft's own ``weight`` as a READ-ONLY property that
delegates to it. ``StreamedLargeLayer.forward()`` used to substitute a plain
``{"weight": ...}`` via ``functional_call``, which setattrs straight onto that
read-only property and raises ``AttributeError``, after the checkpoint is already
sharded and streaming is fully set up.

Decoder-layer projections (q_proj/v_proj) hit the identical get_peft_model-then-wrap
ordering and do not break, because ``_layer_name_map`` already resolves the
``.base_layer.`` indirection peft introduces and substitutes onto THAT name.
``_large_layer_weight_param_name`` mirrors the same resolution for the large-layer
boundary modules, so ``StreamedLargeLayer`` now substitutes onto
``base_layer.weight`` whenever ``inner`` is peft-wrapped, same as every other
streamed weight.

A tied checkpoint never reaches the streamed-large-layer path at all: only
``embed_tokens.weight`` is stored, so ``large_layer_specs`` returns no ``lm_head``
entry and both modules stay resident (untouched by this fix; pinned below).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _requires_train_extra():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")


def _tiny_llama_dir(tmp_path, tie):
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=tie,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = tmp_path / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    if tie:
        state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _lora(target_modules):
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def _build_streamed(tmp_path, *, tie, target_modules, seed=3):
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(tmp_path, tie=tie)
    shards = str(tmp_path / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    model, runtime = build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=_lora(target_modules),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=seed,
    )
    return model, runtime, weights


def _build_resident(weights_dir, target_modules):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(weights_dir, dtype=torch.float32)
    return get_peft_model(base, _lora(target_modules))


def _copy_adapters(src, dst):
    """Force the resident model's adapters to match the streamed model's
    (both are randomly initialised independently by materialize_meta_adapters
    / get_peft_model, so a bit-exactness comparison needs one shared value)."""
    import torch

    src_params = dict(src.named_parameters())
    dst_params = dict(dst.named_parameters())
    matched = 0
    for name, param in src_params.items():
        if "lora_" not in name or param.is_meta:
            continue
        target = dst_params.get(name)
        assert target is not None, f"resident model has no parameter named {name!r}"
        with torch.no_grad():
            target.copy_(param)
        matched += 1
    assert matched > 0, "fixture: no LoRA adapter parameters matched"


def _inputs():
    import torch

    return dict(
        input_ids=torch.randint(0, 64, (1, 8)),
        labels=torch.randint(0, 64, (1, 8)),
    )


class TestUntiedLoraOnStreamedLargeLayerNowWorks:
    @pytest.mark.parametrize("head_target", ["lm_head", "embed_tokens"])
    def test_forward_no_longer_raises(self, tmp_path, head_target):
        """The bug: a LoRA target on either boundary module used to raise
        AttributeError at the first forward on an untied checkpoint."""
        _requires_train_extra()

        model, runtime, _ = _build_streamed(
            tmp_path, tie=False, target_modules=["q_proj", "v_proj", head_target]
        )
        try:
            out = model(**_inputs())
            assert out.loss is not None and out.loss.isfinite()
        finally:
            runtime.close()

    def test_forward_backward_bit_exact_against_resident_lora_on_head(self, tmp_path):
        """The substitution must land on the SAME base weight a resident model
        uses, or the numbers would be silently wrong instead of merely not
        crashing. Same inputs, same adapters (copied in), streamed vs resident."""
        import torch

        _requires_train_extra()
        targets = ["q_proj", "v_proj", "lm_head", "embed_tokens"]

        streamed, runtime, weights = _build_streamed(tmp_path, tie=False, target_modules=targets)
        try:
            resident = _build_resident(weights, targets)
            _copy_adapters(streamed, resident)

            torch.manual_seed(11)
            inputs = _inputs()

            streamed_out = streamed(**inputs)
            resident_out = resident(**inputs)
            assert torch.allclose(streamed_out.logits, resident_out.logits, atol=1e-5)
            assert torch.allclose(streamed_out.loss, resident_out.loss, atol=1e-5)

            streamed_out.loss.backward()
            resident_out.loss.backward()
            streamed_grads = {
                name: p.grad
                for name, p in streamed.named_parameters()
                if "lora_" in name and p.grad is not None
            }
            resident_params = dict(resident.named_parameters())
            assert streamed_grads, "fixture: no LoRA gradients produced"
            for name, grad in streamed_grads.items():
                other = resident_params[name].grad
                assert other is not None, f"resident model produced no grad for {name!r}"
                assert torch.allclose(grad, other, atol=1e-5), f"gradient mismatch at {name!r}"
        finally:
            runtime.close()


class TestUnaffectedConfigsStillTrain:
    def test_untied_without_head_target_builds_and_trains(self, tmp_path):
        """Negative control: an untied checkpoint with no LoRA target on the
        streamed boundary modules is completely unaffected by this fix."""
        _requires_train_extra()

        model, runtime, _ = _build_streamed(
            tmp_path, tie=False, target_modules=["q_proj", "v_proj"]
        )
        try:
            out = model(**_inputs())
            out.loss.backward()
        finally:
            runtime.close()

    def test_tied_lora_on_lm_head_builds_and_trains(self, tmp_path):
        """Negative control: a TIED checkpoint never streams embed_tokens/lm_head
        as a large layer (only embed_tokens.weight is stored, so
        large_layer_specs has no lm_head entry), so this path is untouched and
        this config, which trained fine before this fix, must keep training fine."""
        _requires_train_extra()

        model, runtime, _ = _build_streamed(
            tmp_path, tie=True, target_modules=["q_proj", "v_proj", "lm_head"]
        )
        try:
            out = model(**_inputs())
            out.loss.backward()
        finally:
            runtime.close()
