"""Resident BF16 SFT keeps PEFT preparation without the full-size FP32 copies."""

from __future__ import annotations

import pytest

from soup_cli.utils.kbit_prepare import prepare_kbit_model
from tests.test_kbit_prepare_keeps_dtype import _setup_trainer, _tiny_model, torch


def test_resident_sft_avoids_the_transient_allocation(monkeypatch):
    import peft

    prepare = peft.prepare_model_for_kbit_training
    seen = []

    def inspect_cast(model, **kwargs):
        seen.append(model.get_input_embeddings().weight.numel())
        return prepare(model, **kwargs)

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", inspect_cast)
    model = _setup_trainer(
        monkeypatch,
        task="sft",
        module="soup_cli.trainer.sft",
        cls_name="SFTTrainerWrapper",
        flags=(True, False),
    )

    embedding = model.get_input_embeddings()
    assert seen == [0]
    assert embedding.weight.numel() == 64 * 16
    assert embedding.weight.dtype is torch.bfloat16
    assert not any(param.requires_grad for param in model.parameters())
    assert model.is_gradient_checkpointing
    assert embedding(torch.tensor([[1, 2]])).requires_grad


@pytest.mark.parametrize("load_flag", ["is_loaded_in_4bit", "is_loaded_in_8bit"])
def test_only_weights_main_would_restore_are_shielded(load_flag):
    class Params4bit(torch.nn.Parameter):
        pass

    model = _tiny_model(torch.bfloat16)
    model.is_loaded_in_4bit = False
    setattr(model, load_flag, True)
    model.quantization_method = "bitsandbytes"
    model.model.norm.weight.data = model.model.norm.weight.data.half()
    model.register_parameter("lora_A", torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16)))
    model.register_parameter("existing_fp32", torch.nn.Parameter(torch.ones(2)))
    model.register_parameter("packed", Params4bit(torch.ones(2, dtype=torch.bfloat16)))
    weight = model.get_input_embeddings().weight.detach()

    prepare_kbit_model(
        model,
        use_gradient_checkpointing=False,
        compute_dtype=torch.bfloat16,
        avoid_bf16_upcast=True,
    )

    assert model.get_input_embeddings().weight.data_ptr() == weight.data_ptr()
    assert torch.equal(model.get_input_embeddings().weight, weight)
    assert model.model.norm.weight.dtype is torch.float32
    assert model.lora_A.dtype is torch.float32
    assert model.existing_fp32.dtype is torch.float32
    assert model.packed.dtype is torch.bfloat16
    assert not model.is_gradient_checkpointing


@pytest.mark.parametrize(
    "dtype, method, loaded, expected_dtype",
    [
        (torch.float16, None, True, torch.float32),
        (torch.float32, None, True, torch.float32),
        (torch.bfloat16, "gptq", True, torch.bfloat16),
        (torch.bfloat16, None, False, torch.bfloat16),
    ],
)
def test_other_precision_and_quantization_routes_use_normal_preparation(
    monkeypatch,
    dtype,
    method,
    loaded,
    expected_dtype,
):
    import peft

    model = _tiny_model(torch.bfloat16)
    model.is_loaded_in_4bit = loaded
    model.quantization_method = method
    prepare = peft.prepare_model_for_kbit_training

    def inspect_cast(model, **kwargs):
        assert model.get_input_embeddings().weight.numel() == 64 * 16
        return prepare(model, **kwargs)

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", inspect_cast)
    prepare_kbit_model(
        model,
        use_gradient_checkpointing=False,
        compute_dtype=dtype,
        avoid_bf16_upcast=True,
    )
    assert model.get_input_embeddings().weight.dtype is expected_dtype


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_checkpoint_kwargs_and_input_grad_hook(use_reentrant):
    model = _tiny_model(torch.bfloat16)
    prepare_kbit_model(
        model,
        use_gradient_checkpointing=True,
        compute_dtype=torch.bfloat16,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
        avoid_bf16_upcast=True,
    )

    assert model.is_gradient_checkpointing
    # The lean path must install exactly the input-grad hook PEFT's own preparation installs for
    # these kwargs. Whether PEFT/Transformers install it for use_reentrant=False depends on their
    # versions, so the reference is PEFT itself, run on an identical model.
    import peft

    reference = _tiny_model(torch.bfloat16)
    peft.prepare_model_for_kbit_training(
        reference,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
    )
    tokens = torch.tensor([[1, 2]])
    expected = reference.get_input_embeddings()(tokens).requires_grad
    if use_reentrant:
        assert expected, "PEFT installs the input-grad hook for reentrant checkpointing"
    assert model.get_input_embeddings()(tokens).requires_grad is expected


def test_restores_original_storage_when_peft_raises(monkeypatch):
    import peft

    model = _tiny_model(torch.bfloat16)
    model.get_output_embeddings().weight = model.get_input_embeddings().weight
    original = {name: param.detach() for name, param in model.named_parameters()}

    def fail(model, **kwargs):
        assert model.get_input_embeddings().weight.numel() == 0
        raise RuntimeError("preparation failed")

    monkeypatch.setattr(peft, "prepare_model_for_kbit_training", fail)
    with pytest.raises(RuntimeError, match="^preparation failed$"):
        prepare_kbit_model(
            model,
            use_gradient_checkpointing=True,
            compute_dtype=torch.bfloat16,
            avoid_bf16_upcast=True,
        )

    for name, param in model.named_parameters():
        assert param.data_ptr() == original[name].data_ptr()
        assert torch.equal(param, original[name])
    assert model.get_input_embeddings().weight is model.get_output_embeddings().weight


@pytest.mark.gpu
def test_cuda_preparation_avoids_a_full_size_fp32_embedding_copy():
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("requires native CUDA BF16")

    model = _tiny_model(torch.bfloat16)
    # Move the unquantized test model before setting the bitsandbytes loader marker.
    model.is_loaded_in_4bit = False
    model.set_input_embeddings(torch.nn.Embedding(131072, 32, dtype=torch.bfloat16))
    model.cuda()
    model.is_loaded_in_4bit = True
    weight_bytes = model.get_input_embeddings().weight.nbytes

    def extra_peak(avoid_bf16_upcast: bool) -> int:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        prepare_kbit_model(
            model,
            use_gradient_checkpointing=False,
            compute_dtype=torch.bfloat16,
            avoid_bf16_upcast=avoid_bf16_upcast,
        )
        return torch.cuda.max_memory_allocated() - before

    assert extra_peak(False) >= 2 * weight_bytes
    assert extra_peak(True) < weight_bytes
