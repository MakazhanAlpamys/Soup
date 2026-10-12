"""Persistent tensor accounting, with independent CPU/Kaggle measurements."""

import pytest

from soup_cli.utils.adapter_budget import AdapterTensor, MemoryPart, estimate_adapter_budget


def _expert_tensors(frozen_a: bool = False) -> list[AdapterTensor]:
    tensors = []
    for layer in range(2):
        for expert in range(4):
            for target in ("gate", "up", "down"):
                incoming, outgoing = (128, 64) if target == "down" else (64, 128)
                prefix = f"layers.{layer}.experts.{expert}.{target}"
                tensors.extend([
                    AdapterTensor(f"{prefix}.lora_A.weight", (8, incoming), trainable=not frozen_a),
                    AdapterTensor(f"{prefix}.lora_B.weight", (outgoing, 8)),
                ])
    return tensors


@pytest.mark.parametrize("profile,state,total", [
    ("torch_adamw_fp32", 295104, 590016),
    ("torch_sgd_fp32", 0, 294912),
    ("torch_adagrad_fp32", 147648, 442560),
    ("torch_rmsprop_fp32", 147648, 442560),
    ("hf_adafactor_fp32", 20160, 315072),
])
def test_measured_cpu_optimizer_states(profile: str, state: int, total: int) -> None:
    result = estimate_adapter_budget(_expert_tensors(), profile=profile, device="cpu")
    assert result.stored_parameters == result.trainable_parameters == 36864
    assert result.optimizer == MemoryPart(cpu_bytes=state)
    assert result.total == MemoryPart(cpu_bytes=total)


def test_lorafa_keeps_frozen_a_storage() -> None:
    result = estimate_adapter_budget(
        _expert_tensors(frozen_a=True), profile="peft_lorafa_fp32", device="cpu"
    )
    assert result.stored_parameters == 36864
    assert result.trainable_parameters == 20480
    assert result.weights.cpu_bytes == 147456
    assert result.gradients.cpu_bytes == 81920
    assert result.optimizer == MemoryPart(cpu_bytes=163840)
    assert result.total == MemoryPart(cpu_bytes=393216)


@pytest.mark.parametrize("hidden,state8,state32", [(64, 24576, 24592), (512, 51968, 196624)])
def test_kaggle_t4_states_with_and_without_nf4(hidden: int, state8: int, state32: int) -> None:
    # NF4 changes base storage, not these measured fp32 adapter tensor shapes.
    shapes = [(8, hidden), (2 * hidden, 8), (8, 2 * hidden), (hidden, 8)]
    tensors = [AdapterTensor(str(index), shape) for index, shape in enumerate(shapes)]
    bnb = estimate_adapter_budget(
        tensors, profile="bnb_0.50.2_adamw8bit_fp32", device="cuda"
    )
    adam = estimate_adapter_budget(tensors, profile="torch_adamw_fp32", device="cuda")
    assert bnb.optimizer == MemoryPart(cuda_bytes=state8)
    # Four scalar step tensors were observed on CPU, not CUDA.
    assert adam.optimizer == MemoryPart(cpu_bytes=16, cuda_bytes=state32 - 16)
    assert bnb.weights == adam.weights == MemoryPart(cuda_bytes=hidden * 192)


def test_bnb_threshold_and_shared_maps() -> None:
    result = estimate_adapter_budget(
        [AdapterTensor("small", (4095,)), AdapterTensor("boundary", (4096,)),
         AdapterTensor("tail", (4097,))],
        profile="bnb_0.50.2_adamw8bit_fp32", device="cuda",
    )
    assert result.optimizer == MemoryPart(
        cuda_bytes=4095 * 8 + (4096 + 4097) * 2 + (16 + 17) * 8 + 2048
    )


@pytest.mark.parametrize("profile,device,dtype", [
    ("unknown", "cuda", "float32"),
    ("torch_adamw_fp32", "cuda", "bfloat16"),
    ("bnb_0.50.2_adamw8bit_fp32", "cpu", "float32"),
    ("peft_lorafa_fp32", "cpu", "float32"),
])
def test_unsupported_is_not_zero_or_fit(profile: str, device: str, dtype: str) -> None:
    result = estimate_adapter_budget(
        [AdapterTensor("unclassified", (8, 64), dtype=dtype)], profile=profile, device=device
    )
    assert result.total is None
    assert result.optimizer is None
    assert result.unsupported_reason
    assert result.weights.cpu_bytes + result.weights.cuda_bytes > 0


def test_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        AdapterTensor("bad", (8, 0))
    with pytest.raises(ValueError, match="empty adapter"):
        estimate_adapter_budget([], profile="torch_adamw_fp32", device="cpu")
    item = AdapterTensor("same", (8, 64))
    with pytest.raises(ValueError, match="duplicate"):
        estimate_adapter_budget([item, item], profile="torch_adamw_fp32", device="cpu")


def test_real_peft_meta_shapes_feed_budget() -> None:
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    accelerate = pytest.importorskip("accelerate")
    with accelerate.init_empty_weights(include_buffers=True):
        base = torch.nn.Sequential(torch.nn.Linear(64, 128, bias=False))
        model = peft.get_peft_model(base, peft.LoraConfig(r=8, target_modules=["0"]))
    tensors = [AdapterTensor(name, tuple(param.shape))
               for name, param in model.named_parameters() if param.requires_grad]
    assert all(param.is_meta for param in model.parameters())
    result = estimate_adapter_budget(tensors, profile="torch_adamw_fp32", device="cuda")
    assert result.stored_parameters == 1536
    assert result.total == MemoryPart(cpu_bytes=8, cuda_bytes=24576)


def test_kimi_k2_routed_experts_rank16_reference():
    """Config-only reference, not a claim of runtime K2/FP8 architecture support.

    https://huggingface.co/moonshotai/Kimi-K2-Instruct/raw/main/config.json
    61 layers, first dense, 384 routed experts, hidden 7168, expert width 2048.
    """
    tensors = []
    for layer in range(1, 61):
        for expert in range(384):
            for name, incoming, outgoing in (("gate", 7168, 2048),
                                              ("up", 7168, 2048),
                                              ("down", 2048, 7168)):
                prefix = f"layers.{layer}.experts.{expert}.{name}"
                tensors.extend((AdapterTensor(prefix + ".lora_A.weight", (16, incoming)),
                                AdapterTensor(prefix + ".lora_B.weight", (outgoing, 16))))
    budget = estimate_adapter_budget(tensors, profile="torch_adamw_fp32", device="cuda")
    assert budget.stored_parameters == 10_192_158_720
    assert budget.optimizer.cuda_bytes == 81_537_269_760
    assert budget.total.cuda_bytes == 163_074_539_520
    assert budget.optimizer.cpu_bytes == 60 * 384 * 3 * 2 * 4

