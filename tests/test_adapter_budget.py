"""Persistent adapter tensor accounting across supported optimizer profiles."""

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
def test_cpu_optimizer_states(profile: str, state: int, total: int) -> None:
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
def test_cuda_optimizer_state_and_host_steps(hidden: int, state8: int, state32: int) -> None:
    shapes = [(8, hidden), (2 * hidden, 8), (8, 2 * hidden), (hidden, 8)]
    tensors = [AdapterTensor(str(index), shape) for index, shape in enumerate(shapes)]
    bnb = estimate_adapter_budget(
        tensors, profile="bnb_0.50.2_adamw8bit_fp32", device="cuda"
    )
    adam = estimate_adapter_budget(tensors, profile="torch_adamw_fp32", device="cuda")
    assert bnb.optimizer == MemoryPart(cuda_bytes=state8)
    # The four non-capturable AdamW step tensors are CPU scalars.
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


@pytest.mark.parametrize(
    "profile,device,dtype,reason,weights",
    [
        (
            "unknown", "cuda", "float32", "unverified optimizer profile: 'unknown'",
            MemoryPart(cuda_bytes=2048),
        ),
        (
            "torch_adamw_fp32", "cuda", "bfloat16",
            "optimizer profiles currently require materialized float32 adapters",
            MemoryPart(cuda_bytes=1024),
        ),
        (
            "torch_adamw_fp32", "cuda", "float16",
            "optimizer profiles currently require materialized float32 adapters",
            MemoryPart(cuda_bytes=1024),
        ),
        (
            "torch_adamw_fp32", "cpu", "float64",
            "optimizer profiles currently require materialized float32 adapters",
            MemoryPart(cpu_bytes=4096),
        ),
        (
            "bnb_0.50.2_adamw8bit_fp32", "cpu", "float32",
            "this bitsandbytes profile was verified on CUDA only", MemoryPart(cpu_bytes=2048),
        ),
    ],
)
def test_unsupported_retains_known_storage(
    profile: str, device: str, dtype: str, reason: str, weights: MemoryPart
) -> None:
    result = estimate_adapter_budget(
        [AdapterTensor("unclassified", (8, 64), dtype=dtype)], profile=profile, device=device
    )
    assert result.total is None
    assert result.optimizer is None
    assert result.unsupported_reason == reason
    assert result.weights == weights
    assert result.gradients == weights


@pytest.mark.parametrize(
    "tensors",
    [
        [AdapterTensor("q.lora_A.weight", (2, 3), trainable=False)],
        [AdapterTensor("q.lora_A.weight", (2, 3)), AdapterTensor("q.lora_B.weight", (4, 2))],
        [
            AdapterTensor("q.lora_A.weight", (2, 3), trainable=False),
            AdapterTensor("q.lora_B.weight", (4, 2), trainable=False),
        ],
        [
            AdapterTensor("q.lora_A.weight", (2, 3), trainable=False),
            AdapterTensor("q.lora_B.weight", (4, 2, 2)),
        ],
        [
            AdapterTensor("q.lora_A.weight", (2, 3), trainable=False),
            AdapterTensor("q.lora_B.weight", (4, 1)),
        ],
        [
            AdapterTensor("q.lora_A.weight", (2, 3), trainable=False),
            AdapterTensor("q.lora_B.weight", (4, 2)),
            AdapterTensor("extra", (12,)),
        ],
    ],
    ids=["unpaired", "trainable-a", "frozen-b", "three-axes", "rank-mismatch", "extra"],
)
def test_lorafa_requires_complete_frozen_a_trainable_b_pairs(
    tensors: list[AdapterTensor],
) -> None:
    result = estimate_adapter_budget(tensors, profile="peft_lorafa_fp32", device="cpu")
    assert result.optimizer is None
    assert result.total is None
    assert result.unsupported_reason == (
        "LoRA-FA needs complete 2D A/B pairs with matching rank, A frozen and B trainable"
    )


def test_adafactor_factors_last_two_axes_and_keeps_vector_accumulator() -> None:
    result = estimate_adapter_budget(
        [AdapterTensor("experts", (2, 3, 4)), AdapterTensor("bias", (7,))],
        profile="hf_adafactor_fp32",
        device="cuda",
    )
    assert result.optimizer == MemoryPart(cuda_bytes=92)


@pytest.mark.parametrize(
    "metadata,message",
    [
        ({"name": ""}, "adapter tensor name must be a non-empty string; got ''"),
        ({"name": 8}, "adapter tensor name must be a non-empty string; got 8"),
        ({"shape": ()}, "adapter tensor 'q' shape must be a non-empty tuple; got ()"),
        ({"shape": [8]}, "adapter tensor 'q' shape must be a non-empty tuple; got [8]"),
        (
            {"shape": (8, 0)},
            "adapter tensor 'q' shape dimensions must be positive integers; got (8, 0)",
        ),
        (
            {"shape": (True, 8)},
            "adapter tensor 'q' shape dimensions must be positive integers; got (True, 8)",
        ),
        (
            {"dtype": "int8"},
            "adapter tensor 'q' has unsupported storage dtype: 'int8'",
        ),
        ({"dtype": []}, "adapter tensor 'q' has unsupported storage dtype: []"),
        ({"trainable": 1}, "adapter tensor 'q' trainable must be boolean; got 1"),
    ],
)
def test_invalid_tensor_metadata_names_field_and_value(metadata: dict, message: str) -> None:
    with pytest.raises(ValueError) as exc:
        AdapterTensor(**({"name": "q", "shape": (8, 64)} | metadata))
    assert str(exc.value) == message


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"device": "mps"}, "adapter budget device must be cpu or cuda; got 'mps'"),
        ({"profile": []}, "adapter budget profile must be a non-empty string; got []"),
        ({"profile": ""}, "adapter budget profile must be a non-empty string; got ''"),
        ({"tensors": []}, "empty adapter profile: check target resolution"),
        ({"tensors": None}, "adapter budget tensors must be an iterable of AdapterTensor records"),
        ({"tensors": ["q"]}, "adapter budget tensors[0] must be an AdapterTensor; got str"),
        (
            {"tensors": [AdapterTensor("q", (8,)), AdapterTensor("q", (64,))]},
            "duplicate adapter tensor name 'q'; deduplicate physical parameters",
        ),
    ],
)
def test_invalid_budget_request_names_field_and_value(overrides: dict, message: str) -> None:
    request = {
        "tensors": [AdapterTensor("q", (8,))],
        "profile": "torch_adamw_fp32",
        "device": "cpu",
    }
    with pytest.raises(ValueError) as exc:
        estimate_adapter_budget(**(request | overrides))
    assert str(exc.value) == message


@pytest.mark.parametrize("field,value", [("cpu_bytes", -1), ("cuda_bytes", True)])
def test_invalid_memory_part_cannot_hide_negative_or_non_integer_bytes(
    field: str, value: int,
) -> None:
    with pytest.raises(ValueError) as exc:
        MemoryPart(**{field: value})
    assert str(exc.value) == f"adapter memory {field} must be a non-negative integer; got {value!r}"


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


def test_kimi_k2_routed_experts_rank16_reference() -> None:
    """Size separate per-expert adapters using Kimi K2's published dimensions.

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
