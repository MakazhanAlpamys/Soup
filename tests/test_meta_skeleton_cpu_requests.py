"""#2154: the streaming skeleton never asks the CPU for a full-size fused expert tensor.

``init_empty_weights()`` moves a parameter to ``meta`` only after it was registered, and
the fused MoE experts build theirs with ``torch.empty(num_experts, 2 * intermediate,
hidden)`` first, so every MoE layer committed the whole tensor on the CPU for a moment:
22.5 GB per layer for Kimi K2. The twin here has the same layout at a size where one
fused tensor is 2 MiB, and the factory calls are recorded while the skeleton is built.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

#: The size from which a CPU request counts as one the skeleton must not make.
LARGE = 1 << 20

EXPERTS, HIDDEN, MOE_INTERMEDIATE = 8, 256, 256
_FACTORIES = ("empty", "zeros", "ones", "full", "empty_strided")


def _twin_config():
    from transformers import DeepseekV3Config

    return DeepseekV3Config(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=256,
        moe_intermediate_size=MOE_INTERMEDIATE, num_hidden_layers=3, first_k_dense_replace=1,
        num_attention_heads=4, num_key_value_heads=4, n_shared_experts=1,
        n_routed_experts=EXPERTS, n_group=2, topk_group=1, num_experts_per_tok=2,
        kv_lora_rank=16, q_lora_rank=24, qk_nope_head_dim=16, qk_rope_head_dim=8,
        v_head_dim=16, max_position_embeddings=128,
    )


@pytest.fixture(scope="module")
def twin(tmp_path_factory):
    """A saved bf16 ``deepseek_v3`` twin whose fused ``gate_up_proj`` is 2 MiB."""
    from transformers import DeepseekV3ForCausalLM

    torch.manual_seed(5)
    model = DeepseekV3ForCausalLM(_twin_config()).to(torch.bfloat16).eval()
    root = tmp_path_factory.mktemp("twin2154")
    model.save_pretrained(str(root))
    return str(root)


@pytest.fixture()
def cpu_requests(monkeypatch):
    """Every factory call while the fixture is active: ``(name, shape, device, bytes)``."""
    # torch caches the factories ``torch.device(...)`` applies to on first use; fill that
    # cache with the real ones before wrapping them, or a meta context would miss them.
    with torch.device("meta"):
        torch.empty(1)
    seen: list = []

    def recording(name, original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            seen.append((name, tuple(result.shape), result.device.type, result.nbytes))
            return result

        return call

    for name in _FACTORIES:
        monkeypatch.setattr(torch, name, recording(name, getattr(torch, name)))
    return seen


def _runtime_warnings(caplog):
    """WARNING records from the streaming runtime only: transformers logs a one-time
    deprecation warning the first time a process builds a model."""
    return [
        r
        for r in caplog.records
        if r.name == "soup_cli.utils.layer_stream_runtime" and r.levelname == "WARNING"
    ]


def _large_cpu(seen):
    return [entry for entry in seen if entry[2] == "cpu" and entry[3] >= LARGE]


def test_the_twin_has_a_fused_expert_tensor_of_at_least_the_threshold(twin):
    """Otherwise the tests below would pass on main."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(twin, dtype=torch.bfloat16)
    gate_up = model.model.layers[1].mlp.experts.gate_up_proj
    assert tuple(gate_up.shape) == (EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN)
    assert gate_up.nbytes >= LARGE


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_the_skeleton_makes_no_large_cpu_request(twin, cpu_requests, quant):
    if quant == "nf4":
        pytest.importorskip("bitsandbytes")
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    model = build_meta_skeleton(twin, dtype="bfloat16", quant=quant)

    assert _large_cpu(cpu_requests) == [], "a full-size CPU request while building on meta"
    assert all(p.is_meta for p in model.parameters())
    experts = [p for n, p in model.named_parameters() if ".mlp.experts." in n]
    assert experts and all(p.is_meta for p in experts)
    buffers = dict(model.named_buffers())
    rotary = [b for n, b in buffers.items() if n.endswith("inv_freq")]
    bias = [b for n, b in buffers.items() if n.endswith("e_score_correction_bias")]
    assert rotary and bias
    assert not any(b.is_meta for b in rotary + bias)


def _shape(model):
    """What the rest of the runtime reads off a skeleton."""
    return {
        "parameters": {
            n: (tuple(p.shape), p.dtype, p.is_meta) for n, p in model.named_parameters()
        },
        "buffers": {n: b.clone() for n, b in model.named_buffers()},
        "name_or_path": (model.name_or_path, model.config.name_or_path),
    }


@pytest.mark.parametrize("threshold", [LARGE, 64 << 10], ids=["1MiB", "64KiB"])
@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_the_skeleton_is_the_one_built_without_the_redirect(
    twin, monkeypatch, cpu_requests, quant, threshold
):
    """At 64 KiB the twin's 128 KiB linears go through the redirect too, under ``nf4`` the
    ``Linear4bit`` ones that ``replace_with_bnb_linear`` builds; the buffers stay real."""
    if quant == "nf4":
        pytest.importorskip("bitsandbytes")
    from soup_cli.utils import layer_stream_runtime as runtime

    monkeypatch.setattr(runtime, "SKELETON_META_REDIRECT_BYTES", threshold)
    redirected = runtime.build_meta_skeleton(twin, dtype="bfloat16", quant=quant)
    assert [e for e in cpu_requests if e[2] == "cpu" and e[3] >= threshold] == []
    monkeypatch.setattr(runtime, "SKELETON_META_REDIRECT_BYTES", 1 << 62)  # nothing redirected
    plain = runtime.build_meta_skeleton(twin, dtype="bfloat16", quant=quant)

    mine, theirs = _shape(redirected), _shape(plain)
    assert mine["parameters"] == theirs["parameters"]
    assert mine["buffers"].keys() == theirs["buffers"].keys()
    for name, value in mine["buffers"].items():
        assert torch.equal(value, theirs["buffers"][name]), name
    assert mine["name_or_path"] == theirs["name_or_path"] == (twin, twin)
    if quant == "nf4":
        assert runtime.quantised_layer_suffixes(redirected) == runtime.quantised_layer_suffixes(
            plain
        )


def test_a_large_buffer_is_built_the_old_way_and_named(twin, monkeypatch, caplog, cpu_requests):
    """A buffer must hold real values, so one past the threshold turns the redirect off."""
    from transformers.models.deepseek_v3 import modeling_deepseek_v3

    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    rotary = modeling_deepseek_v3.DeepseekV3RotaryEmbedding
    original_init = rotary.__init__

    def init_with_a_table(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.register_buffer(
            "big_table", torch.ones(LARGE // 4, dtype=torch.float32), persistent=False
        )  # 1 MiB

    monkeypatch.setattr(rotary, "__init__", init_with_a_table)

    with caplog.at_level("WARNING", logger="soup_cli.utils.layer_stream_runtime"):
        model = build_meta_skeleton(twin, dtype="bfloat16", quant="none")

    table = dict(model.named_buffers())["model.rotary_emb.big_table"]
    assert not table.is_meta and bool((table == 1).all())
    assert all(p.is_meta for p in model.parameters())
    warnings = _runtime_warnings(caplog)
    assert len(warnings) == 1 and "model.rotary_emb.big_table" in warnings[0].getMessage()
    # Built again without the redirect: the fused request is back on the CPU.
    assert ("empty", (EXPERTS, 2 * MOE_INTERMEDIATE, HIDDEN)) in {
        entry[:2] for entry in _large_cpu(cpu_requests)
    }


def test_no_warning_when_every_buffer_is_small(twin, caplog):
    from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

    with caplog.at_level("WARNING", logger="soup_cli.utils.layer_stream_runtime"):
        build_meta_skeleton(twin, dtype="bfloat16", quant="none")

    assert _runtime_warnings(caplog) == []


def test_the_redirect_ends_with_the_build_even_when_it_raises(twin, monkeypatch):
    from soup_cli.utils import layer_stream_runtime as runtime

    before = {name: getattr(torch, name) for name in _FACTORIES}

    def failing_build(*args, **kwargs):
        assert torch.empty(LARGE, dtype=torch.float32).is_meta  # the redirect is active here
        raise RuntimeError("from_config failed")

    monkeypatch.setattr(runtime, "_skeleton_from_config", failing_build)
    with pytest.raises(RuntimeError, match="from_config failed"):
        runtime.build_meta_skeleton(twin, dtype="bfloat16", quant="none")

    assert torch.empty(LARGE, dtype=torch.float32).device.type == "cpu"
    assert {name: getattr(torch, name) for name in _FACTORIES} == before  # never patched


def test_the_redirect_applies_to_the_building_thread_only():
    import threading

    from soup_cli.utils.layer_stream_runtime import _large_cpu_requests_on_meta

    other: dict = {}
    with _large_cpu_requests_on_meta(LARGE):
        here = torch.empty(LARGE, dtype=torch.float32)
        thread = threading.Thread(
            target=lambda: other.update(t=torch.empty(LARGE, dtype=torch.float32))
        )
        thread.start()
        thread.join()

    assert here.is_meta
    assert other["t"].device.type == "cpu"


def test_a_small_or_non_cpu_request_is_left_as_it_is():
    from soup_cli.utils.layer_stream_runtime import _large_cpu_requests_on_meta

    target = torch.ones(LARGE, dtype=torch.float32)  # 4 MiB, made before the redirect
    with _large_cpu_requests_on_meta(LARGE):
        small = torch.zeros(4)
        large = torch.empty(LARGE, dtype=torch.float32)
        on_meta = torch.empty(LARGE, device="meta")
        filled = torch.full((2,), 3.0)
        into = torch.zeros(LARGE, dtype=torch.float32, out=target)

    assert small.device.type == "cpu" and bool((small == 0).all())
    assert large.is_meta and on_meta.is_meta
    assert filled.tolist() == [3.0, 3.0]
    assert into is target and not target.is_meta and bool((target == 0).all())


def test_the_streamed_model_builds_and_steps_on_the_twin(twin, tmp_path):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    shards = str(tmp_path / "shards")
    index = shard_checkpoint(twin, shards, dtype="float32", arch="deepseek_v3", num_hidden_layers=3)
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, bias="none", target_modules=["q_b_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    model, runtime = build_streamed_model(
        model_id=twin, shard_dir=shards, index=index, lora_config=lora,
        device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
    )
    try:
        ids = torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(1))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        assert torch.isfinite(loss)
    finally:
        runtime.close()


_FRESH_PROCESS_CHECK = """
import sys

import torch

from soup_cli.utils.layer_stream_runtime import build_meta_skeleton

build_meta_skeleton(sys.argv[1], dtype="bfloat16", quant=sys.argv[2])
with torch.device("meta"):
    made = [torch.empty(2, 2), torch.zeros(2, 2), torch.ones(2, 2), torch.full((2, 2), 1.0)]
print(" ".join(t.device.type for t in made))
"""


@pytest.mark.parametrize("quant", ["none", "nf4"])
def test_a_meta_device_context_still_works_after_the_build(twin, tmp_path, quant):
    """In a fresh process the build may be the first use of a device context, and torch
    caches the factories that ``torch.device(...)`` applies to on that first use."""
    import os
    import subprocess
    import sys

    if quant == "nf4":
        pytest.importorskip("bitsandbytes")
    script = tmp_path / "check.py"
    script.write_text(_FRESH_PROCESS_CHECK, encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    done = subprocess.run(
        [sys.executable, str(script), twin, quant], capture_output=True, text=True, env=env
    )

    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["meta"] * 4
