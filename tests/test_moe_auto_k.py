"""Public MoE auto-k sizing and the shared activation partition."""

from re import escape
from types import SimpleNamespace

import pytest


def _wrapper(tmp_path, monkeypatch, device, available, **overrides):
    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import l2l_spill
    from tests.test_expert_stream import _tiny_deepseek_v3
    from tests.test_l2l_probe_harness import hermetic_ram
    from tests.test_v07201 import _write_tiny_tokenizer

    weights, _ = _tiny_deepseek_v3(tmp_path)
    _write_tiny_tokenizer(weights)
    hermetic_ram(monkeypatch, tmp_path)
    monkeypatch.setattr(
        l2l_spill, "ram_budget", lambda: l2l_spill.RamBudget(None, available, None, {}, None)
    )
    monkeypatch.setenv("SOUP_LAYER_STREAM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("ACCELERATE_USE_CPU", str(device == "cpu").lower())
    monkeypatch.chdir(tmp_path)
    training = {
        "stream_layers": True,
        "stream_expert_layout": "per_expert",
        "stream_source": "disk",
        "stream_disk_kind": "nvme",
        "stream_pin": False,
        "quantization": "none",
        "l2l": True,
        "l2l_auto_k": True,
        "batch_size": 1,
        "gradient_accumulation_steps": 64,
        "epochs": 1,
        "logging_steps": 1,
        "save_steps": 1000,
        "lora": {"r": 2, "alpha": 4, "dropout": 0, "target_modules": ["q_a_proj"]},
    }
    training.update(overrides)
    cfg = SoupConfig.model_validate(
        {
            "base": weights,
            "output": str(tmp_path / "out"),
            "data": {"train": "train.jsonl", "max_length": 64, "chat_template": "chatml"},
            "training": training,
        }
    )
    rows = [
        {"messages": [{"role": "user", "content": "hi " * count},
                      {"role": "assistant", "content": "hello world"}]}
        for count in range(1, 6)
    ]
    return SFTTrainerWrapper(cfg, device=device), {"train": rows}


@pytest.mark.parametrize("available, expected_k", [(450_000, 2), (8_000_000, 64)])
def test_public_deepseek_auto_k_shares_the_hf_window_and_host_partition(
    tmp_path, monkeypatch, capsys, available, expected_k
):
    from tests.conftest import strip_ansi

    wrapper, dataset = _wrapper(
        tmp_path, monkeypatch, "cpu", available, stream_experts={"backend": "grouped"}
    )
    try:
        wrapper.setup(dataset)
        recipe = wrapper._l2l_recipe
        state = wrapper.trainer._soup_l2l
        plan = state.host_budget_plan
        memory = state.memory_plan
        assert recipe.requested_k == wrapper.config.training.gradient_accumulation_steps == 64
        assert recipe.effective_k == state.effective_k == expected_k
        assert plan.max_microbatches == wrapper.trainer.args.gradient_accumulation_steps
        assert plan.max_microbatches == memory.effective_k == expected_k
        assert state.host_budget is plan.activations
        assert plan.routing_bytes == (2 * expected_k + 1) * 64 * 2 * 8
        assert plan.activations.limit_bytes == expected_k * 3 * 64 * 64 * 4 + plan.routing_bytes
        assert plan.weights.reserved_bytes <= available
        assert wrapper._stream_runtime.expert_pool.group_compute is not None
        assert f"L2L auto-k: k={expected_k} of 64" in strip_ansi(capsys.readouterr().out)
    finally:
        wrapper._close_stream_runtime()


def test_auto_k_store_keeps_routing_reserved_across_window_growth(tmp_path, monkeypatch):
    import torch

    from soup_cli.trainer.l2l_trainer import _store_for

    wrapper, dataset = _wrapper(tmp_path, monkeypatch, "cpu", 450_000)
    try:
        wrapper.setup(dataset)
        state = wrapper.trainer._soup_l2l
        plan = state.host_budget_plan
        stepper = SimpleNamespace(causal=wrapper.model, layers=[None] * plan.n_layers)

        def window(first, second):
            return [torch.zeros((1, length), dtype=torch.long) for length in (first, second)]

        first = _store_for(state, stepper, window(8, 16), "cpu")
        assert plan.activations.used_bytes == plan.routing_bytes + 2 * 3 * 16 * 64 * 4
        grown = _store_for(state, stepper, window(64, 8), "cpu")
        assert grown is not first
        assert plan.activations.used_bytes == plan.activations.limit_bytes
        with pytest.raises(ValueError) as caught:
            _store_for(state, stepper, window(65, 8), "cpu")
        assert str(caught.value) == (
            "training.l2l activation window exceeds or differs from the window "
            "reserved by training.l2l_auto_k"
        )
        assert state.store is grown
        assert plan.activations.used_bytes == plan.activations.limit_bytes
    finally:
        wrapper._close_stream_runtime()
    assert plan.activations.used_bytes == 0


@pytest.mark.parametrize(
    "available, optimizer, message",
    [
        (None, "adamw_torch", "training.l2l_auto_k could not establish a safe host RAM budget"),
        (8_000_000, "lomo", "training.l2l_auto_k requires a known optimizer host budget"),
    ],
)
def test_public_auto_k_refusals_drain_the_stream_runtime(
    tmp_path, monkeypatch, available, optimizer, message
):
    wrapper, dataset = _wrapper(tmp_path, monkeypatch, "cpu", available, optimizer=optimizer)
    with pytest.raises(ValueError, match=f"^{escape(message)}$"):
        wrapper.setup(dataset)
    assert wrapper._stream_runtime.source._closed


def test_auto_k_rejects_a_non_deepseek_base_before_loading_a_tokenizer(tmp_path, monkeypatch):
    from transformers import AutoTokenizer, LlamaConfig

    from soup_cli.utils import kimi_k2

    wrapper, dataset = _wrapper(tmp_path, monkeypatch, "cpu", 8_000_000)
    monkeypatch.setattr(kimi_k2, "load_stream_model_config", lambda *args, **kwargs: LlamaConfig())

    def unexpected_tokenizer(*args, **kwargs):
        pytest.fail("auto-k architecture refusal must precede tokenizer and weight loading")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", unexpected_tokenizer)
    with pytest.raises(ValueError) as caught:
        wrapper.setup(dataset)
    assert str(caught.value) == "training.l2l_auto_k requires a DeepSeek-V3 model"


def test_auto_k_checks_the_resolved_dtype_before_sharding(tmp_path, monkeypatch):
    from soup_cli.utils import layer_shard, layer_stream

    wrapper, dataset = _wrapper(tmp_path, monkeypatch, "cpu", 8_000_000)
    monkeypatch.setattr(layer_stream, "resolve_stream_dtype", lambda device: "bfloat16")

    def unexpected_shard(*args, **kwargs):
        pytest.fail("auto-k dtype refusal must precede sharding")

    monkeypatch.setattr(layer_shard, "shard_checkpoint", unexpected_shard)
    with pytest.raises(ValueError) as caught:
        wrapper.setup(dataset)
    assert str(caught.value) == "training.l2l_auto_k requires CPU FP32 or CUDA BF16 streaming"


@pytest.mark.parametrize("available, expected_k", [(739, None), (740, 1), (916, 2)])
def test_auto_k_counts_window_and_routing_workspace_once(monkeypatch, available, expected_k):
    from soup_cli.trainer.stream_setup import StreamingSetupMixin
    from soup_cli.utils import l2l_spill, layer_stream

    mixin = StreamingSetupMixin()
    mixin.device = "cpu"
    mixin.model = SimpleNamespace(parameters=lambda: ())
    cfg = SimpleNamespace(
        data=SimpleNamespace(max_length=1),
        training=SimpleNamespace(
            batch_size=1, optimizer="adamw_torch", gradient_accumulation_steps=8,
            l2l_auto_k=True, l2l_activation_codec="none", stream_expert_layout="per_expert",
        ),
    )
    model = SimpleNamespace(
        hidden_size=8, intermediate_size=16, moe_intermediate_size=4, vocab_size=16,
        num_experts_per_tok=2,
    )
    index = SimpleNamespace(n_layers=2, expert_layers=(1,), expert_count=4)
    monkeypatch.setattr(layer_stream, "estimate_activation_bytes", lambda **kwargs: 300)
    monkeypatch.setattr(layer_stream, "estimate_logits_bytes", lambda **kwargs: 248)
    monkeypatch.setattr(
        l2l_spill, "ram_budget", lambda: l2l_spill.RamBudget(None, available, None, {}, None)
    )
    # Per batch: 64 boundary bytes, 16 routing bytes and 96 input/hidden bytes.
    # Fixed: 548 live workspace bytes and one 16-byte routing comparison.
    if expected_k is None:
        with pytest.raises(MemoryError) as caught:
            mixin._resolve_moe_l2l_plan(cfg, model, index, "float32")
        assert str(caught.value) == (
            "training.l2l_auto_k: MoE L2L k=1 needs 192 host bytes "
            "for activations, routing and window workspace, but only 191 available"
        )
        return
    mixin._resolve_moe_l2l_plan(cfg, model, index, "float32")
    plan = mixin._l2l_host_budget_plan
    assert plan.max_shape == (1, 1, 8)
    assert plan.max_microbatches == expected_k
    assert plan.activations.limit_bytes == expected_k * 80 + 16
    assert plan.weights.reserved_bytes == 548 + expected_k * 176 + 16


@pytest.mark.gpu
def test_deepseek_auto_k_trains_and_saves_the_selected_recipe(tmp_path, monkeypatch):
    import torch
    from transformers import TrainerCallback

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.l2l_recipe import load_recipe

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 required")
    wrapper, dataset = _wrapper(tmp_path, monkeypatch, "cuda", 60_000)
    observed = []

    class ObserveStore(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            store = wrapper.trainer._soup_l2l.store
            assert isinstance(store, BudgetedActivationStore)
            assert 0 < plan.activations.used_bytes <= plan.activations.limit_bytes
            observed.append(store.k)

    try:
        wrapper.setup(dataset)
        recipe = wrapper._l2l_recipe
        plan = wrapper.trainer._soup_l2l.host_budget_plan
        before = {
            name: param.detach().clone()
            for name, param in wrapper.model.named_parameters() if param.requires_grad
        }
        wrapper.trainer.add_callback(ObserveStore())
        wrapper.train()
        assert (recipe.requested_k, recipe.effective_k) == (64, 2)
        assert observed == [2, 2, 2]
        assert wrapper.trainer.state.global_step == 3
        assert plan.activations.used_bytes == 0
        assert load_recipe(wrapper._output_dir) == recipe
        assert any(
            not torch.equal(before[name], param)
            for name, param in wrapper.model.named_parameters() if name in before
        )
        rows = [row for row in wrapper.trainer.state.log_history if "loss" in row]
        assert {row["l2l_effective_k"] for row in rows} == {2}
    finally:
        wrapper._close_stream_runtime()
