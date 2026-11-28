"""Grouped expert execution through the regular streamed L2L model."""

from __future__ import annotations

import pytest

from tests.test_l2l_moe import _assert_same, _build, _l2l, _micro_batches, _yaml


def _config(**training):
    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(_yaml(**training)).training


def test_grouped_experts_default_off():
    assert _config().stream_experts is None


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"stream_expert_layout": "per_expert"},
            "training.stream_experts requires training.l2l=true",
        ),
        (
            {"l2l": True},
            "training.stream_experts requires training.stream_expert_layout='per_expert'",
        ),
        (
            {"l2l": True, "stream_expert_layout": "per_expert", "stream_layers": False},
            "training.stream_experts requires training.stream_layers=true",
        ),
    ],
)
def test_grouped_experts_names_the_missing_requirement(settings, message):
    with pytest.raises(ValueError) as caught:
        _config(stream_experts={}, **settings)
    assert str(caught.value) == f": Value error, {message}"


@pytest.mark.parametrize(
    "field",
    ["group_width", "weight_budget_bytes", "activation_budget_bytes", "call_budget_bytes",
     "native_rows_max", "native_rows_from"],
)
def test_grouped_experts_refuses_boolean_integers(field):
    message = f"training.stream_experts.{field} must be an int, not a bool"
    with pytest.raises(ValueError) as caught:
        _config(l2l=True, stream_expert_layout="per_expert", stream_experts={field: True})
    assert str(caught.value) == f"training -> stream_experts -> {field}: Value error, {message}"


def test_resume_checks_grouped_expert_recipe(tmp_path, monkeypatch):
    from soup_cli.utils.l2l_recipe import (
        resolve_l2l_recipe,
        validate_resume_source,
        write_recipe,
    )

    monkeypatch.chdir(tmp_path)
    common = {"l2l": True, "stream_expert_layout": "per_expert"}
    default = resolve_l2l_recipe(_config(**common))
    group = resolve_l2l_recipe(_config(stream_experts={}, **common))
    changed = resolve_l2l_recipe(_config(stream_experts={"group_width": 2}, **common))
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="checkpoint has no L2L recipe"):
        validate_resume_source(checkpoint, group)
    write_recipe(tmp_path, default)
    with pytest.raises(ValueError, match=r"cannot resume.*training\.stream_experts=None"):
        validate_resume_source(checkpoint, group)
    write_recipe(tmp_path, group)
    assert validate_resume_source(checkpoint, group) == group
    with pytest.raises(ValueError, match="checkpoint used training.stream_experts"):
        validate_resume_source(checkpoint, None)
    with pytest.raises(ValueError, match=r"cannot resume.*training\.stream_experts="):
        validate_resume_source(checkpoint, changed)


@pytest.mark.parametrize("two_reads", [False, True])
@pytest.mark.parametrize("bf16", [False, True], ids=["fp32", "bf16-autocast"])
def test_cpu_l2l_grouped_trained_steps_equal_per_expert(tmp_path, two_reads, bf16):
    import torch

    from soup_cli.config.schema import ExpertStreamConfig

    grouped_dir = tmp_path / "grouped"
    grouped_dir.mkdir()
    common = dict(expert_pool=True, two_read_experts=two_reads, grouped_expert_routes=True)
    plain, plain_runtime = _build(tmp_path, **common)
    try:
        grouped, grouped_runtime = _build(
            grouped_dir, stream_experts=ExpertStreamConfig(), **common,
        )
        try:
            models = plain, grouped
            parameters = [
                {name: param for name, param in model.named_parameters() if param.requires_grad}
                for model in models
            ]
            assert parameters[0].keys() == parameters[1].keys()
            initial = {name: p.detach().clone() for name, p in parameters[0].items()}
            optimizers = [torch.optim.SGD(values.values(), lr=0.25) for values in parameters]
            for run in range(2):
                batches = [ids.add(run) % 128 for ids in _micro_batches(2)]
                baseline = _l2l(plain, plain_runtime, batches, bf16)
                result = _l2l(grouped, grouped_runtime, batches, bf16)
                _assert_same(result[:2], baseline[:2])
                assert result[2].counters.expert_fetches == baseline[2].counters.expert_fetches
                for optimizer in optimizers:
                    optimizer.step()
                for name, parameter in parameters[0].items():
                    assert torch.equal(parameter, parameters[1][name]), name
            assert any(not torch.equal(p, initial[n]) for n, p in parameters[0].items())
            assert grouped_runtime.expert_pool.group_width == 4
            assert grouped_runtime.expert_pool.groups_loaded > 0
        finally:
            grouped_runtime.close()
    finally:
        plain_runtime.close()


def test_call_budget_uses_widest_bucket_and_all_group_members():
    from soup_cli.utils.expert_training import group_call_bytes

    shapes = {"gate_up": (2048, 2048), "down": (2048, 1024)}
    assert group_call_bytes([128] * 4, shapes, packed=False) == 4 * 128 * 2048 * 2 + 1536
    assert group_call_bytes([512] * 4, shapes, packed=False) == 4 * 512 * 2048 * 2 + 1536
    assert group_call_bytes([512, 1], shapes, packed=True) > 512 * 2048 * 2


def test_grouped_sft_can_preserve_hf_tensor_denominator():
    import torch

    from soup_cli.trainer.l2l_trainer import SFT_WINDOW_LOSS

    supplied = torch.tensor(13)
    assert SFT_WINDOW_LOSS.divisor(2, supplied) == 13
    assert SFT_WINDOW_LOSS.divisor(2, supplied, preserve_item_tensor=True) is supplied


@pytest.mark.parametrize("mode", ["split", "ordered"])
def test_grouped_compute_keeps_micro_batch_route_segments(tmp_path, mode):
    from soup_cli.config.schema import ExpertStreamConfig
    from tests.test_l2l_mbr import _capture_window, _run

    directory = tmp_path / "grouped"
    directory.mkdir()
    common = dict(expert_pool=True, two_read_experts=True, grouped_expert_routes=True)
    plain, plain_runtime = _build(tmp_path, **common)
    try:
        grouped, grouped_runtime = _build(directory, stream_experts=ExpertStreamConfig(), **common)
        try:
            options = dict(
                route_preserving_mbr=True, mbr_group_size=3,
                mbr_split_backward=mode == "split", mbr_ordered_backward=mode == "ordered",
                collect_phase_stats=True,
            )
            baseline = _capture_window(plain, plain_runtime)
            candidate = _capture_window(grouped, grouped_runtime)
            _assert_same(_run(candidate, **options), _run(baseline, **options))
            assert candidate[1].counters.expert_fetches == baseline[1].counters.expert_fetches
            assert grouped_runtime.expert_pool.groups_loaded > 0
        finally:
            grouped_runtime.close()
    finally:
        plain_runtime.close()


def test_budget_refusal_precedes_any_group_read(tmp_path):
    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.expert_stream import streamed_experts_class

    model, runtime = _build(
        tmp_path, expert_pool=True, stream_experts=ExpertStreamConfig(call_budget_bytes=1),
    )
    try:
        import torch

        experts = next(module for module in model.modules()
                       if isinstance(module, streamed_experts_class()))
        with pytest.raises(ValueError, match="training.stream_experts.call_budget_bytes"):
            experts(torch.zeros(4, 64), torch.zeros(4, 1, dtype=torch.long), torch.ones(4, 1))
        assert runtime.expert_pool.fetches == runtime.expert_pool.logical_bytes_read == 0
    finally:
        runtime.close()


def test_olmoe_fused_checkpoint_uses_the_same_l2l_pool(tmp_path):
    import torch

    pytest.importorskip("peft")
    from peft import LoraConfig, TaskType
    from safetensors.torch import save_file
    from transformers import OlmoeConfig, OlmoeForCausalLM

    from soup_cli.config.schema import ExpertStreamConfig
    from soup_cli.utils.layer_shard import (
        _discover_safetensors,
        estimate_shard_cache_bytes,
        shard_checkpoint,
    )
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    torch.manual_seed(11)
    base = OlmoeForCausalLM(OlmoeConfig(
        vocab_size=128, hidden_size=64, intermediate_size=32, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, num_experts=4, num_experts_per_tok=2,
        tie_word_embeddings=False, output_router_logits=False,
    ))
    weights = tmp_path / "weights"
    weights.mkdir()
    base.config.save_pretrained(weights)
    save_file(base.state_dict(), str(weights / "model.safetensors"))
    shard_dir = str(tmp_path / "shards")
    cache = dict(dtype="float32", arch="olmoe", expert_layout="per_expert")
    estimate = estimate_shard_cache_bytes(_discover_safetensors(str(weights)), **cache)
    assert estimate.expert_file_bytes > 0
    index = shard_checkpoint(str(weights), shard_dir, **cache)
    assert tuple(index.expert_layers) == (0, 1, 2)
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    def trained_result(backend):
        model, runtime = build_streamed_model(
            model_id=str(weights), shard_dir=shard_dir, index=index, lora_config=lora,
            device="cpu", dtype="float32", pin=False, seed=7, expert_pool=True,
            stream_experts=ExpertStreamConfig(backend=backend),
        )
        try:
            return _l2l(model, runtime, _micro_batches(2), False)[:2]
        finally:
            runtime.close()

    _assert_same(trained_result("grouped"), trained_result("sequential"))
