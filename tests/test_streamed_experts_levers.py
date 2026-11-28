"""Ordinary versus complete expert-streaming levers on a tiny DeepSeek-V3 (CPU).

One checkpoint is sharded per expert and streamed five ways with equal adapters: the fetcher
path, the pool path, the pool with grouped routes, the pool with two reads and the fast row
gather, and the pool with every lever. Next to them runs the resident eager model. Loss and every
LoRA gradient are compared with ``torch.equal``; reads come from ``runtime.expert_pool.fetches``
(the fetcher path counts them on its readers).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")

from tests.test_expert_stream import _lora, _shard, _tiny_deepseek_v3  # noqa: E402

#: name -> (build_streamed_model keywords, fast_row_gather)
PATHS: Dict[str, Tuple[Dict[str, Any], bool]] = {
    "fetcher": ({}, False),
    "pool": (dict(expert_pool=True), False),
    "pool-grouped": (dict(expert_pool=True, grouped_expert_routes=True), False),
    "pool-two-reads-fast": (dict(expert_pool=True, two_read_experts=True), True),
    "pool-all-levers": (
        dict(expert_pool=True, two_read_experts=True, grouped_expert_routes=True), True,
    ),
}


def _lora_grads(net: Any) -> Dict[str, Any]:
    return {
        name.replace(".inner.", "."): parameter.grad.detach().clone()
        for name, parameter in net.named_parameters()
        if "lora_" in name and parameter.grad is not None
    }


def _reads(runtime: Any) -> int:
    return runtime.stats()["expert_fetches"]


@pytest.fixture(scope="module")
def nets(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from peft import get_peft_model
    from transformers import DeepseekV3ForCausalLM

    from soup_cli.utils.expert_stream import streamed_experts_class
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    tmp_path = tmp_path_factory.mktemp("levers")
    weights, _ = _tiny_deepseek_v3(tmp_path)
    index = _shard(tmp_path, weights, expert_layout="per_expert")
    streamed: Dict[str, Tuple[Any, Any]] = {}
    for name, (kwargs, fast) in PATHS.items():
        model, runtime = build_streamed_model(
            model_id=weights, shard_dir=str(tmp_path / "shards"), index=index,
            lora_config=_lora(), device="cpu", dtype="float32", buffers=2, pin=False, seed=3,
            **kwargs,
        )
        for module in model.modules():
            if isinstance(module, streamed_experts_class()):
                module.fast_row_gather = fast
        streamed[name] = (model, runtime)
    first = streamed["fetcher"][0]
    with torch.no_grad():  # integer-arithmetic adapter values, equal on every path
        for key, parameter in first.named_parameters():
            if "lora_B" in key:
                raw = (torch.arange(parameter.numel(), dtype=torch.int64) * 7 + 3) % 11 - 5
                parameter.copy_(raw.to(torch.float32).reshape(parameter.shape) / 256)
    source = {
        key: value for key, value in first.state_dict().items() if "lora_" in key
    }
    for name, (model, _) in streamed.items():
        if name == "fetcher":
            continue
        target = model.state_dict()
        for key, value in source.items():
            target[key].copy_(value)
    resident = DeepseekV3ForCausalLM.from_pretrained(weights, dtype=torch.float32)
    resident.set_experts_implementation("eager")
    reference = get_peft_model(resident, _lora())
    target = {key: value for key, value in reference.state_dict().items() if "lora_" in key}
    renamed = {key.replace(".inner.", "."): value for key, value in source.items()}
    assert renamed and set(renamed) == set(target)
    for key, value in renamed.items():
        target[key].copy_(value)
    for model, _ in streamed.values():
        model.eval()
    reference.eval()
    ids = (torch.arange(36, dtype=torch.int64).reshape(3, 12) * 5 + 1) % 128
    yield reference, streamed, ids
    for _, runtime in streamed.values():
        runtime.close()


def _step(net: Any, ids: Any) -> Tuple[Any, Dict[str, Any]]:
    net.zero_grad(set_to_none=True)
    loss = net(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.detach(), _lora_grads(net)


def test_every_lever_combination_equals_the_resident_model_and_the_default_path(
    nets: Any,
) -> None:
    reference, streamed, ids = nets
    want_loss, want = _step(reference, ids)
    assert want and all(float(grad.abs().sum()) > 0.0 for grad in want.values())
    for name, (model, _) in streamed.items():
        loss, got = _step(model, ids)
        assert torch.equal(loss, want_loss), name
        assert set(got) == set(want), name
        assert all(torch.equal(got[key], want[key]) for key in want), name


def test_two_reads_read_each_hit_expert_twice_where_the_default_reads_it_three_times(
    nets: Any,
) -> None:
    _, streamed, ids = nets
    reads = {}
    for name in ("pool", "pool-two-reads-fast", "pool-all-levers"):
        model, runtime = streamed[name]
        model.zero_grad(set_to_none=True)
        before = _reads(runtime)
        loss = model(input_ids=ids, labels=ids).loss
        forward = _reads(runtime) - before
        loss.backward()
        reads[name] = (forward, _reads(runtime) - before)
    forward = reads["pool"][0]
    assert forward > 0
    assert reads["pool"] == (forward, 3 * forward)
    assert reads["pool-two-reads-fast"] == (forward, 2 * forward)
    assert reads["pool-all-levers"] == (forward, 2 * forward)


def test_bf16_autocast_steps_are_bit_equal_between_the_pool_and_the_fetcher(nets: Any) -> None:
    _, streamed, ids = nets
    results = {}
    for name in ("fetcher", "pool", "pool-all-levers"):
        model, _ = streamed[name]
        model.zero_grad(set_to_none=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        results[name] = (loss.detach(), _lora_grads(model))
    want_loss, want = results["fetcher"]
    for name in ("pool", "pool-all-levers"):
        loss, got = results[name]
        assert torch.equal(loss, want_loss), name
        assert set(got) == set(want) and all(torch.equal(got[key], want[key]) for key in want), name
