"""#1354 - StreamPrefetcher.prime skips resident layer 0 at step boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, TaskType
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig

from soup_cli.utils.layer_shard import shard_checkpoint
from soup_cli.utils.layer_stream_runtime import build_streamed_model
from tests.test_v07203 import _randomise_lora_b, _tiny_stream


def _tiny_stream_with_tie(tmp_path: Path, name: str, tie: bool, seed: int = 3, n_layers: int = 3):
    """Build a streamed model with explicit control over word embedding tying."""
    weights = tmp_path / ("model_tied" if tie else "model_untied")
    if not weights.exists():
        torch.manual_seed(7)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=64,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            tie_word_embeddings=tie,
            max_position_embeddings=128,
        )
        model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
        weights.mkdir(parents=True, exist_ok=True)
        state = {k: v.contiguous() for k, v in model.state_dict().items()}
        if tie:
            state.pop("lm_head.weight", None)
        save_file(state, str(weights / "model.safetensors"))
        config.save_pretrained(str(weights))

    lora = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    shards = str(tmp_path / name)
    index = shard_checkpoint(
        str(weights), shards, dtype="float32", arch="llama", quant="none", quant_device="cpu"
    )
    return build_streamed_model(
        model_id=str(weights),
        shard_dir=shards,
        index=index,
        lora_config=lora,
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=seed,
        tier="ram",
        quant="none",
    )


def test_decoder_loads_over_three_steps(tmp_path: Path) -> None:
    """A CPU test counts decoder loads over 3 steps: [4, 2, 2] on _tiny_stream.

    On main without the fix, step 1 and step 2 each made 3 loads (4, 3, 3) because
    prime() unconditionally re-loaded layer 0 into the slot that already held it.
    """
    model, runtime, _ = _tiny_stream(tmp_path, name="shards_loads", seed=3)
    ids = torch.randint(0, 64, (1, 12))
    model.train()

    loads_per_step = []
    for _ in range(3):
        before = runtime.pool.loads
        model(input_ids=ids, labels=ids).loss.backward()
        loads_per_step.append(runtime.pool.loads - before)

    assert loads_per_step == [4, 2, 2]


def test_unpatched_main_behavior_produces_three_loads_per_subsequent_step(tmp_path: Path) -> None:
    """Unpatched prime() reproducing main yields [4, 3, 3] loads, failing [4, 2, 2]."""
    model, runtime, _ = _tiny_stream(tmp_path, name="shards_unpatched", seed=3)

    def unpatched_prime():
        runtime.prefetcher.prev = None
        runtime.prefetcher.direction = 1
        runtime.prefetcher.primes += 1
        runtime.prefetcher.backward_tail_prefetched = False
        runtime.prefetcher.tail_prefetched = False
        runtime.prefetcher.pool.load_async(
            0, runtime.prefetcher.source, runtime.prefetcher.stream
        )

    runtime.prefetcher.prime = unpatched_prime
    ids = torch.randint(0, 64, (1, 12))
    model.train()

    loads_per_step = []
    for _ in range(3):
        before = runtime.pool.loads
        model(input_ids=ids, labels=ids).loss.backward()
        loads_per_step.append(runtime.pool.loads - before)

    assert loads_per_step == [4, 3, 3]


def test_training_step_following_eval_loads_layer_zero(tmp_path: Path) -> None:
    """A training step that follows an eval or no-grad forward still loads layer 0.

    Eval forward walks 0..L-1 without a backward recompute, so layer 0 is evicted
    and not resident when the next training step begins.
    """
    model, runtime, _ = _tiny_stream(tmp_path, name="shards_eval", seed=3)
    ids = torch.randint(0, 64, (1, 12))
    model.train()

    # Step 0: normal training step (leaves layer 0 resident in slot 0)
    model(input_ids=ids, labels=ids).loss.backward()
    assert runtime.pool.owner[runtime.pool.slot_for(0)] == 0

    # Eval / no-grad forward: walks forward only, evicting layer 0
    model.eval()
    with torch.no_grad():
        model(input_ids=ids)
    assert runtime.pool.owner[runtime.pool.slot_for(0)] != 0

    # Step 1: training step following eval must reload layer 0 at prime()
    model.train()
    before = runtime.pool.loads
    model(input_ids=ids, labels=ids).loss.backward()
    after = runtime.pool.loads
    assert after - before == 3  # prime(0) + forward(2) + backward(0) = 3 loads


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
@pytest.mark.parametrize("task", ["sft", "dpo", "kto"])
def test_gradients_and_adapters_are_torch_equal(
    tmp_path: Path, tie_word_embeddings: bool, task: str
) -> None:
    """Gradients and adapters are bit-exact torch.equal between unpatched and patched prime.

    Covers SFT, DPO, and KTO under both tied and untied architectures.
    """
    p1 = tmp_path / ("m1_tied" if tie_word_embeddings else "m1_untied")
    p2 = tmp_path / ("m2_tied" if tie_word_embeddings else "m2_untied")

    model1, rt1 = _tiny_stream_with_tie(p1, name=f"s1_{task}", tie=tie_word_embeddings, seed=3)
    model2, rt2 = _tiny_stream_with_tie(p2, name=f"s2_{task}", tie=tie_word_embeddings, seed=3)

    _randomise_lora_b(model1, seed=42)
    _randomise_lora_b(model2, seed=42)

    # In model1, unpatch prime to simulate main's unconditional load_async(0)
    def unpatched_prime():
        rt1.prefetcher.prev = None
        rt1.prefetcher.direction = 1
        rt1.prefetcher.primes += 1
        rt1.prefetcher.backward_tail_prefetched = False
        rt1.prefetcher.tail_prefetched = False
        rt1.prefetcher.pool.load_async(0, rt1.prefetcher.source, rt1.prefetcher.stream)

    rt1.prefetcher.prime = unpatched_prime

    opt1 = torch.optim.AdamW([p for p in model1.parameters() if p.requires_grad], lr=1e-3)
    opt2 = torch.optim.AdamW([p for p in model2.parameters() if p.requires_grad], lr=1e-3)

    for step in range(3):
        torch.manual_seed(200 + step)
        ids = torch.randint(0, 64, (2, 8))

        if task == "sft":
            loss1 = model1(input_ids=ids[:1], labels=ids[:1]).loss
            loss2 = model2(input_ids=ids[:1], labels=ids[:1]).loss
        elif task == "dpo":
            # DPO uses concatenated forward (batch size 2 = [chosen, rejected])
            o1, o2 = model1(input_ids=ids).logits, model2(input_ids=ids).logits
            logps1 = (
                o1[:, :-1]
                .log_softmax(-1)
                .gather(2, ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
            )
            logps2 = (
                o2[:, :-1]
                .log_softmax(-1)
                .gather(2, ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
            )
            loss1 = -torch.nn.functional.logsigmoid(0.1 * (logps1[0] - logps1[1]))
            loss2 = -torch.nn.functional.logsigmoid(0.1 * (logps2[0] - logps2[1]))
        else:  # kto
            o1, o2 = model1(input_ids=ids).logits, model2(input_ids=ids).logits
            lp1 = (
                o1[:, :-1]
                .log_softmax(-1)
                .gather(2, ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
            )
            lp2 = (
                o2[:, :-1]
                .log_softmax(-1)
                .gather(2, ids[:, 1:].unsqueeze(-1))
                .squeeze(-1)
                .sum(-1)
            )
            loss1 = (1 - torch.sigmoid(0.1 * lp1[0])) + torch.sigmoid(0.1 * lp1[1])
            loss2 = (1 - torch.sigmoid(0.1 * lp2[0])) + torch.sigmoid(0.1 * lp2[1])

        loss1.backward()
        opt1.step()
        opt1.zero_grad()

        loss2.backward()
        opt2.step()
        opt2.zero_grad()

    for (n1, p1_param), (n2, p2_param) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        if p1_param.requires_grad:
            assert torch.equal(
                p1_param, p2_param
            ), f"Adapter weight mismatch in {n1} for task={task}, tie={tie_word_embeddings}"


@pytest.mark.parametrize("read_ahead", [2, 3])
def test_async_disk_source_no_new_demand_misses(tmp_path: Path, read_ahead: int) -> None:
    """AsyncDiskSource at read-ahead 2 and 3 shows high hit rate without reload."""
    model, runtime, _ = _tiny_stream(
        tmp_path, name=f"disk_ra{read_ahead}", seed=3, tier="disk", read_ahead=read_ahead
    )
    source = runtime.source

    hits = []
    orig_get = source.get

    def logging_get(idx: int, name: str):
        with source._ready:
            hit = idx in source._slot_of
        hits.append(hit)
        return orig_get(idx, name)

    source.get = logging_get

    ids = torch.randint(0, 64, (1, 12))
    model.train()
    for _ in range(3):
        model(input_ids=ids, labels=ids).loss.backward()
        # After each step backward completes, slot 0 holds layer 0
        assert runtime.pool.owner[runtime.pool.slot_for(0)] == 0

    # Cold start on step 0 produces a few initial reads; overall hits dominate (>= 60 of 72)
    assert sum(hits) >= (65 if read_ahead == 3 else 60)
