"""Actual layer-major backward with leased activations and streamed weights."""

import pytest


def test_short_spill_chunk_charges_both_full_sectors_and_alignment_slack(tmp_path):
    import torch

    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_activations import BufferedSpillFile

    path = tmp_path / "spill.bin"
    options = dict(
        n_layers=1, k=1, chunk_shape=(17,), dtype=torch.float32, device="cpu",
        pin=False, spill_layers=1, ring=1,
        spill_factory=lambda size: BufferedSpillFile(str(path), size),
    )
    with pytest.raises(MemoryError) as excinfo:
        BudgetedActivationStore(host_budget=HostMemoryBudget(16383), **options)
    assert str(excinfo.value) == "l2l.activations: needs 16384 bytes; 16383 available"
    assert not path.exists()

    budget = HostMemoryBudget(16384)
    store = BudgetedActivationStore(host_budget=budget, **options)
    try:
        assert budget.used_bytes == 16384
        value = torch.arange(17, dtype=torch.float32)
        store.put(0, 0, value)
        torch.testing.assert_close(store.get(0, 0), value, rtol=0, atol=0)
    finally:
        store.close()
    assert budget.used_bytes == 0
    assert not path.exists()


@pytest.mark.parametrize("spill_layers", [0, 2, 4])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)])
def test_shared_runtime_l2l_matches_disk_l2l(tmp_path, spill_layers, device):
    import torch
    from peft import LoraConfig, TaskType
    from transformers import LlamaConfig, LlamaForCausalLM

    from benchmarks.harness import l2l_schedule
    from soup_cli.utils.budgeted_activations import BudgetedActivationStore
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.l2l_activations import BufferedSpillFile, DirectSpillFile
    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model
    from soup_cli.utils.partial_ram_source import PartialRamBudget

    cuda = device.startswith("cuda")
    backend = DirectSpillFile if cuda else BufferedSpillFile
    torch.set_num_threads(2)
    torch.manual_seed(7)
    weights = tmp_path / "weights"
    base = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                tie_word_embeddings=False,
                attention_dropout=0,
            )
        )
        .to(torch.float32)
        .eval()
    )
    base.save_pretrained(weights, safe_serialization=True)
    del base
    directory = tmp_path / "shards"
    index = shard_checkpoint(str(weights), str(directory), dtype="float32", arch="llama")
    results = []
    for cached in (False, True):
        budget = HostMemoryBudget(10_000_000, reserved_bytes=500)
        # Reserve/allocate activations FIRST. Optional cache must fit what remains.
        store = BudgetedActivationStore(
            host_budget=budget,
            n_layers=4,
            k=2,
            chunk_shape=(1, 32, 32),
            dtype=torch.float32,
            device=device,
            pin=cuda,
            spill_layers=spill_layers,
            ring=2,
            spill_factory=lambda size: backend(str(tmp_path / "spill.bin"), size),
        )
        activation_bytes = budget.used_bytes - 500
        runtime = None
        try:
            model, runtime = build_streamed_model(
                model_id=str(weights),
                shard_dir=str(directory),
                index=index,
                lora_config=LoraConfig(
                    r=4,
                    lora_alpha=8,
                    lora_dropout=0,
                    bias="none",
                    target_modules=["q_proj", "v_proj"],
                    task_type=TaskType.CAUSAL_LM,
                ),
                device=device,
                dtype="float32",
                pin=cuda,
                seed=3,
                tier="disk",
                read_ahead=1,
                partial_ram_budget=PartialRamBudget(
                    10_000_000,
                    500,
                    (65_536 if cuda else 45_000) if cached else 0,
                ),
                shared_host_budget=budget,
            )
            assert runtime.source.host_budget is budget
            assert budget.used_bytes == 500 + activation_bytes + runtime.source.host_allocated_bytes
            if cached:
                assert 0 < len(runtime.source.resident_indices) < 4
            if cuda:
                assert runtime.source.pinned
                assert runtime.source._disk.direct_io
                if spill_layers:
                    assert store.stats.direct
            l2l_schedule.make_non_vacuous(model)
            initial = l2l_schedule.snapshot_adapters(model)
            optimizer = torch.optim.AdamW(l2l_schedule.lora_parameters(model).values(), lr=0.001)
            records = []
            for update in range(2):
                batches = [
                    ((torch.arange(32, device=device).reshape(1, -1) + mb + update) % 64)
                    for mb in range(2)
                ]
                optimizer.zero_grad()
                # Same schedule in both arms: only the weight cache changes.
                call = l2l_schedule.capture_layer_call(model, batches[0])
                losses = l2l_schedule.L2LStep(model, runtime, call).run(batches, store)
                gradients = l2l_schedule.collect_grads(model)
                assert any(torch.count_nonzero(value) for value in gradients.values())
                optimizer.step()
                records.append(([value.item() for value in losses], gradients))
            final = l2l_schedule.snapshot_adapters(model)
            assert any(not torch.equal(final[name], value) for name, value in initial.items())
            results.append((records, final))
        finally:
            if runtime is not None:
                runtime.close()
            assert budget.used_bytes == 500 + activation_bytes
            store.close()
            assert budget.used_bytes == 500
        assert not (tmp_path / "spill.bin").exists()
    for (loss_a, grad_a), (loss_b, grad_b) in zip(results[0][0], results[1][0], strict=True):
        assert loss_a == loss_b
        assert l2l_schedule.compare(grad_a, grad_b)["equal"]
    assert l2l_schedule.compare(results[0][1], results[1][1])["equal"]


def test_shared_budget_is_not_silently_ignored():
    from soup_cli.utils.host_memory_budget import HostMemoryBudget
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    with pytest.raises(ValueError) as excinfo:
        build_streamed_model(
            model_id="must-not-load",
            shard_dir="unused",
            index=None,
            lora_config=None,
            shared_host_budget=HostMemoryBudget(100),
        )
    assert str(excinfo.value) == "shared_host_budget requires an explicit partial_ram_budget"
