"""Real CPU Gloo/Accelerate training over uneven padded multipack bins (#3218)."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest


def _parameter_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _worker(rank, directory):
    import torch
    import torch.distributed as distributed
    from accelerate import Accelerator, PartialState
    from accelerate.data_loader import DataLoaderShard
    from transformers import LlamaConfig, LlamaForCausalLM
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    from soup_cli.utils.multipack_trainer import (
        attach_multipack_state,
        make_multipack_trainer_class,
    )

    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      ACCELERATE_USE_CPU="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    torch.set_num_threads(1)
    root = Path(directory)
    distributed.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    state = PartialState(cpu=True, backend="gloo")
    assert state.device.type == "cpu"
    accelerator = Accelerator(cpu=True)
    results = []
    try:
        for padding_multiple, lengths, global_batches in [
            (None, [65] + [2] * 32, [[0, 1], list(range(2, 33))]),
            (8, [65] + [2] * 32, [list(range(17, 33)), [0], list(range(1, 17))]),
            (None, [65, 65], [[0, 1]]),
        ]:
            observed = []

            class RecordingCollator(DataCollatorForLanguageModeling):
                def torch_call(self, examples):
                    batch = super().torch_call(examples)
                    observed.append({
                        "rows": [example["input_ids"][0] - 4 for example in examples],
                        "positions": batch["input_ids"].numel(),
                    })
                    return batch

            torch.manual_seed(7)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=64, hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, num_key_value_heads=2, pad_token_id=1, eos_token_id=2,
            ))
            before = _parameter_hash(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            model, optimizer = accelerator.prepare(model, optimizer)
            trainer = make_multipack_trainer_class(SimpleNamespace)(
                accelerator=accelerator,
                data_collator=RecordingCollator(pad_token_id=1,
                                                pad_to_multiple_of=padding_multiple),
                train_dataset=[
                    {"input_ids": [index + 4] * length,
                     "labels": [-100] + [index + 4] * (length - 1)}
                    for index, length in enumerate(lengths)
                ],
                args=SimpleNamespace(dataloader_drop_last=False, dataloader_pin_memory=False,
                                     dataloader_num_workers=0),
            )
            attach_multipack_state(trainer, lengths=lengths, max_seq_len=65, batch_size=2, seed=7)
            loader = trainer.get_train_dataloader()
            assert isinstance(loader, DataLoaderShard)
            extended = global_batches + global_batches[:len(global_batches) % 2]
            expected = extended[rank::2]
            assert len(loader) == len(expected)
            steps = 0
            for batch in loader:
                optimizer.zero_grad()
                loss = model(**batch).loss
                assert torch.isfinite(loss)
                accelerator.backward(loss)
                optimizer.step()
                steps += 1
            assert steps == len(expected)
            assert [batch["rows"] for batch in observed] == expected
            assert all(batch["positions"] <= 130 for batch in observed)
            after = _parameter_hash(accelerator.unwrap_model(model))
            assert after != before
            results.append({"steps": steps, "batches": observed,
                            "parameters": after, "expected_rows": len(lengths)})
            distributed.barrier()
        (root / f"rank-{rank}.json").write_text(json.dumps(results), encoding="utf-8")
    finally:
        distributed.destroy_process_group()


@pytest.mark.timeout(150)
def test_two_gloo_ranks_train_bounded_batches_with_even_tail_repeats(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("accelerate")
    if not torch.distributed.is_gloo_available():
        pytest.skip("CPU Gloo backend is unavailable")
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    ranks = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for first, second in zip(*ranks, strict=True):
        assert first["steps"] == second["steps"]
        assert first["parameters"] == second["parameters"]
        consumed = [row for result in (first, second) for batch in result["batches"]
                    for row in batch["rows"]]
        assert set(consumed) == set(range(first["expected_rows"]))
