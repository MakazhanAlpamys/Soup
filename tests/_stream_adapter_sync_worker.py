"""One rank of ``tests/test_stream_adapter_sync.py`` (a plain script, not collected).

Run as ``python _stream_adapter_sync_worker.py --rank R --world-size 2 --port P
--mode none|sync --steps N --out-dir D``, one process per rank, over gloo on
127.0.0.1. Each rank builds its OWN copy of the same tiny streamed Llama (same
seed, own shard dir and own ``SOUP_LAYER_STREAM_CACHE_DIR``, as two ranks on one
host would race one cache), randomises ``lora_B`` identically, then trains with
plain torch on RANK-SPECIFIC token rows. In mode ``sync`` it calls
``AdapterGradSync(model).sync()`` between ``backward()`` and ``optimizer.step()``.

It writes ``rank<R>.json`` (SHA-256 of the concatenated trainable tensors before
and after training, the last ``SyncStats``, and checks made around every sync) and
``rank<R>.safetensors`` (the trainable tensors). An uncaught error exits the
process at once so the peer's socket closes and the peer fails fast too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from datetime import timedelta
from pathlib import Path

_VOCAB = 64
_ROWS = 2
_TOKENS = 8
_SEED = 7
_LR = 1e-2


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("none", "sync"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=None)
    parser.add_argument(
        "--unused-on-rank1",
        action="store_true",
        help="rank 1 drops the first trainable parameter's grad before every sync",
    )
    return parser.parse_args(argv)


def _tiny_llama_dir(root: Path) -> str:
    import torch
    from safetensors.torch import save_file
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(_SEED)
    config = LlamaConfig(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        max_position_embeddings=128,
    )
    model = LlamaForCausalLM(config).to(torch.float32).eval()
    weights = root / "model"
    weights.mkdir(parents=True, exist_ok=True)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    state.pop("lm_head.weight", None)
    save_file(state, str(weights / "model.safetensors"))
    config.save_pretrained(str(weights))
    return str(weights)


def _build(root: Path):
    from peft import LoraConfig, TaskType

    from soup_cli.utils.layer_shard import shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import build_streamed_model

    weights = _tiny_llama_dir(root)
    shards = str(root / "shards")
    index = shard_checkpoint(weights, shards, dtype="float32", arch="llama")
    return build_streamed_model(
        model_id=weights,
        shard_dir=shards,
        index=index,
        lora_config=LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device="cpu",
        dtype="float32",
        buffers=2,
        pin=False,
        seed=3,
    )


def _randomise_lora_b(model) -> None:
    """PEFT starts lora_B at zero, which zeroes every lora_A gradient (a vacuous step)."""
    import torch

    generator = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator).to(param.dtype))


def _trainable(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _digest(named) -> str:
    import torch

    sha = hashlib.sha256()
    for _name, param in named:
        sha.update(param.detach().to(torch.float32).contiguous().numpy().tobytes())
    return sha.hexdigest()


def _flat_grads(params):
    """This rank's gradients as one fp32 vector (zeros where ``grad is None``)."""
    import torch

    return torch.cat(
        [
            (torch.zeros_like(param) if param.grad is None else param.grad).reshape(-1).float()
            for param in params
        ]
    )


def _expected_mean(dist, params, world_size):
    """The mean of every rank's pre-sync gradient, by a route that avoids AdapterGradSync.

    Returns ``(expected, ranks_differ, nonzero)``: the mean vector, whether the ranks'
    own gradients differ, and whether this rank's gradient is not all zeros (so that
    the comparison is not vacuous). ``all_gather`` then ``sum(0) / world_size``
    performs, for two ranks, the very additions the SUM all-reduce does.
    """
    import torch

    mine = _flat_grads(params)
    gathered = [torch.empty_like(mine) for _ in range(world_size)]
    dist.all_gather(gathered, mine)
    stacked = torch.stack(gathered)
    ranks_differ = any(not torch.equal(gathered[0], other) for other in gathered[1:])
    return stacked.sum(0) / world_size, ranks_differ, bool(mine.abs().max().item() > 0)


def _run(args) -> None:
    out_dir = Path(args.out_dir)
    rank_root = out_dir / f"rank{args.rank}"
    rank_root.mkdir(parents=True, exist_ok=True)
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(rank_root / "cache")

    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file

    from soup_cli.utils.adapter_grad_sync import AdapterGradSync, SyncStats

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=120),
    )
    model, runtime = _build(rank_root)
    try:
        _randomise_lora_b(model)
        named = _trainable(model)
        params = [param for _name, param in named]
        n_trainable = sum(param.numel() for param in params)
        initial_sha = _digest(named)
        optimizer = torch.optim.AdamW(params, lr=_LR, weight_decay=0.0)
        sync = None
        if args.mode == "sync":
            kwargs = {} if args.chunk_bytes is None else {"chunk_bytes": args.chunk_bytes}
            sync = AdapterGradSync(model, **kwargs)
        data = torch.Generator().manual_seed(1000 + args.rank)
        dist.barrier()

        stats = SyncStats()
        losses = []
        matches_mean = True
        grads_nonzero = True
        ranks_differ_every_step = True
        for _step in range(args.steps):
            ids = torch.randint(0, _VOCAB, (_ROWS, _TOKENS), generator=data)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            losses.append(float(loss.detach()))
            if sync is not None:
                if args.unused_on_rank1 and args.rank == 1:
                    params[0].grad = None
                expected, ranks_differ, nonzero = _expected_mean(dist, params, args.world_size)
                ranks_differ_every_step &= ranks_differ
                grads_nonzero &= nonzero
                stats = sync.sync()
                matches_mean &= bool(torch.equal(_flat_grads(params), expected))
            optimizer.step()

        final_sha = _digest(named)
        save_file(
            {name: param.detach().contiguous().clone() for name, param in named},
            str(out_dir / f"rank{args.rank}.safetensors"),
        )
        result = {
            "rank": args.rank,
            "mode": args.mode,
            "steps": args.steps,
            "n_trainable": n_trainable,
            "n_params": len(params),
            "initial_sha": initial_sha,
            "final_sha": final_sha,
            "losses": losses,
            "stats": {
                "bytes": stats.bytes,
                "seconds": stats.seconds,
                "n_params": stats.n_params,
                "n_chunks": stats.n_chunks,
            },
            "matches_mean": matches_mean,
            "grads_nonzero": grads_nonzero,
            "ranks_differ_every_step": ranks_differ_every_step,
        }
        (out_dir / f"rank{args.rank}.json").write_text(json.dumps(result), encoding="utf-8")
        dist.barrier()
    finally:
        runtime.close()
        dist.destroy_process_group()


def main(argv=None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        _run(args)
    except BaseException:  # die loudly, so the peer's socket closes and it fails fast too
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    print(f"rank {args.rank}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
