"""One process of ``tests/test_stream_distributed_trainer.py`` (a plain script, not collected).

Run as ``python _stream_distributed_trainer_worker.py --rank R --world-size N --tag T
--out-dir D --base-dir B --rows-file F --batch-size b [--grad-accum g] [--max-grad-norm x]``.
The parent sets the launcher environment: for a rank of a two-process run ``RANK``,
``WORLD_SIZE=2``, ``LOCAL_RANK=0``, ``LOCAL_WORLD_SIZE=1`` (one rank per host),
``MASTER_ADDR``/``MASTER_PORT`` and ``ACCELERATE_USE_CPU=true`` (so accelerate's state is
``MULTI_CPU`` and creates the gloo group itself, from the environment); for the one-rank
reference none of them, and ``--no-rewind-log``: the flight recorder is on by default and a
distributed launch switches it off by itself, so the reference turns it off in its config to
run the same plain SFT step.

What it runs is the production path, on CPU: ``SFTTrainerWrapper(cfg, device="cpu")
.setup(...)`` (config -> ``_setup_streaming_transformers`` -> ``build_streamed_model`` ->
the SFT trainer class) and then ``wrapper.trainer.train()``, i.e. transformers' own
``Trainer`` loop. Three things are touched after ``setup`` and are not production: the
train sampler is made sequential (the equivalence needs a fixed row order that a 1-rank
and a 2-rank run share), ``lora_B`` is randomised (PEFT starts it at zero, which zeroes
every ``lora_A`` gradient in step one), and a callback records the adapters after every
optimizer step. ``wrapper.train()`` is not used: it adds the display/tracker callbacks and
the final save, none of which is under test.

It writes ``<tag>.json`` (facts and logged metrics), ``<tag>.safetensors`` (the trainable
tensors after training, plus ``per_step``: one row of all of them per optimizer step).
An uncaught error exits the process at once so the peer's socket closes and the peer
fails fast too.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

_SEED = 7
_LORA_B_SCALE = 0.05


def _parse(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--rows-file", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1000.0)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--no-rewind-log",
        action="store_true",
        help="the one-rank reference runs the plain SFT step (the recorder is on by default "
        "and a distributed launch switches it off)",
    )
    return parser.parse_args(argv)


def _config(args, rank_root: Path):
    import yaml

    from soup_cli.config.loader import load_config_from_string

    return load_config_from_string(
        yaml.safe_dump(
            {
                "base": args.base_dir,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": 128, "chat_template": "chatml"},
                "training": {
                    "batch_size": args.batch_size,
                    "gradient_accumulation_steps": args.grad_accum,
                    "quantization": "none",
                    "stream_layers": True,
                    "epochs": 1,
                    "lr": args.lr,
                    # Plain SGD on purpose: Adam divides the gradient by its own running
                    # scale, so a gradient that is 2x too large (a SUM instead of a mean)
                    # would give almost the same update and slip through.
                    "optimizer": "sgd",
                    "weight_decay": 0.0,
                    "scheduler": "constant",
                    "warmup_ratio": 0.0,
                    "max_grad_norm": args.max_grad_norm,
                    "logging_steps": 1,
                    "save_steps": 1000,
                    "seed": 42,
                    "rewind_log": not args.no_rewind_log,
                    "lora": {
                        "r": 4,
                        "alpha": 8,
                        "dropout": 0.0,
                        "target_modules": ["q_proj", "v_proj"],
                    },
                },
                "output": str(rank_root / "out"),
            }
        )
    )


def _trainable(model):
    return [(name, param) for name, param in model.named_parameters() if param.requires_grad]


def _digest(named) -> str:
    import torch

    sha = hashlib.sha256()
    for _name, param in named:
        sha.update(param.detach().to(torch.float32).contiguous().numpy().tobytes())
    return sha.hexdigest()


def _flat(named):
    import torch

    return torch.cat([param.detach().to(torch.float32).reshape(-1) for _n, param in named])


def _run(args) -> None:
    out_dir = Path(args.out_dir)
    rank_root = out_dir / args.tag
    rank_root.mkdir(parents=True, exist_ok=True)
    # Two processes on one machine would race one shard cache (they are two hosts in life).
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(rank_root / "cache")

    import torch
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback

    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.adapter_sync_trainer import has_adapter_sync, is_accelerate_prepared

    torch.set_num_threads(1)
    rows = json.loads(Path(args.rows_file).read_text(encoding="utf-8"))
    cfg = _config(args, rank_root)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    try:
        wrapper.setup({"train": rows})
        trainer = wrapper.trainer
        # Read before train(): accelerate's prepare() flags every model it prepared, so
        # after training the flag is set on the 1-rank reference too. Only the value
        # after setup tells whether Soup's C4 bypass ran.
        prepared_after_setup = bool(is_accelerate_prepared(wrapper.model))
        # A fixed row order: row i is the same row in the 1-rank and the 2-rank run.
        trainer._get_train_sampler = lambda train_dataset=None: SequentialSampler(
            trainer.train_dataset if train_dataset is None else train_dataset
        )
        named = _trainable(wrapper.model)
        generator = torch.Generator().manual_seed(_SEED)
        with torch.no_grad():
            for name, param in named:
                if "lora_B" in name:
                    param.copy_(
                        (_LORA_B_SCALE * torch.randn(param.shape, generator=generator)).to(
                            param.dtype
                        )
                    )
        initial_sha = _digest(named)
        per_step = []

        class _Record(TrainerCallback):
            def on_step_end(self, _args, _state, _control, **_kwargs):
                per_step.append(_flat(named))

        trainer.add_callback(_Record())
        trainer.train()

        history = trainer.state.log_history
        stats = getattr(trainer, "last_sync_stats", None)
        state = trainer.accelerator
        facts = {
            "tag": args.tag,
            "rank": args.rank,
            "world_size": int(trainer.args.world_size),
            "process_index": int(trainer.args.process_index),
            "distributed_type": str(state.distributed_type),
            "distributed_stream": bool(wrapper._distributed_stream),
            "trainer_class": type(trainer).__name__,
            "has_adapter_sync": bool(has_adapter_sync(trainer)),
            "accelerate_prepared": prepared_after_setup,
            "ddp_wrapped": isinstance(
                trainer.model_wrapped, torch.nn.parallel.DistributedDataParallel
            ),
            "accepts_loss_kwargs": bool(trainer.model_accepts_loss_kwargs),
            "average_tokens_across_devices": bool(trainer.args.average_tokens_across_devices),
            "global_step": int(trainer.state.global_step),
            "n_trainable": sum(param.numel() for _n, param in named),
            "n_tensors": len(named),
            "initial_sha": initial_sha,
            "final_sha": _digest(named),
            "losses": [entry["loss"] for entry in history if "loss" in entry],
            "grad_norms": [entry["grad_norm"] for entry in history if "grad_norm" in entry],
            "sync_calls": getattr(trainer, "sync_calls", None),
            "sync_bytes": None if stats is None else stats.bytes,
            "sync_chunks": None if stats is None else stats.n_chunks,
            "sync_seconds": None if stats is None else stats.seconds,
        }
        tensors = {name: param.detach().contiguous().clone() for name, param in named}
        tensors["per_step"] = torch.stack(per_step).contiguous()
        save_file(tensors, str(out_dir / f"{args.tag}.safetensors"))
        (out_dir / f"{args.tag}.json").write_text(json.dumps(facts), encoding="utf-8")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    finally:
        wrapper._close_stream_runtime()
        if dist.is_available() and dist.is_initialized():
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
    print(f"{args.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
