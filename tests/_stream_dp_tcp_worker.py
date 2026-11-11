"""One process of ``tests/test_stream_dp_tcp_transport.py`` (a plain script, not collected).

Two subcommands.

``transport`` runs the TCP group alone, with no torch: ``--rank R --world-size W --port P
--rounds N`` builds a ``TcpStarGroup`` (rank 0 listens on ``--port``, the others connect to it),
runs a barrier and ``N`` object gathers, and prints one JSON line per step. ``--die-after K``
makes this rank ``os._exit(1)`` after K gathers (a crashed process: its sockets close),
``--hang-after K`` makes it sleep for an hour after K gathers (a silent peer); ``--timeout S``
sets the group timeout once the group is up. A transport error is printed as
``ERROR <class>: <message>`` and exits 3.

``train`` runs the production path of one B1 rank on CPU, from the config to the saved adapter:
``SFTTrainerWrapper(cfg, device="cpu").setup(...)`` (group, sharding, init check, timeout,
metrics) and then ``wrapper.train(display=..., tracker=..., run_id=...)`` with transformers' own
loop. The parent sets the launcher environment: for a B1 rank ``SOUP_STREAM_DP_RANK``,
``_WORLD_SIZE``, ``_ADDR`` (rank 1) and ``_BIND_PORT`` (rank 0) and NO ``WORLD_SIZE``; for the
one-rank reference none of them, plus ``--no-rewind-log`` (the recorder is on by default and a
distributed run switches it off, so the reference turns it off to run the same plain SFT step).
Three things are touched after ``setup`` and are not production: the train sampler is made
sequential (the equivalence needs a fixed row order that a 1-rank and a 2-rank run share),
``lora_B`` is randomised (PEFT starts it at zero, which zeroes every ``lora_A`` gradient in step
one), and a callback records the adapters after every optimizer step. The display is a recorder
and the tracker a REAL per-rank ``ExperimentTracker``, so only the production rank gates keep rank
1 from writing. It writes ``<tag>.json`` (facts) and ``<tag>.safetensors`` (the trainable tensors
after training and one row of them per optimizer step). An error exits the process at once
(``os._exit``) so that the peer's socket closes and the peer fails fast too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _emit(**fields) -> None:
    print(json.dumps(fields, sort_keys=True), flush=True)


# ---------------------------------------------------------------------------
# transport: the group alone
# ---------------------------------------------------------------------------


def _transport(args) -> int:
    from soup_cli.utils.adapter_grad_transport import GradTransportError, TcpStarGroup

    try:
        group = TcpStarGroup(
            rank=args.rank,
            world_size=args.world_size,
            host="127.0.0.1",
            port=args.port,
            bind_host="127.0.0.1",
            bind_port=args.port,
            token=args.token or None,
            connect_timeout=60,
            timeout_seconds=60,
        )
        if args.timeout is not None:
            group.set_timeout(args.timeout)
        _emit(rank=args.rank, event="joined", timeout=group.timeout_seconds)
        group.barrier()
        _emit(rank=args.rank, event="barrier")
        for round_no in range(args.rounds):
            gathered = group.all_gather_object({"rank": args.rank, "round": round_no})
            ranks = [entry["rank"] for entry in gathered]
            _emit(rank=args.rank, event="gather", round=round_no, ranks=ranks)
            if args.die_after is not None and round_no + 1 == args.die_after:
                _emit(rank=args.rank, event="dying")
                os._exit(1)
            if args.hang_after is not None and round_no + 1 == args.hang_after:
                _emit(rank=args.rank, event="hanging")
                time.sleep(3600)
        group.barrier()
        group.close()
        _emit(rank=args.rank, event="done")
        return 0
    except GradTransportError as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        return 3


# ---------------------------------------------------------------------------
# train: one B1 rank, production path
# ---------------------------------------------------------------------------


def _train_parser(subparsers):
    # The config builder of the C-form worker reads these names off ``args``.
    from _stream_distributed_trainer_worker import _parser

    parser = _parser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--group-timeout", type=float, default=None)
    parser.add_argument("--fault", choices=("none", "exit", "hang"), default="none")
    parser.add_argument("--fault-step", type=int, default=2)
    parser.add_argument("--fault-rank", type=int, default=1)
    return parser


class _Display:
    """Records what the Soup callback asks the live panel to do."""

    def __init__(self):
        self.calls = {"start": 0, "update": 0, "stop": 0}

    def start(self, *args, **kwargs):
        self.calls["start"] += 1

    def update(self, *args, **kwargs):
        self.calls["update"] += 1

    def stop(self, *args, **kwargs):
        self.calls["stop"] += 1


def _train(args) -> None:
    from _stream_distributed_trainer_worker import (
        _LORA_B_SCALE,
        _SEED,
        _config,
        _digest,
        _flat,
        _trainable,
    )

    args.target_modules = [name for name in args.target_modules.split(",") if name]
    out_dir = Path(args.out_dir)
    rank_root = out_dir / args.tag
    rank_root.mkdir(parents=True, exist_ok=True)
    # Two processes on one machine would race one shard cache and one tracker database
    # (they are two hosts in life): each rank gets its own.
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(rank_root / "cache")
    os.environ["SOUP_DB_PATH"] = str(rank_root / "runs.db")

    import torch
    from safetensors.torch import save_file
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback

    import soup_cli.trainer.stream_setup as stream_setup
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.adapter_sync_trainer import has_adapter_sync

    torch.set_num_threads(1)
    if args.group_timeout is not None:
        stream_setup.DISTRIBUTED_STREAM_TIMEOUT_SECONDS = args.group_timeout

    rows = json.loads(Path(args.rows_file).read_text(encoding="utf-8"))
    cfg = _config(args, rank_root)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    trainer._get_train_sampler = lambda train_dataset=None: SequentialSampler(
        trainer.train_dataset if train_dataset is None else train_dataset
    )
    group = wrapper._dp_group

    named = _trainable(wrapper.model)
    generator = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        for name, param in named:
            if "lora_B" in name:
                param.copy_(
                    (_LORA_B_SCALE * torch.randn(param.shape, generator=generator)).to(param.dtype)
                )
    initial_sha = _digest(named)
    per_step = []
    window_tokens = []

    class _Record(TrainerCallback):
        def on_step_end(self, _args, _state, _control, **_kwargs):
            per_step.append(_flat(named))

    trainer.add_callback(_Record())

    # The token count of every optimizer window this rank trained on (what B1 weights by).
    real_step = trainer.training_step

    def _recording_step(*step_args, **step_kwargs):
        items = step_kwargs.get("num_items_in_batch", step_args[2] if len(step_args) > 2 else None)
        if trainer.accelerator.sync_gradients and items is not None:
            window_tokens.append(float(items))
        return real_step(*step_args, **step_kwargs)

    trainer.training_step = _recording_step

    # A fixture for peer death: on the fault rank, die or go silent at the end of a step.
    if args.fault != "none" and args.rank == args.fault_rank:

        class _Fault(TrainerCallback):
            def on_step_end(self, _args, state, _control, **_kwargs):
                if state.global_step != args.fault_step:
                    return
                if args.fault == "exit":
                    print(f"{args.tag}: fault -- exiting at step {state.global_step}", flush=True)
                    sys.stdout.flush()
                    os._exit(1)
                print(f"{args.tag}: fault -- hanging after step {state.global_step}", flush=True)
                time.sleep(3600)

        trainer.add_callback(_Fault())

    # The row ids every micro-batch holds, keyed by the tokenised row (rows are distinct).
    lookup = {}
    for index in range(len(trainer.train_dataset)):
        lookup[tuple(int(token) for token in trainer.train_dataset[index]["input_ids"])] = index
    seen = []
    collate = trainer.data_collator

    def _recording_collate(features):
        seen.append(
            [lookup[tuple(int(token) for token in feature["input_ids"])] for feature in features]
        )
        return collate(features)

    trainer.data_collator = _recording_collate

    display = _Display()
    tracker = ExperimentTracker()
    run_id = tracker.start_run(
        config_dict=cfg.model_dump(),
        device="cpu",
        device_name="cpu",
        gpu_info={},
        experiment_name=args.tag,
    )
    result = wrapper.train(display=display, tracker=tracker, run_id=run_id)

    out_root = Path(result["output_dir"])
    history = trainer.state.log_history
    stats = getattr(trainer, "last_sync_stats", None)
    facts = {
        "tag": args.tag,
        "rank": args.rank,
        "world_size": int(trainer.args.world_size),
        "process_index": int(trainer.args.process_index),
        "distributed_stream": bool(wrapper._distributed_stream),
        "group_class": None if group is None else type(group).__name__,
        "group_rank": None if group is None else int(group.rank),
        "group_world": None if group is None else int(group.world_size),
        "group_timeout": None if group is None else float(group.timeout_seconds),
        "trainer_class": type(trainer).__name__,
        "has_adapter_sync": bool(has_adapter_sync(trainer)),
        "accepts_loss_kwargs": bool(trainer.model_accepts_loss_kwargs),
        "global_step": int(trainer.state.global_step),
        "total_steps": int(result["total_steps"]),
        "n_trainable": sum(param.numel() for _n, param in named),
        "n_train_rows": len(trainer.train_dataset),
        "initial_sha": initial_sha,
        "final_sha": _digest(named),
        "losses": [entry["loss"] for entry in history if "loss" in entry],
        "grad_norms": [entry["grad_norm"] for entry in history if "grad_norm" in entry],
        "window_tokens": window_tokens,
        "sync_calls": getattr(trainer, "sync_calls", None),
        "sync_bytes": None if stats is None else stats.bytes,
        "sync_chunks": None if stats is None else stats.n_chunks,
        "sync_seconds": None if stats is None else stats.seconds,
        "display_calls": display.calls,
        "run_rows": len(tracker.list_runs()),
        "metric_rows": len(tracker.get_metrics(run_id)),
        "rows_seen": seen,
        "output_dir": str(out_root),
        "files": sorted(
            path.relative_to(out_root).as_posix() for path in out_root.rglob("*") if path.is_file()
        ),
    }
    tensors = {name: param.detach().contiguous().clone() for name, param in named}
    tensors["per_step"] = torch.stack(per_step).contiguous()
    save_file(tensors, str(out_dir / f"{args.tag}.safetensors"))
    (out_dir / f"{args.tag}.json").write_text(json.dumps(facts), encoding="utf-8")
    wrapper._close_stream_runtime()  # the group is closed already: training's context did it


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("transport", "train"):
        print("usage: _stream_dp_tcp_worker.py transport|train ...", file=sys.stderr)
        return 2
    command, rest = argv[0], argv[1:]
    if command == "transport":
        parser = argparse.ArgumentParser()
        parser.add_argument("--rank", type=int, required=True)
        parser.add_argument("--world-size", type=int, default=2)
        parser.add_argument("--port", type=int, required=True)
        parser.add_argument("--rounds", type=int, default=3)
        parser.add_argument("--timeout", type=float, default=None)
        parser.add_argument("--die-after", type=int, default=None)
        parser.add_argument("--hang-after", type=int, default=None)
        parser.add_argument("--token", default="")
        return _transport(parser.parse_args(rest))
    args = _train_parser(None).parse_args(rest)
    try:
        _train(args)
    except BaseException:  # die loudly, so the peer's socket closes and it fails fast too
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(1)
    print(f"{args.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
