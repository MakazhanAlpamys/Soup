"""One process of ``tests/test_stream_distributed_runtime.py`` (a plain script, not collected).

It runs the production path of one rank of a data-parallel layer-streaming run on CPU, from
the config to the saved adapter: ``SFTTrainerWrapper(cfg, device="cpu").setup(...)`` (which
includes the cross-rank init check and broadcast, the group timeout and the per-step metrics
callback) and then ``wrapper.train(display=..., tracker=..., run_id=...)``, i.e. the SFT
wrapper's own ``train()`` with its callback, the HF loop, the final save and the tokenizer
save. Compare ``_stream_distributed_trainer_worker.py``, which stops at ``trainer.train()``
because it only compares adapters.

The parent sets the launcher environment (``RANK``, ``WORLD_SIZE``, ``LOCAL_RANK``,
``LOCAL_WORLD_SIZE=1``, ``MASTER_ADDR``/``MASTER_PORT``, ``ACCELERATE_USE_CPU=true``) and
``--rank/--world-size/--tag/--out-dir/--base-dir/--rows-file/--batch-size`` (the arguments of
the trainer worker, whose parser this one extends). What the extra arguments change, per
rank, is what a test varies:

* ``--seed``, ``--lora-r``, ``--target-modules``, ``--base-dir``: the init check (T2);
* ``--save-steps``: periodic ``checkpoint-*`` directories, to see which rank writes them (T6);
* ``--sequential-sampler``: the train sampler is made sequential, as in the trainer worker;
  without it the production sampler runs (T7 wants it);
* ``--group-timeout``: sets ``stream_setup.DISTRIBUTED_STREAM_TIMEOUT_SECONDS`` before the
  setup (T5 and T9 use a short one; the production value is 300 s);
* ``--fault``/``--fault-step``/``--fault-rank``: on that rank, ``exit`` (``os._exit(1)``) or
  ``hang`` (sleep for an hour) at the end of that optimizer step (T5).

Three things are touched after ``setup`` and are not production: ``lora_B`` is randomised (PEFT
starts it at zero, which zeroes every ``lora_A`` gradient in step one), the sampler as said,
and a collator wrapper records the row ids of every micro-batch. The display and the tracker
are a recorder and a per-rank SQLite database: the tracker is a REAL ``ExperimentTracker`` on
every rank, so only the callback's own rank gate keeps rank 1 from writing metrics rows.

It writes ``<tag>.json`` (facts). An error exits the process at once (``os._exit``) so that
the peer's socket closes and the peer fails fast too.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

from _stream_distributed_trainer_worker import (
    _LORA_B_SCALE,
    _SEED,
    _config,
    _digest,
    _parser,
    _trainable,
)


def _parse(argv):
    parser = _parser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=4)
    parser.add_argument("--target-modules", default="q_proj,v_proj")
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sequential-sampler", action="store_true")
    parser.add_argument("--group-timeout", type=float, default=None)
    parser.add_argument("--fault", choices=("none", "exit", "hang"), default="none")
    parser.add_argument("--fault-step", type=int, default=2)
    parser.add_argument("--fault-rank", type=int, default=1)
    args = parser.parse_args(argv)
    args.target_modules = [name for name in args.target_modules.split(",") if name]
    return args


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


def _die(message: str) -> None:
    print(message, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)


def _run(args) -> None:
    out_dir = Path(args.out_dir)
    rank_root = out_dir / args.tag
    rank_root.mkdir(parents=True, exist_ok=True)
    # Two processes on one machine would race one shard cache and one tracker database
    # (they are two hosts in life): each rank gets its own.
    os.environ["SOUP_LAYER_STREAM_CACHE_DIR"] = str(rank_root / "cache")
    os.environ["SOUP_DB_PATH"] = str(rank_root / "runs.db")

    import torch
    import torch.distributed as dist
    from torch.utils.data import SequentialSampler
    from transformers import TrainerCallback

    import soup_cli.trainer.stream_setup as stream_setup
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.trainer.sft import SFTTrainerWrapper

    torch.set_num_threads(1)
    if args.group_timeout is not None:
        stream_setup.DISTRIBUTED_STREAM_TIMEOUT_SECONDS = args.group_timeout

    # Record what the setup asks torch to do with the group timeout, then do it.
    timeout_calls = []
    real_set_timeout = getattr(dist, "set_timeout", None)
    if real_set_timeout is not None:

        def _recording_set_timeout(timeout, group=None):
            timeout_calls.append(timeout.total_seconds())
            return real_set_timeout(timeout, group)

        dist.set_timeout = _recording_set_timeout

    rows = json.loads(Path(args.rows_file).read_text(encoding="utf-8"))
    cfg = _config(args, rank_root)
    wrapper = SFTTrainerWrapper(cfg, device="cpu")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    if args.sequential_sampler:
        trainer._get_train_sampler = lambda train_dataset=None: SequentialSampler(
            trainer.train_dataset if train_dataset is None else train_dataset
        )

    named = _trainable(wrapper.model)
    generator = torch.Generator().manual_seed(_SEED)
    with torch.no_grad():
        for name, param in named:
            if "lora_B" in name:
                param.copy_(
                    (_LORA_B_SCALE * torch.randn(param.shape, generator=generator)).to(param.dtype)
                )
    initial_sha = _digest(named)

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

    if args.fault != "none" and args.rank == args.fault_rank:

        class _Fault(TrainerCallback):
            def on_step_end(self, _args, state, _control, **_kwargs):
                if state.global_step != args.fault_step:
                    return
                if args.fault == "exit":
                    _die(f"{args.tag}: fault -- exiting at the end of step {state.global_step}")
                print(f"{args.tag}: fault -- hanging after step {state.global_step}", flush=True)
                time.sleep(3600)

        trainer.add_callback(_Fault())

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
    facts = {
        "tag": args.tag,
        "rank": args.rank,
        "world_size": int(trainer.args.world_size),
        "process_index": int(trainer.args.process_index),
        "distributed_stream": bool(wrapper._distributed_stream),
        "global_step": int(trainer.state.global_step),
        "total_steps": int(result["total_steps"]),
        "n_trainable": sum(param.numel() for _n, param in named),
        "initial_sha": initial_sha,
        "final_sha": _digest(named),
        "losses": [entry["loss"] for entry in history if "loss" in entry],
        "display_calls": display.calls,
        "run_rows": len(tracker.list_runs()),
        "metric_rows": len(tracker.get_metrics(run_id)),
        "timeout_calls": timeout_calls,
        "rows_seen": seen,
        "output_dir": str(out_root),
        "files": sorted(
            path.relative_to(out_root).as_posix() for path in out_root.rglob("*") if path.is_file()
        ),
    }
    (out_dir / f"{args.tag}.json").write_text(json.dumps(facts), encoding="utf-8")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
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
