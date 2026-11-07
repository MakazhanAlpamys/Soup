"""Run-time pieces of data-parallel layer streaming: group timeout (C8) and step metrics (C7).

* :func:`apply_group_timeout` (C8) shortens the process group's timeout once the heavy
  setup is behind it. accelerate creates the group with the trainer's ``ddp_timeout``
  (1800 s) because the first collective after setup waits for the slower rank's download
  and sharding. After that a peer that goes silent (a host or a network lost, not a
  crash, which closes the socket and fails at once) would keep the survivor in a
  collective for half an hour. The API is ``torch.distributed.set_timeout`` (public from
  torch 2.9 or so; on torch 2.14 it forwards the value to every backend of the group),
  with the older private ``distributed_c10d._set_pg_timeout`` as the fallback; where
  neither exists the group keeps its launch value and the caller says so.
* :func:`attach_stream_dp_metrics` (C7) adds a trainer callback that writes ONE JSON line
  per optimizer step to ``<output_dir>/stream_dp_metrics.rank<R>.jsonl``. Each rank writes
  its own file, so ranks that share a directory (several on one host, a shared
  filesystem) never collide, and a stalled rank shows as a file that stopped growing.

Torch and transformers are imported inside the functions: ``soup`` stays importable
without them.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, Optional

#: File name of one rank's step metrics, inside the run's output directory.
METRICS_FILE_TEMPLATE = "stream_dp_metrics.rank{rank}.jsonl"

SET_TIMEOUT_API = "torch.distributed.set_timeout"
LEGACY_TIMEOUT_API = "torch.distributed.distributed_c10d._set_pg_timeout"


@dataclass(frozen=True)
class GroupTimeout:
    """The outcome of :func:`apply_group_timeout`.

    ``api`` is the torch function that was called, ``None`` when the timeout was left at
    its launch value; ``reason`` then says why.
    """

    api: Optional[str]
    seconds: float
    reason: str = ""


def _positive_seconds(seconds: Any) -> float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError(f"the group timeout must be a number of seconds, got {seconds!r}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"the group timeout must be a positive number of seconds, got {seconds}")
    return float(seconds)


def apply_group_timeout(seconds: Any, *, group: Optional[Any] = None) -> GroupTimeout:
    """Shorten the (default) process group's timeout to ``seconds`` where torch can (C8).

    Never raises for a missing or refusing API: the group then keeps the timeout it was
    created with and the result says so. A bad ``seconds`` is a bug in the caller and does.
    """
    import torch
    import torch.distributed as dist

    value = _positive_seconds(seconds)
    if not dist.is_available() or not dist.is_initialized():
        return GroupTimeout(None, value, "there is no process group")
    timeout = timedelta(seconds=value)
    setter = getattr(dist, "set_timeout", None)
    api = SET_TIMEOUT_API
    if not callable(setter):
        setter = getattr(getattr(dist, "distributed_c10d", None), "_set_pg_timeout", None)
        api = LEGACY_TIMEOUT_API
    if not callable(setter):
        return GroupTimeout(
            None, value, f"torch {torch.__version__} has no API to change a running group's timeout"
        )
    try:
        setter(timeout, group)
    except (RuntimeError, ValueError, TypeError) as exc:
        return GroupTimeout(None, value, f"{api} failed: {type(exc).__name__}: {exc}")
    return GroupTimeout(api, value)


def stream_dp_metrics_path(output_dir: str, rank: int) -> str:
    """Where rank ``rank`` writes its step metrics."""
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise ValueError(f"rank must be a non-negative int, got {rank!r}")
    return os.path.join(output_dir, METRICS_FILE_TEMPLATE.format(rank=rank))


def _finite_or_none(value: Any) -> Optional[float]:
    """A float for JSON: ``None`` for a missing or non-finite number (JSON has no NaN)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def build_stream_dp_metrics_callback(trainer: Any, output_dir: str) -> Any:
    """The C7 callback for ``trainer`` (an adapter-sync trainer), writing under ``output_dir``.

    One JSON object per line, per optimizer step: ``step``, ``loss`` and ``grad_norm``
    (``null`` on a step HF does not log), ``t_step`` (seconds from the start of the
    accumulation window to the end of the optimizer step), ``t_sync`` / ``sync_bytes`` /
    ``sync_chunks`` (the adapter-gradient sync of that step, from the trainer's
    ``last_sync_stats``; zeros if no sync ran), ``world_size`` and ``rank``.

    HF fires ``on_step_end`` before the log event of the same step, so a step that will be
    logged is held back and written by ``on_log`` with its loss; any other step is written
    at once, which keeps the file a live per-step heartbeat. The file is truncated when
    training begins (a resume is refused for this slice, so a leftover file is stale).
    """
    from transformers import TrainerCallback

    class StreamDpMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self._trainer = trainer
            self._output_dir = output_dir
            self._path: Optional[str] = None
            self._begin: Optional[float] = None
            self._calls_seen = 0
            self._pending: Optional[Dict[str, Any]] = None
            self._warned = False

        def _path_for(self, args: Any) -> str:
            if self._path is None:
                self._path = stream_dp_metrics_path(self._output_dir, int(args.process_index))
            return self._path

        def _warn_once(self, exc: OSError) -> None:
            if self._warned:
                return
            self._warned = True
            from rich.console import Console

            from soup_cli.utils.terminal import for_terminal

            Console().print(
                "[yellow]Per-step metrics file could not be written "
                f"({for_terminal(f'{type(exc).__name__}: {exc}')}); "
                "training goes on without it.[/]"
            )

        def _write(self, args: Any, record: Dict[str, Any]) -> None:
            line = json.dumps(record, allow_nan=False, separators=(",", ":"))
            try:
                with open(self._path_for(args), "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            except OSError as exc:
                self._warn_once(exc)

        def _flush(self, args: Any) -> None:
            pending, self._pending = self._pending, None
            if pending is not None:
                self._write(args, pending)

        def on_train_begin(self, args, state, control, **kwargs):
            self._pending = None
            self._calls_seen = int(getattr(self._trainer, "sync_calls", 0) or 0)
            try:
                with open(self._path_for(args), "w", encoding="utf-8"):
                    pass
            except OSError as exc:
                self._warn_once(exc)

        def on_step_begin(self, args, state, control, **kwargs):
            self._begin = time.perf_counter()

        def on_step_end(self, args, state, control, **kwargs):
            self._flush(args)  # a held record that no log event claimed
            now = time.perf_counter()
            calls = int(getattr(self._trainer, "sync_calls", 0) or 0)
            fresh = calls > self._calls_seen
            self._calls_seen = calls
            stats = getattr(self._trainer, "last_sync_stats", None) if fresh else None
            record = {
                "step": int(state.global_step),
                "loss": None,
                "grad_norm": None,
                "t_step": None if self._begin is None else _finite_or_none(now - self._begin),
                "t_sync": float(stats.seconds) if stats is not None else 0.0,
                "sync_bytes": int(stats.bytes) if stats is not None else 0,
                "sync_chunks": int(stats.n_chunks) if stats is not None else 0,
                "world_size": int(args.world_size),
                "rank": int(args.process_index),
            }
            if getattr(control, "should_log", False):
                self._pending = record  # on_log of this very step follows, with the loss
            else:
                self._write(args, record)

        def on_log(self, args, state, control, logs=None, **kwargs):
            pending = self._pending
            if pending is None:
                return
            self._pending = None
            if logs:
                pending["loss"] = _finite_or_none(logs.get("loss"))
                pending["grad_norm"] = _finite_or_none(logs.get("grad_norm"))
            self._write(args, pending)

        def on_train_end(self, args, state, control, **kwargs):
            self._flush(args)

    return StreamDpMetricsCallback()


def attach_stream_dp_metrics(trainer: Any, output_dir: str) -> Any:
    """Build the C7 callback and add it to ``trainer``; return it."""
    callback = build_stream_dp_metrics_callback(trainer, output_dir)
    trainer.add_callback(callback)
    return callback
