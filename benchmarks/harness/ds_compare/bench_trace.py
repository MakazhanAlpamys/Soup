"""Per-step recorder and the metric derivation every arm shares.

One recorder (``TraceState`` behind a ``TrainerCallback``) watches every arm, so the numbers
of a Soup run and of a plain Hugging Face run come out of the same code:

* ``hf_baseline.py`` hands the callback to its ``Trainer`` directly;
* streaming arms run ``soup train`` unmodified, and ``sitecustomize.py`` appends the same
  callback class to ``transformers.trainer.DEFAULT_CALLBACKS`` when the environment variable
  ``SOUP_BENCH_TRACE_OUT`` names an output file. Nothing in Soup is patched; the callback
  only reads events.

The recorder is passive: it stores ``time.perf_counter()`` at the end of every optimizer
step, the logged losses, the trainer's own summary log (``train_runtime`` ...), the token
counts and ``torch.cuda`` peak statistics. ``derive_metrics`` turns that raw trace into the
numbers of protocol section 3.

No ``soup_cli`` import here (the callback runs inside both kinds of process).
"""

from __future__ import annotations

import atexit
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ds_common import atomic_write_json  # noqa: E402

LOSS_STEPS = (64, 128, 192, 256)
STEADY_FROM_STEP = 33  # steady state = steps 33..end (protocol section 3)

_VERSION_DISTS = (
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "deepspeed",
    "trl",
    "datasets",
    "soup-cli",
)


def collect_versions() -> dict[str, Any]:
    """Versions of the packages installed for THIS interpreter (the one that trains).

    Read from package metadata, never by importing: a plain Hugging Face run must not
    import ``soup_cli`` even to report a version. A git install also records its commit.
    """
    from importlib import metadata

    versions: dict[str, Any] = {"python": sys.version.split()[0]}
    for name in _VERSION_DISTS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        direct = metadata.distribution("soup-cli").read_text("direct_url.json")
        if direct:
            import json

            info = json.loads(direct)
            versions["soup_cli_commit"] = (info.get("vcs_info") or {}).get("commit_id")
            versions["soup_cli_editable"] = bool((info.get("dir_info") or {}).get("editable"))
    except Exception:  # noqa: BLE001 - best effort
        pass
    torch = sys.modules.get("torch")
    versions["cuda_runtime"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        versions["cuda_available"] = bool(torch.cuda.is_available()) if torch else False
    except Exception:  # noqa: BLE001
        versions["cuda_available"] = False
    return versions


def _plain(value: Any) -> Any:
    """An enum member (``OptimizerNames.ADAMW_TORCH``) as its value (``adamw_torch``)."""
    import enum

    return value.value if isinstance(value, enum.Enum) else value


def cuda_snapshot() -> dict[str, Any] | None:
    """Peak allocator statistics of device 0, or None without a usable CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        props = torch.cuda.get_device_properties(0)
        return {
            "device_name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        }
    except Exception:  # noqa: BLE001
        return None


class TraceState:
    """Plain-Python accumulator; ``flush`` writes the raw trace atomically."""

    def __init__(self, out_path: str | os.PathLike[str] | None) -> None:
        self.out_path = str(out_path) if out_path else None
        self.begin_perf: float | None = None
        self.begin_wall: float | None = None
        self.steps: list[list[Any]] = []  # [global_step, seconds since begin, cumulative tokens]
        self.logs: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}
        self.args: dict[str, Any] = {}
        self.cuda_begin: dict[str, Any] | None = None
        self.cuda_end: dict[str, Any] | None = None
        self.tokens_counted = 0
        self.counting = False
        self.complete = False
        self.global_step = 0
        self.max_steps = 0
        self.versions: dict[str, Any] = {}
        self.alloc_conf: str | None = None
        self._atexit_registered = False

    # -- events -----------------------------------------------------------------------
    def train_begin(self, args: dict[str, Any], max_steps: int) -> None:
        self.begin_perf = time.perf_counter()
        self.begin_wall = time.time()
        self.args = args
        self.max_steps = int(max_steps)
        # Versions are collected at the end: importing an optional package here would run
        # inside the trainer's own train_runtime window.
        self.alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        self.cuda_begin = cuda_snapshot()
        if not self._atexit_registered:
            atexit.register(self.flush)
            self._atexit_registered = True

    def add_tokens(self, count: int) -> None:
        """Called by a counting collator for every batch it builds."""
        self.counting = True
        self.tokens_counted += int(count)

    def step_end(self, global_step: int) -> None:
        if self.begin_perf is None:
            return
        self.global_step = int(global_step)
        cumulative = self.tokens_counted if self.counting else None
        self.steps.append([int(global_step), time.perf_counter() - self.begin_perf, cumulative])

    def log(self, global_step: int, logs: dict[str, Any]) -> None:
        entry: dict[str, Any] = {"step": int(global_step)}
        for key in ("loss", "num_tokens", "grad_norm", "learning_rate"):
            value = logs.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                entry[key] = float(value)
        if "loss" in entry or "num_tokens" in entry:
            self.logs.append(entry)
        if "train_runtime" in logs:
            self.summary = {
                key: logs[key]
                for key in (
                    "train_runtime",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "train_loss",
                    "total_flos",
                )
                if key in logs
            }

    def train_end(self) -> None:
        self.cuda_end = cuda_snapshot()
        self.versions = collect_versions()
        self.complete = True
        self.flush()

    # -- output -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ds_compare.trace.v1",
            "complete": self.complete,
            "global_step": self.global_step,
            "max_steps": self.max_steps,
            "begin_wall": self.begin_wall,
            "steps": self.steps,
            "logs": self.logs,
            "summary": self.summary,
            "tokens_counted": self.tokens_counted if self.counting else None,
            "args": self.args,
            "cuda_begin": self.cuda_begin,
            "cuda_end": self.cuda_end if self.cuda_end is not None else cuda_snapshot(),
            "versions": self.versions,
            "pytorch_cuda_alloc_conf": self.alloc_conf,
        }

    def flush(self) -> None:
        if not self.out_path or self.begin_perf is None:
            return
        if not self.versions:
            self.versions = collect_versions()
        try:
            atomic_write_json(self.out_path, self.to_dict())
        except Exception as exc:  # noqa: BLE001 - never take training down from an atexit hook
            print(f"[ds_compare] could not write trace {self.out_path}: {exc}", file=sys.stderr)


def make_callback_class(bound_state: TraceState | None = None) -> type:
    """A ``TrainerCallback`` subclass bound to ``bound_state``.

    With ``bound_state=None`` every instance builds its own ``TraceState`` writing to
    ``$SOUP_BENCH_TRACE_OUT`` (the form ``Trainer`` needs when it instantiates a class from
    ``DEFAULT_CALLBACKS``). Only rank 0 records.
    """
    from transformers import TrainerCallback

    class BenchTraceCallback(TrainerCallback):
        def __init__(self) -> None:
            self.state_obj = bound_state or TraceState(os.environ.get("SOUP_BENCH_TRACE_OUT"))

        @property
        def trace(self) -> TraceState:
            return self.state_obj

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: N803 - HF signature
            if getattr(args, "process_index", 0) != 0:
                return
            self.state_obj.train_begin(
                {
                    "per_device_train_batch_size": args.per_device_train_batch_size,
                    "gradient_accumulation_steps": args.gradient_accumulation_steps,
                    "world_size": getattr(args, "world_size", 1),
                    "num_train_epochs": args.num_train_epochs,
                    "learning_rate": args.learning_rate,
                    "lr_scheduler_type": str(_plain(args.lr_scheduler_type)),
                    "weight_decay": args.weight_decay,
                    "max_grad_norm": args.max_grad_norm,
                    "bf16": bool(args.bf16),
                    "optim": str(_plain(args.optim)),
                    "seed": args.seed,
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "deepspeed": bool(args.deepspeed),
                },
                state.max_steps,
            )

        def on_step_end(self, args, state, control, **kwargs):
            self.state_obj.step_end(state.global_step)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                self.state_obj.log(state.global_step, logs)

        def on_train_end(self, args, state, control, **kwargs):
            self.state_obj.train_end()

    return BenchTraceCallback


# ---------------------------------------------------------------------------
# Derivation (pure functions: tested without torch)
# ---------------------------------------------------------------------------


def _cumulative_tokens_by_step(raw: dict[str, Any]) -> dict[int, float]:
    by_step: dict[int, float] = {}
    for step, _seconds, cumulative in raw.get("steps", []):
        if cumulative is not None:
            by_step[int(step)] = float(cumulative)
    if not by_step:
        for entry in raw.get("logs", []):
            if "num_tokens" in entry:
                by_step[int(entry["step"])] = float(entry["num_tokens"])
    return by_step


def derive_metrics(
    raw: dict[str, Any],
    *,
    expected_total_tokens: int | None = None,
    loss_steps: tuple[int, ...] = LOSS_STEPS,
    steady_from: int = STEADY_FROM_STEP,
) -> dict[str, Any]:
    """Protocol section 3 numbers from a raw trace.

    ``tok_per_s`` = trained tokens / ``train_runtime`` (the STEP 3 metric); the runtime is
    the trainer's own ``train_runtime`` when the summary log arrived, else the time of the
    last step. ``steady_tok_per_s`` covers steps ``steady_from``..end from the per-step
    timestamps (``None`` when the run had fewer steps).
    """
    steps = raw.get("steps") or []
    total_steps = int(raw.get("global_step") or (steps[-1][0] if steps else 0))
    summary = raw.get("summary") or {}

    tokens: float | None = None
    tokens_source = None
    if raw.get("tokens_counted"):
        tokens, tokens_source = float(raw["tokens_counted"]), "counted by the collator"
    else:
        counted = [entry["num_tokens"] for entry in raw.get("logs", []) if "num_tokens" in entry]
        if counted:
            tokens, tokens_source = float(counted[-1]), "trainer num_tokens"
        elif expected_total_tokens:
            tokens, tokens_source = float(expected_total_tokens), "expected (row count x length)"

    runtime = summary.get("train_runtime")
    runtime_source = "trainer train_runtime"
    if runtime is None and steps:
        runtime = steps[-1][1]
        runtime_source = "time of the last step (no summary log)"
    tok_per_s = tokens / runtime if tokens is not None and runtime else None

    steady = None
    first = [row for row in steps if row[0] == steady_from - 1]
    last = steps[-1] if steps else None
    if first and last is not None and last[0] > steady_from - 1:
        seconds = last[1] - first[0][1]
        by_step = _cumulative_tokens_by_step(raw)
        if (steady_from - 1) in by_step and int(last[0]) in by_step:
            span_tokens = by_step[int(last[0])] - by_step[steady_from - 1]
        elif tokens is not None and total_steps:
            span_tokens = tokens * (last[0] - (steady_from - 1)) / total_steps
        else:
            span_tokens = None
        if span_tokens is not None and seconds > 0:
            steady = span_tokens / seconds

    losses = [(entry["step"], entry["loss"]) for entry in raw.get("logs", []) if "loss" in entry]
    loss_by_step = dict(losses)
    return {
        "steps": total_steps,
        "tokens_trained": int(tokens) if tokens is not None else None,
        "tokens_source": tokens_source,
        "train_runtime_s": runtime,
        "train_runtime_source": runtime_source,
        "train_samples_per_second": summary.get("train_samples_per_second"),
        "tok_per_s": tok_per_s,
        "steady_tok_per_s": steady,
        "final_loss": losses[-1][1] if losses else None,
        "loss_at": {str(step): loss_by_step[step] for step in loss_steps if step in loss_by_step},
        "loss_curve": [[step, loss] for step, loss in losses],
        "train_loss_mean": summary.get("train_loss"),
        "max_memory_reserved_bytes": (raw.get("cuda_end") or {}).get("max_memory_reserved_bytes"),
        "max_memory_allocated_bytes": (raw.get("cuda_end") or {}).get("max_memory_allocated_bytes"),
        "max_memory_reserved_at_train_begin_bytes": (
            (raw.get("cuda_begin") or {}).get("max_memory_reserved_bytes")
        ),
        "device_name": (raw.get("cuda_end") or raw.get("cuda_begin") or {}).get("device_name"),
        "trace_complete": bool(raw.get("complete")),
        "trace_begin_wall": raw.get("begin_wall"),
        "trace_last_step_seconds": steps[-1][1] if steps else None,
    }
