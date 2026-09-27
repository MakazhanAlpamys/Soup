"""Run a short, fixed-length training job and measure it (#836).

This is the wiring between a real trainer and the two pure halves: the
collector records, :mod:`soup_cli.bench.train_report` judges. Scope is the
transformers SFT trainer, the path every recipe's ``soup train`` takes by
default; other tasks and MLX are refused by name rather than measured through
a path this has not been checked against.
"""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
from importlib import metadata
from typing import Any, Callable, Optional

_VERSIONED = ("torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate")


def bench_scope_error(cfg: Any) -> Optional[str]:
    """Why this config cannot be benchmarked here, or ``None`` when it can."""
    if cfg.task != "sft":
        return f"task: {cfg.task} -- bench train measures the SFT trainer only"
    if getattr(cfg, "backend", "transformers") != "transformers":
        return (
            f"backend: {cfg.backend} -- bench train measures the transformers "
            f"trainer only"
        )
    return None


def config_hash(resolved: dict) -> str:
    """sha256 of the fully resolved config, so a schema-default change that moves
    the run is visible in the report (#716)."""
    blob = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _versions() -> dict:
    found = {}
    for name in _VERSIONED:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def _nvidia_smi(tool: str, query: str, gpu: Optional[str] = None) -> Optional[str]:
    """First line of one ``nvidia-smi`` query, or ``None`` when it cannot be read.

    A non-zero exit is ``None``: the tool prints its failure on stdout ("NVIDIA-SMI
    has failed because ..."), and that is not a value. Never used for memory:
    that is torch's allocator counters, which measure a different quantity from
    nvidia-smi's (#836).
    """
    import subprocess

    command = [tool, query, "--format=csv,noheader,nounits"]
    if gpu is not None:
        command.append(f"--id={gpu}")
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
        if out.returncode != 0:
            return None
        return out.stdout.splitlines()[0].strip() or None
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def _driver_version() -> Optional[str]:
    """Driver version, so a driver update between two reports is visible (#716)."""
    import re

    from soup_cli.utils.layer_stream import _resolve_tool

    tool = _resolve_tool("nvidia-smi")  # absolute path only (CWE-427)
    value = _nvidia_smi(tool, "--query-gpu=driver_version") if tool else None
    return value if value and re.fullmatch(r"\d+(\.\d+)+", value) else None


def _pci_bus_id(props: Any) -> Optional[str]:
    """torch's device as nvidia-smi names it, since nvidia-smi's GPU 0 need not be
    torch's ``cuda:0`` (CUDA_VISIBLE_DEVICES, or a different enumeration order)."""
    try:
        return (
            f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:"
            f"{props.pci_device_id:02X}.0"
        )
    except (AttributeError, TypeError, ValueError):
        return None  # older torch: fall back to nvidia-smi's first GPU


class ClockSampler:
    """Samples the SM clock on a background thread, on ``interval``-second ticks.

    One read after the run lands on an idle card: on the gate-836 box it recorded
    mostly 180 MHz and never above 1417, against a busy median of 2370-2557
    (#1166). Samples are
    ``(time.perf_counter(), MHz)`` -- the collector's clock -- stamped at the
    midpoint of their query, so they can be cut to the counted window afterwards.
    Queries start on an ``interval`` schedule; one slower than that skips
    ticks. The thread only shells out and appends; it never touches torch or
    CUDA. ``read`` replaces the nvidia-smi query in tests.
    """

    def __init__(
        self,
        interval: float = 0.1,
        gpu: Optional[str] = None,
        read: Optional[Callable[[], Optional[str]]] = None,
    ) -> None:
        import threading

        from soup_cli.utils.layer_stream import _resolve_tool

        self.interval = interval
        if read is None:
            tool = _resolve_tool("nvidia-smi")  # absolute path only (CWE-427)
            if tool is not None:
                def read():
                    return _nvidia_smi(tool, "--query-gpu=clocks.sm", gpu)
        self._read = read
        self.available = read is not None
        self.samples: list[tuple[float, int]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        import time

        origin = time.perf_counter()
        tick = 0
        while True:
            began = time.perf_counter()
            value = self._read()
            ended = time.perf_counter()
            if value is not None and value.isdigit():
                self.samples.append(((began + ended) / 2, int(value)))
            # Absolute ticks: a slow query skips ticks instead of pushing every
            # later one back by its own duration.
            tick = max(tick + 1, int((ended - origin) / self.interval) + 1)
            if self._stop.wait(origin + tick * self.interval - time.perf_counter()):
                return

    def start(self) -> "ClockSampler":
        if self.available:
            self._thread.start()
        return self

    def stop(self) -> list[tuple[float, int]]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        return list(self.samples)


def _device_provenance(torch: Any, device: str, sm_clock_mhz_busy: dict) -> dict:
    empty = {
        "device": device, "card": None, "cuda_runtime": None,
        "compute_capability": None, "driver": None, "sm_clock_mhz_busy": sm_clock_mhz_busy,
    }
    if device != "cuda" or not torch.cuda.is_available():
        return empty
    props = torch.cuda.get_device_properties(0)
    return {
        "device": "cuda",
        "card": props.name,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": f"{props.major}.{props.minor}",
        "driver": _driver_version(),
        # Sampled during the counted steps: the boost clock moves a lot under
        # load, and the card is idle again by the time training returns.
        "sm_clock_mhz_busy": sm_clock_mhz_busy,
    }


def run_bench_train(
    cfg: Any,
    *,
    steps: int,
    warmup: int,
    device: str,
    load_dataset: Callable[[Any], dict],
    before_train: Optional[Callable[[Any], None]] = None,
    clock_sampler: Optional[Callable[[], Any]] = None,
) -> dict:
    """Train ``cfg`` for exactly ``steps`` optimizer steps and build the report.

    ``before_train`` receives the wrapper after setup; tests use it to break the
    run on purpose (freeze everything, zero the gradients) so the checks are
    shown to fire through the real trainer and not only on hand-built records.
    ``clock_sampler`` builds the SM-clock sampler; by default a
    :class:`ClockSampler` on CUDA and none elsewhere. Tests inject one on CPU.
    """
    import torch

    from soup_cli.bench.collector import BenchCollector, summarize_trainable
    from soup_cli.bench.train_report import check_trainable, summarize_clock_samples
    from soup_cli.trainer.sft import SFTTrainerWrapper

    if steps <= warmup:
        raise ValueError(
            f"--steps {steps} leaves nothing after --warmup {warmup}: raise "
            f"--steps or lower --warmup."
        )

    with tempfile.TemporaryDirectory(prefix="soup-bench-") as scratch:
        # Log every step (the grad_norm check reads each one), never save, and
        # never write into the config's own output directory.
        training = cfg.training.model_copy(
            update={"logging_steps": 1, "save_steps": steps + 1}
        )
        cfg = cfg.model_copy(update={"training": training, "output": scratch})

        wrapper = SFTTrainerWrapper(cfg, device=device)
        wrapper.setup(load_dataset(cfg))
        trainer = wrapper.trainer
        trainer.args.max_steps = steps
        if hasattr(trainer.args, "eval_strategy"):
            trainer.args.eval_strategy = "no"

        collector = BenchCollector(warmup_steps=warmup)
        if device == "cuda" and torch.cuda.is_available():
            collector._sync = torch.cuda.synchronize
        trainer.add_callback(collector)

        # Tokens come from what training_step receives: that is the batch the
        # loss sees, and it runs in this process -- a collator wrapper would run
        # inside dataloader workers and count into their copy.
        original_step = trainer.training_step

        def training_step(model, inputs, *args, **kwargs):
            collector.observe_batch(inputs)
            return original_step(model, inputs, *args, **kwargs)

        trainer.training_step = training_step

        if before_train is not None:
            before_train(wrapper)

        # With nothing to train, the Trainer dies inside autograd with a
        # message that names no cause. The builder's own rule, asked first.
        nothing = check_trainable(summarize_trainable(trainer.model))
        if nothing is not None:
            raise ValueError(nothing["message"])

        on_cuda = device == "cuda" and torch.cuda.is_available()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        if clock_sampler is None and on_cuda:
            gpu = _pci_bus_id(torch.cuda.get_device_properties(0))
            sampler = ClockSampler(gpu=gpu)
        else:
            sampler = clock_sampler() if clock_sampler is not None else None
        if sampler is not None:
            sampler.start()
        try:
            trainer.train()
        finally:
            samples = sampler.stop() if sampler is not None else []
        if sampler is None:
            unavailable = "not a CUDA run"
        elif not sampler.available:
            unavailable = "no nvidia-smi tool found"
        else:
            unavailable = None
        sm_clock_mhz_busy = summarize_clock_samples(
            samples, collector.counted_window_started, collector.counted_window_ended,
            unavailable_reason=unavailable,
        )
        if on_cuda:
            memory = {
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
        else:
            memory = {"max_memory_allocated_bytes": None, "max_memory_reserved_bytes": None}

        resolved = cfg.model_dump(mode="json")
        resolved.pop("output", None)  # the scratch directory, different every run
        model = trainer.model
        provenance = {
            **_device_provenance(torch, device, sm_clock_mhz_busy),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "versions": _versions(),
            "dtype": str(getattr(model, "dtype", None)),
            "optimizer": str(trainer.args.optim),
            "seed": trainer.args.seed,
            "data_seed": trainer.args.data_seed,
        }
        return collector.build_report(
            warmup_steps=warmup,
            provenance=provenance,
            memory=memory,
            config_hash=config_hash(resolved),
            steps_requested=steps,
            extra={"resolved_config": resolved},
        )
