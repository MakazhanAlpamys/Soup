#!/usr/bin/env python3
"""Forward+backward time and peak memory of the shared-X Q/K/V Fast-LoRA path (#838).

The micro-benchmark #838 leaves open: one attention block's ``q_proj`` / ``k_proj`` /
``v_proj`` at the Llama-3.1-8B shapes, with ``patch_fast_lora_qkv`` applied (the "fast"
arm) and without it (the "plain" arm, peft's own LoRA forward), on the same weights,
the same adapter values and the same input.

``--self-test`` runs the same protocol at tiny shapes on the CPU, in this process and
in one child process, and reads no time or memory figure as a result: it checks the
plumbing (the arms agree, the fast arm took the fast path on every iteration, the plain
arm never did, the arm order, the verdicts and exit status), not speed.

Requirements
------------
- CUDA GPU for the measurement; ``bitsandbytes`` for the ``nf4`` cells.
- No model download: the block is built from random weights at the given shapes.
- ``--self-test`` needs neither a GPU nor a download; seconds.

Usage
-----
    python benchmarks/harness/fast_lora_qkv_bench.py
    python benchmarks/harness/fast_lora_qkv_bench.py --ranks 16 --dtypes nf4 --targets qv
    python benchmarks/harness/fast_lora_qkv_bench.py --self-test

Protocol
--------
A cell is one rank (``--ranks``, default 16 and 64), one base dtype (``bf16``, or
``nf4``: ``bitsandbytes`` ``Linear4bit`` with bf16 compute) and one target set
(``qkv``: all three adapted; ``qv``: ``k_proj`` stays a base-only projection, the
Llama default ``target_modules``). Shapes default to ``d=4096``, q out 4096, k/v out
1024, ``N=2048`` tokens in one row, no bias, LoRA dropout 0, ``lora_alpha = 2r``, the
adapter weights in fp32 over the bf16 or NF4 base as ``get_peft_model`` leaves them,
``lora_B`` drawn at random so that every gradient is non-zero.

Each cell runs ``--processes`` times (default 3), each in a fresh interpreter, so no
process inherits another's allocator or clock state. Inside a process both arms share
one module: the fast arm is the module with ``patch_fast_lora_qkv`` applied, the plain
arm the same module after ``unpatch_fast_lora_qkv``. Before anything is timed, one
forward+backward per arm checks that the outputs, the input gradient and every LoRA
gradient agree (relative error, ``--tolerance``). Then ``--warmup`` (default 20) and
``--iters`` (default 100) forward+backward iterations per arm, the arms alternated
fast, plain, plain, fast, ... One iteration is the three projections and
``torch.autograd.backward`` on fixed output gradients, timed with CUDA events; the
gradients are cleared and the patch toggled between iterations, outside the events.

- "ms" is the median of a process's timed iterations for that arm.
- "peak" is ``torch.cuda.max_memory_allocated()`` over one iteration, after
  ``reset_peak_memory_stats()``, the largest over the timed iterations; both arms
  hold the same resident weights, so the two peaks compare directly.
- "path" is proof of which path ran: the count ``patch_fast_lora_qkv`` returned (must
  be 1), the fast iterations whose cache-hit marker
  (``_soup_fast_lora_qkv_last_cache_hits``) read 2, and the plain iterations that ran
  unpatched (no patch marker, no ``_FastLoraQKVBackward`` grad_fn). Anything short of
  every iteration makes the process VOID.
- The SM clock is sampled with ``soup bench train``'s sampler over the timed
  iterations: min / median / max MHz.

Verdicts and exit status
------------------------
Per cell, from the processes' medians:

- ``faster`` / ``slower``: every fast/plain ratio falls on the same side of 1.00, and
  the gap between the arms' medians is larger than the spread (max - min) inside
  either arm.
- ``no measurable difference``: otherwise.
- ``VOID``: a process failed the agreement check or the path proof.
- ``OOM``: a process ran out of memory.

Exit 0 when no cell is VOID or OOM; 1 when a cell ran out of memory; 3 when a cell is
void; 2 for bad arguments or no CUDA device. ``--self-test`` exits 0 when every
check passes and 1 otherwise.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Optional

EXIT_OK = 0
EXIT_OOM = 1
EXIT_USAGE = 2
EXIT_VOID = 3

DTYPES = ("bf16", "nf4")
TARGETS = {"qkv": ("q_proj", "k_proj", "v_proj"), "qv": ("q_proj", "v_proj")}
FAST, PLAIN = "fast", "plain"
_HITS_MARKER = "_soup_fast_lora_qkv_last_cache_hits"
_PATCH_MARKER = "_soup_fast_lora_qkv"
_FAST_GRAD_FN = "_FastLoraQKVBackward"


@dataclasses.dataclass(frozen=True)
class Shapes:
    hidden: int = 4096
    q_out: int = 4096
    kv_out: int = 1024
    tokens: int = 2048


@dataclasses.dataclass(frozen=True)
class Cell:
    rank: int
    dtype: str
    targets: str

    @property
    def name(self) -> str:
        return f"r{self.rank} {self.dtype} {self.targets}"

    def encode(self) -> str:
        return f"{self.rank}:{self.dtype}:{self.targets}"

    @staticmethod
    def decode(value: str) -> "Cell":
        rank, dtype, targets = value.split(":")
        return Cell(int(rank), dtype, targets)


@dataclasses.dataclass
class ProcessResult:
    cell: str
    fast_ms: Optional[float] = None
    plain_ms: Optional[float] = None
    fast_peak: Optional[int] = None
    plain_peak: Optional[int] = None
    patched: Optional[int] = None
    fast_path_ok: int = 0
    plain_path_ok: int = 0
    iterations: int = 0
    agreement: Optional[float] = None
    clock: Optional[dict] = None
    order: list = dataclasses.field(default_factory=list)
    void: Optional[str] = None
    oom: bool = False


def schedule(count: int) -> list[str]:
    """``count`` iterations per arm, alternated fast, plain, plain, fast, ..."""
    order = []
    for i in range(count):
        order.extend((FAST, PLAIN) if i % 2 == 0 else (PLAIN, FAST))
    return order


# --- building the block ---------------------------------------------------


def build(cell: Cell, shapes: Shapes, device: str, seed: int) -> tuple[Any, Any, Any, tuple]:
    """``(model, attention, x, output_grads)``: one attention block's projections with LoRA."""
    import torch
    import torch.nn as nn
    from peft import LoraConfig, inject_adapter_in_model

    torch.manual_seed(seed)
    sizes = {"q_proj": shapes.q_out, "k_proj": shapes.kv_out, "v_proj": shapes.kv_out}

    def projection(out_features: int) -> nn.Module:
        if cell.dtype == "nf4":
            import bitsandbytes as bnb

            return bnb.nn.Linear4bit(
                shapes.hidden,
                out_features,
                bias=False,
                compute_dtype=torch.bfloat16,
                quant_type="nf4",
            )
        return nn.Linear(shapes.hidden, out_features, bias=False, dtype=torch.bfloat16)

    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name, out_features in sizes.items():
                setattr(self, name, projection(out_features))

        def forward(self, x):
            return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

        def forward(self, x):
            return self.self_attn(x)

    model = Block().to(device)  # Linear4bit quantizes on the move
    inject_adapter_in_model(
        LoraConfig(
            r=cell.rank,
            lora_alpha=2 * cell.rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(TARGETS[cell.targets]),
        ),
        model,
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
                if "lora_B" in name:
                    torch.nn.init.normal_(param, std=0.02)
    model.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    x = torch.randn(1, shapes.tokens, shapes.hidden, device=device, dtype=torch.bfloat16)
    x.requires_grad_(True)
    grads = tuple(
        torch.randn(1, shapes.tokens, sizes[name], device=device, dtype=torch.bfloat16)
        for name in ("q_proj", "k_proj", "v_proj")
    )
    return model, model.self_attn, x, grads


def lora_params(model: Any) -> list:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def step(model: Any, x: Any, grads: tuple) -> tuple:
    """One forward+backward of the three projections; returns the outputs."""
    import torch

    outputs = model(x)
    torch.autograd.backward(outputs, grads)
    return outputs


def clear(model: Any, x: Any) -> None:
    x.grad = None
    for param in lora_params(model):
        param.grad = None


def relative_error(actual: Any, expected: Any) -> float:
    a, e = actual.detach().float(), expected.detach().float()
    norm = e.norm().item()
    return (a - e).norm().item() / (norm if norm else 1.0)


def agreement(
    model: Any, attn: Any, x: Any, grads: tuple, patch: Callable, unpatch: Callable
) -> tuple[float, Optional[int]]:
    """Largest relative error between the arms over outputs, dX and every LoRA grad.

    Leaves the module unpatched. Returns ``(error, what patch returned)``.
    """
    clear(model, x)
    plain_out = [t.detach().clone() for t in step(model, x, grads)]
    plain_grads = [x.grad.detach().clone()] + [p.grad.detach().clone() for p in lora_params(model)]
    clear(model, x)
    patched = patch(model)
    fast_out = step(model, x, grads)
    fast_grads = [x.grad] + [p.grad for p in lora_params(model)]
    pairs = list(zip(fast_out, plain_out)) + list(zip(fast_grads, plain_grads))
    worst = max(relative_error(a, e) for a, e in pairs)
    unpatch(model)
    clear(model, x)
    return worst, patched


def took_fast_path(attn: Any, outputs: tuple) -> bool:
    hits = getattr(attn, _HITS_MARKER, (0, ()))[0]
    return hits == 2 and type(outputs[0].grad_fn).__name__ == _FAST_GRAD_FN


def ran_unpatched(attn: Any, outputs: tuple) -> bool:
    return (
        not getattr(attn, _PATCH_MARKER, False)
        and type(outputs[0].grad_fn).__name__ != _FAST_GRAD_FN
    )


class _CudaTimer:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.pairs: list = []

    def start(self) -> Any:
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def stop(self, started: Any) -> None:
        end = self.torch.cuda.Event(enable_timing=True)
        end.record()
        self.pairs.append((started, end))

    def millis(self) -> list[float]:
        self.torch.cuda.synchronize()
        return [s.elapsed_time(e) for s, e in self.pairs]


class _WallTimer:
    """CPU stand-in for the self-test; its figures are never a result."""

    def __init__(self) -> None:
        self.values: list[float] = []

    def start(self) -> float:
        return time.perf_counter()

    def stop(self, started: float) -> None:
        self.values.append((time.perf_counter() - started) * 1000)

    def millis(self) -> list[float]:
        return list(self.values)


def measure(
    cell: Cell,
    shapes: Shapes,
    device: str,
    *,
    warmup: int,
    iters: int,
    tolerance: float,
    seed: int = 0,
    patch: Optional[Callable] = None,
) -> ProcessResult:
    """One process's run of one cell: agreement check, then the alternated arms."""
    import torch

    from soup_cli.utils.fast_lora_qkv import patch_fast_lora_qkv, unpatch_fast_lora_qkv

    patch = patch or patch_fast_lora_qkv
    result = ProcessResult(cell=cell.encode())
    on_cuda = device == "cuda"
    try:
        model, attn, x, grads = build(cell, shapes, device, seed)
        error, patched = agreement(model, attn, x, grads, patch, unpatch_fast_lora_qkv)
        result.agreement, result.patched = error, patched
        if patched != 1:
            result.void = f"patch_fast_lora_qkv returned {patched}, not 1"
            return result
        if not error <= tolerance:
            result.void = f"the arms disagree before timing: relative error {error:.3g}"
            return result

        def run(arm: str, timer: Any = None) -> bool:
            if arm == FAST:
                patch(model)
            clear(model, x)
            if on_cuda and timer is not None:
                torch.cuda.reset_peak_memory_stats()
            started = timer.start() if timer is not None else None
            outputs = step(model, x, grads)
            if timer is not None:
                timer.stop(started)
            ok = took_fast_path(attn, outputs) if arm == FAST else ran_unpatched(attn, outputs)
            if arm == FAST:
                unpatch_fast_lora_qkv(model)
            return ok

        for arm in schedule(warmup):
            run(arm)
        timers = {arm: _CudaTimer(torch) if on_cuda else _WallTimer() for arm in (FAST, PLAIN)}
        peaks = {FAST: 0, PLAIN: 0}
        sampler = window = None
        if on_cuda:
            torch.cuda.synchronize()
            from soup_cli.bench.train_run import ClockSampler, _pci_bus_id

            sampler = ClockSampler(gpu=_pci_bus_id(torch.cuda.get_device_properties(0))).start()
            window = [time.perf_counter(), None]
        for arm in schedule(iters):
            ok = run(arm, timers[arm])
            result.order.append(arm)
            if ok:
                if arm == FAST:
                    result.fast_path_ok += 1
                else:
                    result.plain_path_ok += 1
            if on_cuda:
                peaks[arm] = max(peaks[arm], torch.cuda.max_memory_allocated())
        times = {arm: timers[arm].millis() for arm in (FAST, PLAIN)}
        if on_cuda:
            window[1] = time.perf_counter()
            from soup_cli.bench.train_report import summarize_clock_samples

            reason = None if sampler.available else "no nvidia-smi tool found"
            result.clock = summarize_clock_samples(
                sampler.stop(), window[0], window[1], unavailable_reason=reason
            )
            result.fast_peak, result.plain_peak = peaks[FAST], peaks[PLAIN]
        result.iterations = iters
        result.fast_ms = statistics.median(times[FAST])
        result.plain_ms = statistics.median(times[PLAIN])
        if result.fast_path_ok != iters or result.plain_path_ok != iters:
            result.void = (
                f"path proof: {result.fast_path_ok}/{iters} fast iterations took the fast "
                f"path, {result.plain_path_ok}/{iters} plain iterations ran unpatched"
            )
    except torch.cuda.OutOfMemoryError:
        result.oom = True
    return result


# --- verdicts -------------------------------------------------------------


def cell_verdict(results: list[ProcessResult]) -> str:
    if any(r.void for r in results):
        return "VOID"
    if any(r.oom for r in results):
        return "OOM"
    fast = [r.fast_ms for r in results]
    plain = [r.plain_ms for r in results]
    ratios = [f / p for f, p in zip(fast, plain)]
    gap = abs(statistics.median(fast) - statistics.median(plain))
    spread = max(max(fast) - min(fast), max(plain) - min(plain))
    if gap > spread and all(r < 1.0 for r in ratios):
        return "faster"
    if gap > spread and all(r > 1.0 for r in ratios):
        return "slower"
    return "no measurable difference"


def exit_status(verdicts: list[str]) -> int:
    if "VOID" in verdicts:
        return EXIT_VOID
    if "OOM" in verdicts:
        return EXIT_OOM
    return EXIT_OK


def _mb(value: Optional[int]) -> str:
    return "-" if value is None else f"{value / 1e6:.1f}MB"


def process_row(index: int, r: ProcessResult) -> str:
    if r.oom:
        return f"  {index}  OOM"
    if r.void and r.fast_ms is None:
        return f"  {index}  VOID: {r.void}"
    clock = r.clock or {}
    clock_text = (
        f"{clock.get('min')}/{clock.get('median')}/{clock.get('max')} MHz"
        if clock.get("median") is not None
        else f"clock n/a ({clock.get('unavailable_reason')})"
    )
    row = (
        f"  {index}  fast {r.fast_ms:9.3f}ms  plain {r.plain_ms:9.3f}ms  "
        f"fast/plain {r.fast_ms / r.plain_ms:.4f}  peak fast {_mb(r.fast_peak)} "
        f"plain {_mb(r.plain_peak)}  path {r.patched} {r.fast_path_ok}/{r.iterations} "
        f"{r.plain_path_ok}/{r.iterations}  agree {r.agreement:.2e}  {clock_text}"
    )
    return row + (f"  VOID: {r.void}" if r.void else "")


# --- processes ------------------------------------------------------------


def run_cell_process(
    cell: Cell,
    shapes: Shapes,
    args: argparse.Namespace,
    device: str,
    work_dir: pathlib.Path,
    index: int,
) -> ProcessResult:
    """One process of one cell in a fresh interpreter."""
    out = work_dir / f"{cell.encode().replace(':', '-')}-{index}.json"
    argv = [
        sys.executable, str(pathlib.Path(__file__).resolve()),
        "--cell", cell.encode(), "--cell-out", str(out), "--device", device,
        "--hidden", str(shapes.hidden), "--q-out", str(shapes.q_out),
        "--kv-out", str(shapes.kv_out), "--tokens", str(shapes.tokens),
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--tolerance", str(args.tolerance),
    ]  # fmt: skip
    sys.stdout.flush()
    status = subprocess.run(argv, check=False).returncode
    if status != 0 or not out.exists():
        raise RuntimeError(f"{cell.name}: process {index} exited {status} without a result")
    return ProcessResult(**json.loads(out.read_text(encoding="utf-8")))


def run_cell_child(args: argparse.Namespace) -> int:
    cell = Cell.decode(args.cell)
    shapes = Shapes(args.hidden, args.q_out, args.kv_out, args.tokens)
    result = measure(
        cell, shapes, args.device, warmup=args.warmup, iters=args.iters, tolerance=args.tolerance
    )
    pathlib.Path(args.cell_out).write_text(json.dumps(dataclasses.asdict(result)), encoding="utf-8")
    return EXIT_OK


def machine_lines() -> list[str]:
    """Everything about the box that is readable without starting CUDA here."""
    import importlib.metadata as md

    import torch

    import soup_cli
    from soup_cli.bench.train_run import _driver_version, _nvidia_smi
    from soup_cli.utils.layer_stream import _resolve_tool

    def version(name: str) -> str:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return "not installed"

    tool = _resolve_tool("nvidia-smi")
    query = (lambda q: _nvidia_smi(tool, q)) if tool else (lambda q: None)
    mem_total = None
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total = f"{int(line.split()[1]) / 1024 / 1024:.1f} GiB"
    except OSError:
        pass
    wsl = "microsoft" in platform.release().lower()
    script = pathlib.Path(__file__).resolve()
    return [
        f"# script {script.name} sha256 {hashlib.sha256(script.read_bytes()).hexdigest()}",
        f"# soup_cli {soup_cli.__version__} from {pathlib.Path(soup_cli.__file__).parent}",
        f"# card {query('--query-gpu=name')} | memory {query('--query-gpu=memory.total')} MiB"
        f" | driver {_driver_version()} | power limit {query('--query-gpu=power.limit')} W",
        f"# torch {torch.__version__} (CUDA runtime {torch.version.cuda}) | bitsandbytes "
        f"{version('bitsandbytes')} | peft {version('peft')} | transformers "
        f"{version('transformers')}",
        f"# {platform.system()} {platform.release()}"
        + (f" | WSL2 memory {mem_total}" if wsl else f" | host memory {mem_total}"),
    ]


def sweep(cells: list[Cell], shapes: Shapes, args: argparse.Namespace) -> int:
    for line in machine_lines():
        print(line)
    print(f"# power source: {args.power_source} (stated by the operator)")
    print(
        f"# shapes d={shapes.hidden} q_out={shapes.q_out} kv_out={shapes.kv_out} "
        f"N={shapes.tokens} | {args.processes} processes per cell | {args.warmup} warm-up + "
        f"{args.iters} timed iterations per arm, alternated | tolerance {args.tolerance}"
    )
    verdicts = []
    with tempfile.TemporaryDirectory(prefix="fast_lora_qkv_bench_") as tmp:
        for cell in cells:
            print(f"{cell.name}")
            results = []
            for index in range(1, args.processes + 1):
                result = run_cell_process(cell, shapes, args, "cuda", pathlib.Path(tmp), index)
                print(process_row(index, result), flush=True)
                results.append(result)
            verdict = cell_verdict(results)
            print(f"  verdict: {verdict}", flush=True)
            verdicts.append(verdict)
    return exit_status(verdicts)


# --- --self-test: tiny shapes on the CPU, no figure read --------------------


def run_self_test() -> int:
    tiny = Shapes(hidden=32, q_out=32, kv_out=16, tokens=8)
    cells = [Cell(2, "bf16", "qkv"), Cell(2, "bf16", "qv")]
    try:
        import bitsandbytes  # noqa: F401

        cells.append(Cell(2, "nf4", "qkv"))
    except ImportError:
        print("# bitsandbytes not installed: the nf4 cell is not self-tested")
    checks: list[tuple[str, bool]] = []
    iters = 4
    for cell in cells:
        r = measure(cell, tiny, "cpu", warmup=2, iters=iters, tolerance=2e-2)
        checks += [
            (f"{cell.name}: not void ({r.void})", r.void is None and not r.oom),
            (f"{cell.name}: patch count is 1", r.patched == 1),
            (f"{cell.name}: the arms agree", r.agreement is not None and r.agreement <= 2e-2),
            (f"{cell.name}: every fast iteration took the fast path", r.fast_path_ok == iters),
            (f"{cell.name}: every plain iteration ran unpatched", r.plain_path_ok == iters),
            (f"{cell.name}: arms alternated", r.order == schedule(iters)),
        ]
    # Negative controls: a patch that does nothing, and one that patches but is unpatched.
    cell = cells[0]
    r = measure(cell, tiny, "cpu", warmup=1, iters=2, tolerance=2e-2, patch=lambda m: 0)
    checks.append(("a patch that applies nothing is VOID", r.void is not None and r.patched == 0))

    r = measure(cell, tiny, "cpu", warmup=1, iters=2, tolerance=2e-2, patch=lambda m: 1)
    checks.append(
        (
            "a patch count that lies is VOID by the path proof",
            r.agreement == 0.0 and r.void is not None and "path proof" in r.void,
        )
    )
    checks.append(
        (
            "the agreement check sees a changed output",
            relative_error(_shifted(), _shifted(0.0)) > 2e-2,
        )
    )

    # The verdict rule on fixed medians.
    def proc(fast: float, plain: float, **extra: Any) -> ProcessResult:
        return ProcessResult(cell="x", fast_ms=fast, plain_ms=plain, **extra)

    rule = [
        ("faster", [proc(9.0, 10.0), proc(9.1, 10.1), proc(9.2, 10.0)]),
        ("slower", [proc(11.0, 10.0), proc(11.1, 10.1), proc(11.2, 10.0)]),
        ("no measurable difference", [proc(9.0, 10.0), proc(10.5, 10.0), proc(9.5, 10.0)]),
        ("no measurable difference", [proc(9.9, 10.0), proc(9.0, 10.0), proc(9.95, 10.0)]),
        ("VOID", [proc(9.0, 10.0, void="path"), proc(9.0, 10.0), proc(9.0, 10.0)]),
        ("OOM", [proc(9.0, 10.0), ProcessResult(cell="x", oom=True), proc(9.0, 10.0)]),
    ]
    for wanted, results in rule:
        checks.append((f"verdict {wanted}", cell_verdict(results) == wanted))
    checks += [
        ("exit 0 with no void or OOM cell", exit_status(["faster", "slower"]) == EXIT_OK),
        ("exit 1 on an OOM cell", exit_status(["faster", "OOM"]) == EXIT_OOM),
        ("exit 3 on a void cell", exit_status(["OOM", "VOID"]) == EXIT_VOID),
    ]
    # One cell through a child process, as the sweep runs it.
    with tempfile.TemporaryDirectory(prefix="fast_lora_qkv_bench_self_test_") as tmp:
        args = argparse.Namespace(warmup=1, iters=2, tolerance=2e-2)
        child = run_cell_process(cells[1], tiny, args, "cpu", pathlib.Path(tmp), 1)
        checks.append(
            (
                "a cell runs in a child process",
                child.void is None and child.plain_path_ok == 2 and child.fast_path_ok == 2,
            )
        )
    failed = 0
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        failed += not ok
    print(f"self-test: {len(checks) - failed} passed, {failed} failed (no time or peak measured)")
    return EXIT_OK if failed == 0 else 1


def _shifted(offset: float = 1.0) -> Any:
    import torch

    return torch.ones(4) + offset


# --- arguments -----------------------------------------------------------------


def _ints(value: str, flag: str) -> list[int]:
    try:
        numbers = [int(part) for part in value.split(",")]
    except ValueError:
        raise ValueError(f"{flag} must be comma-separated integers; got {value!r}") from None
    if not numbers or any(n <= 0 for n in numbers):
        raise ValueError(f"every {flag} entry must be positive")
    return numbers


def _names(value: str, flag: str, allowed: tuple) -> list[str]:
    names = [part.strip() for part in value.split(",")]
    unknown = [n for n in names if n not in allowed]
    if unknown or not names:
        raise ValueError(f"{flag} takes {', '.join(allowed)}; got {value!r}")
    return names


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ranks", default="16,64", help="comma-separated LoRA ranks")
    parser.add_argument("--dtypes", default="bf16,nf4", help="bf16 and/or nf4")
    parser.add_argument("--targets", default="qkv,qv", help="qkv and/or qv (k base-only)")
    parser.add_argument("--processes", type=int, default=3, help="processes per cell")
    parser.add_argument("--warmup", type=int, default=20, help="warm-up iterations per arm")
    parser.add_argument("--iters", type=int, default=100, help="timed iterations per arm")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="largest relative error between the arms before timing",
    )
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--q-out", type=int, default=4096)
    parser.add_argument("--kv-out", type=int, default=1024)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument(
        "--power-source",
        default="not stated",
        help="mains or battery, printed with the machine facts",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="the protocol at tiny shapes on the CPU, reading no figure",
    )
    # One process of one cell, run by the sweep in an interpreter of its own.
    parser.add_argument("--cell", help=argparse.SUPPRESS)
    parser.add_argument("--cell-out", help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.cell is not None:
        if args.cell_out is None:
            print("ERROR: --cell needs --cell-out")
            return EXIT_USAGE
        return run_cell_child(args)
    try:
        ranks = _ints(args.ranks, "--ranks")
        dtypes = _names(args.dtypes, "--dtypes", DTYPES)
        targets = _names(args.targets, "--targets", tuple(TARGETS))
        _ints(
            ",".join(
                str(v)
                for v in (
                    args.processes,
                    args.warmup,
                    args.iters,
                    args.hidden,
                    args.q_out,
                    args.kv_out,
                    args.tokens,
                )
            ),
            "--processes/--warmup/--iters/shape",
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return EXIT_USAGE
    if not 0 < args.tolerance < 1:
        print("ERROR: --tolerance must be between 0 and 1")
        return EXIT_USAGE

    import torch

    if not torch.cuda.is_available():
        print(
            "ERROR: CUDA required: every figure in this table is a CUDA event or a device "
            "peak (try --self-test)"
        )
        return EXIT_USAGE
    os.environ.setdefault("PYTHONHASHSEED", "0")
    cells = [Cell(r, d, t) for d in dtypes for t in targets for r in ranks]
    shapes = Shapes(args.hidden, args.q_out, args.kv_out, args.tokens)
    return sweep(cells, shapes, args)


if __name__ == "__main__":
    sys.exit(main())
