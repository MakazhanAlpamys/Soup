#!/usr/bin/env python3
"""Layer-streaming throughput probe: what bounds the streamed step on THIS card.

Re-runnable form of the instrumentation behind
``benchmarks/probe-v0.73.0-what-bounds-streaming.md``. Those scripts lived in a
scratchpad and left with the machine (#379); this one ships in the repo so the
next card can be measured the same way, through the SHIPPED streaming path
(``shard_checkpoint`` -> ``build_streamed_model``), never a re-implementation.

Everything runs in ONE process so the GEMM ceiling and the streamed step share
a session and a clock: a fraction-of-ceiling across sessions is meaningless
(the old box spread 13% on boost clock alone).

Modes, combinable::

  --ceiling  the shipped 4096^3 GEMM probe plus shape-matched projection GEMMs
             at M = batch * seq, FLOP-weighted per decoder layer
  --step     warm-up + timed steps uninstrumented, then the same steps with the
             CUDA-event instrumentation on (the instrumentation must be free)
  --sweep    sequence-length sweep at the configured batch
  --ablate   interleaved arms per round: A baseline / B no host-to-device
             copies / C no NF4 dequantisation / D neither. B, C and D are
             TIMING-ONLY and compute garbage (stale buffers, a cached zero
             weight); they run LAST because they corrupt the adapters.

Every point is appended to ``--out`` the moment it exists: a 20-minute sweep
once died with an empty log, and that is how the data was lost.

Typical invocation::

    python benchmarks/harness/stream_probe.py \
        --weights unsloth/mistral-7b-instruct-v0.3 --quant nf4 \
        --seq 512 --batch 1 --steps 8 --warmup 3 \
        --ceiling --step --ablate --rounds 2 --out probe.json

A machine without CUDA is an intentional skip and exits 0.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SWEEP_DEFAULT = "16,32,64,128,256,384,512"
ARMS = (
    ("A_baseline", False, False),
    ("B_nocopy", True, False),
    ("C_nodequant", False, True),
    ("D_neither", True, True),
)


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", required=True, help="model id or checkpoint path")
    parser.add_argument(
        "--shards",
        default=None,
        help="shard cache dir (default: the same ~/.soup/layer-stream/<slug> soup train uses)",
    )
    parser.add_argument("--quant", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--tier", choices=("ram", "disk"), default="ram")
    parser.add_argument("--no-pin", action="store_true", help="pageable RAM store")
    parser.add_argument("--buffers", type=int, default=2)
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=8, help="timed steps per point")
    parser.add_argument("--warmup", type=int, default=3, help="untimed steps per point")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--seed", type=int, default=3, help="adapter init seed")
    parser.add_argument("--input-seed", type=int, default=17)
    parser.add_argument("--ceiling", action="store_true")
    parser.add_argument("--step", action="store_true")
    parser.add_argument("--sweep", nargs="?", const=SWEEP_DEFAULT, default=None)
    parser.add_argument("--ablate", action="store_true")
    parser.add_argument("--rounds", type=int, default=2, help="interleaved ablation rounds")
    parser.add_argument(
        "--arms",
        default=None,
        help=(
            "comma-separated subset of the ablation arms (A_baseline,B_nocopy,C_nodequant,"
            "D_neither); default all. On a cold disk tier a baseline step is minutes, so "
            "B and D alone give the compute floor cheaply"
        ),
    )
    parser.add_argument(
        "--evict-gb",
        type=float,
        default=0.0,
        help=(
            "HEURISTIC page-cache eviction before each timed block on the disk tier: "
            "commit this many GB of pageable RAM and free it"
        ),
    )
    parser.add_argument(
        "--lazy-shard-handles",
        action="store_true",
        help=(
            "MEASUREMENT WORKAROUND for the shipped sharder, which holds every source "
            "shard's mmap open at once and dies with an access violation on Windows past "
            "~100 GB of live mappings: during sharding ONLY, safe_open is replaced by a "
            "wrapper that opens at most two files at a time and returns copies. The "
            "runtime and every timed number still come from the shipped code"
        ),
    )
    parser.add_argument("--out", required=True, help="JSON results file, rewritten per point")
    parser.add_argument("--label", default="", help="free text stored beside the results")
    return parser.parse_args()


# ==========================================================================
# sharding workaround (see --lazy-shard-handles)
# ==========================================================================
class _LazyHandleLRU:
    """At most ``capacity`` real ``safe_open`` handles alive at once.

    The shipped sharder enters every source shard into one ExitStack and keeps
    the mappings for the whole of pass 2. On this Windows box that pattern dies
    with an access violation once ~100 GB of mappings are live, whatever the
    file count (measured: 62 x 1.71 GB and 14 x 6.85 GB). Opening a file only
    while a layer needs it is the fix shape; this wrapper applies it from the
    outside so the sharder's own logic is exercised unchanged.
    """

    def __init__(self, real_safe_open: Any, capacity: int = 2):
        self.real = real_safe_open
        self.capacity = capacity
        self.order: List[Any] = []

    def touch(self, handle: Any) -> None:
        if handle in self.order:
            self.order.remove(handle)
        self.order.append(handle)
        while len(self.order) > self.capacity:
            victim = self.order.pop(0)
            victim.release()


class _LazyHandle:
    def __init__(self, lru: _LazyHandleLRU, path: str, framework: str, device: str):
        self.lru = lru
        self.path = path
        self.framework = framework
        self.device = device
        self._real: Any = None

    def _handle(self) -> Any:
        if self._real is None:
            self._real = self.lru.real(self.path, framework=self.framework, device=self.device)
            self._real.__enter__()
        self.lru.touch(self)
        return self._real

    def release(self) -> None:
        if self._real is not None:
            self._real.__exit__(None, None, None)
            self._real = None

    def __enter__(self) -> "_LazyHandle":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()
        if self in self.lru.order:
            self.lru.order.remove(self)

    def keys(self) -> Any:
        return self._handle().keys()

    def metadata(self) -> Any:
        return self._handle().metadata()

    def get_slice(self, key: str) -> Any:
        return self._handle().get_slice(key)

    def get_tensor(self, key: str) -> Any:
        # A COPY, never the zero-copy view: the sharder keeps non-quantised
        # tensors (norms, embeddings, the head) until it writes them, and a view
        # over a released mapping is exactly the access violation being avoided.
        return self._handle().get_tensor(key).clone()


class lazy_shard_handles:  # noqa: N801 — context manager, used like a function
    """Patch ``safetensors.safe_open`` for the duration of one ``with`` block."""

    def __init__(self, capacity: int = 2):
        self.capacity = capacity
        self._saved: Any = None

    def __enter__(self) -> "lazy_shard_handles":
        import safetensors

        self._saved = safetensors.safe_open
        lru = _LazyHandleLRU(self._saved, self.capacity)

        def patched(path: Any, framework: str = "pt", device: str = "cpu") -> _LazyHandle:
            return _LazyHandle(lru, str(path), framework, device)

        safetensors.safe_open = patched
        return self

    def __exit__(self, *exc: Any) -> None:
        import safetensors

        safetensors.safe_open = self._saved


# ==========================================================================
# model + shapes
# ==========================================================================
def make_lora(args: argparse.Namespace):
    from peft import LoraConfig, TaskType

    targets = [name.strip() for name in args.lora_targets.split(",") if name.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type=TaskType.CAUSAL_LM,
    )


def model_shapes(config: Any) -> Dict[str, Any]:
    """Projection shapes and the FLOP-per-token accounting the gates use."""
    hidden = int(config.hidden_size)
    inter = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
    head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    vocab = int(config.vocab_size)
    # q, o ; k, v ; gate, up, down. Biases and norms are noise at this scale.
    per_layer = 2 * hidden * q_out + 2 * hidden * kv_out + 3 * hidden * inter
    decoder = per_layer * layers
    lm_head = vocab * hidden
    # C=6 on the frozen decoder (forward + recompute + dL/dx, no dL/dW); the
    # head is not checkpointed and has no weight grad either: C=4. Embeddings 0.
    flop_per_token = 6 * decoder + 4 * lm_head
    projections = {
        "q_proj": (hidden, q_out, 1),
        "k_proj": (hidden, kv_out, 1),
        "v_proj": (hidden, kv_out, 1),
        "o_proj": (q_out, hidden, 1),
        "gate_proj": (hidden, inter, 1),
        "up_proj": (hidden, inter, 1),
        "down_proj": (inter, hidden, 1),
    }
    return {
        "hidden": hidden,
        "intermediate": inter,
        "layers": layers,
        "kv_out": kv_out,
        "q_out": q_out,
        "vocab": vocab,
        "decoder_params": decoder,
        "lm_head_params": lm_head,
        "flop_per_token": flop_per_token,
        "projections": projections,
        "tied": bool(getattr(config, "tie_word_embeddings", False)),
    }


def build(args: argparse.Namespace, device: str, dtype: str) -> Tuple[Any, ...]:
    from soup_cli.utils.layer_shard import resolve_shard_dir, shard_checkpoint
    from soup_cli.utils.layer_stream_runtime import (
        build_meta_skeleton,
        build_streamed_model,
        quantised_layer_suffixes,
    )
    from soup_cli.utils.spectrum_scan import resolve_model_weights

    weights_dir = resolve_model_weights(args.weights)
    shard_dir = args.shards or resolve_shard_dir(args.weights)
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    probe = build_meta_skeleton(weights_dir, dtype=dtype, quant=args.quant)
    config = probe.config
    arch = str(config.model_type)
    suffixes = quantised_layer_suffixes(probe) if args.quant == "nf4" else ()
    del probe

    started = time.perf_counter()
    import contextlib

    guard: Any = contextlib.nullcontext()
    if args.lazy_shard_handles:
        print(
            "sharding      WORKAROUND: safe_open patched to hold <= 2 handles and return "
            "copies (the shipped sharder dies past ~100 GB of live mappings on Windows)"
        )
        guard = lazy_shard_handles(capacity=2)
    with guard:
        index = shard_checkpoint(
            weights_dir,
            shard_dir,
            dtype=dtype,
            arch=arch,
            quant=args.quant,
            quant_suffixes=suffixes,
            double_quant=True,
            quant_device=device if args.quant == "nf4" else None,
            notify=print,
        )
    shard_seconds = time.perf_counter() - started

    started = time.perf_counter()
    model, runtime = build_streamed_model(
        model_id=weights_dir,
        shard_dir=shard_dir,
        index=index,
        lora_config=make_lora(args),
        device=device,
        dtype=dtype,
        buffers=args.buffers,
        pin=(args.tier == "ram" and not args.no_pin),
        seed=args.seed,
        quant=args.quant,
        double_quant=True,
        tier=args.tier,
    )
    build_seconds = time.perf_counter() - started
    return model, runtime, config, index, weights_dir, shard_dir, shard_seconds, build_seconds


# ==========================================================================
# instrumentation: replicas of the two shipped copy paths, plus stall timing
# ==========================================================================
class Instruments:
    """CUDA-event timing on the copy stream and the compute stream, and the
    two timing-only ablation switches. Installed on the live pool objects the
    prefetcher and the layer wrappers already hold, so the shipped scheduler
    is untouched."""

    def __init__(self, runtime: Any, model: Any, quant: str):
        self.pool = runtime.pool
        self.large_pool = runtime.large_pool
        self.events_on = False
        self.nocopy = False
        self.nodequant = False
        self.patched_linears = 0
        self._copy_pairs: List[Tuple[Any, Any]] = []
        self._stall_pairs: List[Tuple[Any, Any]] = []
        self._zero_cache: Dict[Tuple[Tuple[int, ...], Any], Any] = {}
        self._install_pool()
        self._install_large_pool()
        if quant == "nf4":
            self._install_nodequant(model)

    # -- drift guard: the replica below must stay a copy of the shipped body --
    @staticmethod
    def _guard(kind: Any, method: str, needles: Tuple[str, ...]) -> None:
        source = inspect.getsource(getattr(kind, method))
        missing = [needle for needle in needles if needle not in source]
        if missing:
            raise RuntimeError(
                f"shipped {kind.__name__}.{method} no longer contains {missing!r}; "
                "the probe's replica would measure a different scheduler. Update "
                "benchmarks/harness/stream_probe.py to match the shipped code."
            )

    def _install_pool(self) -> None:
        import torch

        pool = self.pool
        self._guard(
            type(pool),
            "load_async",
            (
                "stream.wait_stream(torch.cuda.current_stream())",
                "dst.copy_(source.get(idx, name), non_blocking=True)",
                "self.events[slot].record(stream)",
            ),
        )
        inst = self

        def load_async(idx: int, source: Any, stream: Any = None) -> int:
            slot = pool.slot_for(idx)
            keys = (
                pool.active_keys_by_layer[0]
                if len(pool.active_keys_by_layer) == 1
                else pool.active_keys_by_layer[idx]
            )
            if pool.is_cuda and stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    start = None
                    if inst.events_on:
                        start = torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                    if not inst.nocopy:
                        for name in keys:
                            dst = pool.buffers[slot][name]
                            dst.copy_(source.get(idx, name), non_blocking=True)
                    if inst.events_on:
                        end = torch.cuda.Event(enable_timing=True)
                        end.record(stream)
                        inst._copy_pairs.append((start, end))
                    pool.events[slot].record(stream)
            else:
                for name in keys:
                    pool.buffers[slot][name].copy_(source.get(idx, name))
            pool.owner[slot] = idx
            pool.loads += 1
            return slot

        original_wait = pool.wait

        def wait(idx: int) -> Any:
            if inst.events_on and pool.is_cuda:
                before = torch.cuda.Event(enable_timing=True)
                before.record()
                out = original_wait(idx)
                after = torch.cuda.Event(enable_timing=True)
                after.record()
                inst._stall_pairs.append((before, after))
                return out
            return original_wait(idx)

        pool.load_async = load_async
        pool.wait = wait

    def _install_large_pool(self) -> None:
        import torch

        large = self.large_pool
        if large is None:
            return
        self._guard(
            type(large),
            "load_async",
            (
                "if self.owner == key:",
                "stream.wait_stream(torch.cuda.current_stream())",
                "dst.copy_(source.get(source_idx, key), non_blocking=True)",
                "self.event.record(stream)",
            ),
        )
        inst = self

        def load_async(key: str, source: Any, stream: Any = None) -> None:
            if large.owner == key:
                return
            if key not in large.specs or key not in large.source_indices:
                raise ValueError(f"large-layer source is missing {key!r}")
            dst = large._view(key)
            source_idx = large.source_indices[key]
            if large.is_cuda and stream is not None:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    start = None
                    if inst.events_on:
                        start = torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                    if not inst.nocopy:
                        dst.copy_(source.get(source_idx, key), non_blocking=True)
                    if inst.events_on:
                        end = torch.cuda.Event(enable_timing=True)
                        end.record(stream)
                        inst._copy_pairs.append((start, end))
                    large.event.record(stream)
            else:
                dst.copy_(source.get(source_idx, key))
            large.owner = key
            large.loads += 1

        original_wait = large.wait

        def wait(key: str) -> Any:
            if inst.events_on and large.is_cuda:
                before = torch.cuda.Event(enable_timing=True)
                before.record()
                out = original_wait(key)
                after = torch.cuda.Event(enable_timing=True)
                after.record()
                inst._stall_pairs.append((before, after))
                return out
            return original_wait(key)

        large.load_async = load_async
        large.wait = wait

    def _install_nodequant(self, model: Any) -> None:
        import types

        import bitsandbytes as bnb
        import torch.nn.functional as functional

        inst = self
        count = 0
        for child in model.modules():
            if not isinstance(child, bnb.nn.Linear4bit):
                continue
            original = child.forward  # the #331 dequant-forward, bound

            def forward(self, x, _original=original):
                if not inst.nodequant:
                    return _original(x)
                quant_state = self.weight.quant_state
                inp_dtype = x.dtype
                if getattr(self, "compute_dtype", None) is not None:
                    x = x.to(self.compute_dtype)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(x.dtype)
                weight = inst.zero_weight(tuple(quant_state.shape), x.dtype, x.device)
                return functional.linear(x, weight, bias).to(inp_dtype)

            child.forward = types.MethodType(forward, child)
            count += 1
        self.patched_linears = count

    def zero_weight(self, shape: Tuple[int, ...], dtype: Any, device: Any) -> Any:
        import torch

        key = (shape, dtype)
        weight = self._zero_cache.get(key)
        if weight is None:
            weight = torch.zeros(shape, dtype=dtype, device=device)
            self._zero_cache[key] = weight
        return weight

    def reset(self) -> None:
        self._copy_pairs = []
        self._stall_pairs = []

    def consume(self) -> Tuple[float, float, int]:
        """(copy seconds on the prefetch stream, compute-stream stall seconds,
        number of copy events). Call only after torch.cuda.synchronize()."""
        copy_ms = sum(start.elapsed_time(end) for start, end in self._copy_pairs)
        stall_ms = sum(before.elapsed_time(after) for before, after in self._stall_pairs)
        copies = len(self._copy_pairs)
        self.reset()
        return copy_ms / 1000.0, stall_ms / 1000.0, copies


# ==========================================================================
# measurement
# ==========================================================================
def gpu_facts(device: str) -> Dict[str, Any]:
    import subprocess

    import torch

    from soup_cli.utils.layer_stream import _resolve_tool

    facts: Dict[str, Any] = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "vram_total_gb": round(torch.cuda.mem_get_info()[1] / 1e9, 3),
    }
    tool = _resolve_tool("nvidia-smi")
    if tool is not None:
        try:
            out = subprocess.run(
                [
                    tool,
                    "--query-gpu=driver_version,pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            driver, gen, width = [part.strip() for part in out.stdout.strip().split(",")]
            facts.update({"driver": driver, "pcie_gen": gen, "pcie_width": width})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return facts


def finite(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def evict_page_cache(gigabytes: float) -> None:
    """Heuristic: commit pageable RAM and release it so the standby list is
    pressured. Reported as a heuristic, never as a guarantee of cold reads."""
    import torch

    if gigabytes <= 0:
        return
    buffer = torch.empty(int(gigabytes * 1e9), dtype=torch.uint8, device="cpu")
    buffer.fill_(1)
    del buffer


def run_steps(
    model: Any,
    optimizer: Any,
    inst: Instruments,
    ids: Any,
    *,
    steps: int,
    warmup: int,
    label: str,
) -> Dict[str, Any]:
    import torch

    from soup_cli.utils.layer_stream_runtime import sm_clock_mhz

    pool = inst.pool
    large = inst.large_pool
    tokens = int(ids.numel())
    records: List[Dict[str, Any]] = []
    clock_start = sm_clock_mhz()
    for index in range(warmup + steps):
        loads_before = pool.loads
        large_before = large.loads if large is not None else 0
        inst.reset()
        torch.cuda.synchronize()
        if index == warmup:
            torch.cuda.reset_peak_memory_stats()
        wall_start = time.perf_counter()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        wall = time.perf_counter() - wall_start
        copy_s, stall_s, copies = inst.consume()
        if index >= warmup:
            records.append(
                {
                    "step_s": wall,
                    "loss": finite(float(out.loss.detach())),
                    "layer_loads": pool.loads - loads_before,
                    "large_loads": (large.loads - large_before) if large is not None else 0,
                    "copy_s": copy_s,
                    "stall_s": stall_s,
                    "copy_events": copies,
                }
            )
    clock_end = sm_clock_mhz()
    step_mean = sum(rec["step_s"] for rec in records) / len(records)
    per_layer_bytes = pool.nbytes // pool.n
    large_bytes = large.nbytes if large is not None else 0
    loads_mean = sum(rec["layer_loads"] for rec in records) / len(records)
    large_mean = sum(rec["large_loads"] for rec in records) / len(records)
    moved = loads_mean * per_layer_bytes + large_mean * large_bytes
    copy_mean = sum(rec["copy_s"] for rec in records) / len(records)
    stall_mean = sum(rec["stall_s"] for rec in records) / len(records)
    return {
        "label": label,
        "tokens_per_step": tokens,
        "steps": steps,
        "warmup": warmup,
        "events_on": inst.events_on,
        "nocopy": inst.nocopy,
        "nodequant": inst.nodequant,
        "step_s_mean": step_mean,
        "step_s_min": min(rec["step_s"] for rec in records),
        "step_s_max": max(rec["step_s"] for rec in records),
        "tok_per_s": tokens / step_mean,
        "layer_loads_per_step": loads_mean,
        "large_loads_per_step": large_mean,
        "bytes_moved_per_step": moved,
        "implied_h2d_gb_per_s": moved / step_mean / 1e9,
        "copy_s_per_step": copy_mean,
        "copy_stream_gb_per_s": (moved / copy_mean / 1e9) if copy_mean > 0 else None,
        "stall_s_per_step": stall_mean,
        "stall_share": stall_mean / step_mean,
        "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "sm_clock_mhz_start": clock_start,
        "sm_clock_mhz_end": clock_end,
        "records": records,
    }


def measure_ceiling(device: str, dtype: str, shapes: Dict[str, Any], tokens: int) -> Dict[str, Any]:
    import torch

    from soup_cli.utils.layer_stream_runtime import measure_gemm_tflops, sm_clock_mhz

    square = measure_gemm_tflops(device)
    torch_dtype = getattr(torch, dtype)
    grouped: Dict[Tuple[int, int], int] = {}
    for _name, (k_dim, n_dim, count) in shapes["projections"].items():
        grouped[(k_dim, n_dim)] = grouped.get((k_dim, n_dim), 0) + count
    per_shape: List[Dict[str, Any]] = []
    weighted_num = 0.0
    weighted_den = 0.0
    for (k_dim, n_dim), count in grouped.items():
        left = torch.randn(tokens, k_dim, device=device, dtype=torch_dtype)
        right = torch.randn(k_dim, n_dim, device=device, dtype=torch_dtype)
        flops = 2.0 * tokens * k_dim * n_dim
        best = 0.0
        for _rep in range(3):
            for _warm in range(3):
                left @ right
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            iters = 20
            for _it in range(iters):
                left @ right
            end.record()
            torch.cuda.synchronize()
            seconds = start.elapsed_time(end) / 1000.0
            best = max(best, flops * iters / seconds / 1e12)
        del left, right
        per_shape.append({"m": tokens, "k": k_dim, "n": n_dim, "count": count, "tflops": best})
        weighted_num += best * flops * count
        weighted_den += flops * count
    torch.cuda.empty_cache()
    square_payload = None
    if square is not None:
        square_payload = {
            "tflops": square.tflops,
            "samples": list(square.samples),
            "sm_clock_mhz": square.sm_clock_mhz,
            "dtype": square.dtype,
        }
    return {
        "square_4096": square_payload,
        "shape_matched": per_shape,
        "shape_matched_weighted_tflops": weighted_num / weighted_den,
        "sm_clock_mhz": sm_clock_mhz(),
        "dtype": dtype,
    }


class Sink:
    """Rewrite the JSON file after every point."""

    def __init__(self, path: str, meta: Dict[str, Any]):
        self.path = Path(path)
        self.payload: Dict[str, Any] = {"meta": meta, "records": []}
        self.flush()

    def add(self, kind: str, record: Dict[str, Any]) -> None:
        self.payload["records"].append({"kind": kind, **record})
        self.flush()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.payload, indent=1), encoding="utf-8")


def describe(summary: Dict[str, Any], flop_per_token: float, ceiling: Optional[float]) -> str:
    eff = flop_per_token * summary["tokens_per_step"] / summary["step_s_mean"] / 1e12
    parts = [
        f"{summary['label']:<24}",
        f"{summary['tok_per_s']:8.1f} tok/s",
        f"step {summary['step_s_mean']:.3f} s",
        f"peak {summary['peak_alloc_gb']:.3f} GB",
        f"loads {summary['layer_loads_per_step']:.1f}+{summary['large_loads_per_step']:.1f}",
        f"eff {eff:.2f} TFLOPS",
    ]
    if ceiling:
        parts.append(f"= {100.0 * eff / ceiling:.1f}% of ceiling")
    if summary["events_on"]:
        parts.append(
            f"copy {summary['copy_s_per_step']:.3f} s "
            f"({summary['copy_stream_gb_per_s'] or 0:.2f} GB/s) "
            f"stall {summary['stall_s_per_step'] * 1000:.1f} ms "
            f"({100 * summary['stall_share']:.2f}%)"
        )
    if summary["sm_clock_mhz_start"] is not None:
        parts.append(f"clock {summary['sm_clock_mhz_start']}->{summary['sm_clock_mhz_end']} MHz")
    return "  ".join(parts)


def main() -> int:
    args = parse_args()
    if not cuda_available():
        print("SKIP: CUDA is required for stream_probe.py")
        return 0
    if not (args.ceiling or args.step or args.sweep or args.ablate):
        print("ERROR: choose at least one of --ceiling --step --sweep --ablate")
        return 2

    import torch

    from soup_cli.utils.layer_stream import resolve_stream_dtype

    device = "cuda"
    dtype = resolve_stream_dtype(device)
    facts = gpu_facts(device)
    print("RUN: layer-streaming throughput probe")
    for key, value in facts.items():
        print(f"{key:<16}{value}")
    print(f"{'dtype':<16}{dtype}")
    print(f"{'quant':<16}{args.quant}")
    print(f"{'tier':<16}{args.tier}{' (pageable)' if args.no_pin else ''}")

    model, runtime, config, index, weights_dir, shard_dir, shard_s, build_s = build(
        args, device, dtype
    )
    shapes = model_shapes(config)
    stats = runtime.stats()
    print(f"{'arch':<16}{config.model_type}  layers {shapes['layers']}")
    print(f"{'shards':<16}{shard_dir}  ({shard_s:.1f} s)")
    print(
        f"{'store':<16}{stats['store_bytes'] / 1e9:.3f} GB "
        f"{'pinned' if stats['pinned'] else 'pageable'} on tier {stats['tier']} "
        f"(disk {stats['disk_bytes'] / 1e9:.3f} GB); build {build_s:.1f} s"
    )
    per_buffer_mb = stats["buffer_bytes"] / stats["buffers"] / 1e6
    print(f"{'buffers':<16}{stats['buffers']} x {per_buffer_mb:.1f} MB")

    import bitsandbytes as bnb

    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = bnb.optim.PagedAdamW8bit(trainable, lr=1e-4)
    inst = Instruments(runtime, model, args.quant)
    if args.quant == "nf4":
        print(f"{'nf4 linears':<16}{inst.patched_linears} patched for the ablation")

    meta = {
        **facts,
        "label": args.label,
        "weights": args.weights,
        "weights_dir": weights_dir,
        "shard_dir": shard_dir,
        "quant": args.quant,
        "dtype": dtype,
        "tier": stats["tier"],
        "pinned": stats["pinned"],
        "store_gb": stats["store_bytes"] / 1e9,
        "disk_gb": stats["disk_bytes"] / 1e9,
        "buffers": stats["buffers"],
        "buffer_bytes": stats["buffer_bytes"],
        "large_buffer_bytes": stats["large_buffer_bytes"],
        "n_layers": stats["n_layers"],
        "total_params": stats["total_params"],
        "shapes": {key: value for key, value in shapes.items() if key != "projections"},
        "lora_r": args.lora_r,
        "lora_targets": args.lora_targets,
        "batch": args.batch,
        "seq": args.seq,
        "steps": args.steps,
        "warmup": args.warmup,
        "shard_seconds": shard_s,
        "build_seconds": build_s,
        "evict_gb": args.evict_gb,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    sink = Sink(args.out, meta)
    flop_per_token = float(shapes["flop_per_token"])
    ceiling_tflops: Optional[float] = None

    generator = torch.Generator(device=device).manual_seed(args.input_seed)

    def make_ids(seq: int) -> Any:
        return torch.randint(
            0, shapes["vocab"], (args.batch, seq), generator=generator, device=device
        )

    if args.ceiling:
        print("ceiling       measuring (same session, same clock)...")
        ceiling = measure_ceiling(device, dtype, shapes, args.batch * args.seq)
        ceiling_tflops = ceiling["shape_matched_weighted_tflops"]
        sink.add("ceiling", ceiling)
        square = ceiling["square_4096"]
        square_tflops = square["tflops"] if square else float("nan")
        print(
            f"ceiling       square 4096^3 {square_tflops:.2f} TFLOPS; "
            f"shape-matched FLOP-weighted {ceiling_tflops:.2f} TFLOPS "
            f"@ {ceiling['sm_clock_mhz']} MHz"
        )

    if args.step:
        ids = make_ids(args.seq)
        evict_page_cache(args.evict_gb)
        inst.events_on = False
        plain = run_steps(
            model, optimizer, inst, ids, steps=args.steps, warmup=args.warmup, label="step_plain"
        )
        sink.add("step", plain)
        print(describe(plain, flop_per_token, ceiling_tflops))
        inst.events_on = True
        evict_page_cache(args.evict_gb)
        timed = run_steps(
            model, optimizer, inst, ids, steps=args.steps, warmup=args.warmup, label="step_events"
        )
        inst.events_on = False
        sink.add("step", timed)
        print(describe(timed, flop_per_token, ceiling_tflops))

    if args.sweep:
        seqs = [int(part) for part in args.sweep.split(",") if part.strip()]
        inst.events_on = True
        for seq in seqs:
            ids = make_ids(seq)
            evict_page_cache(args.evict_gb)
            point = run_steps(
                model,
                optimizer,
                inst,
                ids,
                steps=args.steps,
                warmup=args.warmup,
                label=f"sweep_seq{seq}",
            )
            point["seq"] = seq
            sink.add("sweep", point)
            print(describe(point, flop_per_token, ceiling_tflops))
        inst.events_on = False

    if args.ablate:
        ids = make_ids(args.seq)
        arms = ARMS if args.quant == "nf4" else ARMS[:2]
        if args.arms:
            wanted = {name.strip() for name in args.arms.split(",") if name.strip()}
            unknown = wanted - {arm[0] for arm in ARMS}
            if unknown:
                print(f"ERROR: unknown ablation arm(s) {sorted(unknown)}")
                return 2
            arms = tuple(arm for arm in arms if arm[0] in wanted)
        print(
            "ablate        arms B/C/D are timing-only and compute garbage; "
            f"{args.rounds} interleaved rounds"
        )
        inst.events_on = False
        for round_index in range(args.rounds):
            for name, nocopy, nodequant in arms:
                inst.nocopy = nocopy
                inst.nodequant = nodequant
                evict_page_cache(args.evict_gb)
                point = run_steps(
                    model,
                    optimizer,
                    inst,
                    ids,
                    steps=args.steps,
                    warmup=max(1, args.warmup),
                    label=f"{name}_r{round_index}",
                )
                point["arm"] = name
                point["round"] = round_index
                sink.add("ablate", point)
                print(describe(point, flop_per_token, ceiling_tflops))
        inst.nocopy = False
        inst.nodequant = False

    if args.ceiling:
        print("ceiling       re-measuring at the end of the session...")
        again = measure_ceiling(device, dtype, shapes, args.batch * args.seq)
        sink.add("ceiling_end", again)
        print(
            f"ceiling       shape-matched {again['shape_matched_weighted_tflops']:.2f} TFLOPS "
            f"@ {again['sm_clock_mhz']} MHz"
        )

    runtime.close()
    print(f"wrote         {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
