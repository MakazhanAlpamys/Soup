#!/usr/bin/env python3
"""#842 NF4 projection measurements; see ../gate-842-nf4-streaming.md.

No model downloads. --self-test runs the protocol on CPU without bitsandbytes.
--ablation measures the shipped checkpointed projection against cached dense W.
--grid measures the checkpoint-visible op against explicit dequantise + linear.
Every GPU cell runs in three fresh processes. JSON includes raw samples and paths.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter
from typing import Any
from unittest.mock import patch

from fast_lora_qkv_bench import ProcessResult, cell_verdict

SEED = 842
WARMUP, ITERATIONS, PROCESSES = 20, 100, 3
SHAPES = ((4096, 4096), (1024, 4096), (14336, 4096), (4096, 14336))
TOKENS = (3, 8, 32, 128, 512)


def order(count: int) -> list[str]:
    return [arm for i in range(count) for arm in (("a", "b") if i % 2 == 0 else ("b", "a"))]


def summary(rows: list[dict]) -> dict:
    """Apply #838's noise rule, with B as candidate and A as baseline."""
    if len(rows) != PROCESSES or any(row.get("error") for row in rows):
        return {"verdict": "INVALID", "small_share": False}
    medians = [(r["a"]["total_ms"], r["b"]["total_ms"]) for r in rows]
    if any(not math.isfinite(v) or v <= 0 for pair in medians for v in pair):
        return {"verdict": "INVALID", "small_share": False}
    results = [ProcessResult(cell="nf4", fast_ms=b, plain_ms=a) for a, b in medians]
    reductions = [1 - b / a for a, b in medians]
    return {
        "verdict": cell_verdict(results),
        "reduction_by_process": reductions,
        "small_share": all(0.98 * a <= b <= a for a, b in medians),
    }


def difference(left: Any, right: Any) -> dict:
    import torch

    finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
    return {
        "finite": finite,
        "exact": finite
        and left.shape == right.shape
        and left.dtype == right.dtype
        and torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)),
        "max_abs": float((left.float() - right.float()).abs().max()) if finite else None,
    }


@contextlib.contextmanager
def count_paths():
    """Count executed backend implementations, never infer execution from a gate."""
    import bitsandbytes.functional as functional
    from bitsandbytes.backends.cuda import ops

    counts = Counter()

    def wrap(name, original):
        def counted(*args, **kwargs):
            counts[name] += 1
            return original(*args, **kwargs)

        return counted

    with contextlib.ExitStack() as stack:
        for module, name, label in (
            (ops, "_gemm_4bit_kernel_impl", "kernel"),
            (ops, "_dequant_linear_fallback", "fallback"),
            (functional, "dequantize_4bit", "dequant"),
        ):
            original = getattr(module, name)  # Missing hooks invalidate the run.
            stack.enter_context(patch.object(module, name, wrap(label, original)))
        yield counts


def hardware() -> dict:
    import torch

    from soup_cli.bench.train_run import _driver_version

    props = torch.cuda.get_device_properties(0)
    packages = {}
    for name in ("torch", "bitsandbytes", "peft", "transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    return {
        "card": props.name,
        "sm": f"{props.major}{props.minor}",
        "vram_bytes": props.total_memory,
        "cuda_runtime": torch.version.cuda,
        "driver": _driver_version(),
        "packages": packages,
        "dtype": "native fp16",
        "python": sys.version,
    }


def run_iterations(step, *, warmup: int, iterations: int) -> dict:
    """The CPU self-test and CUDA runner share the arm order and sample collection."""
    for arm in order(warmup):
        step(arm, False)
    rows = {"a": [], "b": []}
    for arm in order(iterations):
        rows[arm].append(step(arm, True))
    return rows


def validate_paths(mode: str, arm: str, forward: dict, backward: dict) -> None:
    kernel = forward.get("kernel", 0)
    fallback = forward.get("fallback", 0)
    dequant = forward.get("dequant", 0)
    if mode == "ablation":
        expected = 1 if arm == "a" else 0
        if arm == "b" and forward.get("cached_linear") != 1:
            raise ValueError(f"cached dense path missing: {forward}")
        if kernel or fallback or dequant != expected or backward.get("dequant", 0) != expected:
            raise ValueError(f"ablation path changed: {arm} {forward} {backward}")
    elif arm == "a":
        if kernel + fallback != 1 or backward.get("dequant", 0) < 1:
            raise ValueError(f"unobserved NF4 path: {forward} {backward}")
    elif kernel or fallback or dequant != 1 or backward.get("dequant", 0) != 1:
        raise ValueError(f"explicit dense path changed: {forward} {backward}")


def measure(mode: str, n: int, k: int, m: int) -> dict:
    import bitsandbytes as bnb
    import torch
    import torch.nn.functional as functional
    from torch.utils.checkpoint import checkpoint

    from soup_cli.bench.train_report import summarize_clock_samples
    from soup_cli.bench.train_run import ClockSampler, _nvidia_smi, _pci_bus_id
    from soup_cli.utils.layer_stream_runtime import (
        checkpoint_visible_nf4_linear,
        install_dequant_forward,
    )

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    report = {"hardware": hardware(), "shape": {"n": n, "k": k, "m": m}, "mode": mode}
    weight = torch.randn(n, k, device="cuda", dtype=torch.float16) / math.sqrt(k)
    packed, state = bnb.functional.quantize_4bit(
        weight,
        blocksize=64,
        quant_type="nf4",
        compress_statistics=True,
    )
    del weight
    x = torch.randn(m, k, device="cuda", dtype=torch.float16, requires_grad=True)
    grad = torch.randn(m, n, device="cuda", dtype=torch.float16) / math.sqrt(n)
    cached = bnb.functional.dequantize_4bit(packed, state) if mode == "ablation" else None
    report["cached_dense_bytes"] = 0 if cached is None else cached.numel() * cached.element_size()
    with count_paths() as counts:
        # Install inside the counter scope: install_dequant_forward captures the function.
        layer = bnb.nn.Linear4bit(k, n, bias=False, compute_dtype=torch.float16)
        layer.weight = bnb.nn.Params4bit(
            packed, requires_grad=False, quant_state=state, bnb_quantized=True
        )
        if install_dequant_forward(layer) != 1:
            raise ValueError("the production repair was not installed")

        def body(arm):
            if arm == "native":
                return bnb.matmul_4bit(x, packed.t(), quant_state=state)
            if mode == "ablation":
                if arm == "a":
                    return layer(x)
                counts["cached_linear"] += 1
                return functional.linear(x, cached)
            if arm == "a":
                return checkpoint_visible_nf4_linear(x, packed, state)
            return functional.linear(x, bnb.functional.dequantize_4bit(packed, state))

        def step(arm, timed):
            x.grad = None
            counts.clear()
            if timed:
                torch.cuda.reset_peak_memory_stats()
                start, middle, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                start.record()
            output = checkpoint(lambda: body(arm), use_reentrant=False)
            if timed:
                middle.record()
            forward = dict(counts)
            counts.clear()
            output.backward(grad)
            if timed:
                end.record()
                end.synchronize()
            backward = dict(counts)
            if arm != "native":
                validate_paths(mode, arm, forward, backward)
            row = {"forward_path": forward, "backward_path": backward}
            if timed:
                row.update(
                    forward_ms=start.elapsed_time(middle),
                    backward_ms=middle.elapsed_time(end),
                    total_ms=start.elapsed_time(end),
                    peak_bytes=torch.cuda.max_memory_allocated(),
                )
                return row
            return output.detach().clone(), x.grad.detach().clone(), row

        native = step("native", False)
        native_fwd = native[2]["forward_path"]
        if native_fwd.get("kernel", 0) + native_fwd.get("fallback", 0) != 1:
            raise ValueError(f"native dispatch missing: {native_fwd}")
        if mode == "ablation" and native_fwd.get("fallback") != 1:
            raise ValueError(f"native ablation did not take the dense fallback: {native_fwd}")
        checks = {}
        for arm in ("a", "b"):
            actual = step(arm, False)
            checks[arm] = {
                "forward": difference(actual[0], native[0]),
                "backward": difference(actual[1], native[1]),
                "paths": actual[2],
            }
            if arm == "a" or mode == "ablation":
                if not all(checks[arm][part]["exact"] for part in ("forward", "backward")):
                    raise ValueError(f"native reference mismatch: {checks}")
            elif not all(checks[arm][part]["finite"] for part in ("forward", "backward")):
                raise ValueError(f"non-finite dense reference: {checks}")
        report["agreement"] = checks
        report["native_paths"] = native[2]
        del native, actual
        for arm in order(WARMUP):
            step(arm, False)
        torch.cuda.synchronize()
        from soup_cli.utils.layer_stream import _resolve_tool

        gpu = _pci_bus_id(torch.cuda.get_device_properties(0))
        smi = _resolve_tool("nvidia-smi")
        sampler = ClockSampler(gpu=gpu).start()
        memory = ClockSampler(
            read=lambda: _nvidia_smi(smi, "--query-gpu=utilization.memory", gpu) if smi else None
        ).start()
        began = time.perf_counter()
        try:
            samples = run_iterations(step, warmup=0, iterations=ITERATIONS)
        finally:
            ended = time.perf_counter()
            clocks = sampler.stop()
            memory_samples = memory.stop()
        report["clocks"] = summarize_clock_samples(clocks, began, ended)
        report["memory_controller_pct"] = summarize_clock_samples(memory_samples, began, ended)
        if not report["memory_controller_pct"]["sample_count"]:
            report["memory_controller_pct"]["unavailable_reason"] = "no in-window memory sample"
        for arm in ("a", "b"):
            report[arm] = {
                key: statistics.median(row[key] for row in samples[arm])
                for key in ("forward_ms", "backward_ms", "total_ms")
            }
            report[arm]["peak_bytes"] = max(row["peak_bytes"] for row in samples[arm])
            report[arm]["projection_tokens_s"] = m * 1000 / report[arm]["total_ms"]
        report["samples"] = samples
        # Isolate the resident dense GEMM ceiling after sampling the two arms.
        with torch.no_grad():
            dense = bnb.functional.dequantize_4bit(packed, state) if cached is None else cached
            result = torch.empty((m, n), device=x.device, dtype=x.dtype)
            for _ in range(WARMUP):
                torch.mm(x, dense.t(), out=result)
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(ITERATIONS):
                torch.mm(x, dense.t(), out=result)
            end.record()
            end.synchronize()
            ms = start.elapsed_time(end) / ITERATIONS
            report["dense_gemm_ceiling"] = {
                "ms": ms,
                "tflops": 2 * m * n * k / (ms * 1e9),
                "projection_tokens_s": m * 1000 / ms,
            }
    return report


def aliasing() -> dict:
    """Four synthetic layers, two packed slots, one backward through the full chain."""
    import copy

    import bitsandbytes as bnb
    import torch
    from torch.utils.checkpoint import checkpoint

    from soup_cli.utils.layer_stream_runtime import install_dequant_forward

    torch.manual_seed(SEED)
    width, tokens = 18944, 8
    dtype = torch.float16
    sources = []
    for _ in range(4):
        weight = torch.randn(width, width, device="cuda", dtype=dtype) / math.sqrt(width)
        sources.append(
            bnb.functional.quantize_4bit(
                weight,
                blocksize=64,
                quant_type="nf4",
                compress_statistics=True,
            )
        )
        del weight
    packed_bytes = sources[0][0].nbytes
    if packed_bytes <= 165 * 1024**2:
        raise ValueError(f"fixture below gate: {packed_bytes} packed bytes")
    slots = [(p.clone(), copy.deepcopy(s)) for p, s in sources[:2]]
    initial = torch.randn(tokens, width, device="cuda", dtype=dtype)
    dy = torch.randn_like(initial) / math.sqrt(width)
    initial_scales = [torch.rand(width, device="cuda", dtype=dtype) + 0.5 for _ in sources]

    def run(arm, counts):
        x = initial.detach().requires_grad_()
        scales = [s.detach().requires_grad_() for s in initial_scales]
        with torch.device("meta"):
            modules = [
                bnb.nn.Linear4bit(width, width, bias=False, compute_dtype=dtype) for _ in sources
            ]
        if arm == "repair":
            for module in modules:
                if install_dequant_forward(module) != 1:
                    raise ValueError("missing repair")
        refills = Counter()
        layers = {}

        def body(index, activation, scale):
            packed, state = sources[index]
            if arm in ("repair", "unsafe"):
                slot = index % len(slots)
                slot_weight, slot_state = slots[slot]
                slot_weight.copy_(packed)
                for target, source in (
                    (slot_state.absmax, state.absmax),
                    (slot_state.offset, state.offset),
                    (slot_state.state2.absmax, state.state2.absmax),
                    (slot_state.state2.code, state.state2.code),
                ):
                    target.copy_(source)
                packed, state = slot_weight, slot_state
                refills[slot] += 1
            hidden = activation * scale
            if arm == "resident":
                return bnb.matmul_4bit(hidden, packed.t(), quant_state=state)
            module = modules[index]
            module.weight = bnb.nn.Params4bit(
                packed, quant_state=state, requires_grad=False, bnb_quantized=True
            )
            return module(hidden)

        counts.clear()
        out = x
        for index in range(len(sources)):
            out = checkpoint(body, index, out, scales[index], use_reentrant=False)
            layers[str(index)] = out.detach()
        forward_counts = dict(counts)
        counts.clear()
        out.backward(dy)
        tensors = {
            "input_grad": x.grad,
            **{f"scale_grad_{i}": s.grad for i, s in enumerate(scales)},
        }
        return (
            layers,
            tensors,
            {"forward": forward_counts, "backward": dict(counts), "slot_refills": dict(refills)},
        )

    report = {
        "hardware": hardware(),
        "packed_bytes_per_layer": packed_bytes,
        "layers": 4,
        "slots": 2,
        "tokens": tokens,
        "width": width,
        "arms": {},
    }
    with count_paths() as counts:
        outputs, grads, paths = run("resident", counts)
        report["resident_paths"] = paths
        for arm in ("unsafe", "bypass", "repair"):
            y, dx, paths = run(arm, counts)
            fwd = {i: difference(y[i], outputs[i]) for i in outputs}
            bwd = {i: difference(dx[i], grads[i]) for i in grads}
            # All arms must have exercised the fused kernel once per layer.
            paths_ok = paths["forward"].get("kernel") == len(sources)
            if arm != "bypass":
                paths_ok &= set(paths["slot_refills"]) == {0, 1}
            exact = all(v["exact"] for v in bwd.values())
            passed = (
                paths_ok
                and all(v["exact"] for v in fwd.values())
                and all(v["finite"] for v in bwd.values())
                and exact == (arm != "unsafe")
            )
            report["arms"][arm] = {"forward": fwd, "backward": bwd, "paths": paths, "pass": passed}
    report["pass"] = report["resident_paths"]["forward"].get("kernel") == len(sources) and all(
        r["pass"] for r in report["arms"].values()
    )
    return report


def self_test() -> dict:
    import torch

    torch.manual_seed(SEED)
    weight = torch.randn(8, 8)
    x = torch.randn(3, 8, requires_grad=True)
    grad = torch.randn(3, 8)
    observed = []

    def step(arm, timed):
        x.grad = None
        result = torch.nn.functional.linear(x, weight) if arm == "a" else x @ weight.t()
        result.backward(grad)
        assert difference(x.grad, grad @ weight)["exact"]
        observed.append((arm, timed))
        return result.detach()

    samples = run_iterations(step, warmup=2, iterations=4)
    assert observed == [(a, False) for a in order(2)] + [(a, True) for a in order(4)]
    assert order(2) == ["a", "b", "b", "a"]
    assert all(difference(a, b)["exact"] for a, b in zip(samples["a"], samples["b"]))
    assert not difference(samples["a"][0], samples["a"][0] + 1)["exact"]
    assert not difference(torch.tensor([float("nan")]), torch.tensor([float("nan")]))["exact"]

    def rows(pairs):
        return [{"a": {"total_ms": a}, "b": {"total_ms": b}} for a, b in pairs]

    assert summary(rows([(10, 8)] * 3))["verdict"] == "faster"
    assert summary(rows([(10, 12)] * 3))["verdict"] == "slower"
    assert summary(rows([(10, 9), (11, 10), (12, 11)]))["verdict"] == "no measurable difference"
    assert summary(rows([(10, 9), (10, 11), (10, 9)]))["verdict"] == "no measurable difference"
    assert summary(rows([(10, 9.9)] * 3))["small_share"]
    assert not summary(rows([(10, 9.9), (10, 9.9), (10, 9)]))["small_share"]
    assert summary([])["verdict"] == "INVALID"
    assert summary(rows([(10, float("nan"))] * 3))["verdict"] == "INVALID"
    assert summary([{"error": "child crashed"}] * 3)["verdict"] == "INVALID"
    for mode, arm, forward, backward in (
        ("ablation", "a", {"dequant": 1}, {"dequant": 1}),
        ("ablation", "b", {"cached_linear": 1}, {}),
        ("grid", "a", {"kernel": 1}, {"dequant": 1}),
        ("grid", "a", {"fallback": 1}, {"dequant": 2}),
        ("grid", "b", {"dequant": 1}, {"dequant": 1}),
    ):
        validate_paths(mode, arm, forward, backward)
    rejected = 0
    for mode, arm, forward, backward in (
        ("ablation", "a", {}, {}),
        ("ablation", "a", {"kernel": 1}, {"dequant": 1}),
        ("ablation", "b", {"dequant": 1}, {}),
        ("grid", "a", {}, {"dequant": 1}),
        ("grid", "a", {"kernel": 1}, {}),
        ("grid", "b", {"kernel": 1}, {"dequant": 1}),
    ):
        try:
            validate_paths(mode, arm, forward, backward)
        except ValueError:
            rejected += 1
    assert rejected == 6
    return {
        "self_test": "PASS",
        "checks": "CPU arms, ABBA, noise rule, path mutations, finite gate",
    }


def main() -> int:
    from rich.console import Console

    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--self-test", action="store_true")
    modes.add_argument("--ablation", action="store_true")
    modes.add_argument("--grid", action="store_true")
    modes.add_argument("--aliasing", action="store_true")
    modes.add_argument("--model", action="store_true")
    parser.add_argument("--json", type=pathlib.Path)
    parser.add_argument("--child", nargs=3, type=int, metavar=("N", "K", "M"))
    args = parser.parse_args()
    if args.self_test:
        payload = self_test()
    elif args.model:
        from nf4_stream_model import model_check

        payload = model_check(count_paths, difference, hardware)
    elif args.aliasing:
        payload = aliasing()
    elif args.child:
        try:
            payload = measure("ablation" if args.ablation else "grid", *args.child)
        except Exception as exc:
            payload = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        cells = (
            [(4096, 4096, 512)] if args.ablation else [(n, k, m) for n, k in SHAPES for m in TOKENS]
        )
        payload = {
            "protocol": {
                "seed": SEED,
                "warmup": WARMUP,
                "iterations": ITERATIONS,
                "processes": PROCESSES,
                "arm_order": "ABBA",
                "checkpoint": "non-reentrant",
                "scale_cache": False,
            },
            "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
            "cells": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            for cell in cells:
                rows = []
                for process in range(PROCESSES):
                    output = pathlib.Path(temporary) / "child.json"
                    output.unlink(missing_ok=True)
                    command = [
                        sys.executable,
                        __file__,
                        "--ablation" if args.ablation else "--grid",
                        "--child",
                        *map(str, cell),
                        "--json",
                        str(output),
                    ]
                    try:
                        child = subprocess.run(
                            command, timeout=1200, capture_output=True, text=True
                        )
                        if not output.exists():
                            raise RuntimeError(f"exit={child.returncode}: {child.stderr[-2000:]}")
                        row = json.loads(output.read_text())
                        if child.returncode and not row.get("error"):
                            raise RuntimeError(f"exit={child.returncode}")
                    except (subprocess.SubprocessError, RuntimeError, ValueError) as exc:
                        row = {"error": str(exc)}
                    rows.append(row)
                    Console(stderr=True).print(
                        f"{cell} process {process + 1}: {row.get('error', 'complete')}"
                    )
                verdict = summary(rows)
                if args.grid and verdict["verdict"] != "INVALID":
                    verdict.pop("small_share")
                    for metric in ("forward_ms", "backward_ms"):
                        results = [
                            ProcessResult(
                                cell=metric, fast_ms=r["b"][metric], plain_ms=r["a"][metric]
                            )
                            for r in rows
                        ]
                        verdict[metric] = cell_verdict(results)
                    verdict["kernel_observed"] = all(
                        r["agreement"]["a"]["paths"]["forward_path"].get("kernel") == 1
                        for r in rows
                    )
                payload["cells"].append(
                    {"shape_n_k_m": cell, "processes": rows, "summary": verdict}
                )
                if args.json:
                    args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    else:
        Console().print_json(data=payload)
    failed = (
        payload.get("error")
        or payload.get("pass") is False
        or any(cell["summary"]["verdict"] == "INVALID" for cell in payload.get("cells", []))
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
