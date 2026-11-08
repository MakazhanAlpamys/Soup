#!/usr/bin/env python3
"""CUDA micro-benchmark for issue #837: Fast-LoRA vs PEFT vs PEFT + Liger.

This measures one synthetic Llama-3.1-8B-shaped SwiGLU MLP block. It does not
load a language model or claim end-to-end training performance.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import pathlib
import platform
import statistics
import subprocess
import threading
import types
from typing import Any

ARMS = ("fast", "plain", "liger")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class ClockSampler:
    """Sample the active SM clock without putting nvidia-smi in the timed path."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @staticmethod
    def _read() -> int | None:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=3,
            )
            return int(result.stdout.strip().splitlines()[0])
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._read()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=4)
        if not self.samples:
            return {"min_mhz": None, "median_mhz": None, "max_mhz": None}
        return {
            "min_mhz": min(self.samples),
            "median_mhz": int(statistics.median(self.samples)),
            "max_mhz": max(self.samples),
        }


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _graph_has(output: Any, fragment: str) -> bool:
    pending = [output.grad_fn]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if fragment.lower() in type(node).__name__.lower():
            return True
        pending.extend(next_node for next_node, _ in node.next_functions)
    return False


def _path_ok(arm: str, output: Any) -> bool:
    fast = _graph_has(output, "FastLoraSwiGLU")
    liger = _graph_has(output, "LigerSiLUMul")
    return (
        (arm == "fast" and fast and not liger)
        or (arm == "liger" and liger and not fast)
        or (arm == "plain" and not fast and not liger)
    )


def _plain_forward(module: Any, hidden_states: Any) -> Any:
    import torch.nn.functional as functional

    return module.down_proj(
        functional.silu(module.gate_proj(hidden_states)) * module.up_proj(hidden_states)
    )


def _replace_with_nf4(mlp: Any) -> Any:
    import bitsandbytes as bnb
    import torch

    for name in PROJECTIONS:
        source = getattr(mlp, name)
        target = bnb.nn.Linear4bit(
            source.in_features,
            source.out_features,
            bias=False,
            compute_dtype=torch.bfloat16,
            compress_statistics=True,
            quant_type="nf4",
        )
        with torch.no_grad():
            target.weight.data.copy_(source.weight.data)
        setattr(mlp, name, target)
    return mlp


def _build(dtype: str, rank: int, seed: int, device: str) -> tuple[Any, Any, Any, Any]:
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from peft import LoraConfig, inject_adapter_in_model

    from soup_cli.utils.fast_lora_mlp import patch_fast_lora_mlp, unpatch_fast_lora_mlp

    config = SimpleNamespace(
        hidden_size=4096,
        intermediate_size=14336,
        hidden_act="silu",
    )

    class PlainSwiGLU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.act_fn = torch.nn.functional.silu
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

        def forward(self, hidden_states: Any) -> Any:
            return _plain_forward(self, hidden_states)

    class Block(nn.Module):
        def __init__(self, mlp: Any) -> None:
            super().__init__()
            self.config = config
            self.mlp = mlp

        def forward(self, hidden_states: Any) -> Any:
            return self.mlp(hidden_states)

    torch.manual_seed(seed)
    mlp = LigerSwiGLUMLP(config)
    mlp.act_fn = torch.nn.functional.silu
    if dtype == "nf4":
        mlp = _replace_with_nf4(mlp)
    else:
        mlp = mlp.to(dtype=torch.bfloat16)

    model = Block(mlp).to(device)
    torch.manual_seed(seed + 1)
    inject_adapter_in_model(
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            bias="none",
            target_modules=list(PROJECTIONS),
        ),
        model,
    )
    model.requires_grad_(False)
    torch.manual_seed(seed + 2)
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
            if "lora_B" in name:
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02)

    original_liger_forward = model.mlp.forward
    plain_forward = types.MethodType(_plain_forward, model.mlp)
    unpatch_fast_lora_mlp(model)
    patched = patch_fast_lora_mlp(model)
    unpatch_fast_lora_mlp(model)
    model.mlp.forward = original_liger_forward
    if patched != 1:
        raise RuntimeError(f"expected to patch one MLP block, got {patched}")
    return (
        model,
        original_liger_forward,
        plain_forward,
        (patch_fast_lora_mlp, unpatch_fast_lora_mlp),
    )


def _set_arm(
    model: Any, arm: str, original_liger_forward: Any, plain_forward: Any, patchers: tuple[Any, Any]
) -> None:
    patch_fast_lora_mlp, unpatch_fast_lora_mlp = patchers
    unpatch_fast_lora_mlp(model)
    model.mlp.forward = plain_forward if arm == "plain" else original_liger_forward
    if arm == "fast" and patch_fast_lora_mlp(model) != 1:
        raise RuntimeError("Fast-LoRA patch did not engage")


def _clear_grads(model: Any, hidden_states: Any) -> None:
    hidden_states.grad = None
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = None


def _schedule(count: int) -> list[str]:
    order: list[str] = []
    for index in range(count):
        rotated = list(ARMS[index % len(ARMS) :]) + list(ARMS[: index % len(ARMS)])
        if index % 2:
            rotated.reverse()
        order.extend(rotated)
    return order


def _capture_arm(
    model: Any,
    hidden_states: Any,
    grad_output: Any,
    arm: str,
    original_liger_forward: Any,
    plain_forward: Any,
    patchers: tuple[Any, Any],
) -> tuple[dict[str, Any], int]:
    import torch

    _set_arm(model, arm, original_liger_forward, plain_forward, patchers)
    _clear_grads(model, hidden_states)
    seen: set[tuple[Any, ...]] = set()
    saved_bytes = 0

    def pack(tensor: Any) -> Any:
        nonlocal saved_bytes
        key = (
            tensor.untyped_storage()._cdata,
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
        )
        if key not in seen:
            seen.add(key)
            saved_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = model(hidden_states)
        torch.autograd.backward(output, grad_output)
    if not _path_ok(arm, output):
        raise RuntimeError(f"path proof failed for {arm}: {type(output.grad_fn).__name__}")

    tensors: dict[str, Any] = {
        "output": output.detach().float().cpu().clone(),
        "input_gradient": hidden_states.grad.detach().float().cpu().clone(),
    }
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            if parameter.grad is None:
                raise RuntimeError(f"missing adapter gradient: {name}")
            tensors[name] = parameter.grad.detach().float().cpu().clone()
    return {"tensors": tensors, "saved_tensor_bytes": saved_bytes}, saved_bytes


def _error(actual: Any, expected: Any) -> dict[str, float]:
    difference = actual - expected
    denom = expected.norm().item()
    return {
        "relative_l2": float(difference.norm().item() / max(denom, 1e-30)),
        "max_abs": float(difference.abs().max().item()),
    }


def _measure_dtype(dtype: str, args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from soup_cli.utils.fast_lora_mlp import unpatch_fast_lora_mlp

    device = "cuda"
    model, liger_forward, plain_forward, patchers = _build(dtype, args.rank, args.seed, device)
    hidden_states = torch.randn(
        1, args.tokens, 4096, device=device, dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn(1, args.tokens, 4096, device=device, dtype=torch.bfloat16)

    captured: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        entry, _ = _capture_arm(
            model, hidden_states, grad_output, arm, liger_forward, plain_forward, patchers
        )
        captured[arm] = entry
    errors: dict[str, dict[str, Any]] = {}
    reference = captured["plain"]["tensors"]
    for arm in ("fast", "liger"):
        measured = captured[arm]["tensors"]
        errors[arm] = {key: _error(measured[key], ref) for key, ref in reference.items()}

    # Drop gradients from parity captures so the resident baseline is clean.
    _clear_grads(model, hidden_states)
    torch.cuda.synchronize()
    # Keep the same module/weights resident for every timed arm.
    baseline_bytes = torch.cuda.memory_allocated()
    samples: dict[str, list[float]] = {arm: [] for arm in ARMS}
    peaks: dict[str, int] = {arm: 0 for arm in ARMS}
    path_counts: dict[str, int] = {arm: 0 for arm in ARMS}
    sampler = ClockSampler()
    sampler.start()
    try:
        for arm in _schedule(args.warmup):
            _set_arm(model, arm, liger_forward, plain_forward, patchers)
            _clear_grads(model, hidden_states)
            output = model(hidden_states)
            torch.autograd.backward(output, grad_output)
            if not _path_ok(arm, output):
                raise RuntimeError(f"warm-up path proof failed for {arm}")
        torch.cuda.synchronize()

        for arm in _schedule(args.iters):
            _set_arm(model, arm, liger_forward, plain_forward, patchers)
            _clear_grads(model, hidden_states)
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = model(hidden_states)
            torch.autograd.backward(output, grad_output)
            end.record()
            torch.cuda.synchronize()
            if not _path_ok(arm, output):
                raise RuntimeError(f"timed path proof failed for {arm}")
            path_counts[arm] += 1
            samples[arm].append(float(start.elapsed_time(end)))
            peaks[arm] = max(peaks[arm], int(torch.cuda.max_memory_allocated()))
    finally:
        clock = sampler.stop()
        unpatch_fast_lora_mlp(model)

    result = {
        "dtype": dtype,
        "shape": {"tokens": args.tokens, "hidden": 4096, "intermediate": 14336, "rank": args.rank},
        "saved_tensor_bytes": {arm: captured[arm]["saved_tensor_bytes"] for arm in ARMS},
        "parity_vs_plain": errors,
        "timing_ms": {
            arm: {
                "median": statistics.median(samples[arm]),
                "p25": statistics.quantiles(samples[arm], n=4)[0],
                "p75": statistics.quantiles(samples[arm], n=4)[2],
                "min": min(samples[arm]),
                "max": max(samples[arm]),
            }
            for arm in ARMS
        },
        "peak_memory_bytes": {
            arm: {
                "absolute": peaks[arm],
                "increment_over_resident_baseline": peaks[arm] - baseline_bytes,
            }
            for arm in ARMS
        },
        "path_proof": {arm: f"{path_counts[arm]}/{args.iters}" for arm in ARMS},
        "clock_mhz_during_measurement": clock,
    }
    del model, hidden_states, grad_output
    torch.cuda.empty_cache()
    return result


def _metadata() -> dict[str, Any]:
    import torch

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    props = torch.cuda.get_device_properties(0)
    try:
        sha = subprocess.run(
            ["git", "-C", str(pathlib.Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        sha = "unknown"
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_sha": sha,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_bytes": int(props.total_memory),
        "nvidia_smi": result.stdout.strip(),
        "packages": {
            name: version(name) for name in ("transformers", "peft", "bitsandbytes", "liger-kernel")
        },
        "protocol": {
            "forward_backward_repeats_per_arm": None,
            "warmup_per_arm": None,
            "cuda_events": True,
            "interleaved_arms": list(ARMS),
            "no_model_download": True,
            "no_end_to_end_claim": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtypes", nargs="+", choices=("bf16", "nf4"), default=("bf16", "nf4"))
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=837)
    parser.add_argument(
        "--output", type=pathlib.Path, default=pathlib.Path("issue837_results.json")
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        assert _schedule(3) == [
            "fast",
            "plain",
            "liger",
            "fast",
            "liger",
            "plain",
            "liger",
            "fast",
            "plain",
        ]
        print("self-test passed: each arm is balanced and interleaved")
        return 0

    import torch
    from rich.console import Console

    console = Console()

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required; no benchmark result was produced")
    if args.tokens <= 0 or args.rank <= 0 or args.iters < 5 or args.warmup < 1:
        parser.error("tokens/rank/warmup must be positive and iters must be at least 5")

    record = _metadata()
    record["protocol"]["forward_backward_repeats_per_arm"] = args.iters
    record["protocol"]["warmup_per_arm"] = args.warmup
    record["settings"] = vars(args) | {"output": str(args.output)}
    record["results"] = []
    for dtype in args.dtypes:
        record["results"].append(_measure_dtype(dtype, args))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in record["results"]:
        console.print(f"\n{row['dtype']} — {row['shape']}")
        for arm in ARMS:
            timing = row["timing_ms"][arm]
            memory = row["peak_memory_bytes"][arm]
            console.print(
                f"{arm:>5}: median {timing['median']:.3f} ms "
                f"(IQR {timing['p25']:.3f}–{timing['p75']:.3f}); "
                f"peak {memory['absolute'] / 2**20:.1f} MiB; "
                f"saved {row['saved_tensor_bytes'][arm] / 2**20:.1f} MiB; "
                f"path {row['path_proof'][arm]}"
            )
    console.print(f"\nFull record: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
