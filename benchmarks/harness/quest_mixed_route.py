"""Audit #674's mixed QuEST route and measure its synthetic fake-quant cost.

This harness deliberately makes no quality, packed-INT4, memory-saving or
throughput claim. It uses a tiny 24-block Llama-shaped fixture to verify the
168 W4 / 161 A4 / 7 A16 accounting and to expose how much dense reference
arithmetic the fake-quant wrappers add on the machine that runs it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def build_model(width: int = 128) -> Any:
    import torch

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = torch.nn.Linear(width, width, bias=False)
            self.k_proj = torch.nn.Linear(width, width, bias=False)
            self.v_proj = torch.nn.Linear(width, width, bias=False)
            self.o_proj = torch.nn.Linear(width, width, bias=False)

    class MLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(width, width, bias=False)
            self.up_proj = torch.nn.Linear(width, width, bias=False)
            self.down_proj = torch.nn.Linear(width, width, bias=False)

    class Block(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()
            self.mlp = MLP()

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList(Block() for _ in range(24))
            self.lm_head = torch.nn.Linear(width, 17, bias=False)
            self.config = SimpleNamespace(
                model_type="llama",
                num_hidden_layers=24,
                hidden_size=width,
                intermediate_size=width,
            )

    return Model()


def synchronize(device: Any) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(layer: Any, value: Any, *, warmup: int, steps: int) -> dict[str, Any]:
    import torch

    samples: list[float] = []
    peak_bytes = 0
    for index in range(warmup + steps):
        layer.zero_grad(set_to_none=True)
        if value.grad is not None:
            value.grad = None
        if value.device.type == "cuda" and index == warmup:
            torch.cuda.reset_peak_memory_stats(value.device)
        synchronize(value.device)
        started = time.perf_counter()
        with torch.autocast(
            value.device.type,
            dtype=torch.bfloat16,
            enabled=value.device.type == "cuda",
        ):
            loss = layer(value).float().square().mean()
        loss.backward()
        synchronize(value.device)
        if index >= warmup:
            samples.append(time.perf_counter() - started)
    if value.device.type == "cuda":
        peak_bytes = int(torch.cuda.max_memory_allocated(value.device))
    gradient = layer.weight.grad
    if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
        raise RuntimeError("instrumented layer did not produce a finite non-zero gradient")
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "peak_allocated_bytes": peak_bytes,
        "steps": steps,
    }


def run(*, device_name: str, warmup: int, steps: int) -> dict[str, Any]:
    import torch

    from soup_cli.utils.quest import (
        A16_MODULES,
        EXPECTED_MODULES,
        assert_route,
        install_mixed_quest,
    )

    if warmup < 0 or steps < 1:
        raise ValueError("warmup must be >= 0 and steps must be >= 1")
    device = torch.device(device_name)
    torch.manual_seed(674)
    raw = build_model().to(device)
    mixed = copy.deepcopy(raw)
    scales = {name: 3.0 for name in EXPECTED_MODULES}
    calibration_sha256 = hashlib.sha256(
        b"synthetic-route-accounting-fixture-not-training-data"
    ).hexdigest()
    metadata = install_mixed_quest(
        mixed,
        activation_scales=scales,
        base_model="synthetic/issue674-route-audit",
        calibration_sha256=calibration_sha256,
    )
    assert_route(mixed, metadata)
    modules = dict(mixed.named_modules())
    observed_a16 = sorted(
        name for name in EXPECTED_MODULES if int(modules[name].quest_activation_bits) == 16
    )
    observed_a4 = sorted(set(EXPECTED_MODULES) - set(observed_a16))
    if observed_a16 != sorted(A16_MODULES) or len(observed_a4) != 161:
        raise RuntimeError("runtime accounting differs from the declared mixed route")

    value = torch.randn(2, 16, 128, device=device, requires_grad=True)
    timings = {
        "raw_block22_q_proj": measure(
            raw.model.layers[22].self_attn.q_proj,
            value.detach().clone().requires_grad_(True),
            warmup=warmup,
            steps=steps,
        ),
        "mixed_a4_block22_q_proj": measure(
            mixed.model.layers[22].self_attn.q_proj,
            value.detach().clone().requires_grad_(True),
            warmup=warmup,
            steps=steps,
        ),
        "mixed_a16_block23_q_proj": measure(
            mixed.model.layers[23].self_attn.q_proj,
            value.detach().clone().requires_grad_(True),
            warmup=warmup,
            steps=steps,
        ),
    }
    return {
        "scope": "synthetic route audit and fake-quant instrumentation only",
        "quality_claimed": False,
        "packed_int4": False,
        "performance_claimed": False,
        "memory_saving_claimed": False,
        "device": str(device),
        "torch": torch.__version__,
        "fixture": {"blocks": 24, "linears": 168, "width": 128},
        "route": {
            "weights_w4": 168,
            "activations_a4": len(observed_a4),
            "activations_a16": len(observed_a16),
            "a16_modules": observed_a16,
        },
        "timings": timings,
        "caveat": (
            "The fixture is tiny and synthetic. Timings measure dense fake-quant "
            "reference overhead on this host; they are not model throughput or "
            "packed-INT4 efficiency evidence."
        ),
    }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(device_name=args.device, warmup=args.warmup, steps=args.steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
