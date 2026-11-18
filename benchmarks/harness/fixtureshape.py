#!/usr/bin/env python3
"""STEP 13 / #379 — where the CI fixture sits relative to the fused NF4 window.

Reconstruction of the METHOD of the historical scratch script ``fixtureshape.py``
whose result is recorded in ``benchmarks/gate-h100-validation.md`` STEP 13. The
original was never in this repository (#379), and this is not the bytes that
were run on it. The other CUDA name of that Reproducing-table row,
``fixture_window.py``, is not part of this slice and stays unpublished.

Recorded on: H100 sm90, 132 SMs, torch 2.13.0+cu130, bitsandbytes 0.50.0.
A re-run on another card or stack is a new baseline, not a comparison: the
fused-kernel cutoffs need bitsandbytes' dispatch on that card, and the record's
cutoffs are not asserted here.

Which mechanism needs which card: the fused 4-bit GEMM and its M-dependent
dispatch are CUDA-only; the CPU counterpart (packed-inference rounding) is
measured by ``fixture_window_cpu.py``, not by this script.

The two shapes in ``tests/test_v07202.py`` (hidden_size=64,
intermediate_size=64: a 64x64 and a 256x64 linear) are measured against
bitsandbytes' own per-M dispatch. For each row the harness records:

  * the dispatcher's decision (``_gemm_4bit_use_custom_fn`` and the row cap);
  * the three real backend hooks counted -- fused kernel, dequant+linear
    fallback, and the ``functional.dequantize_4bit`` proxy the first STEP 13
    attempt mistook for a path signal;
  * the forward difference between ``MatMul4Bit`` and dequantise + ``F.linear``
    and the gradient difference, reported separately, never averaged.

Three real training shapes (Llama-3.1-8B q_proj 4096x4096 and the Qwen2.5-32B
q_proj/down_proj 5120x5120 and gate_proj 27648x5120) are measured as controls.
The record's point is that the window does not overlap the shapes the feature
ships for (batch x seq >= 128), not that it never overlaps them at any M: its
table has 27648x5120 fusing at M <= 32. Windows are printed as a RESULT per
shape, not asserted as H100 numbers: the exact M cutoffs are a property of the
card and dtype.

This is the CUDA half of the Reproducing-table row. The CPU half is already
published as ``fixture_window_cpu.py`` (#1144).

Fail-closed default: it exits 0 only when both a fused kernel call and a
fallback call were counted on the fixture shapes (the fixture straddles the
window), the negative control was detected, every fallback row agrees forward,
and every gradient is bit-exact. Otherwise it exits 2..6. No CUDA or no
bitsandbytes is a SKIP (exit 0), not a false pass. No speed or quality claim.

CPU self-test: ``--self-test`` needs only torch and checks the dispatch reader,
the counter, the window summary and each gate on synthetic/fake data.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

BLOCKSIZE = 64
FIXTURE_SHAPES = "64x64,256x64"  # out_features x in_features from test_v07202.py
DEFAULT_M = "8,16,32,64,128,256,512"
CONTROL_SHAPES = "4096x4096,5120x5120,27648x5120"
SEED = 17

EXIT_OK = 0
EXIT_NO_FUSED = 2
EXIT_NO_FALLBACK = 3
EXIT_CONTROL_FAILED = 4
EXIT_FALLBACK_FORWARD_DIFF = 5
EXIT_GRADIENT_DIFF = 6
EXIT_SKIP = 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture-shapes", default=FIXTURE_SHAPES)
    parser.add_argument("--m", default=DEFAULT_M)
    parser.add_argument("--control-shapes", default=CONTROL_SHAPES)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument("--json", default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sm_clock_mhz_idle():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _dist_version() -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version("bitsandbytes")
    except Exception:  # noqa: BLE001
        return "unknown"


def hardware(dtype) -> dict:
    import torch

    props = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "vram_mib": props.total_memory // (1024 * 1024),
        "driver": _driver_version(),
        "sm_clock_mhz_idle": _sm_clock_mhz_idle(),
        "bitsandbytes_version": _dist_version(),
        "dtype": dtype,
    }


def stats(candidate, other) -> dict:
    import torch

    finite = bool(torch.isfinite(candidate).all() and torch.isfinite(other).all())

    def bits(t):
        return t.detach().contiguous().view(torch.uint8)

    exact = finite and candidate.shape == other.shape and bool(
        torch.equal(bits(candidate), bits(other))
    )
    diff = (candidate.float() - other.float()).abs()
    return {
        "exact": exact,
        "max_abs": float(diff.max()),
        "max_relative_to_scale": float(diff.max() / candidate.float().abs().max())
        if candidate.float().abs().max() > 0 else 0.0,
        "n_diff": int((bits(candidate) != bits(other)).sum()),
        "n_total": candidate.numel(),
        "finite": finite,
    }


def selected_custom(bnb_ops, device_index, dtype, rows, out_features, in_features, blocksize):
    """Ask bitsandbytes' own dispatch which kernel this (shape, M) selects."""
    fn = getattr(bnb_ops, "_gemm_4bit_use_custom_fn", None)
    if fn is None:
        return None
    max_rows = getattr(bnb_ops, "_gemm_4bit_custom_max_m", None)
    if max_rows is not None and rows > max_rows:
        return False
    if in_features % blocksize != 0:
        return False
    try:
        return bool(fn(device_index, dtype, rows, out_features, in_features))
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def counter(ops_module, functional_module):
    """Count the three real backend hooks, not a proxy for the path."""
    from collections import Counter

    counts = Counter()

    def wrap(label, original):
        def counted(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        return counted

    with ExitStack() as stack:
        for module, name, label in (
            (ops_module, "_gemm_4bit_kernel_impl", "kernel"),
            (ops_module, "_dequant_linear_fallback", "fallback"),
            (functional_module, "dequantize_4bit", "dequant"),
        ):
            original = getattr(module, name)
            stack.enter_context(patch.object(module, name, wrap(label, original)))
        yield counts


def windows(rows: list[dict], set_name: str) -> dict:
    """Per shape, the M values that actually reached each path (a result)."""
    result = {}
    for row in rows:
        if row["set"] != set_name:
            continue
        key = f"{row['N']}x{row['K']}"
        entry = result.setdefault(key, {"kernel": [], "fallback": []})
        if row["kernel_calls"] > 0:
            entry["kernel"].append(row["M"])
        if row["fallback_calls"] > 0:
            entry["fallback"].append(row["M"])
    return result


def evaluate(rows: list[dict]) -> dict:
    fixture = [r for r in rows if r["set"] == "fixture"]
    if not any(r["kernel_calls"] > 0 for r in fixture):
        return {
            "verdict": "no_fixture_kernel",
            "exit_code": EXIT_NO_FUSED,
            "reason": "no fixture row counted a fused kernel call; the window cannot be located",
        }
    if not any(r["fallback_calls"] > 0 for r in fixture):
        return {
            "verdict": "no_fixture_fallback",
            "exit_code": EXIT_NO_FALLBACK,
            "reason": (
                "no fixture row counted the dequant+linear fallback; the "
                "window edge cannot be located"
            ),
        }
    if any(not r["control_detected"] for r in rows):
        return {
            "verdict": "control_failed",
            "exit_code": EXIT_CONTROL_FAILED,
            "reason": "the sign-flip-plus-unit negative control rounded away on at least one row",
        }
    if any(
        r["forward"]["exact"] is False for r in rows if r["kernel_calls"] == 0
    ):
        return {
            "verdict": "fallback_forward_diff",
            "exit_code": EXIT_FALLBACK_FORWARD_DIFF,
            "reason": (
                "a row with no counted kernel call showed a forward difference; "
                "fallback arms should agree"
            ),
        }
    if any(not r["gradient"]["exact"] for r in rows):
        return {
            "verdict": "gradient_diff",
            "exit_code": EXIT_GRADIENT_DIFF,
            "reason": "gradient was not bit-exact on at least one row",
        }
    return {
        "verdict": "mechanism_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "the fixture shapes straddle the fused window (both a kernel call and "
            "a fallback call were counted), the control was detected, fallback rows "
            "agree forward, and every gradient is bit-exact"
        ),
    }


def _quantized_linear(weight):
    import bitsandbytes.functional as functional

    return functional.quantize_4bit(
        weight, blocksize=BLOCKSIZE, quant_type="nf4", compress_statistics=True
    )


def measure_row(out_features, in_features, rows, dtype, *, set_name):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    import torch.nn.functional as torch_functional
    from bitsandbytes.backends.cuda import ops as bnb_ops

    generator = torch.Generator(device="cuda").manual_seed(SEED)
    weight = torch.randn(
        (out_features, in_features), device="cuda", dtype=dtype, generator=generator
    )
    packed, state = _quantized_linear(weight)
    inputs = torch.randn((rows, in_features), device="cuda", dtype=dtype, generator=generator)
    decision = selected_custom(
        bnb_ops, 0, dtype, rows, out_features, in_features, BLOCKSIZE
    )

    variant_x = inputs.clone().requires_grad_(True)
    with counter(bnb_ops, functional) as variant_counts:
        dense = functional.dequantize_4bit(packed, state).to(variant_x.dtype)
        variant_out = torch_functional.linear(variant_x, dense)
    variant_out.sum().backward()
    variant_grad = variant_x.grad.detach().clone()

    resident_x = inputs.clone().requires_grad_(True)
    with counter(bnb_ops, functional) as resident_counts:
        resident_out = bnb.matmul_4bit(resident_x, packed.t(), quant_state=state)
    resident_out.sum().backward()
    resident_grad = resident_x.grad.detach().clone()

    control_weight = weight.clone()
    magnitude = float(control_weight.abs().max())
    control_weight[0, 0] = -control_weight[0, 0] - magnitude
    control_packed, control_state = _quantized_linear(control_weight)
    with torch.no_grad():
        control_dense = functional.dequantize_4bit(control_packed, control_state).to(dtype)
        control_out = torch_functional.linear(inputs, control_dense)
    control_detected = not torch.equal(control_out, resident_out.detach())

    result = {
        "set": set_name, "N": out_features, "K": in_features, "M": rows,
        "bnb_selected_custom_kernel": decision,
        "variant_fallback_calls": int(variant_counts["fallback"]),
        "kernel_calls": int(resident_counts["kernel"]),
        "fallback_calls": int(resident_counts["fallback"]),
        "dequant_wrapper_calls": int(resident_counts["dequant"]),
        "forward": stats(variant_out.detach(), resident_out.detach()),
        "gradient": stats(variant_grad, resident_grad),
        "control_detected": bool(control_detected),
    }
    del weight, packed, state, inputs, dense, control_weight, control_packed, control_state
    torch.cuda.empty_cache()
    return result


def run_sweep(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"skipped": "CUDA is required", "exit_code": EXIT_SKIP}
    try:
        import bitsandbytes  # noqa: F401 - availability probe only
    except ImportError:
        return {"skipped": "bitsandbytes is required", "exit_code": EXIT_SKIP}

    # Read the idle clock BEFORE the sweep loads the card; a clock taken after
    # the loops is a busy/leftover reading and must not be labelled idle.
    env = hardware(args.dtype)
    dtype = getattr(torch, args.dtype)
    fixture = [tuple(int(p) for p in s.split("x")) for s in args.fixture_shapes.split(",")]
    controls = [tuple(int(p) for p in s.split("x")) for s in args.control_shapes.split(",")]
    m_values = [int(m) for m in args.m.split(",")]

    rows = []
    for out_features, in_features in fixture:
        for m in m_values:
            rows.append(measure_row(out_features, in_features, m, dtype, set_name="fixture"))
    if not args.no_controls:
        for out_features, in_features in controls:
            for m in m_values:
                rows.append(measure_row(out_features, in_features, m, dtype, set_name="control"))

    verdict = evaluate(rows)
    return {
        "env": env,
        "protocol": {
            "note": "CI fixture vs real training shapes; dequantise+F.linear vs MatMul4Bit",
            "blocksize": BLOCKSIZE, "quant_type": "nf4", "double_quant": True,
            "fixture_shapes": fixture, "control_shapes": controls if not args.no_controls else [],
            "m_values": m_values, "seed": SEED,
        },
        "rows": rows,
        "n_rows": len(rows),
        "fixture_windows": windows(rows, "fixture"),
        "control_windows": windows(rows, "control"),
        "n_forward_diff_on_fused": sum(
            r["kernel_calls"] > 0 and r["forward"]["exact"] is False for r in rows
        ),
        "n_gradient_exact": sum(r["gradient"]["exact"] for r in rows),
        **verdict,
    }


def run_self_test() -> None:
    """Check the reconstruction's logic on CPU, not the original run's bytes."""
    import torch

    a = torch.tensor([100.0, -2.0, 3.0])
    b = a.clone()
    b[1] = -2.5
    s = stats(a, b)
    assert s["exact"] is False and s["n_diff"] == 1 and s["max_abs"] == 0.5
    assert stats(a, a.clone())["exact"] is True

    dispatch = SimpleNamespace(
        _gemm_4bit_use_custom_fn=lambda *args: False, _gemm_4bit_custom_max_m=1536
    )
    assert selected_custom(dispatch, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is False
    assert selected_custom(SimpleNamespace(), 0, "bf16", 8, 4096, 4096, BLOCKSIZE) is None

    ops = SimpleNamespace(
        _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
    )
    fn = SimpleNamespace(dequantize_4bit=lambda: None)
    with counter(ops, fn) as counts:
        ops._dequant_linear_fallback()
        fn.dequantize_4bit()
    assert counts["fallback"] == 1 and counts["dequant"] == 1 and counts["kernel"] == 0

    def row(**updates):
        base = {
            "set": "fixture", "N": 64, "K": 64, "M": 8,
            "kernel_calls": 1, "fallback_calls": 0,
            "forward": {"exact": False}, "gradient": {"exact": True},
            "control_detected": True,
        }
        base.update(updates)
        return base

    fused = row()
    fallback = row(M=64, kernel_calls=0, fallback_calls=1, forward={"exact": True})
    control = row(set="control", N=4096, K=4096, kernel_calls=0,
                  fallback_calls=1, forward={"exact": True})
    assert evaluate([fused, fallback, control])["exit_code"] == EXIT_OK
    assert evaluate([fallback, control])["exit_code"] == EXIT_NO_FUSED
    assert evaluate([fused, control])["exit_code"] == EXIT_NO_FALLBACK
    assert evaluate([fused, fallback, row(control_detected=False)])[
        "exit_code"
    ] == EXIT_CONTROL_FAILED
    assert evaluate([fused, fallback, row(M=64, kernel_calls=0, fallback_calls=1)])[
        "exit_code"
    ] == EXIT_FALLBACK_FORWARD_DIFF
    assert evaluate([fused, fallback, row(gradient={"exact": False})])[
        "exit_code"
    ] == EXIT_GRADIENT_DIFF

    got = windows([fused, fallback, control], "fixture")
    assert got == {"64x64": {"kernel": [8], "fallback": [64]}}
    print("fixtureshape.py --self-test: OK")


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0
    result = run_sweep(args)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.json:
        from pathlib import Path

        Path(args.json).write_text(payload + "\n", encoding="utf-8")
    code = int(result["exit_code"])
    if code != EXIT_OK:
        print(str(result.get("reason", result.get("skipped", ""))), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
