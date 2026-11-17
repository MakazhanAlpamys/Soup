#!/usr/bin/env python3
"""STEP 13, second answer — per-M dispatch sweep with the fused kernel forced on.

Reconstruction for #379 of the historical scratch script ``numerics2.py`` whose
table is in ``benchmarks/gate-h100-validation.md`` STEP 13. The original was
never in this repository (#379). Per layer, no model: the two resident arms are

    variant 2:  F.linear(x, dequantize_4bit(packed, state).to(x.dtype))
    resident:   bnb.matmul_4bit(x, packed.t(), quant_state=state)

on the same NF4 weight and quant_state (blocksize 64, nf4, double quant). Unlike
``numerics.py``, this harness READS the dispatcher's per-M decision instead of
assuming it, FORCES the fused kernel on at training-shaped M by patching the two
module globals bitsandbytes reads at call time, and COUNTS
``_dequant_linear_fallback`` directly so a forcing that silently failed is
distinguishable from a kernel that genuinely agreed. The dispatcher's per-M
decision is recorded, but every fail-closed gate below reads the counted kernel
calls, not that decision: a forced row only counts as fused when the kernel call
was actually observed.

Sets measured here (all forward and gradient reported separately, never averaged):
  * default dispatch: 10 M values in 8..2048 x 6 shapes x 5 seeds = 300 rows;
  * fused forced on:  M = 512 and 2048 x 6 shapes x 5 seeds = 60 rows;
  * fallback cross-check: M in 8..2048 x 6 shapes x 1 seed = 48 rows;
  * shipped CI fixture shapes 64x64 and 256x64, M in 8..512 = 14 rows.

Historical result (H100 sm90, torch 2.13.0+cu130, bitsandbytes 0.50.0, bf16):
gradient bit-exact in 423/423 rows; forward differs only where the fused kernel
genuinely runs, and then by about one bf16 ulp; the record reports forcing
verified with 0 fallback calls on 24/24 forced rows. The defaults here run 60
forced rows. This harness reproduces the METHOD and applies its fail-closed gate
to whatever this host actually does. A re-run on another card or stack is a new
baseline, not a comparison with the recorded numbers.

CPU self-test
-------------
``--self-test`` needs only torch: it checks the stats, the dispatch reader, the
forcing patch on a fake module, the counter, and the gate on synthetic rows.

Exit codes (default invocation):
  0  mechanism reproduced: fused ran, forcing verified, gradient exact everywhere
  2  no fused kernel call was counted anywhere (mechanism absent — cannot show the difference)
  3  invalid negative control
  4  forcing silently failed (a forced row still called the fallback)
  5  gradient not bit-exact on some row
  6  forward never differed on a fused row (fused effect absent)
  SKIP (0) when CUDA / bitsandbytes is unavailable
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
DEFAULT_SHAPES = "4096x4096,4096x14336,14336x4096,5120x5120,5120x27648,27648x5120"
DEFAULT_M = "8,16,32,64,128,256,512,1024,1536,2048"
FORCED_M = "512,2048"
CROSSCHECK_M = "8,16,32,64,128,256,1024,2048"
FIXTURE_SHAPES = "64x64,256x64"
FIXTURE_M = "8,16,32,64,128,256,512"
DEFAULT_SEEDS = "0,1,2,3,4"

EXIT_OK = 0
EXIT_NO_FUSED = 2
EXIT_CONTROL_FAILED = 3
EXIT_FORCING_FAILED = 4
EXIT_GRADIENT_DIFF = 5
EXIT_FORWARD_EFFECT_ABSENT = 6
EXIT_SKIP = 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--m", default=DEFAULT_M, help="default-dispatch M values")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--no-crosscheck", action="store_true")
    parser.add_argument("--no-fixture", action="store_true")
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
def force_fused(bnb_ops):
    """Patch the two globals the registered CUDA kernel reads at call time."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(bnb_ops, "_gemm_4bit_custom_max_m", 10**9))
        stack.enter_context(patch.object(bnb_ops, "_gemm_4bit_use_custom_fn", lambda *a, **k: True))
        yield


@contextmanager
def counter(ops_module, functional_module):
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


def _quantized_linear(weight):
    import bitsandbytes.functional as functional

    return functional.quantize_4bit(
        weight, blocksize=BLOCKSIZE, quant_type="nf4", compress_statistics=True
    )


def measure_row(seed, in_features, out_features, rows, dtype, *, force=False):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    import torch.nn.functional as torch_functional
    from bitsandbytes.backends.cuda import ops as bnb_ops

    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        (out_features, in_features), device="cuda", dtype=dtype, generator=generator
    )
    packed, state = _quantized_linear(weight)
    inputs = torch.randn((rows, in_features), device="cuda", dtype=dtype, generator=generator)

    with ExitStack() as stack:
        if force:
            stack.enter_context(force_fused(bnb_ops))
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

        variant_out = variant_out.detach()
        resident_out = resident_out.detach()
        variant_counts = dict(variant_counts)
        resident_counts = dict(resident_counts)

    control_weight = weight.clone()
    magnitude = float(control_weight.abs().max())
    control_weight[0, 0] = -control_weight[0, 0] - magnitude
    control_packed, control_state = _quantized_linear(control_weight)
    with torch.no_grad():
        control_dense = functional.dequantize_4bit(control_packed, control_state).to(dtype)
        control_out = torch_functional.linear(inputs, control_dense)
    control_detected = not torch.equal(control_out, resident_out)

    result = {
        "seed": seed, "K": in_features, "N": out_features, "M": rows,
        "set": "forced" if force else "default",
        "bnb_selected_custom_kernel": decision,
        "variant_fallback_calls": variant_counts.get("fallback", 0),
        "resident_fallback_calls": resident_counts.get("fallback", 0),
        "resident_kernel_calls": resident_counts.get("kernel", 0),
        "forward": stats(variant_out, resident_out),
        "gradient": stats(variant_grad, resident_grad),
        "control_detected": bool(control_detected),
    }
    del weight, packed, state, inputs, control_weight, control_packed, control_state
    torch.cuda.empty_cache()
    return result


def evaluate(rows: list[dict]) -> dict:
    # Read the path actually taken, not the dispatcher decision: forcing makes
    # the decision True for every forced row whether or not a kernel ran.
    fused = [r for r in rows if r["resident_kernel_calls"] > 0]
    if not fused:
        return {
            "verdict": "no_fused_kernel",
            "exit_code": EXIT_NO_FUSED,
            "reason": (
                "no fused 4-bit kernel call was counted anywhere; the "
                "fused-vs-fallback difference cannot be demonstrated on this host/dtype"
            ),
        }
    if any(not r["control_detected"] for r in rows):
        return {
            "verdict": "control_failed",
            "exit_code": EXIT_CONTROL_FAILED,
            "reason": "the sign-flip-plus-unit negative control rounded away on at least one row",
        }
    forced = [r for r in rows if r["set"] == "forced"]
    if not forced or any(
        r["resident_fallback_calls"] > 0 or r["resident_kernel_calls"] == 0 for r in forced
    ):
        return {
            "verdict": "forcing_failed",
            "exit_code": EXIT_FORCING_FAILED,
            "reason": (
                "a forced row still called the fallback or no kernel call was counted; "
                "forcing cannot be verified"
            ),
        }
    if any(not r["gradient"]["exact"] for r in rows):
        return {
            "verdict": "gradient_diff",
            "exit_code": EXIT_GRADIENT_DIFF,
            "reason": "gradient was not bit-exact on at least one row",
        }
    if not any(r["forward"]["exact"] is False for r in fused):
        return {
            "verdict": "forward_effect_absent",
            "exit_code": EXIT_FORWARD_EFFECT_ABSENT,
            "reason": (
                "no row with a counted kernel call showed the forward difference; "
                "the numerical effect cannot be demonstrated"
            ),
        }
    return {
        "verdict": "mechanism_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "a fused kernel call was counted, forcing engaged with 0 fallback calls "
            "and a counted kernel call on every forced row, the gradient is bit-exact "
            "on all rows, and the forward difference appears only on fused rows"
        ),
    }


def run_sweep(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"skipped": "CUDA is required", "exit_code": EXIT_SKIP}
    try:
        import bitsandbytes  # noqa: F401 - availability probe only
    except ImportError:
        return {"skipped": "bitsandbytes is required", "exit_code": EXIT_SKIP}

    dtype = getattr(torch, args.dtype)
    shapes = [tuple(int(p) for p in s.split("x")) for s in args.shapes.split(",")]
    fixture_shapes = [tuple(int(p) for p in s.split("x")) for s in FIXTURE_SHAPES.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    default_m = [int(m) for m in args.m.split(",")]
    forced_m = [int(m) for m in FORCED_M.split(",")]
    crosscheck_m = [int(m) for m in CROSSCHECK_M.split(",")]
    fixture_m = [int(m) for m in FIXTURE_M.split(",")]

    rows = []
    for seed in seeds:
        for k, n in shapes:
            for m in default_m:
                rows.append(measure_row(seed, k, n, m, dtype))
    for seed in seeds:
        for k, n in shapes:
            for m in forced_m:
                rows.append(measure_row(seed, k, n, m, dtype, force=True))
    if not args.no_crosscheck:
        for k, n in shapes:
            for m in crosscheck_m:
                rows.append(measure_row(seeds[0], k, n, m, dtype))
    if not args.no_fixture:
        for k, n in fixture_shapes:
            for m in fixture_m:
                rows.append(measure_row(0, k, n, m, dtype))

    verdict = evaluate(rows)
    return {
        "env": hardware(args.dtype),
        "protocol": {
            "note": (
                "both arms resident; dequantise+F.linear vs MatMul4Bit; "
                "gradient reported separately"
            ),
            "blocksize": BLOCKSIZE, "quant_type": "nf4", "double_quant": True,
            "shapes": shapes, "default_m": default_m, "forced_m": forced_m,
            "crosscheck_m": crosscheck_m if not args.no_crosscheck else [],
            "fixture_shapes": fixture_shapes if not args.no_fixture else [],
            "fixture_m": fixture_m if not args.no_fixture else [], "seeds": seeds,
        },
        "rows": rows,
        "n_rows": len(rows),
        "n_fused": sum(r["resident_kernel_calls"] > 0 for r in rows),
        "n_forced": sum(r["set"] == "forced" for r in rows),
        "n_gradient_exact": sum(r["gradient"]["exact"] for r in rows),
        "n_forward_diff_on_fused": sum(
            r["resident_kernel_calls"] > 0 and r["forward"]["exact"] is False
            for r in rows
        ),
        **verdict,
    }


def run_self_test() -> None:
    import torch

    a = torch.tensor([100.0, -2.0, 3.0])
    b = a.clone()
    b[1] = -2.5
    s = stats(a, b)
    assert s["exact"] is False and s["n_diff"] == 1 and s["max_abs"] == 0.5
    assert stats(a, a.clone())["exact"] is True

    ops = SimpleNamespace(
        _gemm_4bit_use_custom_fn=lambda *args: False, _gemm_4bit_custom_max_m=1536
    )
    assert selected_custom(ops, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is False
    with force_fused(ops):
        assert selected_custom(ops, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is True
    assert selected_custom(ops, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is False

    fake_ops = SimpleNamespace(
        _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
    )
    fake_fn = SimpleNamespace(dequantize_4bit=lambda: None)
    with counter(fake_ops, fake_fn) as counts:
        fake_ops._dequant_linear_fallback()
        fake_fn.dequantize_4bit()
    assert counts["fallback"] == 1 and counts["dequant"] == 1 and counts["kernel"] == 0

    def row(**updates):
        base = {
            "set": "default", "bnb_selected_custom_kernel": True,
            "resident_kernel_calls": 1, "resident_fallback_calls": 0,
            "control_detected": True,
            "gradient": {"exact": True}, "forward": {"exact": False},
        }
        base.update(updates)
        return base

    forced_ok = row(set="forced", resident_fallback_calls=0)
    assert evaluate([row(), forced_ok])["exit_code"] == EXIT_OK
    # A forced row's dispatcher decision is True under force_fused even when no
    # kernel ran; the gate must read the counted kernel call, not the decision.
    assert evaluate([row(bnb_selected_custom_kernel=True, resident_kernel_calls=0)])[
        "exit_code"
    ] == EXIT_NO_FUSED
    assert evaluate([row(control_detected=False), forced_ok])["exit_code"] == EXIT_CONTROL_FAILED
    assert evaluate([row(), row(set="forced", resident_fallback_calls=1)])[
        "exit_code"
    ] == EXIT_FORCING_FAILED
    assert evaluate([row(), row(set="forced", resident_kernel_calls=0)])[
        "exit_code"
    ] == EXIT_FORCING_FAILED
    assert evaluate([row(gradient={"exact": False}), forced_ok])["exit_code"] == EXIT_GRADIENT_DIFF
    forced_exact = row(set="forced", resident_fallback_calls=0, forward={"exact": True})
    assert evaluate([row(forward={"exact": True}), forced_exact])[
        "exit_code"
    ] == EXIT_FORWARD_EFFECT_ABSENT
    print("numerics2.py --self-test: OK")


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
