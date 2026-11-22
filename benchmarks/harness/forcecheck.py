#!/usr/bin/env python3
"""STEP 13 — count ``_dequant_linear_fallback`` directly (the path control).

Reconstruction for #379 of the historical scratch script ``forcecheck.py``,
used in ``benchmarks/gate-h100-validation.md`` STEP 13 to answer a narrow
question: how do you tell a fused-kernel run that genuinely agreed with
dequantise + linear from a run where the forcing silently failed and everything
took the fallback anyway? The original was never in this repository (#379); the
recorded result was measured on an H100 sm90 with torch 2.13.0+cu130 and
bitsandbytes 0.50.0. A re-run on another card or stack is a new baseline, not a
comparison with that result.

The first attempt's control counted calls to
``bitsandbytes.functional.dequantize_4bit`` and read zero in the forward as
"the fused kernel ran". That is wrong: on CUDA, bitsandbytes' fallback lives in
``backends/cuda/ops.py`` and reaches the dequantiser through a registered torch
op, never through the ``functional`` wrapper. This harness therefore patches
the three real backend hooks and records, per row:

  * ``fallback_calls``      - calls to ``ops._dequant_linear_fallback``;
  * ``kernel_calls``        - calls to ``ops._gemm_4bit_kernel_impl``;
  * ``dequant_wrapper_calls`` - calls to ``functional.dequantize_4bit``, the
    proxy that the first attempt mistook for a path signal.

Only ``bnb.matmul_4bit`` (the resident arm) is measured, at M in 8..2048, with
default dispatch and with the fused kernel forced on (the two globals patched).
The demonstration is structural, not a speed or quality claim:

  * at large M the fallback is counted while the wrapper stays at zero — the
    proxy cannot see the path, the counter can;
  * at small M under DEFAULT dispatch the fused kernel is counted (forced rows
    alone cannot show this);
  * at large M with forcing on, the fallback count is zero and the kernel count
    is positive — the forcing engaged, so an agreement there is a real one.

CPU self-test: ``--self-test`` needs only torch and checks the counter, the
forcing patch on a fake module, and the fail-closed gate on synthetic rows.

Exit codes (default invocation):
  0  counter reproduced the path switch and the forcing (mechanism present)
  2  no fallback was ever counted (the M-dispatch cannot be demonstrated)
  3  the invalid proxy was not distinguishable from the real counter
  4  no default-dispatch row counted the fused kernel at small M
  5  forcing failed (a forced row still called the fallback or missed the kernel)
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
DEFAULT_SHAPES = "4096x4096,5120x27648,5120x5120,27648x5120"
DEFAULT_M = "8,16,32,64,128,256,512,1024,1536,2048"
FORCED_M = "512,2048"
DEFAULT_SEEDS = "0"

EXIT_OK = 0
EXIT_NO_FALLBACK = 2
EXIT_PROXY_NOT_DISTINGUISHABLE = 3
EXIT_NO_KERNEL = 4
EXIT_FORCING_FAILED = 5
EXIT_SKIP = 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--m", default=DEFAULT_M)
    parser.add_argument("--forced-m", default=FORCED_M)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
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
    with ExitStack() as stack:
        stack.enter_context(patch.object(bnb_ops, "_gemm_4bit_custom_max_m", 10**9))
        stack.enter_context(patch.object(bnb_ops, "_gemm_4bit_use_custom_fn", lambda *a, **k: True))
        yield


@contextmanager
def counter(ops_module, functional_module):
    """Patch the three real hooks and count what actually executed."""
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
        with torch.no_grad(), counter(bnb_ops, functional) as counts:
            _ = bnb.matmul_4bit(inputs, packed.t(), quant_state=state)

    result = {
        "seed": seed,
        "K": in_features,
        "N": out_features,
        "M": rows,
        "set": "forced" if force else "default",
        "bnb_selected_custom_kernel": decision,
        "fallback_calls": int(counts["fallback"]),
        "kernel_calls": int(counts["kernel"]),
        "dequant_wrapper_calls": int(counts["dequant"]),
    }
    del weight, packed, state, inputs
    torch.cuda.empty_cache()
    return result


def evaluate(rows: list[dict]) -> dict:
    fallback_rows = [r for r in rows if r["fallback_calls"] > 0]
    if not fallback_rows:
        return {
            "verdict": "no_fallback_observed",
            "exit_code": EXIT_NO_FALLBACK,
            "reason": (
                "the dequant+linear fallback was never counted on any row; "
                "the M-dispatch cannot be observed here"
            ),
        }
    if not any(r["fallback_calls"] > 0 and r["dequant_wrapper_calls"] == 0 for r in rows):
        return {
            "verdict": "proxy_not_distinguishable",
            "exit_code": EXIT_PROXY_NOT_DISTINGUISHABLE,
            "reason": (
                "no row took the fallback while the functional.dequantize_4bit "
                "proxy stayed at zero; the historical proxy-vs-counter distinction "
                "cannot be demonstrated"
            ),
        }
    if not any(r["set"] == "default" and r["kernel_calls"] > 0 for r in rows):
        return {
            "verdict": "no_kernel_observed",
            "exit_code": EXIT_NO_KERNEL,
            "reason": (
                "no default-dispatch row counted the fused kernel at small M; "
                "forced rows alone cannot demonstrate the switch between paths"
            ),
        }
    forced = [r for r in rows if r["set"] == "forced"]
    if not forced or any(
        r["fallback_calls"] > 0 or r["kernel_calls"] == 0 for r in forced
    ):
        return {
            "verdict": "forcing_failed",
            "exit_code": EXIT_FORCING_FAILED,
            "reason": (
                "a forced row still took the fallback or never reached the "
                "kernel; forcing cannot be verified"
            ),
        }
    return {
        "verdict": "mechanism_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "the fallback is counted at large M while the proxy stays at zero, "
            "the kernel is counted at small M under default dispatch, and forcing "
            "at large M moves every row to the kernel with 0 fallback calls"
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
    seeds = [int(s) for s in args.seeds.split(",")]
    default_m = [int(m) for m in args.m.split(",")]
    forced_m = [int(m) for m in args.forced_m.split(",")]

    rows = []
    for seed in seeds:
        for in_features, out_features in shapes:
            for m in default_m:
                rows.append(measure_row(seed, in_features, out_features, m, dtype))
    for seed in seeds:
        for in_features, out_features in shapes:
            for m in forced_m:
                rows.append(measure_row(seed, in_features, out_features, m, dtype, force=True))

    verdict = evaluate(rows)
    return {
        "env": hardware(args.dtype),
        "protocol": {
            "note": "resident MatMul4Bit only; three real backend hooks counted per row",
            "blocksize": BLOCKSIZE, "quant_type": "nf4", "double_quant": True,
            "shapes": shapes, "default_m": default_m, "forced_m": forced_m, "seeds": seeds,
        },
        "rows": rows,
        "n_rows": len(rows),
        "n_fallback_rows": sum(r["fallback_calls"] > 0 for r in rows),
        "n_kernel_rows": sum(r["kernel_calls"] > 0 for r in rows),
        "n_proxy_blind_rows": sum(
            r["fallback_calls"] > 0 and r["dequant_wrapper_calls"] == 0 for r in rows
        ),
        "n_forced_rows": sum(r["set"] == "forced" for r in rows),
        **verdict,
    }


def run_self_test() -> None:
    ops = SimpleNamespace(
        _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
    )
    fn = SimpleNamespace(dequantize_4bit=lambda: None)
    with counter(ops, fn) as counts:
        ops._dequant_linear_fallback()
        fn.dequantize_4bit()
    assert counts["fallback"] == 1 and counts["dequant"] == 1 and counts["kernel"] == 0

    dispatch = SimpleNamespace(
        _gemm_4bit_use_custom_fn=lambda *args: False, _gemm_4bit_custom_max_m=1536
    )
    assert selected_custom(dispatch, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is False
    with force_fused(dispatch):
        assert selected_custom(dispatch, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is True
    assert selected_custom(dispatch, 0, "bf16", 2048, 4096, 4096, BLOCKSIZE) is False

    def row(**updates):
        base = {
            "set": "default", "fallback_calls": 0, "kernel_calls": 1,
            "dequant_wrapper_calls": 0,
        }
        base.update(updates)
        return base

    large = row(fallback_calls=1, kernel_calls=0, dequant_wrapper_calls=0)
    small = row(fallback_calls=0, kernel_calls=1)
    forced = row(set="forced", fallback_calls=0, kernel_calls=1)
    assert evaluate([large, small, forced])["exit_code"] == EXIT_OK
    assert evaluate([small, forced])["exit_code"] == EXIT_NO_FALLBACK
    blind = row(fallback_calls=1, kernel_calls=0, dequant_wrapper_calls=2)
    assert evaluate([blind, small, forced])["exit_code"] == EXIT_PROXY_NOT_DISTINGUISHABLE
    # Forced rows alone must not satisfy the "kernel counted at small M" gate.
    assert evaluate([large, forced])["exit_code"] == EXIT_NO_KERNEL
    assert evaluate([large])["exit_code"] == EXIT_NO_KERNEL
    assert evaluate([large, small, row(set="forced", fallback_calls=1, kernel_calls=0)])[
        "exit_code"
    ] == EXIT_FORCING_FAILED
    assert evaluate([large, small, row(set="forced", fallback_calls=0, kernel_calls=0)])[
        "exit_code"
    ] == EXIT_FORCING_FAILED
    print("forcecheck.py --self-test: OK")


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
