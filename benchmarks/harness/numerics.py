#!/usr/bin/env python3
"""STEP 13, first attempt — the VACUOUS numerics answer (kept on purpose).

This is a per-layer reconstruction, for #379, of the historical
``benchmarks/harness/numerics.py`` whose result is recorded in
``benchmarks/gate-h100-validation.md`` STEP 13. It reproduces the RESULT and,
more importantly, the REASON it was vacuous; it does not re-run a model.

The question STEP 13 asked was whether Soup's "variant 2" repair
(dequantise the streamed NF4 weight with ``dequantize_4bit`` and multiply with a
native ``F.linear``) changes the arithmetic of ``bitsandbytes.MatMul4Bit``.
Both arms here are resident: there is no streaming and no checkpointing. The
comparison is

    variant 2:  F.linear(x, dequantize_4bit(packed, state).to(x.dtype))
    resident:   bnb.matmul_4bit(x, packed.t(), quant_state=state)

on the SAME packed NF4 weight and quant_state, at real Qwen2.5-32B projection
shapes, blocksize 64, nf4, double quant, M = 2048 tokens, 5 seeds x 5 shapes.

THE POINT OF THIS HARNESS IS THAT THE ANSWER IS MEANINGLESS. At M = 2048,
bitsandbytes' own dispatcher has already crossed ``_gemm_4bit_custom_max_m``
(1536 on CUDA), so ``bnb.matmul_4bit`` takes ``_dequant_linear_fallback`` —
which is itself dequantise + linear. Both arms run the same code. Forward and
gradient being bit-exact on 25/25 rows is true and says nothing about the fused
kernel.

So the default invocation FAILS CLOSED. It exits 0 only when the vacuity is
actually demonstrated: every resident row reached the fallback, the two arms
agree bit-exactly on forward and gradient, and the negative control (a sign
flip plus a whole unit, which cannot round away) is detected. If the fused
kernel runs at M = 2048 on this host, the thing this harness demonstrates is
absent and it exits 2 instead of presenting the agreement as clean numerics.

The historical first control counted ``bitsandbytes.functional.dequantize_4bit``
and saw zero in the forward, which was mis-read as "the fused kernel ran". The
fallback reaches the dequantiser through a registered torch op, not that
wrapper. This harness counts ``_dequant_linear_fallback`` directly.

CPU self-test
-------------
``--self-test`` needs only torch. It exercises the stats, the dispatch reader,
the counter and the fail-closed gate on synthetic/fake data; no CUDA, no
bitsandbytes import, no download.

Exit codes (default invocation):
  0  vacuity reproduced (fallback taken on every row, both arms exact, control detected)
  2  fused kernel ran at M = 2048 — vacuity absent, agreement cannot be read as clean
  3  invalid control (the perturbed weight did not change the result)
  4  gradient or forward differed where the fallback was taken
  5  CUDA / bitsandbytes unavailable for the default sweep (SKIP, exits 0)
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

# K x N (in_features x out_features) of real Qwen2.5-32B projections.
DEFAULT_SHAPES = "5120x27648,5120x27648,27648x5120,5120x5120,5120x1024"
DEFAULT_M = 2048
DEFAULT_SEEDS = "0,1,2,3,4"
BLOCKSIZE = 64
FLOATED_POINT = 1.0

EXIT_OK = 0
EXIT_NOT_VACUOUS = 2
EXIT_CONTROL_FAILED = 3
EXIT_NOT_EXACT = 4
EXIT_SKIP = 0


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", default=DEFAULT_SHAPES)
    parser.add_argument("--m", type=int, default=DEFAULT_M)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--json", default=None, help="write the full record here")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _dist_version() -> str:
    import importlib.metadata as metadata

    try:
        return metadata.version("bitsandbytes")
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _sm_clock_mhz_idle():
    """Best-effort current SM clock; an idle reading, not a busy-window figure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def hardware() -> dict:
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
        "n_diff": int((bits(candidate) != bits(other)).sum()),
        "n_total": candidate.numel(),
        "finite": finite,
    }


def selected_custom(bnb_ops, device_index, dtype, rows, out_features, in_features, blocksize):
    """Ask bitsandbytes' own private dispatch which kernel an M would take."""
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
    except Exception:  # noqa: BLE001 - a private symbol that changed signature
        return None


@contextmanager
def counter(ops_module, functional_module):
    """Count the executed backend implementations, never infer a path from a gate.

    ``functional.dequantize_4bit`` is counted too, but the fallback reaches the
    dequantiser through a registered torch op, so a zero there must NOT be read
    as "the fused kernel ran" — that was the historical mistake.
    """
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
            original = getattr(module, name)  # missing hook invalidates the run
            stack.enter_context(patch.object(module, name, wrap(label, original)))
        yield counts


def _quantized_linear(weight):
    import bitsandbytes.functional as functional

    packed, state = functional.quantize_4bit(
        weight,
        blocksize=BLOCKSIZE,
        quant_type="nf4",
        compress_statistics=True,
    )
    return packed, state


def measure_row(seed, in_features, out_features, rows):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional
    import torch
    import torch.nn.functional as torch_functional
    from bitsandbytes.backends.cuda import ops as bnb_ops

    generator = torch.Generator(device="cuda").manual_seed(seed)
    weight = torch.randn(
        (out_features, in_features), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    packed, state = _quantized_linear(weight)
    inputs = torch.randn(
        (rows, in_features), device="cuda", dtype=torch.bfloat16, generator=generator
    )

    decision = selected_custom(
        bnb_ops, 0, torch.bfloat16, rows, out_features, in_features, BLOCKSIZE
    )

    # variant 2: dequantise + native linear, both forward and gradient.
    variant_x = inputs.clone().requires_grad_(True)
    with counter(bnb_ops, functional) as variant_counts:
        dense = functional.dequantize_4bit(packed, state).to(variant_x.dtype)
        variant_out = torch_functional.linear(variant_x, dense)
    variant_out.sum().backward()
    variant_grad = variant_x.grad.detach().clone()

    # resident: bitsandbytes MatMul4Bit, both forward and gradient.
    resident_x = inputs.clone().requires_grad_(True)
    with counter(bnb_ops, functional) as resident_counts:
        resident_out = bnb.matmul_4bit(resident_x, packed.t(), quant_state=state)
    resident_out.sum().backward()
    resident_grad = resident_x.grad.detach().clone()

    # Negative control from the record: sign flip plus a whole unit, asserted to
    # survive bf16 rounding (the first attempt's 1e-3 relative perturbation
    # rounded away and compared a tensor against itself).
    control_weight = weight.clone()
    magnitude = float(control_weight.abs().max())
    control_weight[0, 0] = -control_weight[0, 0] - magnitude
    control_packed, control_state = _quantized_linear(control_weight)
    with torch.no_grad():
        control_dense = functional.dequantize_4bit(control_packed, control_state).to(inputs.dtype)
        control_out = torch_functional.linear(inputs, control_dense)
    control_detected = not torch.equal(control_out, resident_out.detach())

    result = {
        "seed": seed,
        "K": in_features,
        "N": out_features,
        "M": rows,
        "bnb_selected_custom_kernel": decision,
        "variant_fallback_calls": int(variant_counts["fallback"]),
        "resident_fallback_calls": int(resident_counts["fallback"]),
        "resident_kernel_calls": int(resident_counts["kernel"]),
        "forward": stats(variant_out.detach(), resident_out.detach()),
        "gradient": stats(variant_grad, resident_grad),
        "control_detected": bool(control_detected),
    }
    del weight, packed, state, inputs, dense, control_weight, control_packed, control_state
    torch.cuda.empty_cache()
    return result


def evaluate(rows: list[dict]) -> dict:
    """The vacuity gate. Fail closed when agreement cannot be claimed as clean."""
    if not rows:
        return {"verdict": "no_rows", "exit_code": EXIT_NOT_VACUOUS, "reason": "no rows measured"}

    fused = [r for r in rows if r["bnb_selected_custom_kernel"] is True]
    if fused:
        return {
            "verdict": "not_vacuous",
            "exit_code": EXIT_NOT_VACUOUS,
            "reason": (
                "the fused 4-bit kernel ran at M=%d on at least one row; at this M "
                "the historical run was vacuous because every arm fell back. The "
                "agreement here is not evidence of numerical cleanliness."
                % rows[0]["M"]
            ),
        }
    if any(not r["resident_fallback_calls"] for r in rows):
        return {
            "verdict": "not_vacuous",
            "exit_code": EXIT_NOT_VACUOUS,
            "reason": "at least one resident row did not take the dequant+linear fallback",
        }
    if any(not r["control_detected"] for r in rows):
        return {
            "verdict": "control_failed",
            "exit_code": EXIT_CONTROL_FAILED,
            "reason": "the sign-flip-plus-unit negative control rounded away on at least one row",
        }
    if any(
        not r["forward"]["exact"] or not r["gradient"]["exact"] for r in rows
    ):
        return {
            "verdict": "not_exact",
            "exit_code": EXIT_NOT_EXACT,
            "reason": "a row that took the fallback still differed between the two arms",
        }
    return {
        "verdict": "vacuity_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "every row took the fallback at M=%d, both arms agree bit-exactly on "
            "forward and gradient, and the negative control was detected: the "
            "bit-exact answer is VACUOUS, not evidence about the fused kernel."
            % rows[0]["M"]
        ),
    }


def run_sweep(args) -> dict:
    import torch

    if not torch.cuda.is_available():
        return {"skipped": "CUDA is required", "exit_code": EXIT_SKIP}

    shapes = [tuple(int(p) for p in s.split("x")) for s in args.shapes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for seed in seeds:
        for in_features, out_features in shapes:
            rows.append(measure_row(seed, in_features, out_features, args.m))

    verdict = evaluate(rows)
    return {
        "env": hardware(),
        "protocol": {
            "note": "both arms resident; variant 2 vs MatMul4Bit on the same NF4 weight",
            "blocksize": BLOCKSIZE,
            "quant_type": "nf4",
            "double_quant": True,
            "M": args.m,
            "seeds": seeds,
            "shapes": shapes,
        },
        "rows": rows,
        "n_rows": len(rows),
        "n_fused": sum(r["bnb_selected_custom_kernel"] is True for r in rows),
        "n_fallback": sum(bool(r["resident_fallback_calls"]) for r in rows),
        "n_control_detected": sum(r["control_detected"] for r in rows),
        **verdict,
    }


def run_self_test() -> None:
    """Pure-Python/torch checks of the pieces, with no CUDA and no bnb."""
    import torch

    a = torch.tensor([1.0, 2.0, 3.0])
    b = a.clone()
    assert stats(a, b)["exact"] is True
    b[1] = 2.5
    s = stats(a, b)
    assert s["exact"] is False and s["n_diff"] == 1 and s["max_abs"] == 0.5

    ops = SimpleNamespace(
        _gemm_4bit_use_custom_fn=lambda *args: False, _gemm_4bit_custom_max_m=1536
    )
    # past the cap, bnb does not even ask its dispatch: no fused kernel here.
    assert selected_custom(ops, 0, "bf16", 2048, 27648, 5120, BLOCKSIZE) is False
    assert selected_custom(ops, 0, "bf16", 8, 27648, 5120, BLOCKSIZE) is False

    good = {
        "M": 2048,
        "bnb_selected_custom_kernel": False,
        "resident_fallback_calls": 1,
        "forward": {"exact": True},
        "gradient": {"exact": True},
        "control_detected": True,
    }
    assert evaluate([good])["exit_code"] == EXIT_OK

    fused = dict(good, bnb_selected_custom_kernel=True)
    assert evaluate([fused])["exit_code"] == EXIT_NOT_VACUOUS
    no_fb = dict(good, resident_fallback_calls=0)
    assert evaluate([no_fb])["exit_code"] == EXIT_NOT_VACUOUS
    no_ctrl = dict(good, control_detected=False)
    assert evaluate([no_ctrl])["exit_code"] == EXIT_CONTROL_FAILED
    not_exact = dict(good, gradient={"exact": False})
    assert evaluate([not_exact])["exit_code"] == EXIT_NOT_EXACT

    # The counter observes the path: a fake fallback and a fake kernel are
    # counted only when they actually run, independent of any gate.
    fake_ops = SimpleNamespace(
        _gemm_4bit_kernel_impl=lambda: None, _dequant_linear_fallback=lambda: None
    )
    fake_fn = SimpleNamespace(dequantize_4bit=lambda: None)
    with counter(fake_ops, fake_fn) as counts:
        fake_ops._dequant_linear_fallback()
        fake_fn.dequantize_4bit()
    assert counts["fallback"] == 1 and counts["dequant"] == 1 and counts["kernel"] == 0

    print("numerics.py --self-test: OK")


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
