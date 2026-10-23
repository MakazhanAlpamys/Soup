#!/usr/bin/env python3
"""Per-layer CPU NF4 reconstruction for #379 / STEP 13.

This is NOT a replay of the full-model tiny-Llama logits numbers recorded in
``tests/test_v07202.py`` (1.948e-03 at 8 tokens, ~1.972e-03 from 32..256 on the
H100 host CPU, and 2.509e-03 on CI). Those numbers compare a streamed two-layer
model with a resident model.

Instead this harness reconstructs, one ``Linear4bit`` at a time, both STEP
13 questions that were lost with the scratchpad: where the CI fixture sits
relative to the CPU fused-kernel window (the historical
``fixture_window_cpu.py``) and whether bitsandbytes' CPU *inference* packing
path creates the rounding-scale numerical difference investigated by the
historical ``cpu_mode_probe.py``.

Requirements / attribution boundary
-----------------------------------
- AVX512-BF16 is required for the mechanism this gate is intended to
  demonstrate. Without it, the default invocation FAILS; use ``--survey`` for a
  capability-only report.
- Importing bitsandbytes on an AVX512-BF16 host may fetch/use
  ``kernels-community/quantization-bitsandbytes``; that happens while the CPU
  backend module is imported, before a row reaches the packed inference path.
  ``packing_format_for_cpu`` proves that packing ran. The JSON separately
  records best-effort lower-level kernel attribution and whether the CPU
  ``gemv_4bit`` dispatch is registered. No "no downloads" promise is made.
- The "variant 2" arm imports Soup's shipped ``install_dequant_forward``
  unmodified. NF4 uses ``compress_statistics=True`` (double quant), matching
  layer-stream shards.

Default invocation is a mechanism gate:
1. the training arm must stay unpacked and equal Soup variant 2 EXACTLY;
2. at least one inference row must enter CPU packing;
3. every packed inference row must differ from variant 2;
4. every packed difference must remain rounding-scale: strictly greater than
   zero and at most 2% of ``max(abs(variant2))``.

Each row also computes an emulated bf16 reference from the same quantised
weight: bf16-rounded input and dequantised weight, fp32 accumulation, then a
bf16-rounded output. The JSON publishes its relative error and the ratio between
the 2% envelope and the worst emulated row, so the envelope is justified by the
same run. A previous review-time forced broken-packing simulation produced
order-one errors, but that patched experiment is not an arm of this harness and
is not used as acceptance evidence.

Exit codes: 0 mechanism reproduced; 2 packing unavailable; 3 invalid training
control; 4 packed effect absent on at least one row; 5 packed effect is too
large to be the rounding-scale mechanism.

``--survey`` waives only exit 2 (missing capability). A broken control or
nonsensical packed result still fails closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass

M_VALUES = (8, 16, 32, 64, 128, 256)
FIXTURE_SHAPES = ((64, 64), (256, 64))

EXIT_OK = 0
EXIT_MECHANISM_ABSENT = 2
EXIT_CONTROL_FAILED = 3
EXIT_EFFECT_ABSENT = 4
EXIT_EFFECT_OUT_OF_RANGE = 5

# The default envelope is intentionally loose. Every run publishes a
# per-row emulated-bf16 reference plus the multiple by which this ceiling
# exceeds the worst emulated row, so the justification is reproducible.
PACKED_EFFECT_REL_ENVELOPE = 0.02


@dataclass(frozen=True)
class Row:
    out_features: int
    in_features: int
    m: int
    inference_packed_for_cpu: bool
    training_packed_for_cpu: bool
    variant2_max_abs: float
    emulated_bf16_vs_variant2_max_abs: float
    emulated_bf16_rel: float
    inference_vs_variant2_max_abs: float
    inference_vs_variant2_rel: float
    training_vs_variant2_max_abs: float
    inference_vs_training_max_abs: float


def _quantized_linear(weight, *, compute_dtype):
    import bitsandbytes as bnb
    import bitsandbytes.functional as functional

    packed, state = functional.quantize_4bit(
        weight,
        blocksize=64,
        compress_statistics=True,
        quant_type="nf4",
    )
    layer = bnb.nn.Linear4bit(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        compute_dtype=compute_dtype,
        compress_statistics=True,
        quant_type="nf4",
        device="cpu",
    )
    layer.weight = bnb.nn.Params4bit(
        data=packed,
        requires_grad=False,
        quant_state=state,
        blocksize=state.blocksize,
        compress_statistics=True,
        quant_type=state.quant_type,
        bnb_quantized=True,
    )
    return layer


def _row_seed(seed: int, out_features: int, in_features: int, m: int) -> int:
    """Derive a deterministic stream from the complete fixture tuple."""
    return seed + out_features * 1_000_003 + in_features * 1_009 + m


def _emulated_bf16_reference(layer, x):
    """Emulate the rounding-only bf16 path using this row's quantised weight."""
    import bitsandbytes.functional as bnb_functional
    import torch

    dense_quantized = bnb_functional.dequantize_4bit(
        layer.weight,
        layer.weight.quant_state,
    ).to(torch.float32)
    return torch.nn.functional.linear(
        x.to(torch.bfloat16).to(torch.float32),
        dense_quantized.to(torch.bfloat16).to(torch.float32),
    ).to(torch.bfloat16).to(torch.float32)


def measure_row(out_features: int, in_features: int, m: int, *, seed: int = 17) -> Row:
    import torch

    from soup_cli.utils.layer_stream_runtime import install_dequant_forward

    row_seed = _row_seed(seed, out_features, in_features, m)
    generator = torch.Generator().manual_seed(row_seed)
    weight = torch.randn(
        (out_features, in_features),
        generator=generator,
        dtype=torch.float32,
    )
    x = torch.randn(
        (m, in_features),
        generator=generator,
        dtype=torch.float32,
    )

    variant2_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    patched = install_dequant_forward(variant2_layer)
    if patched != 1:
        raise RuntimeError(
            f"install_dequant_forward patched {patched} layers, expected 1"
        )
    variant2_layer.train()
    variant2_x = x.clone().requires_grad_(True)
    variant2 = variant2_layer(variant2_x).detach()

    emulated_bf16 = _emulated_bf16_reference(variant2_layer, x)

    training_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    training_layer.train()
    training_x = x.clone().requires_grad_(True)
    training = training_layer(training_x).detach()
    training_packed = bool(
        getattr(
            training_layer.weight.quant_state,
            "packing_format_for_cpu",
            False,
        )
    )

    inference_layer = _quantized_linear(weight, compute_dtype=torch.float32)
    inference_layer.eval()
    with torch.no_grad():
        inference = inference_layer(x)
    inference_packed = bool(
        getattr(
            inference_layer.weight.quant_state,
            "packing_format_for_cpu",
            False,
        )
    )

    inference_diff = float((inference - variant2).abs().max())
    emulated_bf16_diff = float((emulated_bf16 - variant2).abs().max())
    variant2_max = float(variant2.abs().max())
    if variant2_max > 0.0:
        inference_rel = inference_diff / variant2_max
        emulated_bf16_rel = emulated_bf16_diff / variant2_max
    elif inference_diff == 0.0:
        inference_rel = 0.0
        emulated_bf16_rel = 0.0 if emulated_bf16_diff == 0.0 else float("inf")
    else:
        inference_rel = float("inf")
        emulated_bf16_rel = (
            0.0 if emulated_bf16_diff == 0.0 else float("inf")
        )

    return Row(
        out_features=out_features,
        in_features=in_features,
        m=m,
        inference_packed_for_cpu=inference_packed,
        training_packed_for_cpu=training_packed,
        variant2_max_abs=variant2_max,
        emulated_bf16_vs_variant2_max_abs=emulated_bf16_diff,
        emulated_bf16_rel=emulated_bf16_rel,
        inference_vs_variant2_max_abs=inference_diff,
        inference_vs_variant2_rel=inference_rel,
        training_vs_variant2_max_abs=float(
            (training - variant2).abs().max()
        ),
        inference_vs_training_max_abs=float(
            (inference - training).abs().max()
        ),
    )


def _evaluate_rows(rows: list[Row]) -> dict:
    # A broken control is a harness/Soup failure on every host; do not hide it
    # behind a missing-AVX512 capability verdict.
    if any(row.training_packed_for_cpu for row in rows) or any(
        row.training_vs_variant2_max_abs != 0.0 for row in rows
    ):
        return {
            "verdict": "control_failed",
            "exit_code": EXIT_CONTROL_FAILED,
            "reason": (
                "training-style Linear4bit did not remain an exact unpacked "
                "control against Soup install_dequant_forward."
            ),
            "attribution": "invalid-control",
        }

    packed_rows = [row for row in rows if row.inference_packed_for_cpu]
    if not packed_rows:
        return {
            "verdict": "mechanism_absent",
            "exit_code": EXIT_MECHANISM_ABSENT,
            "reason": (
                "CPU inference packing did not run; this host cannot demonstrate "
                "the cpu_mode_probe mechanism."
            ),
            "attribution": "none",
        }

    if any(row.inference_vs_variant2_max_abs == 0.0 for row in packed_rows):
        return {
            "verdict": "packed_effect_absent",
            "exit_code": EXIT_EFFECT_ABSENT,
            "reason": (
                "the packed inference path ran, but at least one packed row did "
                "not differ from Soup variant 2."
            ),
            "attribution": "packed-path-detected-no-universal-divergence",
        }

    if any(
        not (0.0 < row.inference_vs_variant2_rel <= PACKED_EFFECT_REL_ENVELOPE)
        for row in packed_rows
    ):
        return {
            "verdict": "packed_effect_out_of_range",
            "exit_code": EXIT_EFFECT_OUT_OF_RANGE,
            "reason": (
                "the packed inference path ran, but at least one packed row "
                "differs by more than the 2% rounding-scale envelope."
            ),
            "attribution": "packed-path-detected-non-rounding-scale-effect",
        }

    return {
        "verdict": "mechanism_reproduced",
        "exit_code": EXIT_OK,
        "reason": (
            "every packed row differs from Soup variant 2 within the 2% "
            "rounding-scale envelope while the training control is exact."
        ),
        "attribution": (
            "bitsandbytes packed CPU inference path; exact lower-level kernel "
            "reported separately"
        ),
    }


_MISSING_KERNEL_MARKER = object()


def _cpu_packed_kernel_attribution() -> str:
    """Best-effort bnb 0.50.x attribution without making it a gate."""
    try:
        import bitsandbytes.backends.cpu.ops as cpu_ops
    except Exception:
        return "unknown"

    marker = getattr(
        cpu_ops,
        "gemm_4bit_forward_kernel",
        _MISSING_KERNEL_MARKER,
    )
    if marker is _MISSING_KERNEL_MARKER:
        registered = _cpu_gemv_registered()
        if registered is False:
            return "not-registered"
        return "unknown"
    if marker is None:
        return "native-bitsandbytes-cpu-gemv"
    return "kernels-community"


def _cpu_gemv_registered() -> bool | None:
    try:
        import torch

        return bool(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "bitsandbytes::gemv_4bit", "CPU"
            )
        )
    except Exception:
        return None


def run_probe(*, m_values=M_VALUES, shapes=FIXTURE_SHAPES) -> dict:
    import bitsandbytes
    import bitsandbytes.functional as functional
    import torch

    rows = [
        measure_row(out_features, in_features, m)
        for out_features, in_features in shapes
        for m in m_values
    ]
    verdict = _evaluate_rows(rows)
    emulated_worst = max((row.emulated_bf16_rel for row in rows), default=0.0)
    envelope_multiple = (
        PACKED_EFFECT_REL_ENVELOPE / emulated_worst
        if emulated_worst > 0.0
        else None
    )
    return {
        "torch": torch.__version__,
        "bitsandbytes": bitsandbytes.__version__,
        "platform": platform.platform(),
        "has_avx512bf16": bool(functional.has_avx512bf16()),
        "kernels_package_installed": (
            importlib.util.find_spec("kernels") is not None
        ),
        "cpu_packed_kernel": _cpu_packed_kernel_attribution(),
        "cpu_gemv_4bit_registered": _cpu_gemv_registered(),
        "rows": [asdict(row) for row in rows],
        "packed_inference_rows": sum(
            row.inference_packed_for_cpu for row in rows
        ),
        "packed_training_rows": sum(
            row.training_packed_for_cpu for row in rows
        ),
        "training_control_exact": all(
            row.training_vs_variant2_max_abs == 0.0 for row in rows
        ),
        "packed_effect_rel_envelope": PACKED_EFFECT_REL_ENVELOPE,
        "emulated_bf16_worst_rel": emulated_worst,
        "envelope_over_emulated_worst": envelope_multiple,
        **verdict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", help="optional JSON output path")
    parser.add_argument(
        "--survey",
        action="store_true",
        help=(
            "capability-only survey: waive exit 2 when CPU packing is absent; "
            "control/effect failures still fail closed"
        ),
    )
    args = parser.parse_args(argv)

    result = run_probe()
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        from pathlib import Path

        Path(args.output).write_text(
            payload + "\n",
            encoding="utf-8",
        )

    code = int(result["exit_code"])
    if args.survey and code == EXIT_MECHANISM_ABSENT:
        return EXIT_OK
    if code != EXIT_OK:
        print(str(result["reason"]), file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
