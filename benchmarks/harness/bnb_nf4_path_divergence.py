"""Standalone probe: bitsandbytes' fused NF4 GEMM vs ``dequantize_4bit`` + ``F.linear``.

No Soup import, no model download. One CUDA device. This is the measurement behind
https://github.com/MakazhanAlpamys/Soup/issues/776 and ``benchmarks/nf4-on-blackwell.md``:
on an RTX 5070 Laptop GPU (sm_120) the two ways of multiplying by the SAME packed NF4 weight
give slightly different numbers at small row counts, with no layer streaming anywhere.

Three arms per (shape, dtype, rows):

- ``fused``     ``bnb.matmul_4bit(x, packed.t(), quant_state=state)`` with no grad needed,
                which calls ``torch.ops.bitsandbytes.gemm_4bit`` directly;
- ``autograd``  the same call with ``x.requires_grad_()``, i.e. ``MatMul4Bit.apply``;
- ``dequant``   ``F.linear(x, dequantize_4bit(packed, state))``.

``reference`` is the same dequantised weight multiplied in fp32 (TF32 off), so each arm's own
rounding error is measured against the same exact-ish product, not just against each other.

Usage::

    python bnb_nf4_path_divergence.py --json out.json
    python bnb_nf4_path_divergence.py --bnb-path C:/scratch/bnb-main-target --json main.json

``--bnb-path`` puts a directory (a ``pip install --target`` of another bitsandbytes build) in
front of ``sys.path``; the resolved ``bitsandbytes.__file__`` is printed and recorded so a
silent fall-back to the installed copy is visible.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

DEFAULT_SHAPES = "4096x4096,4096x14336,14336x4096"  # K x N (in_features x out_features)
DEFAULT_ROWS = "1,4,8,12,32,33,64,256,512,1536,1537,2048"
DEFAULT_DTYPES = "float16,bfloat16"


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shapes", default=DEFAULT_SHAPES, help="KxN list, comma separated")
    parser.add_argument("--rows", default=DEFAULT_ROWS, help="M values, comma separated")
    parser.add_argument("--dtypes", default=DEFAULT_DTYPES)
    parser.add_argument("--weight-std", type=float, default=1.0)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--no-double-quant", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bnb-path", default=None, help="directory to put first on sys.path")
    parser.add_argument("--json", default=None, help="write the full record here")
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


def _dist_version(bnb_path) -> str:
    import importlib.metadata as metadata

    try:
        if bnb_path:
            dists = list(metadata.distributions(path=[bnb_path]))
            for dist in dists:
                if (dist.metadata["Name"] or "").lower() == "bitsandbytes":
                    return dist.version
        return metadata.version("bitsandbytes")
    except Exception:  # noqa: BLE001 - provenance only
        return "unknown"


def _selected_custom(bnb_ops, device_index, dtype, rows, out_features, in_features, blocksize):
    """True/False when bnb's own private dispatch can be asked, else None."""
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


def _stats(candidate, other):
    diff = (candidate.float() - other.float()).abs()
    return {
        "max_abs": diff.max().item(),
        "n_diff": int((diff != 0).sum().item()),
        "n_total": diff.numel(),
    }


def main() -> int:
    args = _parse_args()
    if args.bnb_path:
        sys.path.insert(0, os.path.abspath(args.bnb_path))

    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required")
        return 0

    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    try:
        import bitsandbytes.backends.cuda.ops as bnb_ops
    except Exception:  # noqa: BLE001
        bnb_ops = None

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": props.name,
        "capability": f"sm_{props.major}{props.minor}",
        "vram_mib": props.total_memory // (1024 * 1024),
        "driver": _driver_version(),
        "bitsandbytes_module_version": bnb.__version__,
        "bitsandbytes_dist_version": _dist_version(args.bnb_path),
        "bitsandbytes_file": os.path.abspath(bnb.__file__),
        "double_quant": not args.no_double_quant,
        "blocksize": args.blocksize,
        "weight_std": args.weight_std,
        "seed": args.seed,
    }
    for key, value in env.items():
        print(f"{key:>28}  {value}")

    shapes = [tuple(int(part) for part in item.split("x")) for item in args.shapes.split(",")]
    rows_list = [int(item) for item in args.rows.split(",")]
    dtypes = [getattr(torch, name) for name in args.dtypes.split(",")]

    records = []
    header = (
        f"{'K':>6} {'N':>6} {'dtype':>9} {'M':>5} {'custom':>7} {'fused-vs-deq':>13} "
        f"{'autog-vs-deq':>13} {'autog-vs-fused':>14} {'fused-vs-ref':>13} {'deq-vs-ref':>11} "
        f"{'max|out|':>9}"
    )
    print()
    print(header)
    for in_features, out_features in shapes:
        for dtype in dtypes:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            weight = torch.randn(
                out_features, in_features, device=device, dtype=dtype, generator=generator
            )
            weight = weight * args.weight_std
            packed, state = quantize_4bit(
                weight,
                blocksize=args.blocksize,
                quant_type="nf4",
                compress_statistics=not args.no_double_quant,
            )
            dense = dequantize_4bit(packed, state).to(dtype)
            dense32 = dense.float()
            del weight
            for rows in rows_list:
                inputs = torch.randn(
                    rows, in_features, device=device, dtype=dtype, generator=generator
                )
                with torch.no_grad():
                    fused = bnb.matmul_4bit(inputs, packed.t(), quant_state=state)
                    fused_again = bnb.matmul_4bit(inputs, packed.t(), quant_state=state)
                    dequant = functional.linear(inputs, dense)
                    dequant_again = functional.linear(inputs, dense)
                    reference = functional.linear(inputs.float(), dense32)
                grad_inputs = inputs.clone().requires_grad_(True)
                autograd = bnb.matmul_4bit(grad_inputs, packed.t(), quant_state=state)
                autograd = autograd.detach()

                fused_vs_dequant = _stats(fused, dequant)
                autograd_vs_dequant = _stats(autograd, dequant)
                autograd_vs_fused = _stats(autograd, fused)
                fused_vs_ref = _stats(fused, reference)
                dequant_vs_ref = _stats(dequant, reference)
                record = {
                    "K": in_features,
                    "N": out_features,
                    "dtype": str(dtype).replace("torch.", ""),
                    "M": rows,
                    "bnb_selected_custom_kernel": _selected_custom(
                        bnb_ops, 0, dtype, rows, out_features, in_features, args.blocksize
                    ),
                    "fused_vs_dequant": fused_vs_dequant,
                    "autograd_vs_dequant": autograd_vs_dequant,
                    "autograd_vs_fused": autograd_vs_fused,
                    "fused_vs_fp32_reference": fused_vs_ref,
                    "dequant_vs_fp32_reference": dequant_vs_ref,
                    "max_abs_output": reference.abs().max().item(),
                    "fused_repeat_exact": bool(torch.equal(fused, fused_again)),
                    "dequant_repeat_exact": bool(torch.equal(dequant, dequant_again)),
                }
                records.append(record)
                custom = record["bnb_selected_custom_kernel"]
                print(
                    f"{in_features:>6} {out_features:>6} {record['dtype']:>9} {rows:>5} "
                    f"{('?' if custom is None else str(custom)):>7} "
                    f"{fused_vs_dequant['max_abs']:>13.6e} "
                    f"{autograd_vs_dequant['max_abs']:>13.6e} "
                    f"{autograd_vs_fused['max_abs']:>14.6e} "
                    f"{fused_vs_ref['max_abs']:>13.6e} {dequant_vs_ref['max_abs']:>11.6e} "
                    f"{record['max_abs_output']:>9.2f}"
                )
                del inputs, fused, fused_again, dequant, dequant_again, reference, autograd
            del dense, dense32, packed, state
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump({"env": env, "records": records}, handle, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
