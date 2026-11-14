"""Check the locally built FA2 against explicit FP64 causal GQA attention."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rich.console import Console


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import flash_attn
    import torch
    import torch.nn.functional as functional
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel

    reports = []
    for dtype in (torch.float16, torch.bfloat16):
        torch.manual_seed(378)
        tensors = [torch.randn(2, 128, heads, 128, device="cuda", dtype=dtype,
                               requires_grad=True) for heads in (4, 1, 1)]
        references = [tensor.detach().double().requires_grad_(True) for tensor in tensors]
        native = [tensor.detach().clone().requires_grad_(True) for tensor in tensors]
        actual = flash_attn_func(*tensors, causal=True, dropout_p=0.0)
        query = references[0].transpose(1, 2)
        key, value = [tensor.repeat_interleave(4, dim=2).transpose(1, 2)
                      for tensor in references[1:]]
        scores = query @ key.transpose(-2, -1) / math.sqrt(128)
        mask = torch.ones(128, 128, device="cuda", dtype=torch.bool).triu(1)
        expected = (scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
                    @ value).transpose(1, 2)
        with sdpa_kernel([SDPBackend.MATH]):
            native_output = functional.scaled_dot_product_attention(
                *(tensor.transpose(1, 2) for tensor in native),
                is_causal=True, dropout_p=0.0, enable_gqa=True,
            ).transpose(1, 2)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient.double())
        native_output.backward(gradient)
        comparisons = [("output", actual, expected, native_output), *[
            (name, tensor.grad, reference.grad, native_tensor.grad)
            for name, tensor, reference, native_tensor in zip(
                ("q_grad", "k_grad", "v_grad"), tensors, references, native, strict=True,
            )
        ]]
        metrics = {}
        for name, result, reference, native_result in comparisons:
            reference = reference.detach()
            denominator = reference.norm()
            metrics[name] = {
                "fa2_max_absolute_error": float((result.detach().double() - reference).abs().max()),
                "fa2_relative_l2_error": float((result.detach().double() - reference).norm()
                                               / denominator),
                "sdpa_math_max_absolute_error": float(
                    (native_result.detach().double() - reference).abs().max()
                ),
                "sdpa_math_relative_l2_error": float(
                    (native_result.detach().double() - reference).norm() / denominator
                ),
            }
        limit = 0.01 if dtype == torch.bfloat16 else 0.002
        reports.append({"dtype": str(dtype), "relative_l2_limit": limit, "checks": metrics,
                        "passed": all(row["fa2_relative_l2_error"] <= limit
                                      for row in metrics.values())})
    record = {
        "status": "pass" if all(row["passed"] for row in reports) else "fail",
        "shape": "B2/S128/Hq4/Hkv1/D128, causal, dropout0, seed378",
        "reference": (
            "explicit FP64 attention/softmax and autograd; identical quantized inputs/dout"
        ),
        "torch": torch.__version__, "flash_attn": flash_attn.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "reports": reports,
    }
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    Console().print(record)
    if record["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
