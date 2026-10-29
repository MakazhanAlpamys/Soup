#!/usr/bin/env python3
"""CPU-only numerical probes for Issue #342's GRPO non-finite gradients.

This harness is a collection of bounded arithmetic and small CPU-autograd
experiments. It does not load a model, download data, initialise CUDA, replay
the reported H100 incident, or identify a root cause. Its purpose is narrower:
make several claims from the investigation directly reproducible and make their
evidence boundary visible in one machine-readable result.

Run::

    python benchmarks/harness/grpo_nan_numeric_probes.py --selftest --json

The Tensor operations are deliberately created on CPU. ``--selftest`` asserts
each documented claim and exits non-zero if a supported Torch build changes one
of these small controls.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Any

from rich.console import Console

K_TOKENS = 65_536
OPERAND_BOUND = 10.0


def _torch() -> Any:
    """Import Torch only when a probe is executed."""
    import torch

    return torch


def _all_finite(tensor: Any) -> bool:
    torch = _torch()
    return bool(torch.isfinite(tensor).all())


def _split_k_bound() -> dict[str, float | int]:
    """Return the conditional dense-contraction bound used to reject overflow."""
    torch = _torch()
    bound = K_TOKENS * OPERAND_BOUND * OPERAND_BOUND
    fp32_max = float(torch.finfo(torch.float32).max)
    return {
        "k": K_TOKENS,
        "operand_bound": OPERAND_BOUND,
        "bound": bound,
        "fp32_max": fp32_max,
        "orders_below_fp32_max": math.log10(fp32_max / bound),
    }


def _dtype_ranges() -> dict[str, float]:
    """Report the finite ranges relevant to the proposed overflow stories."""
    torch = _torch()
    bf16_max = float(torch.finfo(torch.bfloat16).max)
    return {
        "fp16_max": float(torch.finfo(torch.float16).max),
        "bf16_max": bf16_max,
        "fp32_max": float(torch.finfo(torch.float32).max),
        "ln_bf16_max": math.log(bf16_max),
    }


def _constructed_masked_exponential() -> dict[str, bool | float | int]:
    """Construct one finite scalar loss whose backward contains ``0 * inf``.

    This only proves that an ostensibly masked, overflowing exponential can
    yield a non-finite gradient in this deliberately constructed graph. It does
    not establish that the graph occurred in stock TRL GRPO or in Issue #342.
    """
    torch = _torch()
    log_ratio = torch.tensor(
        [[0.0, 0.0, 100.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
        device="cpu",
    )
    old_logps = torch.zeros_like(log_ratio)
    completion_mask = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.bfloat16, device="cpu")
    advantage = torch.ones((1, 1), dtype=torch.bfloat16, device="cpu")

    ratio = torch.exp(log_ratio - old_logps)
    clipped_ratio = torch.clamp(ratio, 0.8, 1.2)
    loss = (
        -torch.minimum(ratio * advantage, clipped_ratio * advantage) * completion_mask
    ).sum() / completion_mask.sum()
    loss.backward()

    gradient = log_ratio.grad
    assert gradient is not None
    return {
        "loss": float(loss.detach()),
        "loss_finite": _all_finite(loss),
        "gradient_finite": _all_finite(gradient),
        "nonfinite_gradient_elements": int((~torch.isfinite(gradient)).sum().item()),
    }


def _default_on_policy_ratio() -> dict[str, bool]:
    """Exercise the old-log-probability detach used by default on-policy GRPO."""
    torch = _torch()
    policy_logps = torch.tensor(
        [[-12.0, -3.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
        device="cpu",
    )
    ratio = torch.exp(policy_logps - policy_logps.detach())
    ratio.sum().backward()

    gradient = policy_logps.grad
    assert gradient is not None
    return {
        "ratio_all_one": bool(torch.equal(ratio, torch.ones_like(ratio))),
        "ratio_finite": _all_finite(ratio),
        "gradient_finite": _all_finite(gradient),
    }


def _boolean_selection() -> dict[str, bool | float]:
    """Show that boolean selection can avoid multiplying an existing infinity."""
    torch = _torch()
    intermediate = torch.tensor([1.5, float("inf")], device="cpu", requires_grad=True)
    selected = torch.where(
        torch.tensor([True, False], device="cpu"),
        intermediate,
        torch.zeros_like(intermediate),
    )
    loss = selected.sum()
    loss.backward()

    gradient = intermediate.grad
    assert gradient is not None
    return {
        "loss_finite": _all_finite(loss),
        "gradient_finite": _all_finite(gradient),
        "inactive_gradient": float(gradient[1]),
    }


def _adamw_skip_semantics() -> dict[str, float | int]:
    """Compare AdamW behavior for ``grad is None`` and an all-zero gradient."""
    torch = _torch()
    none_parameter = torch.nn.Parameter(torch.tensor([2.0], device="cpu"))
    zero_parameter = torch.nn.Parameter(torch.tensor([2.0], device="cpu"))
    none_optimizer = torch.optim.AdamW([none_parameter], lr=0.1, weight_decay=0.5)
    zero_optimizer = torch.optim.AdamW([zero_parameter], lr=0.1, weight_decay=0.5)
    none_parameter.grad = None
    zero_parameter.grad = torch.zeros_like(zero_parameter)
    none_optimizer.step()
    zero_optimizer.step()
    return {
        "none_grad_weight": float(none_parameter.detach()),
        "none_grad_state_entries": len(none_optimizer.state),
        "zero_grad_weight": float(zero_parameter.detach()),
        "zero_grad_state_entries": len(zero_optimizer.state),
    }


def _nanmean_visibility() -> dict[str, bool | float]:
    """Show the reporting limitation of a finite ``nanmean`` scalar."""
    torch = _torch()
    gathered = torch.tensor([float("nan"), 0.242], device="cpu")
    return {
        "input_has_nonfinite": not _all_finite(gathered),
        "nanmean": float(torch.nanmean(gathered)),
    }


def _view_backward() -> dict[str, bool]:
    """Check ordinary CPU autograd through the non-contiguous logits-style view."""
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(342)
    hidden_states = torch.randn((2, 8, 4), generator=generator, device="cpu", requires_grad=True)
    sliced_states = hidden_states[:, -3:, :]
    projection = torch.nn.Linear(4, 7, device="cpu")
    projection(sliced_states).sum().backward()

    gradient = hidden_states.grad
    assert gradient is not None
    return {
        "slice_is_contiguous": sliced_states.is_contiguous(),
        "hidden_gradient_finite": _all_finite(gradient),
    }


def run_probes() -> dict[str, Any]:
    """Run every CPU-only probe and return JSON-safe result values."""
    torch = _torch()
    return {
        "evidence_boundary": (
            "CPU arithmetic and autograd controls only; not an H100 reproduction or root-cause "
            "determination."
        ),
        "execution_device": "cpu",
        "torch_version": torch.__version__,
        "split_k_bound": _split_k_bound(),
        "dtype_ranges": _dtype_ranges(),
        "constructed_masked_exponential": _constructed_masked_exponential(),
        "default_on_policy_ratio": _default_on_policy_ratio(),
        "boolean_selection": _boolean_selection(),
        "adamw_skip_semantics": _adamw_skip_semantics(),
        "nanmean_visibility": _nanmean_visibility(),
        "view_backward": _view_backward(),
    }


def assert_expected_results(results: dict[str, Any]) -> None:
    """Fail loudly if a probe no longer demonstrates its stated narrow claim."""
    split_k = results["split_k_bound"]
    assert split_k["bound"] == 6_553_600.0
    assert split_k["orders_below_fp32_max"] > 31.0

    ranges = results["dtype_ranges"]
    assert ranges["fp16_max"] == 65_504.0
    assert ranges["bf16_max"] > 1e38
    assert 88.7 < ranges["ln_bf16_max"] < 88.8

    masked = results["constructed_masked_exponential"]
    assert masked["loss_finite"] is True
    assert masked["gradient_finite"] is False
    assert masked["nonfinite_gradient_elements"] > 0

    on_policy = results["default_on_policy_ratio"]
    assert on_policy["ratio_all_one"] is True
    assert on_policy["ratio_finite"] is True
    assert on_policy["gradient_finite"] is True

    selection = results["boolean_selection"]
    assert selection["loss_finite"] is True
    assert selection["gradient_finite"] is True
    assert selection["inactive_gradient"] == 0.0

    adamw = results["adamw_skip_semantics"]
    assert math.isclose(adamw["none_grad_weight"], 2.0, rel_tol=0.0, abs_tol=1e-6)
    assert adamw["none_grad_state_entries"] == 0
    assert math.isclose(adamw["zero_grad_weight"], 1.9, rel_tol=0.0, abs_tol=1e-6)
    assert adamw["zero_grad_state_entries"] == 1

    nanmean = results["nanmean_visibility"]
    assert nanmean["input_has_nonfinite"] is True
    assert math.isclose(nanmean["nanmean"], 0.242, rel_tol=0.0, abs_tol=1e-6)

    view = results["view_backward"]
    assert view["slice_is_contiguous"] is False
    assert view["hidden_gradient_finite"] is True


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="assert every probe result")
    parser.add_argument("--json", action="store_true", help="write the result as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the probes and return a process exit status."""
    args = _parse_args(argv)
    console = Console(file=sys.stdout, force_terminal=False, color_system=None)
    results = run_probes()
    if args.selftest:
        try:
            assert_expected_results(results)
        except AssertionError as error:
            results["selftest_passed"] = False
            results["selftest_error"] = str(error) or "probe assertion failed"
            if args.json:
                console.print_json(data=results)
            else:
                console.print("[bold red]FAIL[/bold red] GRPO numerical probe self-test")
                console.print(results["selftest_error"])
            return 1
        results["selftest_passed"] = True

    if args.json:
        console.print_json(data=results)
    else:
        status = "passed" if args.selftest else "completed"
        console.print(f"[bold green]{status.upper()}[/bold green] CPU GRPO numerical probes")
        console.print(results["evidence_boundary"])
        console.print_json(data=results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
