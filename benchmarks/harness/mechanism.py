#!/usr/bin/env python3
"""Reproduce the NF4 pooled-buffer aliasing mechanism.

This is the standalone mechanism harness described by issue #379 and the
#331 measurement record.

Protocol
--------
A synthetic Llama-shaped NF4 workload is built with:

- 32 layers
- hidden size 5120
- intermediate size 27648
- sequence length 128
- two recycled GPU buffer slots

Each decoder layer exercises the seven large linear shapes present in the
32B-sized configuration. Two LoRA-style trainable projections provide four
gradient tensors per layer.

The five arms are:

- control: pooled packed weights + pooled quantization state
- sync: control + torch.cuda.synchronize()
- clone: private copies of packed weights AND quantization state
- clone_quantstate: private quantization state, pooled packed weights
- clone_packed: private packed weights, pooled quantization state

The ``clone`` arm is the comparison baseline within this harness; it is not
an independent resident NF4 reference.

Expected result
---------------
The mechanism is an aliasing problem, not a missing CUDA synchronization:

- control -> mismatch
- sync -> mismatch
- clone -> exact
- clone_quantstate -> mismatch
- clone_packed -> mismatch

The harness also has a mutation mode. ``--bypass-pool`` disables recycled
buffers for the control arm. The mechanism must then disappear; because that
means the harness was mutated away from the mechanism it claims to reproduce,
the command exits non-zero.

No model download is required.

Requirements
------------
- CUDA-capable GPU
- PyTorch
- bitsandbytes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import bitsandbytes as bnb
import torch
import torch.nn.functional as functional
from bitsandbytes.functional import QuantState
from torch.utils.checkpoint import checkpoint

DEVICE = "cuda"
DTYPE = torch.bfloat16

N_LAYERS = 32
HIDDEN = 5120
KV_DIM = 1024
INTERMEDIATE = 27648
TOKENS = 128
POOL_SLOTS = 2
LORA_RANK = 16
CORRECTNESS_REPEATS = 3

SEED_WEIGHTS = 11
SEED_INPUT = 23
SEED_LORA = 29

PROJECTIONS = (
    ("q", HIDDEN, HIDDEN),
    ("k", KV_DIM, HIDDEN),
    ("v", KV_DIM, HIDDEN),
    ("o", HIDDEN, HIDDEN),
    ("gate", INTERMEDIATE, HIDDEN),
    ("up", INTERMEDIATE, HIDDEN),
    ("down", HIDDEN, HIDDEN),
)


@dataclass
class QuantBundle:
    """Packed NF4 weight plus its quantization state."""

    packed: torch.Tensor
    state: QuantState


@dataclass
class ArmState:
    """Persistent model parameters and pooled buffers for one arm."""

    parameters: list[dict[str, torch.Tensor]]
    pools: dict[str, "BundlePool"]


class BundlePool:
    """Round-robin pooled storage for packed weights and quantization state."""

    def __init__(self, n_slots: int, source: QuantBundle):
        self.packed = [
            torch.empty_like(source.packed)
            for _ in range(n_slots)
        ]

        self.absmax = [
            torch.empty_like(source.state.absmax)
            for _ in range(n_slots)
        ]

        self.offset = (
            [
                torch.empty_like(source.state.offset)
                for _ in range(n_slots)
            ]
            if source.state.nested
            else None
        )

        self.state2_absmax = (
            [
                torch.empty_like(source.state.state2.absmax)
                for _ in range(n_slots)
            ]
            if source.state.nested
            else None
        )

        self.cursor = 0
        self.template = source.state

    def acquire(self, source: QuantBundle) -> QuantBundle:
        slot = self.cursor % len(self.packed)
        self.cursor += 1

        packed = self.packed[slot]
        packed.copy_(source.packed)

        absmax = self.absmax[slot]
        absmax.copy_(source.state.absmax)

        offset = None
        state2 = None

        if source.state.nested:
            assert self.offset is not None
            assert self.state2_absmax is not None

            offset = self.offset[slot]
            offset.copy_(source.state.offset)

            nested_absmax = self.state2_absmax[slot]
            nested_absmax.copy_(source.state.state2.absmax)

            state2 = QuantState(
                absmax=nested_absmax,
                shape=source.state.state2.shape,
                code=source.state.state2.code,
                blocksize=source.state.state2.blocksize,
                quant_type=source.state.state2.quant_type,
                dtype=source.state.state2.dtype,
            )

        state = QuantState(
            absmax=absmax,
            shape=source.state.shape,
            code=source.state.code,
            blocksize=source.state.blocksize,
            quant_type=source.state.quant_type,
            dtype=source.state.dtype,
            offset=offset,
            state2=state2,
        )

        return QuantBundle(packed=packed, state=state)


def clone_quant_state(state: QuantState) -> QuantState:
    """Deep-copy tensor components of a QuantState."""

    state2 = None

    if state.nested:
        state2 = QuantState(
            absmax=state.state2.absmax.clone(),
            shape=state.state2.shape,
            code=state.state2.code.clone(),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )

    offset = state.offset.clone() if state.offset is not None else None

    return QuantState(
        absmax=state.absmax.clone(),
        shape=state.shape,
        code=state.code.clone(),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=offset,
        state2=state2,
    )


def load_sources() -> list[dict[str, QuantBundle]]:
    """Build the complete synthetic 32B-shaped NF4 source set."""

    generator = torch.Generator(device=DEVICE).manual_seed(SEED_WEIGHTS)

    sources: list[dict[str, QuantBundle]] = []

    for _ in range(N_LAYERS):
        layer: dict[str, QuantBundle] = {}

        for name, out_features, in_features in PROJECTIONS:
            weight = torch.randn(
                (out_features, in_features),
                generator=generator,
                device=DEVICE,
                dtype=DTYPE,
            ) / 32

            packed, state = bnb.functional.quantize_4bit(
                weight,
                blocksize=64,
                quant_type="nf4",
                compress_statistics=True,
            )

            layer[name] = QuantBundle(
                packed=packed,
                state=state,
            )

            del weight

        sources.append(layer)

    return sources


def make_lora_parameters() -> list[dict[str, torch.Tensor]]:
    """Create identical LoRA-style tensors for every arm."""

    generator = torch.Generator(device=DEVICE).manual_seed(SEED_LORA)

    parameters: list[dict[str, torch.Tensor]] = []

    for _ in range(N_LAYERS):
        layer = {}

        for name, out_features, _in_features in (
            ("q", HIDDEN, HIDDEN),
            ("v", KV_DIM, HIDDEN),
        ):
            a = (
                torch.randn(
                    (LORA_RANK, HIDDEN),
                    generator=generator,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                * 0.02
            )

            b = (
                torch.randn(
                    (out_features, LORA_RANK),
                    generator=generator,
                    device=DEVICE,
                    dtype=DTYPE,
                )
                * 0.02
            )

            layer[f"{name}.A"] = a.requires_grad_(True)
            layer[f"{name}.B"] = b.requires_grad_(True)

        parameters.append(layer)

    return parameters


def project(
    x: torch.Tensor,
    bundle: QuantBundle,
) -> torch.Tensor:
    """Run one NF4 linear using the packed weight and quantization state."""

    return bnb.matmul_4bit(
        x,
        bundle.packed,
        quant_state=bundle.state,
    )


def acquire_bundle(
    pools: dict[str, BundlePool],
    sources: dict[str, QuantBundle],
    name: str,
    arm: str,
) -> QuantBundle:
    """Acquire one layer weight according to the selected mechanism arm."""

    pooled = pools[name].acquire(sources[name])

    if arm == "control" or arm == "sync":
        return pooled

    if arm == "clone":
        return QuantBundle(
            packed=pooled.packed.clone(),
            state=clone_quant_state(pooled.state),
        )

    if arm == "clone_quantstate":
        return QuantBundle(
            packed=pooled.packed,
            state=clone_quant_state(pooled.state),
        )

    if arm == "clone_packed":
        return QuantBundle(
            packed=pooled.packed.clone(),
            state=pooled.state,
        )

    raise ValueError(f"unsupported arm: {arm!r}")


def make_arm_state(
    sources: list[dict[str, QuantBundle]],
    initial_parameters: list[dict[str, torch.Tensor]],
) -> ArmState:
    """Create one persistent parameter/pool state for a correctness arm."""

    parameters = [
        {
            name: tensor.detach().clone().requires_grad_(True)
            for name, tensor in layer.items()
        }
        for layer in initial_parameters
    ]

    pools = {
        name: BundlePool(
            POOL_SLOTS,
            sources[0][name],
        )
        for name, _, _ in PROJECTIONS
    }

    return ArmState(parameters=parameters, pools=pools)


def run_once(
    sources: list[dict[str, QuantBundle]],
    state: ArmState,
    arm: str,
    *,
    bypass_pool: bool,
) -> dict[str, torch.Tensor]:
    """Run one backward pass using persistent arm state."""

    torch.manual_seed(SEED_INPUT)

    parameters = state.parameters
    pools = state.pools

    for layer in parameters:
        for parameter in layer.values():
            parameter.grad = None

    generator = torch.Generator(device=DEVICE).manual_seed(SEED_INPUT)

    x = torch.randn(
        (TOKENS, HIDDEN),
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
        requires_grad=True,
    )

    for layer_index in range(N_LAYERS):

        q_a = parameters[layer_index]["q.A"]
        q_b = parameters[layer_index]["q.B"]
        v_a = parameters[layer_index]["v.A"]
        v_b = parameters[layer_index]["v.B"]

        def body(hidden, q_a, q_b, v_a, v_b):
            source = sources[layer_index]

            bundles = {}

            for name, _out_features, _in_features in PROJECTIONS:
                if bypass_pool:
                    bundles[name] = source[name]
                else:
                    bundles[name] = acquire_bundle(
                        pools,
                        source,
                        name,
                        arm,
                    )

            if arm == "sync":
                torch.cuda.synchronize()

            q = project(hidden, bundles["q"])
            k = project(hidden, bundles["k"])
            v = project(hidden, bundles["v"])
            o = project(hidden, bundles["o"])
            gate = project(hidden, bundles["gate"])
            up = project(hidden, bundles["up"])
            down = project(hidden, bundles["down"])

            q_lora = functional.linear(
                functional.linear(hidden, q_a),
                q_b,
            )

            v_lora = functional.linear(
                functional.linear(hidden, v_a),
                v_b,
            )

            output = (
                q
                + o
                + down
                + 0.1 * q_lora
                + 0.1 * functional.pad(
                    v_lora,
                    (0, HIDDEN - KV_DIM),
                )
                + 0.0001 * k.mean(dim=-1, keepdim=True)
                + 0.0001 * v.mean(dim=-1, keepdim=True)
                + 0.000001 * gate.mean(dim=-1, keepdim=True)
                + 0.000001 * up.mean(dim=-1, keepdim=True)
            )

            return output

        x = checkpoint(
            body,
            x,
            q_a,
            q_b,
            v_a,
            v_b,
            use_reentrant=False,
        )

    loss = x.float().pow(2).mean()
    loss.backward()

    gradients = {}

    for layer_index, layer in enumerate(parameters):
        for name, parameter in layer.items():
            gradient = parameter.grad

            if gradient is None:
                raise RuntimeError(
                    f"missing gradient for layer {layer_index} {name}"
                )

            gradients[f"{layer_index}.{name}"] = (
                gradient.detach().float().clone()
            )

    return gradients


def _assert_finite_gradients(
    gradients: dict[str, torch.Tensor],
    label: str,
) -> None:
    """Reject NaN/Inf gradients before exactness checks."""

    for name, gradient in gradients.items():
        if not torch.isfinite(gradient).all():
            raise RuntimeError(
                f"{label} contains non-finite gradient values: {name}"
            )


def max_gradient_diff(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> float:
    """Largest absolute gradient difference."""

    if set(left) != set(right):
        raise RuntimeError("gradient key sets differ")

    _assert_finite_gradients(left, "left gradients")
    _assert_finite_gradients(right, "right gradients")

    return max(
        (left[name] - right[name]).abs().max().item()
        for name in left
    )


def exact_gradient_count(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> tuple[int, int]:
    """Return exact gradient-tensor count and total."""

    if set(reference) != set(candidate):
        raise RuntimeError("gradient key sets differ")

    exact = sum(
        torch.equal(reference[name], candidate[name])
        for name in reference
    )

    return exact, len(reference)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the NF4 pooled-buffer aliasing mechanism "
            "from Soup's #331/#379 measurements."
        )
    )

    parser.add_argument(
        "--bypass-pool",
        action="store_true",
        help=(
            "mutation/negative control: bypass recycled buffers for the "
            "control path. The expected control mismatch must disappear; "
            "the harness then exits non-zero."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not torch.cuda.is_available():
        print("SKIP: CUDA is required for mechanism.py")
        return 0

    print(f"torch         {torch.__version__}")
    print(f"bitsandbytes  {bnb.__version__}")
    print(f"gpu           {torch.cuda.get_device_name(0)}")
    print(
        f"shape         {N_LAYERS} layers, hidden={HIDDEN}, "
        f"intermediate={INTERMEDIATE}, dtype={DTYPE}"
    )
    print(
        f"pool          {POOL_SLOTS} recycled slots, "
        f"sequence={TOKENS}"
    )
    print("quant         NF4")
    print()

    if args.bypass_pool:
        print("mode          NEGATIVE CONTROL: pooled reuse bypassed")
        print()

    sources = load_sources()
    initial_parameters = make_lora_parameters()

    print("built         synthetic NF4 source set")

    arm_names = (
        "control",
        "sync",
        "clone_quantstate",
        "clone_packed",
    )
    all_arms = (
        "control",
        "sync",
        "clone",
        "clone_quantstate",
        "clone_packed",
    )
    arm_states = {
        name: make_arm_state(sources, initial_parameters)
        for name in all_arms
    }

    exact_results = {name: [] for name in all_arms}
    diff_results = {name: [] for name in all_arms}

    for repetition in range(CORRECTNESS_REPEATS):
        print(
            f"correctness  {repetition + 1}/{CORRECTNESS_REPEATS}"
        )

        print("running       clone")
        clone = run_once(
            sources,
            arm_states["clone"],
            "clone",
            bypass_pool=False,
        )

        clone_reference = {
            name: gradient.clone()
            for name, gradient in clone.items()
        }
        clone_exact, clone_total = exact_gradient_count(
            clone_reference,
            clone,
        )
        clone_diff = max_gradient_diff(clone_reference, clone)
        exact_results["clone"].append(
            f"{clone_exact}/{clone_total}"
        )
        diff_results["clone"].append(clone_diff)

        for arm in arm_names:
            print(f"running       {arm}")
            candidate = run_once(
                sources,
                arm_states[arm],
                arm,
                bypass_pool=args.bypass_pool if arm == "control" else False,
            )

            exact, total = exact_gradient_count(clone, candidate)
            diff = max_gradient_diff(clone, candidate)

            exact_results[arm].append(f"{exact}/{total}")
            diff_results[arm].append(diff)

            del candidate

        del clone

    print()
    print(f"{'arm':>18}  {'exact':>28}  {'max_abs_diff':>45}")

    for name in all_arms:
        print(
            f"{name:>18}  "
            f"{', '.join(exact_results[name]):>28}  "
            f"{', '.join(f'{diff:.6e}' for diff in diff_results[name]):>45}"
        )

    print()

    if args.bypass_pool:
        if any(diff != 0.0 for diff in diff_results["control"]):
            print(
                "ERROR: bypassing the pool did not remove the "
                "control mismatch."
            )
            print("RESULT: negative control was NOT caught.")
            return 2

        print(
            "NEGATIVE CONTROL: bypassing pool reuse removed "
            "the control mismatch."
        )
        print("RESULT: mutation detected; exiting non-zero.")
        return 1

    if all(diff == 0.0 for diff in diff_results["control"]):
        print(
            "ERROR: control did not reproduce the NF4 "
            "pooled-buffer mismatch."
        )
        return 1

    if diff_results["control"][-1] == 0.0:
        print(
            "ERROR: control ended exact; expected the historical "
            "stale-buffer mismatch after repeated backwards."
        )
        return 1

    if any(diff == 0.0 for diff in diff_results["sync"]):
        print(
            "ERROR: synchronize() removed the mismatch; "
            "the aliasing mechanism was not reproduced."
        )
        return 1

    if any(diff == 0.0 for diff in diff_results["clone_quantstate"]):
        print(
            "ERROR: cloning quantization state alone removed "
            "the mismatch."
        )
        return 1

    if any(diff == 0.0 for diff in diff_results["clone_packed"]):
        print(
            "ERROR: cloning packed weights alone removed "
            "the mismatch."
        )
        return 1

    if any(diff != 0.0 for diff in diff_results["clone"]):
        print("ERROR: clone reference is not self-consistent.")
        return 1

    print(
        "RESULT: mechanism reproduced — control and sync mismatch, "
        "single-sided clones mismatch, full clone matches."
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
