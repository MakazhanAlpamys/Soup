"""Setup and workspace bounds for grouped computation in the streamed expert pool."""

from __future__ import annotations

import sys
import threading
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence

_FIELD = "training.stream_experts"
_MATH_LOCK = threading.Lock()
_HYBRID_SHAPES = {"gate_up": (2048, 2048), "down": (2048, 1024)}
_CUBLAS_WORKSPACE = 4 * 2**20


def group_call_bytes(
    rows: Sequence[int], shapes: dict, *, packed: bool, itemsize: int = 2,
    quant_specs: dict | None = None,
) -> int:
    """Bound the largest projection call for a group, including packed backward."""
    def align(size: int) -> int:
        return (size + 511) // 512 * 512

    active = [count for count in rows if count]
    if not active:
        return 0
    metadata = (512 if packed else 3 * 512)
    result = 0
    for part, (columns, inner) in shapes.items():
        width = max(columns, inner) if packed else columns
        size = sum(align(count * width * itemsize) for count in active) + metadata
        if packed and shapes == _HYBRID_SHAPES and min(active) <= 128:
            size += align(columns * inner * itemsize)
            spec = (quant_specs or {}).get(part)
            if spec is not None and spec.nested:
                blocks = (columns * inner + spec.blocksize - 1) // spec.blocksize
                size += 2 * align(blocks * 4)
        result = max(result, size)
    return result


def group_activation_bytes(
    *, tokens: int, top_k: int, experts: int, hidden: int, intermediate: int, width: int,
) -> int:
    """Conservative live tensor storage, including all token routes, masks and gathers."""
    routes = tokens * top_k
    output = (tokens * hidden * 4 + 511) // 512 * 512
    routing = routes * experts * 8 + routes * 16 + experts * 32
    temporaries = routes * (3 * hidden + 4 * intermediate) * 4
    return output + routing + temporaries + width * 8192


def expert_group_width(options: Any) -> int:
    return 1 if options.backend == "sequential" else options.group_width


def expert_training_reservations(reader: Any, options: Any, *, dtype: str, device: str) -> tuple:
    """Plan host staging and the admitted weight and workspace ceilings."""
    from soup_cli.utils.expert_stream import plan_grouped_expert_buffer

    packed = options.backend == "fused_nf4"
    plan = plan_grouped_expert_buffer(
        reader, dtype=dtype, group_width=expert_group_width(options), packed_only=packed,
    )
    if plan.weight_bytes > options.weight_budget_bytes:
        raise ValueError(f"{_FIELD}.weight_budget_bytes is too small for the expert arena")
    cuda = str(device).startswith("cuda")
    arena = options.weight_budget_bytes if cuda else plan.weight_bytes
    blas = _CUBLAS_WORKSPACE if cuda and options.backend in ("grouped", "triton") else 0
    return arena + options.activation_budget_bytes + options.call_budget_bytes + blas, (
        plan.host_staging_bytes
    )


def validate_segmented_group_backend(backend: str, *, device: Any = None) -> None:
    """Segmented VJP recomputes native projections and the ordinary SwiGLU."""
    if backend != "sequential" and (
        backend != "grouped" or (device is not None and str(device).split(":", 1)[0] != "cpu")
    ):
        raise ValueError(
            "training.stream_expert_gemm='segmented' requires "
            "training.stream_experts.backend='sequential' or CPU 'grouped'"
        )


def validate_expert_training(
    options: Any, *, device: str, dtype: str, expert_gemm: str = "native",
) -> None:
    """Refuse backend/device mismatches before cache reads and allocations."""
    if expert_gemm == "segmented":
        validate_segmented_group_backend(options.backend, device=device)
    kind = str(device).split(":", 1)[0]
    if kind not in ("cpu", "cuda"):
        raise ValueError(f"{_FIELD} requires CPU or CUDA")
    if kind == "cpu":
        if options.backend in ("triton", "fused_nf4"):
            raise ValueError(f"{_FIELD}.backend={options.backend!r} requires CUDA BF16")
        return
    if options.backend == "sequential":
        return
    if dtype != "bfloat16":
        raise ValueError(f"{_FIELD}.backend={options.backend!r} requires CUDA BF16")
    import torch

    if torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError(f"{_FIELD} requires CUDA compute capability 8.0 or newer")
    if options.backend in ("grouped", "triton"):
        path = Path(options.cublas_library or "")
        if (sys.platform != "linux" or not path.is_absolute() or not path.is_file()
                or path.name not in ("libcublas.so.12", "libcublas.so.13")):
            raise ValueError(
                f"{_FIELD}.cublas_library must name an absolute Linux cuBLAS 12/13 library"
            )


def _math_policy(stack: ExitStack) -> None:
    """Own and restore the process BLAS policy for the explicit CUDA backend."""
    import torch

    from soup_cli.utils.expert_group_gemm import bf16_policy, set_bf16_policy

    if not _MATH_LOCK.acquire(blocking=False):
        raise RuntimeError("another expert runtime owns the CUDA math policy")
    stack.callback(_MATH_LOCK.release)
    cuda = torch.backends.cuda
    prior_backend = cuda.preferred_blas_library()
    prior_tf32 = cuda.matmul.allow_tf32
    prior_reduction, prior_split = bf16_policy(cuda.matmul)
    if prior_split is None:
        raise ValueError(f"{_FIELD} requires a torch build with a BF16 split-K control")
    stack.callback(set_bf16_policy, cuda.matmul, prior_reduction, prior_split)
    stack.callback(setattr, cuda.matmul, "allow_tf32", prior_tf32)
    stack.callback(cuda.preferred_blas_library, prior_backend)
    cuda.preferred_blas_library("cublaslt")
    cuda.matmul.allow_tf32 = False
    set_bf16_policy(cuda.matmul, False, False)


class ExpertGroupCompute:
    """Projection backend and budgets owned by the existing ExpertBufferPool."""

    def __init__(self, reader: Any, options: Any, *, device: str, dtype: str) -> None:
        validate_expert_training(options, device=device, dtype=dtype)
        self.options = options
        self.shapes = dict(reader.shapes)
        self.quant_specs = dict(reader.quant_specs)
        self.width = expert_group_width(options)
        self.packed = options.backend == "fused_nf4"
        self.itemsize = 4 if dtype == "float32" else 2
        self.stack = ExitStack()
        self.gemm = None
        self.reference_policy = False
        if not str(device).startswith("cuda") or options.backend == "sequential":
            return
        import torch

        try:
            if self.packed:
                from soup_cli.utils.expert_group_fused import plan_fused_group
                from soup_cli.utils.expert_group_fused_runtime import PackedGemm

                hybrid = self.shapes == _HYBRID_SHAPES
                if hybrid:
                    capability = torch.cuda.get_device_capability(device)
                    if sys.platform != "linux" or capability != (8, 6):
                        raise ValueError(f"{_FIELD}.backend='fused_nf4' on OLMoE needs Linux sm_86")
                    _math_policy(self.stack)
                    self.reference_policy = True
                elif find_spec("triton") is None:
                    raise ValueError(f"{_FIELD}.backend='fused_nf4' requires Triton")
                for shape in self.shapes.values():
                    plan_fused_group([0], shape, native_small_rows=hybrid)
                self.gemm = PackedGemm(
                    backend="hybrid_wmma" if hybrid else "triton",
                    call_budget_bytes=options.call_budget_bytes,
                )
            else:
                from soup_cli.utils.expert_group_gemm import GroupedGemm

                needs_triton = bool(self.quant_specs) or options.backend == "triton"
                if needs_triton and find_spec("triton") is None:
                    raise ValueError(f"{_FIELD} needs Triton for CUDA NF4 restore or activation")
                _math_policy(self.stack)
                self.reference_policy = True
                with torch.cuda.device(device):
                    self.gemm = GroupedGemm(
                        Path(options.cublas_library), call_budget_bytes=options.call_budget_bytes,
                        native_olmoe_rows_max=options.native_rows_max,
                        native_olmoe_rows_from=options.native_rows_from,
                        supported_shapes={(shape[1], shape[0]) for shape in self.shapes.values()},
                    )
                self.stack.callback(self.gemm.close)
        except BaseException:
            self.stack.close()
            raise

    def close(self) -> None:
        self.stack.close()

    def check_reference_policy(self) -> None:
        if self.reference_policy:
            from soup_cli.utils.expert_group_gemm import require_reference_policy

            require_reference_policy()

    def check_activations(self, *, tokens: int, top_k: int, experts: int) -> None:
        size = group_activation_bytes(
            tokens=tokens, top_k=top_k, experts=experts, hidden=self.shapes["gate_up"][1],
            intermediate=self.shapes["down"][1], width=self.width,
        )
        if size > self.options.activation_budget_bytes:
            raise ValueError(f"{_FIELD}.activation_budget_bytes exceeded before expert reads")

    def check_call(self, rows: Sequence[int]) -> None:
        if self.packed and max(rows, default=0) > 512:
            raise ValueError(f"{_FIELD}.backend='fused_nf4' supports at most 512 rows per expert")
        size = group_call_bytes(
            rows, self.shapes, packed=self.packed, itemsize=self.itemsize,
            quant_specs=self.quant_specs,
        )
        if size > self.options.call_budget_bytes:
            raise ValueError(f"{_FIELD}.call_budget_bytes exceeded before expert reads")

    def forward(self, inputs: list, pairs: list, act_fn: Any) -> list:
        import torch.nn.functional as functional

        def linear(states: list, weights: list) -> list:
            if self.gemm is None:
                return [functional.linear(x, weight) for x, weight in zip(states, weights)]
            return self.gemm.linear([state.detach().contiguous() for state in states], weights)

        first = linear(inputs, [pair[0] for pair in pairs])
        if self.options.backend == "triton":
            from soup_cli.utils.expert_group_activation import fused_swiglu

            middle = [
                fused_swiglu(matrix, output_budget_bytes=self.options.activation_budget_bytes)
                for matrix in first
            ]
        else:
            middle = [act_fn(gate) * up for gate, up in (value.chunk(2, dim=-1) for value in first)]
        del first
        return linear(middle, [pair[1] for pair in pairs])
