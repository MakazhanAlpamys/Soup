"""Grouped CUDA BF16 GEMM with a native row policy."""

from __future__ import annotations

import ctypes
import sys
import threading
from pathlib import Path
from typing import Any

SUPPORTED_SHAPES = {(32, 32), (16, 32), (2048, 2048), (1024, 2048)}


def plan_grouped_linear(
    shapes: list[tuple[int, int, int]], *, native_olmoe_rows_max: int = 16,
    native_olmoe_rows_from: int | None = None,
    supported_shapes: set[tuple[int, int]] | None = None,
) -> dict:
    """CUDA tensor-storage bound for this call, separate from persistent workspace."""
    if type(native_olmoe_rows_max) is not int or native_olmoe_rows_max not in (16, 128):
        raise ValueError("explicit OLMoE native row policy must be 16 or 128")
    if (native_olmoe_rows_from is not None
            and (type(native_olmoe_rows_from) is not int
                 or not native_olmoe_rows_max < native_olmoe_rows_from < 2**31)):
        raise ValueError("large-row native boundary must exceed the small-row cutoff")
    if not 1 <= len(shapes) <= 4:
        raise ValueError("1 through 4 matrix triples required")
    supported_shapes = SUPPORTED_SHAPES if supported_shapes is None else supported_shapes
    output = 0
    grouped = native = 0
    for rows, inner, columns in shapes:
        if (any(type(value) is not int for value in (rows, inner, columns))
                or not 0 <= rows < 2**31 or not 0 < inner < 2**31 or not 0 < columns < 2**31
                or (inner, columns) not in supported_shapes):
            raise ValueError("unsupported grouped GEMM shape")
        output += (rows * columns * 2 + 511) // 512 * 512
        cutoff = native_olmoe_rows_max if inner >= 1024 else 16
        native_row = (0 < rows <= cutoff or (inner >= 1024
                      and native_olmoe_rows_from is not None
                      and rows >= native_olmoe_rows_from))
        grouped += rows > 0 and not native_row
        native += native_row
    metadata = 3 * 512 if grouped else 0
    return {"output_bytes": output, "pointer_bytes": metadata,
            "call_bytes": output + metadata, "grouped_matrices": grouped,
            "native_matrices": native}


def require_reference_policy() -> None:
    import torch

    cuda = torch.backends.cuda
    if (str(cuda.preferred_blas_library()) != "_BlasBackend.Cublaslt"
            or bf16_policy(cuda.matmul) != (False, False)):
        raise ValueError("grouped GEMM requires caller-selected cuBLASLt and BF16 (False, False)")


def bf16_policy(matmul: Any) -> tuple[bool, bool | None]:
    reduced = matmul.allow_bf16_reduced_precision_reduction
    if isinstance(reduced, tuple):
        return reduced
    return reduced, getattr(matmul, "allow_bf16_reduced_precision_reduction_split_k", None)


def set_bf16_policy(matmul: Any, reduced: bool, split_k: bool) -> None:
    # Torch 2.11 accepts a tuple; earlier builds expose independent switches.
    try:
        matmul.allow_bf16_reduced_precision_reduction = (reduced, split_k)
    except (RuntimeError, TypeError):
        matmul.allow_bf16_reduced_precision_reduction = reduced
        matmul.allow_bf16_reduced_precision_reduction_split_k = split_k


def column_major_problem(rows: int, inner: int, columns: int) -> dict:
    """Row-major H @ W.T becomes column-major W.T @ H.T without copies."""
    if any(type(value) is not int or not 1 <= value < 2**31
           for value in (rows, inner, columns)):
        raise ValueError("nonempty 32-bit matrix dimensions required")
    return {"transa": 1, "transb": 0, "m": columns, "n": rows, "k": inner,
            "lda": inner, "ldb": inner, "ldc": columns, "group_size": 1}


class GroupedGemm:
    """Internal BF16 grouped compute for validated OLMoE geometry; forward only."""

    def __init__(
        self, library: Path, *, call_budget_bytes: int, native_olmoe_rows_max: int = 16,
        native_olmoe_rows_from: int | None = None,
        supported_shapes: set[tuple[int, int]] | None = None,
    ) -> None:
        import torch

        if type(call_budget_bytes) is not int or call_budget_bytes < 1:
            raise ValueError("positive integer call budget required")
        require_reference_policy()
        if (sys.platform != "linux" or not library.is_absolute() or not library.is_file()
                or library.name not in ("libcublas.so.12", "libcublas.so.13")):
            raise ValueError("explicit absolute Linux cuBLAS 12/13 shared library required")
        self.device = torch.device("cuda", torch.cuda.current_device())
        if (not torch.cuda.is_bf16_supported(including_emulation=False)
                or torch.cuda.get_device_capability(self.device)[0] < 8):
            raise ValueError("native BF16 sm_80+ required")
        plan_grouped_linear([(0, 32, 32)], native_olmoe_rows_max=native_olmoe_rows_max,
                            native_olmoe_rows_from=native_olmoe_rows_from)
        self.native_olmoe_rows_max = native_olmoe_rows_max
        self.native_olmoe_rows_from = native_olmoe_rows_from
        self.supported_shapes = supported_shapes
        self.call_budget_bytes = call_budget_bytes
        self.calls = self.grouped_matrices = self.native_matrices = 0
        self._poisoned = False
        self.lib = ctypes.CDLL(str(library))
        pointer, integer = ctypes.c_void_p, ctypes.c_int
        signatures = {
            "cublasCreate_v2": [ctypes.POINTER(pointer)],
            "cublasDestroy_v2": [pointer],
            "cublasSetStream_v2": [pointer, pointer],
            "cublasSetWorkspace_v2": [pointer, pointer, ctypes.c_size_t],
            "cublasGetVersion_v2": [pointer, ctypes.POINTER(integer)],
            "cublasGemmGroupedBatchedEx": [
                pointer, pointer, pointer, pointer, pointer, pointer, pointer, pointer,
                integer, pointer, pointer, integer, pointer, pointer, pointer, integer,
                pointer, integer, pointer, integer,
            ],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, integer
        self.handle = pointer()
        self.workspace = None
        self._lock = threading.Lock()
        self._check(self.lib.cublasCreate_v2(ctypes.byref(self.handle)), "create")
        try:
            version = integer()
            self._check(self.lib.cublasGetVersion_v2(self.handle, ctypes.byref(version)), "version")
            self.version = version.value
            self.workspace = torch.empty(4 * 2**20, device=self.device, dtype=torch.uint8)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int, operation: str) -> None:
        if status:
            raise RuntimeError(f"cuBLAS {operation} returned status {status}")

    def _prepare(self, inputs: list, weights: list) -> tuple[list, Any, dict]:
        import torch

        if not self.handle or not 1 <= len(inputs) <= 4 or len(inputs) != len(weights):
            raise ValueError("live handle and 1 through 4 input/weight pairs required")
        if any(hidden.ndim != 2 or weight.ndim != 2 for hidden, weight in zip(inputs, weights)):
            raise ValueError("matrices must be two-dimensional")
        plan = plan_grouped_linear([(hidden.shape[0], weight.shape[1], weight.shape[0])
                                    for hidden, weight in zip(inputs, weights)],
                                   native_olmoe_rows_max=self.native_olmoe_rows_max,
                                   native_olmoe_rows_from=self.native_olmoe_rows_from,
                                   supported_shapes=self.supported_shapes)
        if plan["call_bytes"] > self.call_budget_bytes:
            raise ValueError("grouped GEMM output/metadata exceeds call budget")
        selected = []
        singles = []
        outputs = []
        problems = []
        for hidden, weight in zip(inputs, weights):
            if (hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]
                    or hidden.dtype != torch.bfloat16 or weight.dtype != hidden.dtype
                    or hidden.device != self.device or weight.device != self.device
                    or not hidden.is_contiguous() or not weight.is_contiguous()
                    or hidden.requires_grad or weight.requires_grad
                    or hidden.data_ptr() % 16 or weight.data_ptr() % 16):
                raise ValueError("contiguous aligned non-autograd CUDA BF16 pairs required")
        for hidden, weight in zip(inputs, weights):
            output = torch.empty((hidden.shape[0], weight.shape[0]), device=self.device,
                                 dtype=torch.bfloat16)
            outputs.append(output)
            cutoff = self.native_olmoe_rows_max if weight.shape[1] >= 1024 else 16
            if (0 < hidden.shape[0] <= cutoff or (weight.shape[1] >= 1024
                    and self.native_olmoe_rows_from is not None
                    and hidden.shape[0] >= self.native_olmoe_rows_from)):
                singles.append((hidden, weight, output))
            elif hidden.shape[0]:
                problems.append(column_major_problem(hidden.shape[0], hidden.shape[1],
                                                     weight.shape[0]))
                selected.append((hidden, weight, output))
        if not selected and not singles:
            return outputs, lambda: outputs, plan
        host = {}
        for name in problems[0] if problems else ():
            host[name] = (ctypes.c_int * len(problems))(*(problem[name] for problem in problems))
        host["alpha"] = (ctypes.c_float * len(problems))(*([1.0] * len(problems)))
        host["beta"] = (ctypes.c_float * len(problems))(*([0.0] * len(problems)))
        pointers = [torch.tensor([entry[index].data_ptr() for entry in selected],
                                 device=self.device, dtype=torch.int64)
                    for index in (1, 0, 2)] if selected else []

        def launch() -> list:
            try:
                if not self.handle:
                    raise RuntimeError("grouped cuBLAS GEMM is closed")
                with torch.cuda.device(self.device):
                    stream = torch.cuda.current_stream(self.device)
                    if selected:
                        self._check(self.lib.cublasSetStream_v2(
                            self.handle, stream.cuda_stream,
                        ), "stream")
                        # Setting the stream resets cuBLAS workspace, so set ours afterwards.
                        self._check(self.lib.cublasSetWorkspace_v2(
                            self.handle, self.workspace.data_ptr(), self.workspace.numel(),
                        ), "workspace")
                        self._check(self.lib.cublasGemmGroupedBatchedEx(
                            self.handle, host["transa"], host["transb"], host["m"], host["n"],
                            host["k"], host["alpha"], pointers[0].data_ptr(), 14, host["lda"],
                            pointers[1].data_ptr(), 14, host["ldb"], host["beta"],
                            pointers[2].data_ptr(), 14, host["ldc"], len(problems),
                            host["group_size"], 68,
                        ), "grouped GEMM")
                    for hidden, weight, output in singles:
                        torch.mm(hidden, weight.T, out=output)
                return outputs
            finally:
                # Keep host descriptors, pointer arrays and selected weights/outputs
                # alive until cuBLAS has finished, including failed API calls.
                _keep_alive = selected, singles, host, pointers
                try:
                    torch.cuda.synchronize(self.device)
                except BaseException:
                    self._poisoned = True
                    raise

        return outputs, launch, plan

    def linear(self, inputs: list, weights: list) -> list:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("grouped GEMM is already active")
        try:
            if not self.handle or self._poisoned:
                raise RuntimeError("grouped GEMM is closed or completion is unknown")
            require_reference_policy()
            outputs, launch, plan = self._prepare(inputs, weights)
            launch()
            self.calls += 1
            self.grouped_matrices += plan["grouped_matrices"]
            self.native_matrices += plan["native_matrices"]
            return outputs
        finally:
            self._lock.release()

    def stats(self) -> dict:
        return {"cublas_version": self.version, "workspace_bytes": 4 * 2**20,
                "call_budget_bytes": self.call_budget_bytes, "native_rows_max": 16,
                "native_olmoe_rows_max": self.native_olmoe_rows_max,
                "native_olmoe_rows_from": self.native_olmoe_rows_from,
                "calls": self.calls, "grouped_matrices": self.grouped_matrices,
                "native_matrices": self.native_matrices, "closed": not bool(self.handle),
                "poisoned": self._poisoned,
                "scope": "workspace/output bound; CUDA/cuBLAS contexts and weights separate"}

    def close(self) -> None:
        import torch

        if not self._lock.acquire(blocking=False):
            raise RuntimeError("cannot close an active grouped cuBLAS GEMM")
        try:
            torch.cuda.synchronize(self.device)
            if self.handle:
                self._check(self.lib.cublasDestroy_v2(self.handle), "destroy")
                self.handle = ctypes.c_void_p()
            self.workspace = None
        finally:
            self._lock.release()
