"""Opt-in, bounded crash capture for Soup issue #342.

This module is deliberately outside ``src/``: importing or using Soup normally
does not install hooks, synchronize CUDA, retain training rows, or write files.
Install it only after Soup's ``GRPOTrainerWrapper.setup()`` has initialized its
inner ``GRPOTrainer``, for example::

    from benchmarks.harness.grpo_crash_capsule import Issue342CrashCapsule

    capsule = Issue342CrashCapsule(
        wrapper.trainer,
        output_dir="/raid/issue342-capsule-001",
        max_total_bytes=1 << 30,
        acknowledge_training_rows=True,
    ).install()
    with capsule.sdpa_backend("baseline"):
        wrapper.train()

Use one fresh process per SDPA arm.  The capture changes timing and is forensic
instrumentation, not a performance benchmark or a root-cause determination.
The ``.pt`` payload contains tokenized training rows, RNG state, and trainable
weights.  Do not share it; load only a capsule you created, using
``torch.load(path, weights_only=True)``.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import platform
import random
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DEFAULT_MAX_TOTAL_BYTES = 2 << 30
_SCHEMA_VERSION = 1


class CapsuleError(RuntimeError):
    """Raised when an explicitly requested capsule cannot be captured safely."""


class CapsuleSizeLimitError(CapsuleError):
    """Raised when a capture would exceed the caller-selected disk cap."""


def _torch() -> Any:
    """Import Torch lazily so normal benchmark discovery has no CUDA side effect."""
    import torch

    return torch


def _validated_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _rank_and_world(torch: Any) -> tuple[int, int]:
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_available() and distributed.is_initialized():
        return int(distributed.get_rank()), int(distributed.get_world_size())
    return 0, 1


def _distributed_api(torch: Any, world_size: int) -> Any | None:
    if world_size == 1:
        return None
    distributed = getattr(torch, "distributed", None)
    if distributed is None or not distributed.is_available() or not distributed.is_initialized():
        raise CapsuleError("world_size > 1 requires initialized torch.distributed")
    if not callable(getattr(distributed, "broadcast_object_list", None)):
        raise CapsuleError("world_size > 1 requires torch.distributed.broadcast_object_list")
    return distributed


def _safe_filename(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise CapsuleError("capsule filename must be a non-empty string")
    candidate = Path(name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or name in {".", ".."}:
        raise CapsuleError(f"capsule filename must be one relative component, got {name!r}")
    return name


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


class CapsuleWriter:
    """Create one new rank directory and write bounded, never-overwritten files."""

    def __init__(self, directory: Path, max_bytes: int) -> None:
        self.directory = directory
        self.max_bytes = _validated_positive_int(max_bytes, "max_bytes")
        self.bytes_written = 0

    @classmethod
    def create(
        cls,
        output_dir: str | os.PathLike[str],
        *,
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
        rank: int = 0,
        world_size: int = 1,
        torch_module: Any | None = None,
    ) -> "CapsuleWriter":
        """Create a new user-named root and a per-rank child without overwriting.

        For multi-rank runs rank zero creates the root, synchronizes, then each
        rank creates its own child.  Dividing the total cap across ranks makes
        the aggregate output bounded even when every rank sees the anomaly.
        """
        total = _validated_positive_int(max_total_bytes, "max_total_bytes")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        world_size = _validated_positive_int(world_size, "world_size")
        if rank >= world_size:
            raise CapsuleError(f"rank {rank} is outside world_size {world_size}")
        if total < world_size:
            raise ValueError(
                "max_total_bytes must be at least world_size so every rank can have a "
                "non-zero bounded share"
            )
        root = Path(output_dir).expanduser()

        def create_root() -> None:
            if root.exists() or root.is_symlink():
                raise FileExistsError(
                    f"refusing to overwrite capsule directory: {root}. Choose a new path."
                )
            if not root.parent.is_dir():
                raise FileNotFoundError(
                    f"capsule parent must already exist: {root.parent}. Create it explicitly first."
                )
            root.mkdir(mode=0o700)

        if world_size == 1:
            create_root()
        else:
            if torch_module is None:
                raise CapsuleError("multi-rank capture requires torch_module for synchronization")
            distributed = _distributed_api(torch_module, world_size)
            creation_error: str | None = None
            if rank == 0:
                try:
                    create_root()
                except OSError as exc:
                    creation_error = f"{type(exc).__name__}: {exc}"
            status = [creation_error]
            distributed.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise CapsuleError(f"rank zero could not create capsule root: {status[0]}")
            distributed.barrier()
        if not root.is_dir() or root.is_symlink():
            raise CapsuleError(f"capsule root was not safely created: {root}")
        rank_dir = root / f"rank{rank:03d}"
        try:
            rank_dir.mkdir(mode=0o700)
        except FileExistsError as exc:
            raise CapsuleError(f"refusing to overwrite capsule rank directory: {rank_dir}") from exc
        # Each rank gets the same floor share.  Consequently every rank can
        # fill its own cap without the aggregate exceeding max_total_bytes.
        return cls(rank_dir, total // world_size)

    def _write(self, name: str, data: bytes) -> Path:
        name = _safe_filename(name)
        if self.bytes_written + len(data) > self.max_bytes:
            raise CapsuleSizeLimitError(
                f"capsule rank cap {self.max_bytes} bytes would be exceeded by {name} "
                f"({self.bytes_written + len(data)} bytes)"
            )
        target = self.directory / name
        try:
            with target.open("xb") as handle:
                handle.write(data)
        except FileExistsError as exc:
            raise CapsuleError(f"refusing to overwrite capsule file: {target}") from exc
        self.bytes_written += len(data)
        return target

    def write_json(self, name: str, payload: Mapping[str, Any]) -> Path:
        encoded = (
            json.dumps(dict(payload), sort_keys=True, indent=2, default=_json_default).encode(
                "utf-8"
            )
            + b"\n"
        )
        return self._write(name, encoded)

    def write_torch(self, name: str, payload: Mapping[str, Any], torch_module: Any) -> Path:
        buffer = io.BytesIO()
        torch_module.save(dict(payload), buffer)
        return self._write(name, buffer.getvalue())


def _tensor_stats(tensor: Any, torch: Any) -> dict[str, Any]:
    detached = tensor.detach()
    finite = bool(torch.isfinite(detached).all().item())
    result: dict[str, Any] = {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "numel": int(detached.numel()),
        "finite": finite,
    }
    if detached.numel() and finite:
        result["abs_max"] = float(detached.abs().max().item())
    return result


def _clone_tensor_tree(value: Any, torch: Any) -> Any:
    """Make a weights-only-loadable CPU copy of tensor-only loss inputs."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, Mapping):
        return {str(key): _clone_tensor_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clone_tensor_tree(item, torch) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _loss_inputs(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = kwargs.get("inputs")
    if isinstance(candidate, Mapping):
        return candidate
    for value in args:
        if isinstance(value, Mapping):
            return value
    return {}


def _loss_tensor(result: Any, torch: Any) -> Any | None:
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (tuple, list)) and result and isinstance(result[0], torch.Tensor):
        return result[0]
    return None


def _random_state_json() -> Any:
    """Serialize Python's RNG state without putting arbitrary objects in a pickle."""

    def convert(value: Any) -> Any:
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return repr(value)

    return convert(random.getstate())


def _callback(owner: "Issue342CrashCapsule") -> Any:
    """Build lazily so CPU writer tests do not require Transformers."""
    try:
        import transformers
    except ImportError:
        callback_base = object
    else:
        callback_base = transformers.TrainerCallback

    class CaptureCallback(callback_base):
        def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, kwargs
            owner.capture_pre_optimizer(state)
            return control

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            del args, state, kwargs
            owner.finish_step()
            return control

    return CaptureCallback()


@contextlib.contextmanager
def sdpa_backend(mode: str) -> Iterator[None]:
    """Select an SDPA family; unsupported selection is evidence, not a fallback."""
    names = {
        "math": ("MATH",),
        "mem_efficient": ("EFFICIENT_ATTENTION",),
        "flash": ("FLASH_ATTENTION",),
    }
    if mode == "baseline":
        # Do not rewrite global SDPA switches for the baseline arm.  This
        # preserves the process's native policy and avoids calling a missing
        # backend a successful forced selection.
        yield
        return
    if mode not in names:
        raise ValueError(f"unknown SDPA mode {mode!r}; choose {sorted((*names, 'baseline'))}")
    torch = _torch()
    attention = getattr(getattr(torch, "nn", None), "attention", None)
    current = getattr(attention, "sdpa_kernel", None)
    backend_type = getattr(attention, "SDPBackend", None)
    if callable(current) and backend_type is not None:
        missing = [name for name in names[mode] if not hasattr(backend_type, name)]
        if missing:
            raise RuntimeError(
                "torch.nn.attention does not expose requested SDPA backend(s): "
                + ", ".join(missing)
            )
        backends = [getattr(backend_type, name) for name in names[mode]]
        with current(backends):
            yield
        return
    legacy = getattr(getattr(getattr(torch, "backends", None), "cuda", None), "sdp_kernel", None)
    if not callable(legacy):
        raise RuntimeError(
            "this Torch build exposes neither torch.nn.attention.sdpa_kernel nor cuda.sdp_kernel"
        )
    enabled = set(names[mode])
    kwargs = {
        "enable_flash": "FLASH_ATTENTION" in enabled,
        "enable_math": "MATH" in enabled,
        "enable_mem_efficient": "EFFICIENT_ATTENTION" in enabled,
        "enable_cudnn": "CUDNN_ATTENTION" in enabled,
    }
    try:
        parameters = inspect.signature(legacy).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    accepts_cudnn = accepts_keywords or any(
        parameter.name == "enable_cudnn" for parameter in parameters
    )
    if not accepts_cudnn:
        kwargs.pop("enable_cudnn")
    with legacy(**kwargs):
        yield


class Issue342CrashCapsule:
    """Rank-local capture around a stock TRL GRPO trainer, installed explicitly."""

    def __init__(
        self,
        trainer: Any,
        *,
        output_dir: str | os.PathLike[str],
        max_total_bytes: int = _DEFAULT_MAX_TOTAL_BYTES,
        acknowledge_training_rows: bool = False,
        instrument_lora_operands: bool = True,
    ) -> None:
        if not acknowledge_training_rows:
            raise CapsuleError(
                "set acknowledge_training_rows=True: a capsule contains tokenized training rows, "
                "RNG state, and trainable weights"
            )
        self.torch = _torch()
        self.trainer = trainer
        self.model = trainer.model
        self.rank, self.world_size = _rank_and_world(self.torch)
        self.writer = CapsuleWriter.create(
            output_dir,
            max_total_bytes=max_total_bytes,
            rank=self.rank,
            world_size=self.world_size,
            torch_module=self.torch,
        )
        self.instrument_lora_operands = bool(instrument_lora_operands)
        self._current: dict[str, Any] | None = None
        self._serial = 0
        self._bad_seen = False
        self._flushed = False
        self._installed = False
        self._parameter_handles: list[Any] = []
        self._module_handles: list[Any] = []
        self._original_loss: Callable[..., Any] | None = None
        self._loss_method_name: str | None = None
        self._capture_callback: Any | None = None
        self._closed = False
        self._sdpa_mode: str | None = None
        self._lock = threading.Lock()

    def _environment(self) -> dict[str, Any]:
        cuda: dict[str, Any] = {"available": bool(self.torch.cuda.is_available())}
        if cuda["available"]:
            index = self.torch.cuda.current_device()
            device = self.torch.cuda.get_device_properties(index)
            cuda.update(
                {
                    "device_index": index,
                    "name": device.name,
                    "capability": list(self.torch.cuda.get_device_capability(index)),
                    "total_memory": int(device.total_memory),
                }
            )
        cuda_backend = getattr(getattr(self.torch, "backends", None), "cuda", None)
        cudnn_backend = getattr(getattr(self.torch, "backends", None), "cudnn", None)
        sdp_enabled = {
            name: bool(getattr(cuda_backend, name)())
            for name in (
                "flash_sdp_enabled",
                "mem_efficient_sdp_enabled",
                "math_sdp_enabled",
                "cudnn_sdp_enabled",
            )
            if callable(getattr(cuda_backend, name, None))
        }
        matmul = getattr(cuda_backend, "matmul", None)
        return {
            "schema_version": _SCHEMA_VERSION,
            "captured_at_ns": time.time_ns(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": str(self.torch.__version__),
            "torch_cuda": str(getattr(self.torch.version, "cuda", None)),
            "cuda": cuda,
            "requested_sdpa_mode": self._sdpa_mode,
            "sdp_enabled": sdp_enabled,
            "deterministic_algorithms": bool(self.torch.are_deterministic_algorithms_enabled()),
            "cudnn_deterministic": getattr(cudnn_backend, "deterministic", None),
            "cudnn_benchmark": getattr(cudnn_backend, "benchmark", None),
            "cudnn_version": (
                cudnn_backend.version()
                if callable(getattr(cudnn_backend, "version", None))
                else None
            ),
            "allow_tf32_matmul": getattr(matmul, "allow_tf32", None),
            "allow_tf32_cudnn": getattr(cudnn_backend, "allow_tf32", None),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "packages": {
                package: _package_version(package)
                for package in ("transformers", "trl", "peft", "accelerate")
            },
            "trainer_class": type(self.trainer).__qualname__,
            "model_class": type(self.model).__qualname__,
            "rank": self.rank,
            "world_size": self.world_size,
            "rank_cap_bytes": self.writer.max_bytes,
        }

    def _synchronize(self, boundary: str) -> None:
        """Surface asynchronous device errors at a named forensic boundary."""
        if not self.torch.cuda.is_available():
            return
        try:
            self.torch.cuda.synchronize()
        except Exception as exc:
            try:
                self.writer.write_json(
                    f"cuda_error_{self._serial:06d}_{boundary}.json",
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "reason": "cuda_synchronize_failure",
                        "boundary": boundary,
                        "error": repr(exc),
                        "rank": self.rank,
                        "world_size": self.world_size,
                    },
                )
            except CapsuleError:
                pass
            raise

    def _rng_state(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "torch_cpu": self.torch.get_rng_state().detach().to(device="cpu", copy=True),
            "python_random": _random_state_json(),
        }
        if self.torch.cuda.is_available():
            payload["torch_cuda"] = [
                state.detach().to(device="cpu", copy=True)
                for state in self.torch.cuda.get_rng_state_all()
            ]
        return payload

    def _trainable_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                state[name] = parameter.detach().to(device="cpu", copy=True)
        return state

    def _trainable_state_nbytes(self) -> int:
        """Return the raw byte count of the pre-step trainable-state snapshot."""
        return sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    def _start_forward(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        if self._current is None:
            self._synchronize("before_loss")
            self._serial += 1
            state = getattr(self.trainer, "state", None)
            self._current = {
                "serial": self._serial,
                "global_step": getattr(state, "global_step", None),
                "environment": self._environment(),
                "rng": self._rng_state(),
                "pre_step_trainable": self._trainable_state(),
                "microbatches": [],
                "first_nonfinite_parameter": None,
                "nonfinite_parameter_names_before_clip": [],
                "first_nonfinite_scalar_grad": None,
                "first_nonfinite_activation": None,
                "lora_observations": {},
            }
        copied_inputs = _clone_tensor_tree(_loss_inputs(args, kwargs), self.torch)
        self._current["microbatches"].append(copied_inputs)

    def _finish_forward(self, result: Any) -> None:
        loss = _loss_tensor(result, self.torch)
        if loss is None or self._current is None:
            return
        self._synchronize("after_loss")
        self._current["loss"] = _tensor_stats(loss, self.torch)
        if not self._current["loss"]["finite"]:
            self._bad_seen = True
        if loss.requires_grad:

            def record_scalar_grad(gradient: Any) -> Any:
                if not bool(self.torch.isfinite(gradient).all().item()):
                    with self._lock:
                        if self._current is not None and (
                            self._current["first_nonfinite_scalar_grad"] is None
                        ):
                            self._current["first_nonfinite_scalar_grad"] = _tensor_stats(
                                gradient, self.torch
                            )
                            self._bad_seen = True
                return gradient

            loss.register_hook(record_scalar_grad)

    def _parameter_hook(self, name: str) -> Callable[[Any], Any]:
        def record(gradient: Any) -> Any:
            if not bool(self.torch.isfinite(gradient).all().item()):
                with self._lock:
                    if self._current is not None and (
                        self._current["first_nonfinite_parameter"] is None
                    ):
                        self._current["first_nonfinite_parameter"] = {
                            "name": name,
                            "stats": _tensor_stats(gradient, self.torch),
                        }
                    if (
                        self._current is not None
                        and name not in self._current["nonfinite_parameter_names_before_clip"]
                    ):
                        self._current["nonfinite_parameter_names_before_clip"].append(name)
                    self._bad_seen = True
            return gradient

        return record

    def _lora_forward_hook(self, name: str) -> Callable[..., None]:
        def record(module: Any, args: tuple[Any, ...]) -> None:
            del module
            if self._current is None or not args or not isinstance(args[0], self.torch.Tensor):
                return
            observations = self._current["lora_observations"].setdefault(name, [])
            stats = _tensor_stats(args[0], self.torch)
            observations.append({"input": stats})
            if not stats["finite"] and self._current["first_nonfinite_activation"] is None:
                self._current["first_nonfinite_activation"] = {
                    "module": name,
                    "stage": "input",
                    "stats": stats,
                }
                self._bad_seen = True

        return record

    def _lora_backward_hook(self, name: str) -> Callable[..., None]:
        def record(module: Any, grad_input: Any, grad_output: Any) -> None:
            del module, grad_input
            if (
                self._current is None
                or not grad_output
                or not isinstance(grad_output[0], self.torch.Tensor)
            ):
                return
            observations = self._current["lora_observations"].setdefault(name, [])
            stats = _tensor_stats(grad_output[0], self.torch)
            if not stats["finite"] and self._current["first_nonfinite_activation"] is None:
                self._current["first_nonfinite_activation"] = {
                    "module": name,
                    "stage": "grad_output",
                    "stats": stats,
                }
                self._bad_seen = True
            if observations:
                observations[-1]["grad_output"] = stats
                input_stats = observations[-1].get("input", {})
                if input_stats.get("finite") and stats.get("finite"):
                    last_dim = grad_output[0].shape[-1] if grad_output[0].ndim else 1
                    rows = max(1, int(grad_output[0].numel() // max(1, last_dim)))
                    observations[-1]["fp32_dot_abs_bound"] = (
                        rows * float(input_stats["abs_max"]) * float(stats["abs_max"])
                    )
            else:
                observations.append({"grad_output": stats})

        return record

    def _wrap_loss(self) -> None:
        if callable(getattr(self.trainer, "_compute_loss", None)):
            method_name = "_compute_loss"
        else:
            method_name = "compute_loss"
        original = getattr(self.trainer, method_name, None)
        if not callable(original):
            raise CapsuleError("trainer exposes neither _compute_loss nor compute_loss")
        self._original_loss = original
        self._loss_method_name = method_name

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not (getattr(self.model, "training", False) and self.torch.is_grad_enabled()):
                return original(*args, **kwargs)
            self._start_forward(args, kwargs)
            result = original(*args, **kwargs)
            self._finish_forward(result)
            return result

        setattr(self.trainer, method_name, wrapped)

    def _callback_list(self) -> list[Any]:
        """Return the mutable callback list required for pre-watchdog capture."""
        handler = getattr(self.trainer, "callback_handler", None)
        callbacks = getattr(handler, "callbacks", None)
        if not isinstance(callbacks, list):
            raise CapsuleError(
                "capture requires Trainer.callback_handler.callbacks to be a mutable list "
                "so it can run before Soup's gradient watchdog"
            )
        return callbacks

    def _place_before_watchdog(self, callback: Any, callbacks: list[Any]) -> None:
        """Fail closed unless the forensic callback can precede Soup's watchdog.

        A capture that runs after ``GRPOStabilityCallback`` may see gradients
        already cleared by ``target.zero_grad()``.  The standard Transformers
        callback handler exposes a mutable list, but this is deliberately
        checked rather than assumed so an unsupported trainer version cannot
        silently produce an empty, misleading capsule.
        """
        if callback not in callbacks:
            raise CapsuleError("trainer.add_callback did not register the forensic callback")
        callbacks.remove(callback)
        callbacks.insert(0, callback)
        if not callbacks or callbacks[0] is not callback:
            raise CapsuleError("could not place the forensic callback before Soup's watchdog")

    def _gradient_stats(self) -> tuple[dict[str, Any], list[str]]:
        records: dict[str, Any] = {}
        bad: list[str] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                records[name] = None
                continue
            stats = _tensor_stats(parameter.grad, self.torch)
            records[name] = stats
            if not stats["finite"]:
                bad.append(name)
        return records, bad

    def _summary(self, reason: str, bad: list[str], error: str | None = None) -> dict[str, Any]:
        assert self._current is not None
        return {
            "schema_version": _SCHEMA_VERSION,
            "reason": reason,
            "serial": self._current["serial"],
            "global_step": self._current["global_step"],
            "nonfinite_gradient_names": bad,
            "first_nonfinite_parameter": self._current["first_nonfinite_parameter"],
            "nonfinite_parameter_names_before_clip": self._current[
                "nonfinite_parameter_names_before_clip"
            ],
            "first_nonfinite_scalar_grad": self._current["first_nonfinite_scalar_grad"],
            "first_nonfinite_activation": self._current["first_nonfinite_activation"],
            "loss": self._current.get("loss"),
            "microbatch_count": len(self._current["microbatches"]),
            "rank": self.rank,
            "world_size": self.world_size,
            "error": error,
        }

    def _flush(self, reason: str, bad: list[str]) -> None:
        if self._current is None or self._flushed:
            return
        payload = dict(self._current)
        payload["schema_version"] = _SCHEMA_VERSION
        payload["reason"] = reason
        gradient_stats, nonfinite_names = self._gradient_stats()
        payload["post_clip_pre_watchdog_gradient_stats"] = gradient_stats
        payload["nonfinite_gradient_names"] = nonfinite_names
        name = f"capsule_{self._current['serial']:06d}.pt"
        summary_name = f"capsule_{self._current['serial']:06d}.json"
        try:
            capsule_path = self.writer.write_torch(name, payload, self.torch)
            summary = self._summary(reason, bad)
            summary["payload"] = capsule_path.name
            self.writer.write_json(summary_name, summary)
        except CapsuleSizeLimitError as exc:
            try:
                self.writer.write_json(summary_name, self._summary(reason, bad, error=str(exc)))
            except CapsuleSizeLimitError as summary_exc:
                warnings.warn(
                    "capsule payload and size-limit summary could not be written: "
                    f"payload={exc}; summary={summary_exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._flushed = True

    def capture_pre_optimizer(self, state: Any) -> None:
        """Run after Trainer clipping and before Soup's watchdog clears gradients."""
        del state
        if self._current is None:
            return
        self._synchronize("pre_optimizer")
        _, bad = self._gradient_stats()
        self._bad_seen = self._bad_seen or bool(bad)
        if self._bad_seen:
            self._flush("nonfinite_before_watchdog", bad)

    def finish_step(self) -> None:
        """Drop normal-step rows and weights; anomaly data was already persisted."""
        self._current = None
        self._bad_seen = False
        self._flushed = False

    def install(self) -> "Issue342CrashCapsule":
        if self._closed:
            raise CapsuleError(
                "a closed capsule cannot be reinstalled; choose a fresh output directory"
            )
        if self._installed:
            return self
        try:
            if bool(getattr(getattr(self.trainer, "args", None), "use_liger_kernel", False)):
                raise CapsuleError(
                    "Liger loss bypasses the wrapped GRPO loss path; crash capture is unsupported"
                )
            trainable_bytes = self._trainable_state_nbytes()
            if trainable_bytes > self.writer.max_bytes:
                raise CapsuleSizeLimitError(
                    "capsule rank cap cannot fit the pre-step trainable snapshot: "
                    f"{trainable_bytes} bytes required, {self.writer.max_bytes} bytes available"
                )
            # Check before adding hooks or registering any callback.  A custom
            # trainer that cannot expose ordering is unsupported rather than a
            # source of partially attached instrumentation.
            callbacks = self._callback_list()
            self.writer.write_json("manifest.json", self._environment())
            self.writer.write_json(
                "capture_policy.json",
                {
                    "captures_only_on_anomaly": True,
                    "contains_training_rows": True,
                    "load_only_your_own_capsule": True,
                    "serialization": "Torch pickle (.pt)",
                    "torch_load": "weights_only=True where supported",
                },
            )
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    self._parameter_handles.append(
                        parameter.register_hook(self._parameter_hook(name))
                    )
            if self.instrument_lora_operands:
                for name, module in self.model.named_modules():
                    if isinstance(module, self.torch.nn.Linear) and (
                        "lora_A" in name or "lora_B" in name
                    ):
                        self._module_handles.append(
                            module.register_forward_pre_hook(self._lora_forward_hook(name))
                        )
                        self._module_handles.append(
                            module.register_full_backward_hook(self._lora_backward_hook(name))
                        )
            self._wrap_loss()
            callback = _callback(self)
            self._capture_callback = callback
            self.trainer.add_callback(callback)
            self._place_before_watchdog(callback, callbacks)
        except Exception:
            # An unsupported Trainer shape must leave no hooks or monkeypatches
            # active.  Keep the new directory as an audit record rather than
            # deleting it or risking an overwrite of user-selected storage.
            self.close()
            raise
        self._installed = True
        return self

    @contextlib.contextmanager
    def sdpa_backend(self, mode: str) -> Iterator[None]:
        if self._sdpa_mode is not None:
            raise CapsuleError("SDPA backend contexts cannot be nested")
        self._sdpa_mode = mode
        try:
            with sdpa_backend(mode):
                yield
        finally:
            self._sdpa_mode = None

    def close(self) -> None:
        if self._closed:
            return
        for handle in self._parameter_handles:
            handle.remove()
        self._parameter_handles.clear()
        for handle in self._module_handles:
            handle.remove()
        self._module_handles.clear()
        if self._loss_method_name is not None and self._original_loss is not None:
            setattr(self.trainer, self._loss_method_name, self._original_loss)
        self._loss_method_name = None
        self._original_loss = None
        callbacks = getattr(getattr(self.trainer, "callback_handler", None), "callbacks", None)
        if isinstance(callbacks, list) and self._capture_callback in callbacks:
            callbacks.remove(self._capture_callback)
        self._capture_callback = None
        self._installed = False
        self._closed = True
