"""#2050 wrapper around the unchanged #361 throughput recipe.

All modes use deterministic SDPA math; this qualification is part of the row.
The optional audit is untimed evidence, never a throughput measurement. It hashes
every logit, loss, adapter gradient and optimizer update and counts executed
matrix/dequantization operators, including backward and recomputation. Instrumentation
is removed entirely for timing arms apart from a first, warm-up cache-budget check.
No production configuration or forward implementation is changed on disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console


def tensor_hash(tensor: Any) -> str:
    import torch

    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    import bitsandbytes.optim as optim
    import issue361_nf4_throughput as base
    import torch
    from stream_selective_checkpoint import POLICIES, install, saved_projection_bytes
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils._python_dispatch import TorchDispatchMode

    from soup_cli.utils import layer_stream_runtime as streaming

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy", choices=("full", "qkv", "qkvo"), default="full")
    parser.add_argument("--audit", help="untimed exactness/operator evidence JSON")
    parser.add_argument("--profile", action="store_true", help="count ops in a separate audit")
    own, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    args = base.parse_args()
    # The existing shard writer admits outputs under the checkout's CWD. The
    # pair runner starts in harness/, so normalize to this explicit source tree.
    os.chdir(Path(__file__).parents[2])
    if args.quant != "nf4" or args.batch != 1 or args.buffers != 2:
        raise ValueError("#2050 requires NF4, batch 1, two buffers")
    if own.audit and (args.warmup != 0 or args.steps != 2):
        raise ValueError("audit requires exactly two updates and zero warm-up")
    if own.profile and not own.audit:
        raise ValueError("operator profiling is only permitted in an untimed audit")
    gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid or torch.cuda.device_count() != 1:
        raise ValueError("select exactly one GPU by UUID with CUDA_VISIBLE_DEVICES")
    # This setting is identical in gates, profiles and measured arms.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_build, original_optimizer = streaming.build_streamed_model, optim.PagedAdamW8bit
    state: dict[str, Any] = {
        "policy": own.policy,
        "attention_backend": "sdpa_math",
        "deterministic_algorithms": True,
        "cache_checked": False,
        "steps": [],
        "operators": {},
        "profile": own.profile,
        "gpu_uuid": gpu_uuid,
    }
    model = runtime = controller = None
    phase = "setup"
    counts: Counter = Counter()
    handles = []

    def gpu_values(fields: str) -> list[int]:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return [int(value.strip()) for value in output.strip().split(",")]

    temperature, memory_mib = gpu_values("temperature.gpu,memory.used")
    power = float(
        subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                gpu_uuid,
                "--query-gpu=power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip()
    )
    if temperature > 65 or memory_mib >= 1024 or power != 350:
        raise RuntimeError(f"GPU admission refused: {temperature} C, {memory_mib} MiB, {power} W")
    state["admission"] = {"temperature_c": temperature, "memory_mib": memory_mib, "power_w": power}

    class SelectedGpuSampler(base.GpuSampler):
        def _poll(self) -> None:
            while not self._stop.is_set():
                try:
                    self.samples.append(
                        tuple(gpu_values("utilization.gpu,clocks.sm,temperature.gpu"))
                    )
                except (ValueError, subprocess.SubprocessError):
                    pass
                self._stop.wait(self.interval)

    class Operators(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if phase in ("forward", "backward") and (
                name.startswith(("aten.mm.", "aten.addmm.", "aten.bmm.")) or "bitsandbytes" in name
            ):
                shapes = [list(v.shape) for v in args if isinstance(v, torch.Tensor)]
                counts[json.dumps([phase, name, shapes])] += 1
            return func(*args, **(kwargs or {}))

    def build(**kwargs: Any) -> Any:
        nonlocal model, runtime, controller
        model, runtime = original_build(**kwargs)
        if own.audit:
            # Make first-step A gradients nontrivial, using the same fixed B in every arm.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "lora_B" in name:
                        param.fill_(0.01)
        if own.policy != "full":
            controller = install(model, own.policy)

        def before(module: Any, inputs: Any) -> None:
            nonlocal phase
            phase = "forward"

        def after(module: Any, inputs: Any, output: Any) -> None:
            nonlocal phase
            if not state["cache_checked"]:
                config = model.config
                kv = config.hidden_size // config.num_attention_heads * config.num_key_value_heads
                expected = (
                    0
                    if controller is None
                    else saved_projection_bytes(
                        own.policy,
                        config.num_hidden_layers,
                        config.hidden_size,
                        kv,
                        args.seq,
                        element_bytes=torch.tensor(
                            [], dtype=getattr(torch, kwargs["dtype"])
                        ).element_size(),
                    )
                )
                actual = 0 if controller is None else controller.cache_bytes()
                expected_count = (
                    0
                    if controller is None
                    else (config.num_hidden_layers * len(POLICIES[own.policy]))
                )
                actual_count = 0 if controller is None else len(controller.cached_tensors())
                if (actual, actual_count) != (expected, expected_count):
                    raise RuntimeError(
                        f"cache budget/count mismatch: {actual}/{actual_count}, "
                        f"expected {expected}/{expected_count}"
                    )
                state.update(cache_checked=True, saved_bytes=actual, saved_tensors=actual_count)
            if own.audit:
                state["steps"].append(
                    {"logits": tensor_hash(output.logits), "loss": tensor_hash(output.loss)}
                )
            else:
                for handle in handles:
                    handle.remove()
            phase = "backward"

        if own.audit:
            handles.append(model.register_forward_pre_hook(before))
        handles.append(model.register_forward_hook(after))
        return model, runtime

    def optimizer_factory(*args: Any, **kwargs: Any) -> Any:
        optimizer = original_optimizer(*args, **kwargs)
        original_step = optimizer.step

        def step(*step_args: Any, **step_kwargs: Any) -> Any:
            nonlocal phase
            state["steps"][-1]["gradients"] = {
                name: tensor_hash(param.grad)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            phase = "optimizer"
            result = original_step(*step_args, **step_kwargs)
            state["steps"][-1]["parameters"] = {
                name: tensor_hash(param)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            return result

        optimizer.step = step
        return optimizer

    streaming.build_streamed_model = build
    original_sampler, original_clock = base.GpuSampler, streaming.sm_clock_mhz
    base.GpuSampler = SelectedGpuSampler
    streaming.sm_clock_mhz = lambda: gpu_values("clocks.sm")[0]
    if own.audit:
        optim.PagedAdamW8bit = optimizer_factory
    try:
        with sdpa_kernel(SDPBackend.MATH):
            if own.profile:
                with Operators():
                    rc = base.main()
            else:
                rc = base.main()
        state["operators"] = dict(sorted(counts.items()))
        if controller is not None:
            state["selected_ops"] = controller.ops
        if not state["cache_checked"]:
            raise RuntimeError("no forward/cache audit occurred")
        if own.audit:
            state["status"] = "ok" if rc == 0 else "failed"
            Path(own.audit).write_text(json.dumps(state, indent=2), encoding="utf-8")
        if args.json_path:
            path = Path(args.json_path)
            row = json.loads(path.read_text(encoding="utf-8"))
            if not row.get("gpu", {}).get("n") or row["gpu"]["temp_max"] > 75:
                raise RuntimeError("missing GPU samples or temperature exceeded 75 C")
            row["selective_checkpoint"] = {
                k: v for k, v in state.items() if k not in ("steps", "operators")
            }
            row["issue"] = 2050
            row["measurement_kind"] = "untimed-audit" if own.audit else "throughput"
            path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        return rc
    finally:
        streaming.build_streamed_model, optim.PagedAdamW8bit = original_build, original_optimizer
        base.GpuSampler, streaming.sm_clock_mhz = original_sampler, original_clock
        for handle in handles:
            handle.remove()
        if controller is not None:
            controller.remove()
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        Console(stderr=True).print_exception(show_locals=False)
        raise SystemExit(1)
