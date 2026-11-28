"""Fresh-process dense streaming residency experiment; never rents hardware."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any


def host_memory() -> dict:
    """Process RSS and locked memory, distinct from the shared host's free RAM."""
    status = Path("/proc/self/status")
    values = {}
    if status.exists():
        for line in status.read_text().splitlines():
            name, _, value = line.partition(":")
            if name in {"VmRSS", "VmHWM", "VmLck", "VmPin"}:
                values[name + "_bytes"] = int(value.split()[0]) * 1024
    return values


@contextlib.contextmanager
def residency_wrapper(indices: list[int], provenance: dict, **_unused: Any):
    """Install dedicated slots after ordinary upstream setup, before any training."""
    import torch
    from transformers import TrainerCallback

    import soup_cli.trainer.sft as sft
    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers
    from benchmarks.harness.streaming_residency_controls.retention_bench import (
        baseline_patch_metadata,
    )

    original = sft.SFTTrainerWrapper

    class ResidencyWrapper(original):
        def setup(self, *args: Any, **kwargs: Any):
            result = super().setup(*args, **kwargs)
            runtime = self._stream_runtime
            if not runtime or runtime.n_layers != 32 or runtime.tier != "ram":
                raise ValueError("requires the 32-layer dense RAM-streamed fixture")
            provenance["baseline_metadata_compatibility"] = baseline_patch_metadata(self)
            torch.cuda.synchronize()
            provenance["allocated_before_install"] = torch.cuda.memory_allocated()
            provenance["host_memory_before_install"] = host_memory()
            state = install_persistent_layers(runtime, indices)
            provenance["residency"] = state
            provenance["allocated_after_install"] = torch.cuda.memory_allocated()
            provenance["host_memory_after_install"] = host_memory()
            provenance["runtime_after_install"] = runtime.stats()
            provenance["per_update_loads"] = []
            provenance["attention_backend"] = (
                self.model.get_base_model().config._attn_implementation
            )

            class LoadAccounting(TrainerCallback):
                def on_step_begin(self, _args, _state, _control, **_kwargs):
                    self.loads = runtime.pool.loads
                    self.large = getattr(runtime.large_pool, "loads", 0)
                    self.volume = state["dynamic_logical_bytes"]

                def on_step_end(self, _args, state_info, _control, **_kwargs):
                    provenance["per_update_loads"].append({
                        "step": state_info.global_step,
                        "decoder_loads": runtime.pool.loads - self.loads,
                        "large_loads": getattr(runtime.large_pool, "loads", 0) - self.large,
                        "decoder_logical_bytes": state["dynamic_logical_bytes"] - self.volume,
                    })

            self.trainer.add_callback(LoadAccounting())
            return result

    sft.SFTTrainerWrapper = ResidencyWrapper
    try:
        yield
    finally:
        provenance["host_memory_at_exit"] = host_memory()
        sft.SFTTrainerWrapper = original


def main() -> None:
    import torch

    from benchmarks.harness import liger_streaming_bench, liger_streaming_parity
    from soup_cli.config.loader import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--vram-budget-gib", type=float, default=10)
    args = parser.parse_args()
    indices = [int(value) for value in args.indices.split(",") if value]
    if args.capture and args.semantics:
        raise ValueError("choose one capture mode")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one visible CUDA GPU required")
    total = torch.cuda.get_device_properties(0).total_memory
    budget = int(args.vram_budget_gib * 2**30)
    if budget > total or budget <= 0:
        raise ValueError("invalid CUDA allocator budget")
    torch.cuda.set_per_process_memory_fraction(budget / total)
    provenance = {
        "status": "STARTED", "indices": indices, "allocator_budget_bytes": budget,
        "physical_gpu_bytes": total, "budget_excludes_context": True,
        "torch": torch.__version__, "pid": os.getpid(),
        "affinity": sorted(os.sched_getaffinity(0)), "started_unix": time.time(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.semantics:
            from benchmarks.harness import streaming_optimization_bench as semantics

            def wrapper(_arm: str, _provenance: dict, *_args: Any):
                return residency_wrapper(indices, provenance)

            semantics.study_wrapper = wrapper
            semantics.attention_semantics(args.config, "baseline", args.output)
        else:
            with residency_wrapper(indices, provenance):
                if args.capture:
                    liger_streaming_parity.capture(args.config, "baseline", args.output)
                else:
                    liger_streaming_bench.run(
                        load_config(args.config), arm="baseline", warmup=args.warmup,
                        steps=args.steps, output=args.output,
                    )
        if provenance["residency"]["preload_loads"] != len(indices):
            raise ValueError("persistent layer preloads incomplete")
        provenance["status"] = "COMPLETED_NOT_A_NUMERICAL_OR_PERFORMANCE_VERDICT"
    except BaseException as exc:
        provenance.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "residency-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
