"""Physical-8-GiB research entry: residency, then resident-only compilation.

No hardware provisioning or production configuration changes. The frozen source
overlay supplies the existing capture, comparator and synchronized timing code.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def initialize_patch_metadata(wrapper):
    """Frozen ordinary upstream has no optional research-Liger bookkeeping field."""
    if wrapper.config.training.use_liger:
        raise ValueError("ordinary unpatched reference required")
    if not hasattr(wrapper, "_streaming_liger_patches"):
        wrapper._streaming_liger_patches = {}


def select_resident_layers(
    layers: int, layer_bytes: int, baseline_reserved: int, budget: int, reserve: int,
) -> list[int]:
    """Choose a fixed middle block before qualifying either candidate."""
    if layers < 2 or layer_bytes <= 0 or min(baseline_reserved, budget, reserve) < 0:
        raise ValueError("invalid measured capacity inputs")
    count = min(layers // 2, max(0, (budget - baseline_reserved - reserve) // layer_bytes))
    start = (layers - count) // 2
    return list(range(start, start + count))


def compile_resident_layers(
    model: Any, indices: list[int], *, precise: bool = False, backend: str = "inductor",
) -> int:
    """Compile tensor computation only; keep pooled-weight ownership eager."""
    import torch
    import torch._dynamo

    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    if backend not in {"inductor", "aot_eager"}:
        raise ValueError("unsupported research backend")
    if torch._dynamo.config.suppress_errors:
        raise ValueError("compiler error suppression is forbidden")
    selected = set(indices)
    if not selected or len(selected) != len(indices):
        raise ValueError("need unique resident indices")
    found, pending, seen = {}, [model], set()
    while pending:
        module = pending.pop()
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, StreamedDecoderLayer):
            if module.idx in selected:
                owner = module.pool.owner[module.pool.slot_for(module.idx)]
                if module.quant_specs or owner != module.idx:
                    raise ValueError("selected dense layer is not resident")
                persistent = getattr(module.pool, "_persistent_layers", {}).get("indices", [])
                if module.idx not in persistent:
                    raise ValueError("selected layer has no dedicated storage")
                found[module.idx] = module
        else:
            pending.extend(module._modules.values())
    if set(found) != selected:
        raise ValueError("resident layer selection was not fully reached")
    options = {"triton.cudagraphs": False} if backend == "inductor" else None
    if options is not None and precise:
        options.update(epilogue_fusion=False, pattern_matcher=False, max_fusion_size=1,
                       force_same_precision=True, emulate_precision_casts=True)
    if backend == "inductor":
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
        torch._dynamo.config.fail_on_recompile_limit_hit = True
    prepared = []
    for index in indices:
        inner = found[index].inner
        if getattr(inner, "_soup_resident_compiled", False):
            raise ValueError("layer already compiled")
        prepared.append(torch.compile(inner.forward, backend=backend, fullgraph=True,
                                      dynamic=False, options=options))
    for index, forward in zip(indices, prepared, strict=True):
        found[index].inner.forward = forward
        found[index].inner._soup_resident_compiled = True
    return len(prepared)


def compiler_counters() -> dict:
    from torch._dynamo.utils import counters

    return {group: dict(counters[group]) for group in ("stats", "frames", "graph_break")}


def host_memory() -> dict:
    import resource

    values = {"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    for line in Path("/proc/self/status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"VmRSS", "VmHWM", "VmLck", "VmPin"}:
            values[name + "_bytes"] = int(value.split()[0]) * 1024
    return values


@contextlib.contextmanager
def study_wrapper(arm: str, indices: list[int], provenance: dict, warmup: int | None):
    """Install research modifications after ordinary Soup setup."""
    import torch
    from transformers import TrainerCallback

    import soup_cli.trainer.sft as sft
    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers

    original = sft.SFTTrainerWrapper

    class ResearchWrapper(original):
        def setup(self, *args: Any, **kwargs: Any):
            setup_started = time.perf_counter()
            result = super().setup(*args, **kwargs)
            initialize_patch_metadata(self)
            provenance["ordinary_setup_seconds"] = time.perf_counter() - setup_started
            runtime = self._stream_runtime
            if not runtime or runtime.tier != "ram" or self._streaming_liger_patches:
                raise ValueError("requires ordinary dense RAM-streamed PEFT training")
            provenance["layer_count"] = runtime.n_layers
            provenance["per_layer_bytes"] = runtime.pool.nbytes // runtime.pool.n
            provenance["host_memory_after_setup"] = host_memory()
            provenance["attention_backend"] = (
                self.model.get_base_model().config._attn_implementation
            )
            state = install_persistent_layers(runtime, indices)
            provenance["residency"] = state
            provenance["compiled_decoders"] = 0
            compile_wrap_started = time.perf_counter()
            if arm.startswith("resident_compile"):
                provenance["compiled_decoders"] = compile_resident_layers(
                    self.model, indices, precise=arm == "resident_compile_precise",
                )
            provenance["compile_wrapper_seconds"] = time.perf_counter() - compile_wrap_started
            provenance["all_update_wall_seconds"] = []
            provenance["compiler_before"] = compiler_counters()
            provenance["setup_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
            provenance["setup_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
            provenance["all_update_peak_allocated_bytes"] = []
            provenance["all_update_peak_reserved_bytes"] = []
            provenance["per_update_loads"] = []

            class Observe(TrainerCallback):
                def on_step_begin(self, _args, step, _control, **_kwargs):
                    torch.cuda.synchronize()
                    self.update_started = time.perf_counter()
                    self.loads = runtime.pool.loads
                    self.bytes = state["dynamic_logical_bytes"]
                    if warmup is not None and step.global_step == warmup:
                        provenance["compiler_after_warmup"] = compiler_counters()

                def on_step_end(self, _args, step, _control, **_kwargs):
                    torch.cuda.synchronize()
                    provenance["all_update_wall_seconds"].append(
                        time.perf_counter() - self.update_started,
                    )
                    provenance["all_update_peak_allocated_bytes"].append(
                        torch.cuda.max_memory_allocated(),
                    )
                    provenance["all_update_peak_reserved_bytes"].append(
                        torch.cuda.max_memory_reserved(),
                    )
                    provenance["per_update_loads"].append({
                        "step": step.global_step, "loads": runtime.pool.loads - self.loads,
                        "bytes": state["dynamic_logical_bytes"] - self.bytes,
                    })
                    if (arm.startswith("resident_compile") and warmup is not None
                            and step.global_step > warmup
                            and compiler_counters() != provenance["compiler_after_warmup"]):
                        raise ValueError("compilation occurred inside measured timing window")

            self.trainer.add_callback(Observe())
            return result

    sft.SFTTrainerWrapper = ResearchWrapper
    try:
        yield
        provenance["compiler_after"] = compiler_counters()
        if arm.startswith("resident_compile"):
            before = provenance["compiler_before"]["stats"].get("unique_graphs", 0)
            after = provenance["compiler_after"]["stats"].get("unique_graphs", 0)
            if after <= before:
                raise ValueError("no compiled graph executed")
    finally:
        from torch._dynamo.utils import compilation_time_metrics

        provenance["compiler_time_metrics_seconds_inclusive"] = {
            key: list(values) for key, values in compilation_time_metrics.items()
        }
        provenance["host_memory_at_exit"] = host_memory()
        provenance["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        provenance["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        provenance["whole_run_peak_allocated_bytes"] = max(
            [provenance["peak_allocated_bytes"], provenance.get("setup_peak_allocated_bytes", 0)]
            + provenance.get("all_update_peak_allocated_bytes", []),
        )
        provenance["whole_run_peak_reserved_bytes"] = max(
            [provenance["peak_reserved_bytes"], provenance.get("setup_peak_reserved_bytes", 0)]
            + provenance.get("all_update_peak_reserved_bytes", []),
        )
        sft.SFTTrainerWrapper = original


def main() -> None:
    import torch

    from benchmarks.harness import liger_streaming_bench, liger_streaming_parity
    from soup_cli.config.loader import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["baseline", "resident", "resident_compile",
                                          "resident_compile_precise"], required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.capture and args.semantics:
        raise ValueError("capture modes are exclusive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one visible CUDA GPU required")
    total = torch.cuda.get_device_properties(0).total_memory
    if not 7.5 * 2**30 <= total <= 8.1 * 2**30:
        raise ValueError("requires a real physical GPU with approximately 8 GiB")
    torch.cuda.set_per_process_memory_fraction(7 * 2**30 / total)
    indices = [int(value) for value in args.indices.split(",") if value]
    if (args.arm == "baseline") != (not indices):
        raise ValueError("baseline has no resident layers; candidates need selected layers")
    provenance = {"status": "STARTED", "arm": args.arm, "indices": indices,
                  "physical_gpu_bytes": total, "allocator_budget_bytes": 7 * 2**30,
                  "budget_excludes_context": True, "torch": torch.__version__,
                  "research_bench_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "pid": os.getpid(), "affinity": sorted(os.sched_getaffinity(0))}
    process_started = time.perf_counter()
    try:
        if args.semantics:
            from benchmarks.harness import streaming_optimization_bench as semantics

            semantics.study_wrapper = lambda _arm, _provenance: study_wrapper(
                args.arm, indices, provenance, None,
            )
            semantics.attention_semantics(args.config, "baseline", args.output)
        else:
            with study_wrapper(args.arm, indices, provenance,
                               None if args.capture else args.warmup):
                if args.capture:
                    liger_streaming_parity.capture(args.config, "baseline", args.output)
                else:
                    liger_streaming_bench.run(load_config(args.config), arm="baseline",
                                             warmup=args.warmup, steps=args.steps,
                                             output=args.output)
        provenance["status"] = "COMPLETED_NOT_A_NUMERICAL_OR_PERFORMANCE_VERDICT"
    except BaseException as error:
        provenance.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        provenance["whole_entry_seconds"] = time.perf_counter() - process_started
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "residency-compile-provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
        )


if __name__ == "__main__":
    main()
