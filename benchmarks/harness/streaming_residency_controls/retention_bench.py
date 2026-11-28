"""Four research arms at one physical8GiB/allocator7GiB budget; no compilation."""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

from benchmarks.harness.streaming_residency_controls.activation_retention import install_retention


def baseline_patch_metadata(wrapper):
    """Bridge report-only metadata absent in frozen upstream, refusing real patches."""
    training = wrapper.config.training
    patches = getattr(wrapper, "_streaming_liger_patches", {})
    if (training.use_liger or training.fast_lora or getattr(wrapper, "_liger_applied", False)
            or patches):
        raise ValueError("requires unpatched ordinary PEFT streaming")
    wrapper._streaming_liger_patches = {}
    return {"metadata_only": True, "production_arithmetic_changed": False}


def host_memory():
    return {line.split(":")[0] + "_bytes": int(line.split()[1]) * 1024
            for line in Path("/proc/self/status").read_text().splitlines()
            if line.split(":")[0] in {"VmRSS", "VmHWM", "VmLck", "VmPin"}}


@contextlib.contextmanager
def study_wrapper(arm, indices, provenance):
    import torch
    from transformers import TrainerCallback

    import soup_cli.trainer.sft as sft
    from benchmarks.harness.streaming_residency_candidate import install_persistent_layers
    from soup_cli.utils.layer_stream_runtime import StreamedDecoderLayer

    original = sft.SFTTrainerWrapper

    def remember_peaks():
        for key, value in (("whole_run_peak_allocated_bytes", torch.cuda.max_memory_allocated()),
                           ("whole_run_peak_reserved_bytes", torch.cuda.max_memory_reserved())):
            provenance[key] = max(provenance.get(key, 0), value)

    class ResearchWrapper(original):
        def setup(self, *args, **kwargs):
            result = super().setup(*args, **kwargs)
            runtime = self._stream_runtime
            provenance["baseline_metadata_compatibility"] = baseline_patch_metadata(self)
            if not runtime or runtime.tier != "ram":
                raise ValueError("requires ordinary dense RAM-streamed training")
            provenance.update(layer_count=runtime.n_layers,
                              per_layer_bytes=runtime.pool.nbytes // runtime.pool.n,
                              host_memory_after_setup=host_memory(),
                              attention_backend=self.model.get_base_model().config._attn_implementation)
            state = install_persistent_layers(
                runtime, indices if arm in {"resident", "both"} else [])
            provenance["residency"] = state
            if arm in {"retained", "both"}:
                provenance["retention"] = install_retention(self.model, indices,
                                                            resident=arm == "both")
            provenance["per_update"] = []
            calls = {}
            pending, seen = [self.model], set()
            while pending:
                module = pending.pop()
                if module is None or id(module) in seen:
                    continue
                seen.add(id(module))
                if isinstance(module, StreamedDecoderLayer):
                    previous = module.inner.forward
                    index = module.idx

                    def counted(*args, _previous=previous, _index=index, **kwargs):
                        calls[_index] = calls.get(_index, 0) + 1
                        return _previous(*args, **kwargs)

                    module.inner.forward = counted
                else:
                    pending.extend(module._modules.values())

            class Observe(TrainerCallback):
                def on_step_begin(self, _args, _step, _control, **_kwargs):
                    remember_peaks()
                    self.loads = runtime.pool.loads
                    self.volume = state["dynamic_logical_bytes"]
                    self.calls = sum(calls.values())

                def on_step_end(self, _args, step, _control, **_kwargs):
                    remember_peaks()
                    provenance["per_update"].append({
                        "step": step.global_step, "decoder_loads": runtime.pool.loads - self.loads,
                        "decoder_logical_bytes": state["dynamic_logical_bytes"] - self.volume,
                        "inner_forward_calls": sum(calls.values()) - self.calls})

            self.trainer.add_callback(Observe())
            provenance["inner_forward_calls"] = calls
            remember_peaks()
            return result

    sft.SFTTrainerWrapper = ResearchWrapper
    try:
        yield
    finally:
        remember_peaks()
        provenance.update(host_memory_at_exit=host_memory(),
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
        sft.SFTTrainerWrapper = original


def main():
    import torch

    from benchmarks.harness import liger_streaming_bench, liger_streaming_parity
    from soup_cli.config.loader import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--arm", required=True,
                        choices=["baseline", "resident", "retained", "both"])
    parser.add_argument("--indices", default="")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--warmup", default=6, type=int)
    parser.add_argument("--steps", default=8, type=int)
    args = parser.parse_args()
    if args.capture and args.semantics:
        raise ValueError("capture modes exclusive")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one CUDA GPU required")
    total = torch.cuda.get_device_properties(0).total_memory
    if not 7.5 * 2**30 <= total <= 8.1 * 2**30 or not torch.cuda.is_bf16_supported():
        raise ValueError("requires physical8GiB and native BF16")
    torch.cuda.set_per_process_memory_fraction(7 * 2**30 / total)
    indices = [int(value) for value in args.indices.split(",") if value]
    if (args.arm == "baseline") != (not indices):
        raise ValueError("baseline alone has no selected layers")
    provenance = {"arm": args.arm, "indices": indices, "physical_gpu_bytes": total,
                  "allocator_budget_bytes": 7 * 2**30, "torch": torch.__version__,
                  "pid": os.getpid(), "started_unix": time.time()}
    try:
        if args.semantics:
            from benchmarks.harness import streaming_optimization_bench as semantics

            semantics.study_wrapper = lambda *_args: study_wrapper(args.arm, indices, provenance)
            semantics.attention_semantics(args.config, "baseline", args.output)
        else:
            with study_wrapper(args.arm, indices, provenance):
                if args.capture:
                    liger_streaming_parity.capture(args.config, "baseline", args.output)
                else:
                    liger_streaming_bench.run(load_config(args.config), arm="baseline",
                                             warmup=args.warmup, steps=args.steps,
                                             output=args.output)
        provenance["status"] = "COMPLETED_REQUIRES_INDEPENDENT_COMPARISON"
    except BaseException as error:
        provenance.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.mkdir(parents=True, exist_ok=True)
        provenance["whole_process_seconds"] = time.time() - provenance["started_unix"]
        (args.output / "cache-provenance.json").write_text(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
