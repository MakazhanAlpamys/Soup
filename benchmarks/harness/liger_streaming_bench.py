"""One local Soup SFT streaming arm; never provisions hardware or uploads results.

Run from the checkout: python -m benchmarks.harness.liger_streaming_bench --help
CUDA only. Correctness tests must qualify an arm before interpreting its timing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


def build_config(config: Any, *, arm: str, output: str) -> Any:
    """Revalidate deliberate overrides rather than mutating a validated model."""
    from soup_cli.config.schema import SoupConfig

    if config.task != "sft" or not config.training.stream_layers or config.training.fast_lora:
        raise ValueError("benchmark requires streamed SFT with fast_lora=false")
    if not Path(config.base).is_dir():
        raise ValueError("base must be a local immutable model snapshot directory")
    if not isinstance(config.data.train, str) or not Path(config.data.train).is_file():
        raise ValueError("data.train must be one local frozen dataset file")
    if config.data.streaming:
        raise ValueError("benchmark requires a finite local dataset")
    choices = {
        "baseline": None,
        "rms_norm": ["rms_norm"],
        "swiglu": ["swiglu"],
        "rms_norm+swiglu": ["rms_norm", "swiglu"],
    }
    if arm not in choices:
        raise ValueError(f"unknown benchmark arm: {arm}")
    payload = config.model_dump()
    payload["output"] = output
    payload["data"]["val_split"] = 0
    payload["training"].update(
        use_liger=arm != "baseline",
        eval_steps=None,
        # Measurements exclude periodic saves; final saving still uses Soup.
        save_steps=1_000_000_000,
    )
    return SoupConfig.model_validate(payload)


def _gpu_snapshot() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,pstate,clocks.sm,power.draw,"
                "temperature.gpu,memory.used",
                "--format=csv",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE: {exc}"


def run(
    config: Any, *, arm: str, warmup: int, steps: int, output: Path, diagnostics: bool = False,
) -> None:
    import torch
    from transformers import TrainerCallback, set_seed

    from benchmarks.harness.liger_streaming_parity import configure_determinism, initialize_adapters
    from soup_cli.data.loader import load_dataset
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils.seeding import resolve_training_seed

    configure_determinism()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if warmup < 1 or steps < 2:
        raise ValueError("warmup must be >=1 and measured steps >=2")
    output.mkdir(parents=True, exist_ok=False)
    config = build_config(config, arm=arm, output=str(output / "adapter"))
    (output / "resolved-config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")
    report: dict[str, Any] = {
        "status": "STARTED",
        "arm": arm,
        "warmup": warmup,
        "steps": steps,
        "diagnostic_instrumented": diagnostics,
        "scope": "synchronized optimizer-step window; excludes first-batch fetch/save/eval",
        "data_sha256": hashlib.sha256(Path(config.data.train).read_bytes()).hexdigest(),
        "gpu_before": _gpu_snapshot(),
        "seconds": [],
        "peak_allocated_bytes": [],
        "peak_reserved_bytes": [],
    }
    set_seed(resolve_training_seed(config.training))
    wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
    try:
        setup_start = time.perf_counter()
        wrapper.setup(load_dataset(config.data))
        adapters = initialize_adapters(wrapper.model)
        digest = hashlib.sha256()
        for name, value in sorted(adapters.items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().float().numpy().tobytes())
        report["initial_adapter_sha256"] = digest.hexdigest()
        report["setup_seconds"] = time.perf_counter() - setup_start
        report["patch_counts"] = wrapper._streaming_liger_patches
        trainer = wrapper.trainer
        report["decoder_dtype"] = str(wrapper.model.get_base_model().model.norm.weight.dtype)
        report["amp"] = {"fp16": trainer.args.fp16, "bf16": trainer.args.bf16}
        trainer.args.max_steps = warmup + steps
        trainer.args.disable_tqdm = True
        trainer.args.logging_steps = 1
        trainer.args.logging_nan_inf_filter = False
        report["input_shapes"] = []

        def observe(_module, _args, kwargs):
            report["input_shapes"].append(list(kwargs["input_ids"].shape))

        handle = wrapper.model.register_forward_pre_hook(observe, with_kwargs=True)

        class Measure(TrainerCallback):
            def on_step_begin(self, _args, state, _control, **_kwargs):
                if state.global_step >= warmup:
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    self.started = time.perf_counter()

            def on_step_end(self, _args, state, _control, **_kwargs):
                if state.global_step > warmup:
                    torch.cuda.synchronize()
                    report["seconds"].append(time.perf_counter() - self.started)
                    report["peak_allocated_bytes"].append(torch.cuda.max_memory_allocated())
                    report["peak_reserved_bytes"].append(torch.cuda.max_memory_reserved())

        trainer.add_callback(Measure())
        torch.cuda.synchronize()
        train_start = time.perf_counter()
        if diagnostics:
            from benchmarks.harness.liger_streaming_diagnostics import StreamingDiagnostics

            diagnosis = StreamingDiagnostics(wrapper._stream_runtime, output, warmup)
            trainer.add_callback(diagnosis.callback())
            measurement = diagnosis
        else:
            measurement = contextlib.nullcontext()
        with measurement:
            wrapper.train()
        handle.remove()
        torch.cuda.synchronize()
        report["training_wall_seconds_including_final_save"] = time.perf_counter() - train_start
        report["training_log_history"] = trainer.state.log_history
        report["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        for entry in trainer.state.log_history:
            for key in ("loss", "grad_norm"):
                if key in entry and not math.isfinite(float(entry[key])):
                    raise ValueError(f"non-finite {key}: timing is not a valid training result")
        if any(
            not torch.isfinite(param).all().item()
            for param in wrapper.model.parameters()
            if param.requires_grad
        ):
            raise ValueError("non-finite adapter weights after training")
        if len(report["seconds"]) != steps:
            raise ValueError("incomplete measured step window")
        report["median_seconds_per_optimizer_step"] = statistics.median(report["seconds"])
        report["status"] = "TIMING_CAPTURED_NOT_A_CORRECTNESS_OR_QUALITY_VERDICT"
    except BaseException as exc:
        report["status"] = "FAILED"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        wrapper._close_stream_runtime()
        report["gpu_after"] = _gpu_snapshot()
        (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    from soup_cli.config.loader import load_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--arm", choices=["baseline", "rms_norm", "swiglu", "rms_norm+swiglu"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args()
    run(
        load_config(args.config),
        arm=args.arm,
        warmup=args.warmup,
        steps=args.steps,
        output=args.output,
        diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()
