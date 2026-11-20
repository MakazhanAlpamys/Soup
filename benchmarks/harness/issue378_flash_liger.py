"""One real Soup SFT arm for the #378 sequence/checkpointing sweep.

Counter/profiler probes and timed runs are separate fresh processes. Select one
CUDA GPU with CUDA_VISIBLE_DEVICES; pass a pinned local Llama-3.1-8B snapshot.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from rich.console import Console

ARMS = {"sdpa": (False, False), "fa2": (True, False),
        "liger": (False, True), "fa2_liger": (True, True)}
console = Console()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def gpu_sample() -> dict:
    fields = "name,uuid,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu"
    selected = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = subprocess.run([
        "nvidia-smi", "-i", selected, f"--query-gpu={fields}", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    return dict(zip(fields.split(","), result.stdout.strip().split(", "), strict=True))


def activation_counters() -> Counter:
    """Install only in a separate probe process; timed runs never call this."""
    import torch.nn.functional as functional

    counts = Counter()

    def wrap(module, name, label):
        original = getattr(module, name)

        @functools.wraps(original)
        def counted(*args, **kwargs):
            counts[label] += 1
            return original(*args, **kwargs)

        setattr(module, name, counted)

    wrap(functional, "scaled_dot_product_attention", "sdpa")
    # Transformers lazy-imports these aliases; install before model setup.
    try:
        import flash_attn
        import flash_attn.flash_attn_interface as interface
    except ImportError:
        return counts
    for module in (flash_attn, interface):
        for name in ("flash_attn_func", "flash_attn_varlen_func"):
            wrap(module, name, "fa2")
    return counts


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--sequence", type=int, required=True)
    parser.add_argument("--checkpoint", choices=("off", "on"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    import torch
    from transformers import TrainerCallback

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.sft import SFTTrainerWrapper

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one visible CUDA GPU")
    if min(args.sequence, args.steps) < 1 or args.warmup < 0:
        raise ValueError("Positive sequence/steps and nonnegative warmup required")
    if args.profile and not args.probe:
        raise ValueError("Profiling is confined to separate activation probes")
    torch.set_num_threads(4)
    before = gpu_sample()
    counters = activation_counters() if args.probe else None
    fa2, liger = ARMS[args.arm]
    config = SoupConfig.model_validate({
        "base": args.model, "task": "sft", "backend": "transformers",
        "data": {"train": "in-memory-throughput-fixture", "max_length": args.sequence,
                 "chat_template": "llama3"},
        "output": str(args.output / "adapter"),
        "training": {
            "quantization": "none", "batch_size": 4, "gradient_accumulation_steps": 1,
            "epochs": 1, "lr": 2e-5, "warmup_ratio": 0.0, "scheduler": "constant",
            "optimizer": "adamw_torch", "seed": 378, "data_seed": 378,
            "gradient_checkpointing": args.checkpoint == "on",
            "use_flash_attn": fa2, "use_liger": liger,
            "logging_steps": 100000, "save_steps": 100000,
            "lora": {"r": 16, "alpha": 32, "dropout": 0.0, "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
            ]},
        },
    })
    body = "A telescope collects light through a lens and forms an image. " * (
        args.sequence // 8 + 32
    )
    # Rows overshoot the requested size; realized collated shape is asserted below.
    rows = [{"messages": [{"role": "user", "content": "Explain how telescopes work."},
                          {"role": "assistant", "content": body}]}
            for _ in range(4 * (args.warmup + args.steps))]
    wrapper = SFTTrainerWrapper(config, device="cuda", report_to="none")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    trainer.args.max_steps = args.warmup + args.steps
    trainer.args.save_strategy = "no"
    trainer.args.dataloader_num_workers = 0
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    wrapper.model.config.use_cache = False
    data = list(trainer.train_dataset)
    batch = trainer.data_collator(data[:4])
    realized = list(batch["input_ids"].shape)
    if realized != [4, args.sequence]:
        raise RuntimeError(f"Wrong realized shape: {realized}, expected [4, {args.sequence}]")
    # TRL drops the dataset's labels column in its Liger path and reconstructs
    # labels in the collator from assistant_masks. Hash actual model inputs,
    # rather than treating that equivalent storage representation as a new task.
    workload_hash = hashlib.sha256()
    for offset in range(0, len(data), 4):
        collated = trainer.data_collator(data[offset:offset + 4])
        for key in ("input_ids", "labels"):
            tensor = collated[key].contiguous()
            if list(tensor.shape) != realized:
                raise RuntimeError(f"Variable collated shape for {key}")
            workload_hash.update(key.encode() + tensor.numpy().tobytes())
        mask = collated.get("attention_mask")
        if mask is not None and not bool(mask.all()):
            raise RuntimeError("The throughput fixture must not introduce padding")
    targets = batch["labels"][:, 1:].ne(-100).sum(dim=1).tolist()
    if len(set(targets)) != 1 or targets[0] <= 0:
        raise RuntimeError("Expected the same nonzero target count in every fixture row")
    first_batch_hash = hashlib.sha256()
    for key in ("input_ids", "labels"):
        first_batch_hash.update(key.encode() + batch[key].contiguous().numpy().tobytes())
    module_counts = Counter(type(m).__name__ for m in wrapper.model.modules())
    module_summary = {key: value for key, value in module_counts.items()
                      if any(word in key for word in ("Liger", "RMSNorm", "MLP", "Attention"))}
    if bool(module_counts["LigerRMSNorm"]) != liger or bool(
        module_counts["LigerSwiGLUMLP"]
    ) != liger:
        raise RuntimeError(
            f"Liger module activation does not match the requested arm: {module_summary}"
        )
    actual_attn = wrapper.model.config._attn_implementation
    if actual_attn != ("flash_attention_2" if fa2 else "sdpa"):
        raise RuntimeError(f"Unexpected actual attention implementation: {actual_attn}")
    if counters is not None:
        for module in wrapper.model.modules():
            name = type(module).__name__
            if name in ("LigerRMSNorm", "LigerSwiGLUMLP"):
                module.register_forward_hook(
                    lambda module, inputs, output, name=name: counters.update([name])
                )
    loss_values = []
    original_loss = trainer.compute_loss

    def recording_loss(*call_args, **call_kwargs):
        value = original_loss(*call_args, **call_kwargs)
        loss_values.append((value[0] if isinstance(value, tuple) else value).detach())
        return value

    trainer.compute_loss = recording_loss
    samples, telemetry = [], [before, gpu_sample()]

    class Measurement(TrainerCallback):
        def on_step_begin(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()

        def on_step_end(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.started
            samples.append({"step": state.global_step, "seconds": elapsed,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            if state.global_step in (args.warmup, args.warmup + args.steps):
                telemetry.append(gpu_sample())

    trainer.add_callback(Measurement())
    profile_events = []
    started = time.perf_counter()
    if args.profile:
        with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA,
        ]) as profiler:
            trainer.train()
        profile_events = [{"name": row.key, "device_time_us": row.device_time_total,
                           "calls": row.count} for row in profiler.key_averages()
                          if row.device_time_total > 0]
        profile_events.sort(key=lambda row: row["device_time_us"], reverse=True)
        profile_events = profile_events[:40]
    else:
        trainer.train()
    trainer_seconds = time.perf_counter() - started
    if counters is not None:
        if fa2:
            assert counters["fa2"] > 0 and counters["sdpa"] == 0, dict(counters)
        else:
            assert counters["sdpa"] > 0 and counters["fa2"] == 0, dict(counters)
        if liger:
            assert counters["LigerRMSNorm"] > 0 and counters["LigerSwiGLUMLP"] > 0
    losses = [float(value.cpu()) for value in loss_values]
    if len(samples) != args.steps + args.warmup or not all(
        __import__("math").isfinite(value) for value in losses
    ):
        raise RuntimeError("Incomplete steps or non-finite loss")
    measured = samples[args.warmup:]
    seconds = sum(row["seconds"] for row in measured)
    return {
        "status": "ok", "arm": args.arm, "sequence": args.sequence,
        "checkpoint": args.checkpoint, "probe": args.probe,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"), "model_snapshot": Path(args.model).name,
        "tokenized_data_sha256": workload_hash.hexdigest(),
        "first_batch_sha256": first_batch_hash.hexdigest(),
        "realized_batch_shape": realized,
        "supervised_targets_per_row": targets[0],
        "module_counts": module_summary, "attention_implementation": actual_attn,
        "activation_counts": None if counters is None else dict(counters),
        "profile_top_events": profile_events,
        "dtype_counts": dict(Counter(str(p.dtype) for p in wrapper.model.parameters())),
        "trainable_parameters": sum(
            p.numel() for p in wrapper.model.parameters() if p.requires_grad
        ),
        "runtime": {name: package_version(name) for name in (
            "torch", "transformers", "peft", "trl", "triton", "liger-kernel", "flash-attn",
        )},
        "cuda": torch.version.cuda, "warmup_steps": args.warmup, "measured_steps": args.steps,
        "steps": samples, "losses": losses, "telemetry": telemetry,
        "trainer_total_seconds": trainer_seconds,
        "summary": {
            "seconds_per_step": seconds / args.steps,
            "median_seconds_per_step": statistics.median(row["seconds"] for row in measured),
            "input_tokens_per_second": 4 * args.sequence * args.steps / seconds,
            "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in samples),
            "steady_peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in measured),
            "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in samples),
        },
    }


def main() -> None:
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args)
    except Exception as exc:
        import torch

        result = {"status": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                  "arm": args.arm, "sequence": args.sequence, "checkpoint": args.checkpoint,
                  "probe": args.probe, "error": str(exc), "traceback": traceback.format_exc()}
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        console.print(result["status"], result["error"])
        raise SystemExit(2 if result["status"] == "oom" else 1) from exc
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    console.print(result["summary"])


if __name__ == "__main__":
    main()
