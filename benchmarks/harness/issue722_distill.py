"""Measure the shipped dense/chunked distillation paths on the same real models.

Run from a Soup checkout with its training dependencies installed. Each invocation
is one arm in a fresh process; see the accompanying record for the paired order.
Downloads/setup and final adapter serialization are outside the step window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from rich.console import Console

console = Console()


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def gpu_sample() -> dict:
    """Whole-device telemetry is context, never attributed to this process."""
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    fields = "uuid,name,memory.used,utilization.gpu,clocks.sm,power.draw,temperature.gpu"
    result = subprocess.run(
        ["nvidia-smi", "-i", selector, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return dict(zip(fields.split(","), result.stdout.strip().split(", "), strict=True))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="Pinned local HF snapshot directory")
    parser.add_argument("--teacher", required=True, help="Pinned local HF snapshot directory")
    parser.add_argument("--arm", choices=("dense", "chunked"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=2048)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=722)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if min(args.batch, args.sequence, args.chunk, args.warmup, args.steps) < 1:
        raise ValueError("Batch, sequence, chunk, warmup and measured steps must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    started = time.perf_counter()

    import numpy as np
    import torch
    from transformers import AutoTokenizer, TrainerCallback

    from soup_cli.config.schema import SoupConfig
    from soup_cli.trainer.distill import DistillTrainerWrapper

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    # This is a throughput fixture, not a held-out model-quality corpus.
    rows = [
        {"messages": [
            {"role": "user", "content": "Explain how a telescope works. " * 24},
            {"role": "assistant", "content": (
                "A telescope collects light using a lens or a mirror. "
                "Its aperture sets the light-gathering area and angular resolution. "
                "An eyepiece magnifies the image formed at the focal plane. "
            ) * args.sequence},
        ]}
        for _ in range(args.batch * (args.warmup + args.steps))
    ]
    student_tokenizer = AutoTokenizer.from_pretrained(args.student, local_files_only=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher, local_files_only=True)
    if student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise RuntimeError("The teacher and student must have identical token-ID vocabularies")
    config = SoupConfig.model_validate({
        "base": args.student, "task": "distill", "backend": "transformers",
        "data": {"train": "benchmark-generated-in-memory", "max_length": args.sequence},
        "output": str(args.output / "adapter"),
        "training": {
            "teacher_model": args.teacher,
            "distill_divergence": "forward_kl", "distill_temperature": 2.0,
            "distill_chunk_size": args.chunk if args.arm == "chunked" else None,
            "distill_checkpoint": args.arm == "chunked",
            "batch_size": args.batch, "gradient_accumulation_steps": 1,
            "epochs": 1, "gradient_checkpointing": True,
            "quantization": "none", "seed": args.seed, "data_seed": args.seed,
            "lr": 2e-5, "warmup_ratio": 0.0, "optimizer": "adamw_torch",
            "scheduler": "constant", "logging_steps": 1000, "save_steps": 100000,
            "lora": {"r": 8, "alpha": 16, "dropout": 0.0,
                     "target_modules": ["q_proj", "v_proj"]},
        },
    })
    wrapper = DistillTrainerWrapper(config, device="cuda", report_to="none")
    wrapper.setup({"train": rows})
    trainer = wrapper.trainer
    trainer.args.max_steps = args.warmup + args.steps
    trainer.args.save_strategy = "no"
    trainer.args.disable_tqdm = True
    trainer.args.dataloader_num_workers = 0
    trainer.args.dataloader_pin_memory = True
    trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    wrapper.model.config.use_cache = False
    wrapper.teacher.config.use_cache = False
    dataset = list(trainer.train_dataset)
    lengths = [len(row["input_ids"]) for row in dataset]
    if set(lengths) != {args.sequence}:
        raise RuntimeError(f"Expected exact sequence length {args.sequence}, got {set(lengths)}")
    active_per_row = [sum(t != -100 for t in row["labels"][1:]) for row in dataset]
    if min(active_per_row) <= 0:
        raise RuntimeError("Every row must retain supervised causal targets")
    trainable = [(n, p) for n, p in wrapper.model.named_parameters() if p.requires_grad]
    initial = {n: p.detach().float().cpu().clone().numpy() for n, p in trainable}
    initial_hash = hashlib.sha256(b"".join(initial[n].tobytes() for n in initial)).hexdigest()
    loss_values = []
    original_compute_loss = trainer.compute_loss

    def recording_loss(*call_args, **call_kwargs):
        result = original_compute_loss(*call_args, **call_kwargs)
        # Scalar detachment does not synchronize CUDA; values are read after training.
        loss_values.append((result[0] if isinstance(result, tuple) else result).detach())
        return result

    trainer.compute_loss = recording_loss
    samples = []
    telemetry = [gpu_sample()]
    first_gradients = {}

    class Measurement(TrainerCallback):
        def on_step_begin(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.baseline = torch.cuda.memory_allocated()
            self.started = time.perf_counter()

        def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
            if state.global_step == 0:
                for name, parameter in trainable:
                    if parameter.grad is None:
                        raise RuntimeError(f"Missing gradient: {name}")
                    first_gradients[name] = parameter.grad.detach().float().cpu().numpy().copy()

        def on_step_end(self, training_args, state, control, **kwargs):
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self.started
            samples.append({
                "step": state.global_step, "seconds": elapsed,
                "start_allocated_bytes": self.baseline,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            })
            if state.global_step == args.warmup or state.global_step == args.warmup + args.steps:
                telemetry.append(gpu_sample())

    trainer.add_callback(Measurement())
    setup_seconds = time.perf_counter() - started
    train_started = time.perf_counter()
    # The actual Soup-created Trainer performs forward, CE/KD, backward and AdamW.
    trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started
    final = {n: p.detach().float().cpu().numpy().copy() for n, p in trainable}
    if set(first_gradients) != set(initial):
        raise RuntimeError("First-step gradient capture must cover every trainable parameter")
    np.savez_compressed(args.output / "initial-adapter.npz", **initial)
    np.savez_compressed(args.output / "first-gradients.npz", **first_gradients)
    np.savez_compressed(args.output / "final-adapter.npz", **final)
    losses = [float(value.cpu()) for value in loss_values]
    if len(samples) != args.warmup + args.steps or len(losses) != len(samples):
        raise RuntimeError("Unexpected training step/loss count")
    if not all(np.isfinite(losses)) or not all(
        np.isfinite(value).all() for value in (*first_gradients.values(), *final.values())
    ):
        raise RuntimeError("Non-finite loss, gradient or adapter")
    measured = samples[args.warmup:]
    seconds = sum(s["seconds"] for s in measured)
    record = {
        "schema": 1, "issue": 722, "arm": args.arm,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config": config.model_dump(mode="json"),
        "student_snapshot": Path(args.student).name, "teacher_snapshot": Path(args.teacher).name,
        "tokenized_data_sha256": digest(dataset), "initial_adapter_sha256": initial_hash,
        "shape": {"batch": args.batch, "sequence": args.sequence,
                  "vocabulary": wrapper.model.config.vocab_size,
                  "active_targets_per_row": dict(Counter(active_per_row))},
        "runtime": {"python": __import__("platform").python_version(),
                    "torch": torch.__version__, "cuda": torch.version.cuda,
                    **{name: version(name) for name in (
                        "transformers", "peft", "accelerate", "datasets", "soup-cli")}},
        "dtype": {"student": dict(Counter(str(p.dtype) for p in wrapper.model.parameters())),
                  "teacher": dict(Counter(str(p.dtype) for p in wrapper.teacher.parameters())),
                  "trainable": dict(Counter(str(p.dtype) for _, p in trainable)),
                  "autocast_bf16": trainer.args.bf16},
        "setup_seconds": setup_seconds, "trainer_total_seconds": training_seconds,
        "warmup_steps": args.warmup, "measured_steps": args.steps,
        "steps": samples, "losses": losses, "telemetry": telemetry,
        "summary": {
            "seconds_per_step": seconds / args.steps,
            "median_seconds_per_step": statistics.median(s["seconds"] for s in measured),
            "input_tokens_per_second": args.batch * args.sequence * args.steps / seconds,
            "supervised_tokens_per_second": (
                args.batch * active_per_row[0] * args.steps / seconds
            ),
            "peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in samples),
            "steady_peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in measured),
            "peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in samples),
        },
        "timing_scope": "on_step_begin to on_step_end, CUDA synchronized; includes optimizer; "
                        "excludes data-loader fetch before on_step_begin, setup and adapter save; "
                        "first warmup step also includes gradient capture, never timed as steady",
    }
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    console.print(record["summary"])


if __name__ == "__main__":
    main()
