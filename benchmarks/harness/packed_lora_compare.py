#!/usr/bin/env python3
"""Run ``soup train`` on one recipe with or without ``training.packed_lora`` and meter the run.

Two invocations of this script on the same ``--data``, ``--seed`` and recipe, one without
``--packed`` (plain Soup) and one with it, give the rows of
``benchmarks/packed-lora-wired-2026-11-08.md``: the loss of every optimizer step, the tokens per
second over the steps after ``--warmup``, and torch's peak allocated memory.

    python benchmarks/harness/packed_lora_compare.py --out /tmp/plain \
        --model Qwen/Qwen2.5-1.5B-Instruct --data rows.jsonl --batch 2 --accum 4 --steps 105
    python benchmarks/harness/packed_lora_compare.py --out /tmp/packed \
        --model Qwen/Qwen2.5-1.5B-Instruct --data rows.jsonl --batch 8 --accum 1 --steps 105 \
        --packed '{enabled: true}'

The recipe (the "case-1" recipe): chatml rows, ``max_length`` 512, LoRA r 16 / alpha 32 /
dropout 0 on all seven projections, lr 2e-4 constant, ``adamw_torch``, gradient clipping 1.0,
4-bit NF4, ``gradient_checkpointing: true``, seed 3407. ``--batch B --accum A`` reads
B x A rows per optimizer step in the same order whatever B and A are, so a card that cannot hold a
plain batch of 8 can still be compared with a packed batch of 8. ``rows.jsonl`` is one
``{"messages": [...]}`` per line; ``--steps`` x ``--batch`` x ``--accum`` rows are read.

The meter is a ``TrainerCallback`` that only reads: wall time of each optimizer step (CUDA
synchronised at both ends), the non-pad tokens of its micro-batches, and
``torch.cuda.max_memory_allocated()``. Requires CUDA. Prints one ``SUMMARY {json}`` line and writes
``steps.jsonl`` and ``log.jsonl`` into ``--out``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

RECIPE = """\
base: {model}
task: sft
data:
  train: {data}
  format: chatml
  train_on_responses_only: false
  val_split: 0.0
  max_length: {max_length}
training:
{packed}  epochs: 1
  lr: 0.0002
  batch_size: {batch}
  gradient_accumulation_steps: {accum}
  seed: {seed}
  data_seed: {seed}
  scheduler: constant
  warmup_ratio: 0.0
  weight_decay: 0.0
  max_grad_norm: 1.0
  optimizer: adamw_torch
  gradient_checkpointing: true
  quantization: 4bit
  logging_steps: 1
  save_steps: 1000000
  lora:
    r: 16
    alpha: 32
    dropout: 0.0
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
output: {out}/soup_out
"""


def install_meter(out_dir: str, warmup: int, stop_after: int) -> None:
    """Add the meter to every ``transformers.Trainer`` built from now on."""
    import torch
    import transformers

    steps_path = os.path.join(out_dir, "steps.jsonl")
    log_path = os.path.join(out_dir, "log.jsonl")

    class Meter(transformers.TrainerCallback):
        def __init__(self):
            self.begin = 0.0
            self.tokens = 0

        def on_step_begin(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            self.begin, self.tokens = time.perf_counter(), 0

        def on_step_end(self, args, state, control, **kwargs):
            torch.cuda.synchronize()
            end = time.perf_counter()
            with open(steps_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "step": state.global_step, "seconds": end - self.begin,
                    "alloc_peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "reserved_peak_mib": torch.cuda.max_memory_reserved() / 2**20}) + "\n")
            if stop_after and state.global_step >= stop_after:
                control.should_training_stop = True

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                with open(log_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    original_init = transformers.Trainer.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Meter())

    transformers.Trainer.__init__ = init


def summarise(out_dir: str, warmup: int) -> dict:
    def read(name):
        path = os.path.join(out_dir, name)
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    steps, logs = read("steps.jsonl"), read("log.jsonl")
    summary = {"steps_done": len(steps), "loss": [round(e["loss"], 4) for e in logs],
               "grad_norm": [round(e["grad_norm"], 3) for e in logs if "grad_norm" in e]}
    measured = [s for s in steps if s["step"] > warmup]
    tokens = {e["step"]: e.get("num_tokens") for e in logs}
    if measured and all(tokens.get(s["step"]) for s in steps):
        first = measured[0]["step"]
        done = tokens[measured[-1]["step"]] - tokens[first - 1] if first - 1 in tokens else None
        seconds = sum(s["seconds"] for s in measured)
        if done:
            summary.update(measured_steps=len(measured), tokens_per_s=round(done / seconds, 1),
                           step_s_median=round(sorted(s["seconds"] for s in measured)[
                               len(measured) // 2], 3))
    if steps:
        summary.update(alloc_peak_mib=round(steps[-1]["alloc_peak_mib"], 1),
                       reserved_peak_mib=round(steps[-1]["reserved_peak_mib"], 1),
                       step1_s=round(steps[0]["seconds"], 2))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True,
                        help="chatml rows, one {\"messages\": [...]} a line")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--steps", type=int, default=105, help="optimizer steps, warm-up included")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stop-after", type=int, default=0,
                        help="end the run after this many steps (the data are those of --steps)")
    parser.add_argument("--allow-oom-attempt", action="store_true",
                        help="pass --allow-oom-attempt to soup train (its hardware-fit gate)")
    parser.add_argument("--packed", default=None,
                        help="YAML flow mapping for training.packed_lora, e.g. '{enabled: true}'")
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError:
        print("packed_lora_compare: needs torch with CUDA; skipping", file=sys.stderr)
        return 0
    if not torch.cuda.is_available():
        print("packed_lora_compare: no CUDA device; skipping", file=sys.stderr)
        return 0

    os.makedirs(args.out, exist_ok=True)
    rows = args.steps * args.batch * args.accum
    sliced = os.path.join(args.out, "train_rows.jsonl")
    with open(args.data, encoding="utf-8") as src, open(sliced, "w", encoding="utf-8") as dst:
        written = 0
        for line in src:
            if written == rows:
                break
            if line.strip():
                dst.write(line if line.endswith("\n") else line + "\n")
                written += 1
    if written < rows:
        print(f"packed_lora_compare: {args.data} has {written} rows, {rows} needed",
              file=sys.stderr)
        return 2
    recipe = RECIPE.format(
        model=args.model, data=sliced, max_length=args.max_length, batch=args.batch,
        accum=args.accum, seed=args.seed, out=args.out,
        packed=f"  packed_lora: {args.packed}\n" if args.packed else "")
    config = os.path.join(args.out, "soup.yaml")
    with open(config, "w", encoding="utf-8") as handle:
        handle.write(recipe)

    install_meter(args.out, args.warmup, args.stop_after)
    from soup_cli.cli import app

    sys.argv = ["soup", "train", "--config", config, "--yes"]
    if args.allow_oom_attempt:
        sys.argv.append("--allow-oom-attempt")
    code = 0
    try:
        app()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    print("SUMMARY " + json.dumps({"packed": args.packed, "batch": args.batch,
                                   "accum": args.accum, **summarise(args.out, args.warmup)}))
    return code


if __name__ == "__main__":
    sys.exit(main())
