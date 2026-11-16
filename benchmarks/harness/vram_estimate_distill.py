#!/usr/bin/env python3
"""Peak VRAM of resident ``task: distill`` steps, for the estimate's teacher and KD terms.

#ISSUE: the VRAM estimate (``utils/vram_estimate.py``) priced a token-level distillation
run as a plain LoRA step on the student. The frozen teacher, its logits and the fp32
divergence over the vocabulary were not counted. This harness measures what a distill
step really holds: one real ``soup train`` setup (``DistillTrainerWrapper.setup``) and
``STEPS`` training steps per row, with ``torch.cuda.max_memory_allocated()`` read over
them, the same protocol as ``vram_estimate_calibration.py`` (#2049).

Each row runs in a fresh process; the parent only spawns and collects.

Usage
-----
    python benchmarks/harness/vram_estimate_distill.py --rows rows.json --out out.jsonl
    python benchmarks/harness/vram_estimate_distill.py --row '{"student": ..., ...}'

A row is a JSON object::

    {"student": "<local model dir>", "teacher": "<local model dir>", "seq": 512,
     "batch": 1, "kernel": "dense" | "checkpoint", "chunk": 128,
     "divergence": "forward_kl", "gradient_checkpointing": true, "lora_r": 8}

``kernel: dense`` is the shipped default (``distill_chunk_size`` unset,
``distill_checkpoint: false``); ``checkpoint`` sets ``distill_checkpoint: true`` and
``distill_chunk_size: chunk``. The two models must share a tokenizer (token-level KD).

What each result line holds
---------------------------
- ``realised`` / ``realised_batch``: the collated batch's shape; a row whose shape is
  not ``seq`` x ``batch`` is void.
- ``peak_allocated`` / ``peak_reserved``: over the steps, after ``reset_peak_memory_stats``.
- ``before_step_allocated``: student, adapters and teacher, before the first step.
- ``supervised``: the causal targets (``labels[1:] != -100``) per row.
- ``student`` / ``teacher``: the ``config.json`` fields the estimate reads and the exact
  parameter count from the safetensors headers.
- ``estimate``: Soup's estimate of the run (``soup_cli.utils.hardware_fit``), by term.
- ``stack``: package versions and the device name.

Exit status: 0 when every row produced a result line, 1 otherwise.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import yaml

#: Overshoot factor of each row over the requested length (truncation lands on it).
OVERSHOOT = 3

#: Training steps per row (AdamW's state exists only from the first step on).
STEPS = 2

RESULT_PREFIX = "VRAM_DISTILL_RESULT "


def _calibration_helpers() -> Any:
    """``model_facts`` and ``stack`` from the #2049 harness next to this file."""
    import importlib.util

    path = pathlib.Path(__file__).with_name("vram_estimate_calibration.py")
    spec = importlib.util.spec_from_file_location("vram_estimate_calibration", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config_doc(row: dict, work_dir: pathlib.Path) -> dict:
    lora_r = int(row.get("lora_r", 8))
    checkpointed = row.get("kernel", "dense") == "checkpoint"
    training: dict[str, Any] = {
        "teacher_model": row["teacher"],
        "distill_divergence": row.get("divergence", "forward_kl"),
        "distill_temperature": 2.0,
        "distill_chunk_size": int(row.get("chunk", 128)) if checkpointed else None,
        "distill_checkpoint": checkpointed,
        "batch_size": int(row["batch"]),
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "quantization": "none",
        "gradient_checkpointing": row.get("gradient_checkpointing", True),
        "optimizer": "adamw_torch",
        "logging_steps": 1,
        "save_steps": 100000,
        "lora": {
            "r": lora_r,
            "alpha": max(lora_r * 2, 1),
            "target_modules": row.get("target_modules", ["q_proj", "v_proj"]),
        },
    }
    return {
        "base": row["student"],
        "task": "distill",
        "backend": "transformers",
        "data": {
            "train": str(work_dir / "d.jsonl"),
            "max_length": int(row["seq"]),
            "chat_template": "chatml",
            "val_split": 0.0,
        },
        "output": str(work_dir / "out"),
        "training": training,
    }


def rows_for(seq: int, batch: int) -> list[dict]:
    body = "word " * (seq * OVERSHOOT)
    message = {"role": "assistant", "content": body}
    return [
        {"messages": [{"role": "user", "content": "hi"}, message]}
        for _ in range(batch * STEPS)
    ]


def estimate_terms(cfg: Any) -> dict:
    from soup_cli.utils.hardware_fit import estimate_peak_vram_gb, hardware_fit_input_or_reason

    inp = hardware_fit_input_or_reason(cfg, allow_network=False)
    if not hasattr(inp, "seq_len"):
        return {"unavailable": inp.reason}
    est = estimate_peak_vram_gb(inp).estimate
    return {"terms": dict(est.terms), "total": est.total_bytes}


def run_row(row: dict) -> dict:
    """One row in this process: set up through Soup, run the steps, read the peak."""
    import torch

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.distill import DistillTrainerWrapper

    helpers = _calibration_helpers()
    fraction = os.environ.get("SOUP_HARNESS_MEMORY_FRACTION")
    if fraction:
        torch.cuda.set_per_process_memory_fraction(float(fraction))
    result: dict[str, Any] = {
        "row": row,
        "stack": helpers.stack(),
        "student": helpers.model_facts(row["student"]),
        "teacher": helpers.model_facts(row["teacher"]),
    }
    free, total = torch.cuda.mem_get_info()
    result["free_before"], result["total"] = int(free), int(total)
    with tempfile.TemporaryDirectory(prefix="vram_estimate_distill_") as tmp:
        work_dir = pathlib.Path(tmp)
        data = rows_for(int(row["seq"]), int(row["batch"]))
        (work_dir / "d.jsonl").write_text(
            "\n".join(json.dumps(r) for r in data), encoding="utf-8"
        )
        (work_dir / "out").mkdir()
        cfg = load_config_from_string(yaml.safe_dump(config_doc(row, work_dir)))
        result["estimate"] = estimate_terms(cfg)
        wrapper = DistillTrainerWrapper(cfg, device="cuda")
        wrapper.setup({"train": data})
        trainer = wrapper.trainer
        batch = next(iter(trainer.get_train_dataloader()))
        result["realised"] = int(batch["input_ids"].shape[-1])
        result["realised_batch"] = int(batch["input_ids"].shape[0])
        result["supervised"] = [
            int((labels[1:] != -100).sum()) for labels in batch["labels"]
        ]
        trainer.args.max_steps = STEPS
        result["steps"] = STEPS
        gc.collect()
        torch.cuda.empty_cache()
        result["before_step_allocated"] = int(torch.cuda.memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        try:
            result["loss"] = float(trainer.train().training_loss)
            result["peak_allocated"] = int(torch.cuda.max_memory_allocated())
            result["peak_reserved"] = int(torch.cuda.max_memory_reserved())
            result["oom"] = False
        except torch.cuda.OutOfMemoryError:
            result["oom"] = True
            result["peak_allocated"] = None
            result["peak_reserved"] = None
    result["void"] = result["realised"] != int(row["seq"]) or result["realised_batch"] != int(
        row["batch"]
    )
    return result


def run_all(rows: list[dict], out: pathlib.Path) -> int:
    failures = 0
    with out.open("a", encoding="utf-8") as sink:
        for index, row in enumerate(rows):
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--row", json.dumps(row)],
                capture_output=True,
                text=True,
            )
            lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)]
            if proc.returncode != 0 or not lines:
                failures += 1
                tail = (proc.stdout + proc.stderr)[-2000:]
                print(f"[{index}] FAILED exit {proc.returncode}: {json.dumps(row)}\n{tail}")
                sink.write(
                    json.dumps({"row": row, "error": f"exit {proc.returncode}", "tail": tail})
                    + "\n"
                )
                continue
            result = json.loads(lines[-1][len(RESULT_PREFIX):])
            sink.write(json.dumps(result) + "\n")
            sink.flush()
            peak = result.get("peak_allocated")
            shown = "OOM" if result.get("oom") else f"{peak / 1e9:.3f} GB"
            void = " VOID" if result.get("void") else ""
            print(f"[{index}] {shown}{void}  {json.dumps(row)}", flush=True)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--row", help="run one row (a JSON object) in this process")
    parser.add_argument("--rows", help="JSON file with a list of rows, each run in its own process")
    parser.add_argument("--out", help="JSON Lines file the results are appended to")
    args = parser.parse_args(argv)
    if args.row:
        result = run_row(json.loads(args.row))
        print(RESULT_PREFIX + json.dumps(result), flush=True)
        return 0
    if not (args.rows and args.out):
        parser.error("--rows and --out are required unless --row is given")
    rows = json.loads(pathlib.Path(args.rows).read_text(encoding="utf-8"))
    return run_all(rows, pathlib.Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
