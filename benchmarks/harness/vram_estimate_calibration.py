#!/usr/bin/env python3
"""Peak VRAM of resident (not streamed) sft steps, for calibrating ``utils/vram_estimate.py``.

#2049: the train pre-flight and ``soup profile`` predicted different peaks for the
same run, and the gate records Soup already has are almost all streamed runs, which
the resident pre-flight never sees. This harness measures the resident side: one
real ``soup train`` setup (``SFTTrainerWrapper.setup``) and ``STEPS`` training steps
per row, with ``torch.cuda.max_memory_allocated()`` read over them.

Two steps, not one: AdamW creates its state inside the first ``optimizer.step()``,
after that step's backward pass, so only from the second step on do the optimizer
state, the gradients and the activations share the card. A one-step reading misses
that peak for full fine-tuning.

Each row runs in a fresh process. #1693 showed that arms built one after another
in the same process do not start from the same state (pinned host memory from the
first arm was still held by the second), so the parent only spawns and collects.

Usage
-----
    python benchmarks/harness/vram_estimate_calibration.py --rows rows.json --out out.jsonl
    python benchmarks/harness/vram_estimate_calibration.py --row '{"base": ..., ...}'

A row is a JSON object::

    {"base": "<local model dir>", "quantization": "none" | "4bit",
     "lora_r": 8, "target_modules": "auto", "seq": 512, "batch": 1,
     "gradient_checkpointing": false, "optimizer": "adamw_torch"}

``lora_r: 0`` is full fine-tuning. ``base`` should be a local snapshot directory,
so the record can name the revision and the estimator can read its config offline.

What each result line holds
---------------------------
- ``realised``: the sequence length of the collated batch. ``data.max_length``
  truncates rather than pads, so every row overshoots it and the realised length
  is read back; a row whose realised length is not ``seq`` is void.
- ``peak_allocated`` / ``peak_reserved``: ``max_memory_allocated`` and
  ``max_memory_reserved`` over the steps, after ``reset_peak_memory_stats``.
- ``before_step_allocated``: ``memory_allocated`` just before the first step (the
  loaded model and adapters; the optimizer state does not exist yet).
- ``free_before`` / ``total``: ``mem_get_info`` before the model is loaded.
- ``oom``: the step ran out of memory (then there is no peak).
- ``model``: ``config.json`` fields the estimator reads, and the exact parameter
  count from the safetensors headers.
- ``stack``: torch / transformers / trl / peft / bitsandbytes versions, device name.

Exit status: 0 when every row produced a result line, 1 otherwise.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as md
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

#: Training steps per row; the peak is read over all of them (see the docstring).
STEPS = 2

#: The config.json fields recorded with each row.
CONFIG_FIELDS = (
    "model_type",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "tie_word_embeddings",
    "torch_dtype",
    "dtype",
)

RESULT_PREFIX = "VRAM_ROW_RESULT "


def _exact_param_count(model_dir: pathlib.Path) -> int | None:
    """Parameter count from the safetensors headers alone (no weights read)."""
    from soup_cli.utils.checkpoint_files import checkpoint_files

    total = 0
    selected = checkpoint_files(str(model_dir))
    if not selected.weights:
        return None
    for name in selected.weights:
        with open(model_dir / name, "rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            count = 1
            for dim in meta["shape"]:
                count *= int(dim)
            total += count
    return total


def model_facts(base: str) -> dict:
    path = pathlib.Path(base)
    facts: dict[str, Any] = {"base": base}
    config_path = path / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
        for field in CONFIG_FIELDS:
            if field in config:
                facts[field] = config[field]
            elif field in text:
                facts[field] = text[field]
        facts["params"] = _exact_param_count(path)
    return facts


def stack() -> dict:
    import torch

    out = {}
    for pkg in ("torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    out["cuda"] = torch.version.cuda
    out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


def config_doc(row: dict, work_dir: pathlib.Path) -> dict:
    lora_r = int(row.get("lora_r", 8))
    training: dict[str, Any] = {
        "batch_size": int(row["batch"]),
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "quantization": row.get("quantization", "none"),
        "gradient_checkpointing": row.get("gradient_checkpointing", False),
        "optimizer": row.get("optimizer", "adamw_torch"),
        "logging_steps": 1,
        "save_steps": 100000,
        "lora": {
            "r": lora_r,
            "alpha": max(lora_r * 2, 1),
            "target_modules": row.get("target_modules", "auto"),
        },
    }
    return {
        "base": row["base"],
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
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


def run_row(row: dict) -> dict:
    """One row in this process: set up through Soup, run one step, read the peak."""
    import torch

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    result: dict[str, Any] = {"row": row, "stack": stack(), "model": model_facts(row["base"])}
    free, total = torch.cuda.mem_get_info()
    result["free_before"], result["total"] = int(free), int(total)
    with tempfile.TemporaryDirectory(prefix="vram_estimate_calibration_") as tmp:
        work_dir = pathlib.Path(tmp)
        data = rows_for(int(row["seq"]), int(row["batch"]))
        (work_dir / "d.jsonl").write_text(
            "\n".join(json.dumps(r) for r in data), encoding="utf-8"
        )
        (work_dir / "out").mkdir()
        cfg = load_config_from_string(yaml.safe_dump(config_doc(row, work_dir)))
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        wrapper.setup({"train": data})
        trainer = wrapper.trainer
        batch = next(iter(trainer.get_train_dataloader()))
        result["realised"] = int(batch["input_ids"].shape[-1])
        result["realised_batch"] = int(batch["input_ids"].shape[0])
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
