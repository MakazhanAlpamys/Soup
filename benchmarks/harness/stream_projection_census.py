"""Untimed native-auto memory census for the #2162 feasibility stage.

Reads an existing model and shard cache. No optimizer, downloads or service changes.
One fresh process per shape; a driver error anywhere in the inventory refuses the
run before importing torch. This is not a throughput measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console

GPU_FIELDS = "index,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.limit"


def check_inventory(stdout: str, stderr: str, returncode: int, gpu: str, cap: int) -> dict:
    if returncode or stderr.strip() or "GPU is lost" in stdout:
        raise RuntimeError("GPU inventory is unhealthy; no census started")
    rows = list(csv.reader(io.StringIO(stdout)))
    candidates = [row for row in rows if len(row) == 7 and row[1].strip() == gpu]
    if len(candidates) != 1:
        raise RuntimeError("selected GPU is missing or ambiguous")
    row = [value.strip() for value in candidates[0]]
    free = (int(row[2]) - int(row[3])) * 2**20
    if free < cap + 2 * 2**30 or int(row[4]) > 5 or int(row[5]) > 60:
        raise RuntimeError("selected GPU lacks idle memory/temperature headroom")
    return dict(uuid=row[1], free_bytes=free, temperature=int(row[5]), power_limit=float(row[6]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seq", type=int, choices=(2048, 4096), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cap = 8 * 2**30
    if args.out.exists():
        raise FileExistsError("preserve the existing census; use a new output path")
    if not (args.weights / "config.json").is_file() or not (args.shards / "index.json").is_file():
        raise FileNotFoundError("existing offline weights and shard index are required")
    inventory = subprocess.run(
        ["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    )
    admission = check_inventory(
        inventory.stdout, inventory.stderr, inventory.returncode, args.gpu, cap
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if "torch" in sys.modules:
        raise RuntimeError("CUDA selection must precede torch import")
    import torch
    from peft import LoraConfig, TaskType
    from stream_projection_retention import observe_auto_plan, projection_plan

    import soup_cli
    from soup_cli.utils.layer_shard import read_shard_index
    from soup_cli.utils.layer_stream_runtime import build_streamed_model, streamed_decoder_layers

    source_root = os.path.normcase(os.path.realpath(Path(__file__).parents[2] / "src"))
    imported = os.path.normcase(os.path.realpath(soup_cli.__file__))
    if os.path.commonpath([source_root, imported]) != source_root:
        raise RuntimeError("soup_cli came from another checkout; set PYTHONPATH to this src")
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(cap / total_memory)
    console = Console()
    messages = []
    config = json.loads((args.weights / "config.json").read_text())
    if config["model_type"] != "mistral" or config["num_hidden_layers"] != 32:
        raise ValueError("this census is fixed to the existing 32-layer Mistral model")
    index = read_shard_index(str(args.shards))
    model, runtime = build_streamed_model(
        model_id=str(args.weights), shard_dir=str(args.shards), index=index,
        lora_config=LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type=TaskType.CAUSAL_LM,
        ),
        device="cuda", dtype="bfloat16", buffers=2, pin=True, seed=42,
        quant="nf4", double_quant=True, tier="ram",
    )
    try:
        if not runtime.pinned or runtime.tier != "ram":
            raise RuntimeError("census requires the planned pinned RAM source")
        model.train()
        generator = torch.Generator(device="cuda").manual_seed(7)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and "lora_B" in name:
                    param.copy_(torch.randn(
                        param.shape, generator=generator, device=param.device, dtype=param.dtype,
                    ) * 0.01)
        native = observe_auto_plan(
            model, rows=1, seq_len=args.seq, vocab_size=config["vocab_size"],
            override_bytes=cap, notify=messages.append,
        )
        plan = projection_plan(
            streamed_decoder_layers(model), native, batch=1, seq=args.seq, element_bytes=2,
        )
        result: dict[str, Any] = dict(
            stage="untimed-native-census", written_utc=datetime.now(timezone.utc).isoformat(),
            shape=[1, args.seq], cap_bytes=cap, admission=admission,
            allocator_cap_bytes=cap,
            native=native, projections=plan, messages=messages, optimizer_updates=0,
            attention_implementation=model.config._attn_implementation,
            kernel_profiled=False, deterministic_algorithms=True,
            shard_index_sha256=hashlib.sha256(
                (args.shards / "index.json").read_bytes()
            ).hexdigest(),
            packages={name: importlib.metadata.version(name) for name in (
                "torch", "transformers", "peft", "bitsandbytes",
            )},
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        console.print(result)
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
