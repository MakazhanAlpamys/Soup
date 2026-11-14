"""Real quant-check performance smokes: pinned Qwen FP16/NF4 and a tiny CPU control.

Run with one visible CUDA GPU, from the checkout root. Weights and outputs stay
under the ignored .codex directory. These short toy suites verify the reporting
path, not model quality or a reproducible inference speedup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def prepare(root: Path) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LlamaConfig,
        LlamaForCausalLM,
        PreTrainedTokenizerFast,
    )

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Select exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    snapshot = snapshot_download(
        "Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
    )
    before, after = root / "qwen-fp16", root / "qwen-nf4"
    shutil.copytree(snapshot, before, copy_function=shutil.copyfile)
    tokenizer = AutoTokenizer.from_pretrained(before, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        before, device_map={"": "cuda:0"}, dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4",
        ),
    )
    model.save_pretrained(after)
    tokenizer.save_pretrained(after)
    del model
    torch.cuda.empty_cache()
    questions = [
        ("What is 2 + 2? Answer with just the number.", "4"),
        ("What is the capital of France? Answer with one word.", "Paris"),
        ("What color is a clear daytime sky? Answer with one word.", "blue"),
    ]
    tasks = [{"prompt": tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
    ), "expected": answer, "scoring": "contains"} for question, answer in questions]
    (root / "gpu-tasks.jsonl").write_text("".join(json.dumps(row) + "\n" for row in tasks))

    tiny_tokenizer = Tokenizer(models.WordLevel({
        "<pad>": 0, "<unk>": 1, "<eos>": 2, "<bos>": 3,
        "alpha": 4, "beta": 5, "gamma": 6, "delta": 7,
    }, unk_token="<unk>"))
    tiny_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tiny_tokenizer, unk_token="<unk>", pad_token="<pad>",
        eos_token="<eos>", bos_token="<bos>",
    )
    torch.manual_seed(2677)
    tiny = LlamaForCausalLM(LlamaConfig(
        vocab_size=8, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=1024,
        pad_token_id=0, eos_token_id=2, bos_token_id=3,
    ))
    for side in ("tiny-before", "tiny-after"):
        tiny.save_pretrained(root / side)
        hf_tokenizer.save_pretrained(root / side)
    (root / "cpu-tasks.jsonl").write_text(json.dumps({
        "prompt": "alpha beta", "expected": "gamma", "scoring": "contains",
    }) + "\n")


def run(root: Path) -> None:
    for name, before, after, tasks, device in (
        ("h200", "qwen-fp16", "qwen-nf4", "gpu-tasks.jsonl", "cuda:0"),
        ("cpu", "tiny-before", "tiny-after", "cpu-tasks.jsonl", "cpu"),
    ):
        command = [
            sys.executable, "-m", "soup_cli", "eval", "quant-check",
            "--before", str(root / before), "--after", str(root / after),
            "--tasks", str(root / tasks), "--performance", "--device", device, "--format", "json",
        ]
        result = subprocess.run(command, capture_output=True, text=True, env=os.environ.copy())
        (root / f"{name}.json").write_text(result.stdout)
        (root / f"{name}.stderr.log").write_text(result.stderr)
        if result.returncode not in (0, 2):
            raise RuntimeError(f"{name} failed, exit {result.returncode}: {result.stderr}")
        payload = json.loads(result.stdout)
        performance = payload["performance"]
        assert performance["before"]["device"] == performance["after"]["device"] == device
        if name == "cpu":
            assert performance["before"]["peak_gpu_memory_bytes"] is None
            assert performance["after"]["peak_gpu_memory_bytes"] is None
        else:
            assert 0 < performance["after"]["peak_gpu_memory_bytes"] < (
                performance["before"]["peak_gpu_memory_bytes"]
            )
        console.print({"device": device, "exit": result.returncode, "quality": payload["rows"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".codex/quant-check-smoke"))
    parser.add_argument(
        "--reuse", action="store_true", help="Reuse checkpoints from a prior prepare",
    )
    args = parser.parse_args()
    if not args.reuse:
        args.root.mkdir(parents=True, exist_ok=False)
        prepare(args.root)
    run(args.root)


if __name__ == "__main__":
    main()
