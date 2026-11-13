"""A/B of ``training.prefix_tree`` on a real model, through the shipped trainer.

The model, tokenizer, dataset rows and collator are the ones ``soup train``
builds (``SFTTrainerWrapper.setup``), resident or layer-streamed. The same
examples are then run three ways, forward + backward, no optimizer step (so
every arm sees the same weights and the losses are comparable):

  one     one example per step             what a card that fits batch 1 does today
  batch   the ordinary batch per group     the same gradient as the tree arm
  tree    the group as prefix-tree rows    training.prefix_tree

A "group" is one planned prefix-tree row: the examples the sampler put together
under ``max_length`` tokens. All three arms hold the same examples, so their
losses are the same number computed three ways. ``one`` against ``batch`` (two
ORDINARY layouts, the per-example losses weighted by their trained tokens) is
what the model's arithmetic disagrees with itself by in its dtype; ``tree``
against ``batch`` is reported next to it and the harness exits non-zero when
the tree's disagreement is more than four times that noise.
Arms are interleaved per round and their order rotates. Peak VRAM is the
allocator's, reset per arm.

``--train`` instead runs ``wrapper.train()`` once and reports what a user sees.

Examples:

  python benchmarks/harness/prefix_tree_ab.py --model path/to/Qwen2.5-0.5B-Instruct \
      --data one-schema --stream --quant 4bit --out result.json
  python benchmarks/harness/prefix_tree_ab.py --model ... --data fixture --train --prefix-tree
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "src" / "soup_cli" / "data" / "_fixtures" / "gate" / "tool_call.jsonl"


def fixture_rows(mode: str, repeat: int) -> List[Dict[str, Any]]:
    """Soup's own tool-calling fixture as chat rows.

    ``fixture``: each row keeps its own tool list as the system message (40 rows,
    27 distinct lists). ``one-schema``: every row gets the longest list, the
    shape of a dataset written against one tool schema.
    """
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines() if line]
    split = []
    for row in rows:
        header, _, question = row["prompt"].rpartition("\n\n")
        expected = row["expected"]
        answer = expected if isinstance(expected, str) else json.dumps(expected)
        split.append((header, question, answer))
    if mode == "one-schema":
        longest = max((header for header, _q, _a in split), key=len)
        split = [(longest, question, answer) for _h, question, answer in split]
    out = []
    for copy in range(repeat):
        for header, question, answer in split:
            suffix = "" if copy == 0 else f" (variant {copy})"
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": header},
                        {"role": "user", "content": question + suffix},
                        {"role": "assistant", "content": answer},
                    ]
                }
            )
    return out


def build(args: argparse.Namespace, work: Path, *, prefix_tree: bool) -> Any:
    import yaml

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper

    training: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "quantization": args.quant,
        "stream_layers": bool(args.stream),
        "prefix_tree": prefix_tree,
        "epochs": args.epochs,
        "logging_steps": 1,
        "save_steps": 100000,
        "lr": 1e-4,
        "seed": 3,
        "lora": {"r": 8, "alpha": 16, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]},
    }
    cfg = load_config_from_string(
        yaml.safe_dump(
            {
                "base": args.model,
                "task": "sft",
                "backend": "transformers",
                "modality": "text",
                "data": {"train": "train.jsonl", "max_length": args.max_length},
                "training": training,
                "output": str(work / "out"),
            }
        )
    )
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.setup({"train": fixture_rows(args.data, args.repeat)})
    return wrapper


def run_ab(args: argparse.Namespace, work: Path) -> Dict[str, Any]:
    import torch

    from soup_cli.utils.prefix_tree import PrefixTreeCollator
    from soup_cli.utils.prefix_tree_trainer import prefix_tree_state

    wrapper = build(args, work, prefix_tree=True)
    trainer = wrapper.trainer
    model = wrapper.model
    state = prefix_tree_state(trainer)
    dataset = trainer.train_dataset
    plain = trainer.data_collator
    tree = PrefixTreeCollator(
        plain,
        max_tokens=state.max_tokens,
        pad_token_id=state.pad_token_id,
        additive_dtype=state.additive_dtype,
    )
    groups = [list(group) for group in state.sampler.rows][: args.groups or None]
    cuda = torch.cuda.is_available()
    # training_step reads this; Trainer.train() sets it, and train() is not run here.
    trainer.current_gradient_accumulation_steps = 1

    def step(batch: Dict[str, Any]) -> float:
        model.zero_grad(set_to_none=True)
        loss = trainer.training_step(model, dict(batch))
        return float(loss.detach().float().cpu())

    def sync() -> None:
        if cuda:
            torch.cuda.synchronize()

    def arm(name: str) -> Dict[str, Any]:
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        losses: List[float] = []
        tokens = 0
        steps = 0
        sync()
        start = time.perf_counter()
        for group in groups:
            features = [dataset[index] for index in group]
            if name == "one":
                batches = [plain([feature]) for feature in features]
            elif name == "batch":
                batches = [plain(features)]
            else:
                batches = [tree(features)]
            group_losses = []
            weights = []
            for batch in batches:
                tokens += int(batch["input_ids"].numel())
                steps += 1
                # Trained tokens: the stock loss is their mean.
                weights.append(int((batch["labels"][:, 1:] != -100).sum()))
                group_losses.append(step(batch))
            total = max(sum(weights), 1)
            losses.append(sum(loss * weight for loss, weight in zip(group_losses, weights)) / total)
        sync()
        seconds = time.perf_counter() - start
        return {
            "seconds": seconds,
            "steps": steps,
            "row_tokens": tokens,
            "losses": losses,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else None,
        }

    names = ["one", "batch", "tree"]
    arm("tree")  # warm-up: allocator, kernels, the stream's first sweep
    rounds: List[Dict[str, Any]] = []
    for index in range(args.rounds):
        order = names[index % 3 :] + names[: index % 3]
        results = {}
        for name in order:
            try:
                results[name] = arm(name)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[name] = None
        rounds.append(results)
    wrapper._close_stream_runtime()

    examples = sum(len(group) for group in groups)
    summary: Dict[str, Any] = {}
    for name in names:
        runs = [r[name] for r in rounds if r[name] is not None]
        if not runs:
            summary[name] = None
            continue
        seconds = statistics.median(run["seconds"] for run in runs)
        summary[name] = {
            "seconds_per_pass": seconds,
            "examples_per_second": examples / seconds,
            "steps": runs[0]["steps"],
            "row_tokens": runs[0]["row_tokens"],
            "peak_gb": max(run["peak_gb"] or 0.0 for run in runs) or None,
            "spread": (max(r["seconds"] for r in runs) - min(r["seconds"] for r in runs)) / seconds,
        }
    def disagreement(left: str, right: str) -> Any:
        worst_seen = None
        for results in rounds:
            if not (results.get(left) and results.get(right)):
                continue
            pairs = zip(results[left]["losses"], results[right]["losses"])
            value = max(abs(a - b) / max(abs(a), 1e-9) for a, b in pairs)
            worst_seen = value if worst_seen is None else max(worst_seen, value)
        return worst_seen

    worst = disagreement("batch", "tree")
    noise = disagreement("one", "batch")
    tree_vs_one = disagreement("one", "tree")
    stats = state.sampler.stats
    return {
        "model": args.model,
        "data": args.data,
        "stream": bool(args.stream),
        "quant": args.quant,
        "max_length": args.max_length,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "examples": examples,
        "groups": len(groups),
        "planned": {
            "examples": stats.examples,
            "rows": stats.rows,
            "source_tokens": stats.source_tokens,
            "tree_tokens": stats.tree_tokens,
            "ratio": stats.ratio,
        },
        "arms": summary,
        "worst_relative_loss_difference_batch_vs_tree": worst,
        "worst_relative_loss_difference_one_vs_batch": noise,
        "worst_relative_loss_difference_one_vs_tree": tree_vs_one,
        "losses_first_round": {
            name: None if rounds[0].get(name) is None else rounds[0][name]["losses"]
            for name in names
        },
    }


def run_train(args: argparse.Namespace, work: Path) -> Dict[str, Any]:
    import torch

    wrapper = build(args, work, prefix_tree=bool(args.prefix_tree))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        wrapper.train()
    finally:
        wrapper._close_stream_runtime()
    seconds = time.perf_counter() - start
    trainer = wrapper.trainer
    logged = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    final = [entry for entry in trainer.state.log_history if "train_runtime" in entry]
    return {
        "model": args.model,
        "data": args.data,
        "stream": bool(args.stream),
        "quant": args.quant,
        "prefix_tree": bool(args.prefix_tree),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "examples_per_epoch": len(trainer.train_dataset),
        "optimizer_steps": int(trainer.state.global_step),
        "wall_seconds": seconds,
        "train_runtime": final[-1]["train_runtime"] if final else None,
        "first_loss": logged[0] if logged else None,
        "last_loss": logged[-1] if logged else None,
        "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", choices=["fixture", "one-schema"], default="one-schema")
    parser.add_argument("--repeat", type=int, default=1, help="copies of the 40 fixture rows")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--groups", type=int, default=0, help="first N groups only (0 = all)")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train", action="store_true", help="run wrapper.train() instead")
    parser.add_argument("--prefix-tree", action="store_true", help="with --train")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="prefix_tree_ab_") as tmp:
        result = run_train(args, Path(tmp)) if args.train else run_ab(args, Path(tmp))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    if not args.train:
        worst = result["worst_relative_loss_difference_batch_vs_tree"]
        noise = result["worst_relative_loss_difference_one_vs_batch"]
        if worst is not None and noise is not None and worst > max(4 * noise, 1e-4):
            sys.stderr.write(
                f"batch and tree losses disagree by {worst:.3g}, over four times the "
                f"ordinary layouts' own {noise:.3g}\n"
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
