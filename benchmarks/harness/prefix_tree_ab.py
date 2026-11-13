"""A/B of ``training.prefix_tree`` on a real model, through the shipped trainer.

The model, tokenizer, dataset rows and collator are the ones ``soup train``
builds (``SFTTrainerWrapper.setup``), resident or layer-streamed.

Default mode: the same examples run three ways, forward + backward, no
optimizer step (so every arm sees the same weights and the losses are
comparable):

  one     one example per step             what a card that fits batch 1 does today
  batch   the ordinary batch per group     the same gradient as the tree arm
  tree    the group as prefix-tree rows    training.prefix_tree

A "group" is one planned prefix-tree row: the examples the sampler put together
under ``max_length`` tokens. All three arms hold the same examples, so their
losses are the same number computed three ways. ``one`` against ``batch`` (two
ORDINARY layouts, the per-example losses weighted by their trained tokens) is
what the model's arithmetic disagrees with itself by in its dtype; ``tree``
against ``batch`` is reported next to it and the harness exits non-zero when
the tree's disagreement is more than four times that noise. Arms are
interleaved per round and their order rotates. Peak VRAM is the allocator's,
reset per arm. The token counts under "planned" are for the WHOLE training
set; ``--groups`` limits only what is timed.

``--train`` instead runs ``wrapper.train()`` once, with the held-out loss
(``--val`` rows the run never trains on) measured before and after.

Data (``--data``):

  fixture       Soup's own 40 tool-calling rows, each with its own tool list
  one-schema    the same rows, every one given the longest tool list
  glaive        glaiveai/glaive-function-calling-v2 as published: each row
                carries its own function definition(s)
  glaive-fixed  the same dataset as an agent with ONE schema: the ``--tools``
                most frequent functions, every row given all of them
  dolly         databricks/databricks-dolly-15k: plain instructions, no system
                prompt of its own

The public sets are read from ``--data-file`` (the dataset's own JSON / JSONL),
in file order, keeping rows whose chat-template length fits ``--max-length``;
the first ``--rows`` train, the next ``--val`` are held out.

Examples:

  python benchmarks/harness/prefix_tree_ab.py --model path/to/model \
      --data glaive-fixed --data-file glaive-function-calling-v2.json \
      --rows 600 --val 100 --stream --quant 4bit --groups 12 --out result.json
  python benchmarks/harness/prefix_tree_ab.py --model ... --data dolly \
      --data-file databricks-dolly-15k.jsonl --train --prefix-tree --epochs 3
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "src" / "soup_cli" / "data" / "_fixtures" / "gate" / "tool_call.jsonl"
GLAIVE_PREAMBLE = (
    "You are a helpful assistant with access to the following functions. Use them if required -"
)
_FUNCTION_NAME = re.compile(r'"name":\s*"([A-Za-z0-9_]+)",\s*"description"')


def _chat(system: str, user: str, assistant: str) -> Dict[str, Any]:
    messages = [{"role": "system", "content": system}] if system else []
    messages += [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    return {"messages": messages}


def fixture_rows(mode: str) -> List[Dict[str, Any]]:
    """Soup's own tool-calling fixture as chat rows (``fixture`` / ``one-schema``)."""
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
    return [_chat(header, question, answer) for header, question, answer in split]


def _glaive_turns(chat: str) -> Tuple[str, str]:
    """The first user message and the assistant's first reply."""
    user, _, rest = chat.partition("ASSISTANT:")
    user = user.replace("USER:", "", 1).strip()
    reply = rest.split("<|endoftext|>")[0].strip()
    return user, reply


def _glaive_functions(system: str) -> List[Tuple[str, str]]:
    """``(name, compact JSON)`` of each function a glaive system prompt defines."""
    text = system.partition("-\n")[2]
    decoder = json.JSONDecoder()
    out = []
    at = 0
    while True:
        at = text.find("{", at)
        if at < 0:
            break
        try:
            value, end = decoder.raw_decode(text, at)
        except ValueError:
            break
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            out.append((value["name"], json.dumps(value, separators=(",", ":"))))
        at = end
    return out


def glaive_rows(path: str, mode: str, tools: int) -> List[Dict[str, Any]]:
    """glaive-function-calling-v2, first turn of each conversation, in file order.

    ``glaive``: the row's own system prompt. ``glaive-fixed``: the ``tools``
    most frequent functions among single-function rows become one schema (each
    function in its most common spelling, compact JSON, sorted by name), and the
    rows about those functions are kept.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed = []
    for row in data:
        functions = _glaive_functions(row["system"])
        if not functions:
            continue
        user, reply = _glaive_turns(row["chat"])
        if user and reply:
            parsed.append((row["system"].replace("SYSTEM:", "", 1).strip(), functions, user, reply))
    if mode == "glaive":
        return [_chat(system, user, reply) for system, _f, user, reply in parsed]
    single = [item for item in parsed if len(item[1]) == 1]
    counts = collections.Counter(item[1][0][0] for item in single)
    chosen = sorted(name for name, _count in counts.most_common(tools))
    spellings: Dict[str, collections.Counter] = {name: collections.Counter() for name in chosen}
    for _system, functions, _user, _reply in single:
        name, text = functions[0]
        if name in spellings:
            spellings[name][text] += 1
    schema = GLAIVE_PREAMBLE + "\n" + "\n".join(
        spellings[name].most_common(1)[0][0] for name in chosen
    )
    return [
        _chat(schema, user, reply)
        for _system, functions, user, reply in single
        if functions[0][0] in spellings
    ]


def dolly_rows(path: str) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        user = row["instruction"].strip()
        if row.get("context", "").strip():
            user += "\n\n" + row["context"].strip()
        if user and row["response"].strip():
            out.append(_chat("", user, row["response"].strip()))
    return out


def load_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(train rows, held-out rows)`` for ``--data``."""
    if args.data in ("fixture", "one-schema"):
        return fixture_rows(args.data), []
    if not args.data_file:
        raise SystemExit(f"--data {args.data} needs --data-file")
    if args.data in ("glaive", "glaive-fixed"):
        rows = glaive_rows(args.data_file, args.data, args.tools)
    else:
        rows = dolly_rows(args.data_file)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    wanted = args.rows + args.val
    kept: List[Dict[str, Any]] = []
    for row in rows:
        encoded = tokenizer.apply_chat_template(row["messages"], tokenize=True)
        length = len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)
        # A margin: the trainer's own tokenisation may differ by a few tokens, and
        # a row cut inside its answer is refused by the trainer.
        if length <= args.max_length - 8:
            kept.append(row)
            if len(kept) == wanted:
                break
    if len(kept) < wanted:
        raise SystemExit(f"only {len(kept)} rows fit --max-length {args.max_length}")
    return kept[: args.rows], kept[args.rows :]


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
        "save_steps": 1000000,
        "lr": args.lr,
        "seed": args.seed,
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
    train_rows, val_rows = load_rows(args)
    dataset = {"train": train_rows}
    if val_rows:
        dataset["val"] = val_rows
    wrapper = SFTTrainerWrapper(cfg)
    wrapper.setup(dataset)
    return wrapper


def _planned(wrapper: Any, args: argparse.Namespace) -> Dict[str, Any]:
    """Token counts of the whole training set, ordinary and as prefix trees."""
    from soup_cli.utils.prefix_tree import PrefixTreeBatchSampler
    from soup_cli.utils.prefix_tree_trainer import examples_from_dataset, prefix_tree_state

    state = prefix_tree_state(wrapper.trainer)
    if state is not None:
        stats = state.sampler.stats
    else:
        trainer = wrapper.trainer
        examples = examples_from_dataset(trainer.train_dataset, trainer.data_collator)
        stats = PrefixTreeBatchSampler(
            examples, max_tokens=args.max_length, batch_size=args.batch_size, seed=args.seed
        ).stats
    return {
        "examples": stats.examples,
        "rows": stats.rows,
        "source_tokens": stats.source_tokens,
        "tree_tokens": stats.tree_tokens,
        "ratio": stats.ratio,
        "examples_per_row": stats.examples / max(stats.rows, 1),
    }


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

    names = ["one", "tree"] if args.no_batch_arm else ["one", "batch", "tree"]
    arm("tree")  # warm-up: allocator, kernels, the stream's first sweep
    rounds: List[Dict[str, Any]] = []
    for index in range(args.rounds):
        shift = index % len(names)
        order = names[shift:] + names[:shift]
        results = {}
        for name in order:
            try:
                results[name] = arm(name)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                results[name] = None
        rounds.append(results)
    planned = _planned(wrapper, args)
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
            "tokens_per_second": runs[0]["row_tokens"] / seconds,
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

    return {
        "model": args.model,
        "data": args.data,
        "stream": bool(args.stream),
        "quant": args.quant,
        "max_length": args.max_length,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "torch": torch.__version__,
        "timed_examples": examples,
        "timed_groups": len(groups),
        "planned": planned,
        "arms": summary,
        "worst_relative_loss_difference_batch_vs_tree": disagreement("batch", "tree"),
        "worst_relative_loss_difference_one_vs_batch": disagreement("one", "batch"),
        "worst_relative_loss_difference_one_vs_tree": disagreement("one", "tree"),
        "losses_first_round": {
            name: None if rounds[0].get(name) is None else rounds[0][name]["losses"]
            for name in names
        },
    }


def run_train(args: argparse.Namespace, work: Path) -> Dict[str, Any]:
    import torch

    wrapper = build(args, work, prefix_tree=bool(args.prefix_tree))
    trainer = wrapper.trainer
    # One held-out row per forward: a wider evaluation batch is its own VRAM question.
    trainer.args.per_device_eval_batch_size = 1
    trainer.args.prediction_loss_only = True
    cuda = torch.cuda.is_available()

    def held_out() -> Any:
        if trainer.eval_dataset is None:
            return None
        return float(trainer.evaluate()["eval_loss"])

    planned = _planned(wrapper, args)
    before = held_out()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        wrapper.train()
        seconds = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else None
        after = held_out()
    finally:
        wrapper._close_stream_runtime()
    logged = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    final = [entry for entry in trainer.state.log_history if "train_runtime" in entry]
    return {
        "model": args.model,
        "data": args.data,
        "stream": bool(args.stream),
        "quant": args.quant,
        "prefix_tree": bool(args.prefix_tree),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "train_examples": len(trainer.train_dataset),
        "held_out_examples": 0 if trainer.eval_dataset is None else len(trainer.eval_dataset),
        "planned": planned,
        "optimizer_steps": int(trainer.state.global_step),
        "wall_seconds": seconds,
        "train_runtime": final[-1]["train_runtime"] if final else None,
        "first_logged_loss": logged[0] if logged else None,
        "last_logged_loss": logged[-1] if logged else None,
        "held_out_loss_before": before,
        "held_out_loss_after": after,
        "peak_gb": peak,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--data",
        choices=["fixture", "one-schema", "glaive", "glaive-fixed", "dolly"],
        default="one-schema",
    )
    parser.add_argument("--data-file", default="", help="the public dataset's JSON / JSONL")
    parser.add_argument("--rows", type=int, default=600, help="training rows (public sets)")
    parser.add_argument("--val", type=int, default=0, help="held-out rows (public sets)")
    parser.add_argument("--tools", type=int, default=4, help="functions in glaive-fixed's schema")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--quant", choices=["4bit", "none"], default="4bit")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--groups", type=int, default=0, help="time the first N groups (0 = all)")
    parser.add_argument("--no-batch-arm", action="store_true", help="skip the ordinary-batch arm")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3)
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
