"""Planning cost of ``training.prefix_tree`` on a synthetic set, on the CPU.

Each example is 400 tokens: one of 50 system prompts of 300 tokens, a 60-token
question and a 40-token answer (the trained part). The set is read through
``examples_from_dataset``, as the trainer reads it, and then planned by
``PrefixTreeBatchSampler`` into rows of ``--max-length`` tokens.

Timing and memory are separate runs: ``--trace`` turns on ``tracemalloc``,
which slows Python down, so its times are not the planning time.

  python benchmarks/harness/prefix_tree_plan_cost.py --examples 10000 --out time.json
  python benchmarks/harness/prefix_tree_plan_cost.py --examples 10000 --trace --out mem.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

IGNORE = -100
SYSTEMS = 50
SYSTEM_TOKENS = 300
QUESTION_TOKENS = 60
ANSWER_TOKENS = 40


class SyntheticSet:
    def __init__(self, count: int) -> None:
        rng = random.Random(0)
        self.count = count
        self.systems = [
            [rng.randrange(1000, 30000) for _ in range(SYSTEM_TOKENS)] for _ in range(SYSTEMS)
        ]

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        rng = random.Random(index)
        system = self.systems[rng.randrange(SYSTEMS)]
        question = [rng.randrange(1000, 30000) for _ in range(QUESTION_TOKENS)]
        answer = [rng.randrange(1000, 30000) for _ in range(ANSWER_TOKENS)]
        prompt = SYSTEM_TOKENS + QUESTION_TOKENS
        return system + question + answer, [IGNORE] * prompt + answer


def collate(features: List[Tuple[List[int], List[int]]]) -> Dict[str, Any]:
    import torch

    width = max(len(ids) for ids, _ in features)
    ids = torch.zeros((len(features), width), dtype=torch.long)
    mask = torch.zeros((len(features), width), dtype=torch.long)
    labels = torch.full((len(features), width), IGNORE, dtype=torch.long)
    for row, (tokens, row_labels) in enumerate(features):
        ids[row, : len(tokens)] = torch.tensor(tokens)
        mask[row, : len(tokens)] = 1
        labels[row, : len(tokens)] = torch.tensor(row_labels)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=int, default=10_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--trace", action="store_true", help="measure memory (slower)")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    import torch

    from soup_cli.utils.prefix_tree import PrefixTreeBatchSampler
    from soup_cli.utils.prefix_tree_trainer import examples_from_dataset

    dataset = SyntheticSet(args.examples)
    if args.trace:
        tracemalloc.start()
    start = time.perf_counter()
    examples = examples_from_dataset(dataset, collate)
    read_s = time.perf_counter() - start
    held = None
    if args.trace:
        held, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
    start = time.perf_counter()
    sampler = PrefixTreeBatchSampler(
        examples, max_tokens=args.max_length, batch_size=1, seed=0
    )
    plan_s = time.perf_counter() - start
    stats = sampler.stats
    result: Dict[str, Any] = {
        "examples": args.examples,
        "max_length": args.max_length,
        "tokens": stats.source_tokens,
        "rows": stats.rows,
        "ratio": round(stats.ratio, 4),
        "trace": args.trace,
        "read_s": round(read_s, 3),
        "plan_s": round(plan_s, 3),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "machine": platform.platform(),
        "processor": platform.processor(),
    }
    if args.trace:
        after, peak = tracemalloc.get_traced_memory()
        result.update(
            held_mb=round(held / 1e6, 1),
            held_bytes_per_token=round(held / stats.source_tokens, 1),
            plan_peak_mb=round(peak / 1e6, 1),
            plan_peak_bytes_per_token=round(peak / stats.source_tokens, 1),
            after_mb=round(after / 1e6, 1),
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
