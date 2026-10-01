"""Write chat prompts of widely varying, UNSORTED length for the end-to-end gate (d).

    python cuda_graph_decode_prompts.py --count 24 --output prompts.jsonl

Unsorted on purpose: a batch whose longest prompt arrives late is what makes a
static cache grow mid-run, and it is what the warm-up on the longest prompt removes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOPICS = ("database indexes", "TCP congestion control", "garbage collection", "CUDA streams",
          "B-trees", "hash maps", "compilers", "RAID levels")
FILLER = "Give concrete examples and explain the trade-offs involved. "


def prompts(count: int) -> list[str]:
    # (index * 7) % 23 visits 0..22 filler sentences in a scrambled order.
    return [f"Explain {TOPICS[index % len(TOPICS)]}. " + FILLER * ((index * 7) % 23)
            for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Path(args.output).open("w", encoding="utf-8") as handle:
        for text in prompts(args.count):
            handle.write(json.dumps({"prompt": text}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
