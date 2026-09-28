"""Does --cuda-graphs re-record per request, or grow memory, over a large batch?

Written for a review finding on the --cuda-graphs port: transformers builds a new
StaticCache per generate() call, so every request may present new K/V addresses to
the graph tree. Mirrors `soup infer --cuda-graphs` (`_load_model`, `_warm_cuda_graphs`
on the longest prompt, then `_generate(..., cuda_graphs=True)` per row) and records per
row: latency, allocated/reserved bytes, the tree manager's recording counter and the
inductor skip counters. Not a gate run; greedy. Run from a Soup checkout:

    python benchmarks/harness/cuda_graph_decode_rerecord_probe.py \
        --model DIR --rows 256 --max-tokens 32 --output probe.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _prompts(count: int) -> list[str]:
    spec = importlib.util.spec_from_file_location(
        "cg_prompts", REPO / "benchmarks" / "harness" / "cuda_graph_decode_prompts.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.prompts(count)


def _tree_state(torch) -> dict:
    try:
        from torch._inductor import cudagraph_trees

        manager = cudagraph_trees.get_container(0).tree_manager
    except Exception as exc:  # the manager may not exist before the first graph call
        return {"manager": f"unavailable: {type(exc).__name__}: {exc}"}
    if manager is None:
        return {"manager": None}
    rerecords = {}
    for node_id, per_function in getattr(manager, "num_rerecord", {}).items():
        for function_id, count in per_function.items():
            node = getattr(node_id, "id", node_id)
            function = getattr(function_id, "id", function_id)
            rerecords[f"{node}:{function}"] = count
    return {
        "recordings": getattr(manager, "debug_fail_counter", None),
        "warmed_up_functions": len(getattr(manager, "warmed_up_functions", ()) or ()),
        "unexpected_rerecords": rerecords,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "src"))
    import torch
    from torch._dynamo.utils import counters

    import soup_cli
    from soup_cli.commands.infer import _generate, _load_model, _warm_cuda_graphs

    if not Path(soup_cli.__file__).is_relative_to(REPO):
        raise SystemExit(f"soup_cli imported from {soup_cli.__file__}, not this checkout")
    prompts = _prompts(args.rows)
    model, tokenizer = _load_model(args.model, None, "cuda", is_local=True)
    started = time.perf_counter()
    _warm_cuda_graphs(model, tokenizer, prompts, args.max_tokens)
    torch.cuda.synchronize()
    record = {
        "model": args.model,
        "rows": args.rows,
        "max_tokens": args.max_tokens,
        "soup_cli": soup_cli.__file__,
        "torch": torch.__version__,
        "warmup_seconds": time.perf_counter() - started,
        "after_warmup": {
            "tree": _tree_state(torch),
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        },
        "per_row": [],
    }
    for index, text in enumerate(prompts):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        _, tokens = _generate(
            model,
            tokenizer,
            [{"role": "user", "content": text}],
            max_tokens=args.max_tokens,
            temperature=0.0,
            cuda_graphs=True,
        )
        torch.cuda.synchronize()
        row = {
            "index": index,
            "seconds": time.perf_counter() - begin,
            "tokens": int(tokens),
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }
        if index in (0, 1, 2, 31, 63, 127, 128, 129, 191, args.rows - 1):
            row["tree"] = _tree_state(torch)
        record["per_row"].append(row)
    record["inductor_counters"] = {key: int(value) for key, value in counters["inductor"].items()}
    record["stats_counters"] = {key: int(value) for key, value in counters["stats"].items()}
    record["max_reserved"] = torch.cuda.max_memory_reserved()
    rows = record["per_row"]
    first, last = rows[: min(32, len(rows))], rows[-min(32, len(rows)) :]
    record["summary"] = {
        "tok_per_s_first_32": sum(r["tokens"] for r in first) / sum(r["seconds"] for r in first),
        "tok_per_s_last_32": sum(r["tokens"] for r in last) / sum(r["seconds"] for r in last),
        "reserved_after_row_0": rows[0]["reserved"],
        "reserved_after_last": rows[-1]["reserved"],
        "recordings_after_warmup": record["after_warmup"]["tree"].get("recordings"),
        "recordings_at_end": _tree_state(torch).get("recordings"),
        "cudagraph_skips": record["inductor_counters"].get("cudagraph_skips", 0),
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
