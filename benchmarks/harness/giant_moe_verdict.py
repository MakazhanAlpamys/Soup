"""Check the giant MoE pilot journal for completed training and exact reload."""

import argparse
import csv
import json
import math
from pathlib import Path

from rich.console import Console


def verdict(events: list[dict], index: dict, physical_vram_mib: int) -> dict:
    failures = []

    def require(condition, message):
        if not condition:
            failures.append(message)

    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    by_kind = {}
    for row in events:
        by_kind.setdefault(row.get("event"), []).append(row)
    env = next(iter(by_kind.get("environment", [])), {})
    shard = next(iter(by_kind.get("sharded", [])), {})
    maximum_length = next(iter(by_kind.get("data", [])), {}).get("max_length", 64)
    require(7000 <= physical_vram_mib <= 8500, "physical GPU is not the 8 GB class")
    require(index.get("total_params", 0) >= 500_000_000_000, "source is not a real giant")
    require(
        shard.get("source_parameters") == index.get("total_params"),
        "journal/index parameter counts differ",
    )
    require(index.get("arch") == "deepseek_v3", "wrong architecture")
    require(
        index.get("n_layers") == 61 and index.get("expert_count") == 256,
        "unexpected DeepSeek-V3 geometry",
    )
    require(index.get("expert_layers") == list(range(3, 61)), "incomplete expert-layer index")
    require(
        sum(str(entry[0]).endswith(".safetensors") for entry in index.get("source_files", []))
        == 163,
        "incomplete source manifest",
    )
    require(
        index.get("quant") == "nf4" and index.get("double_quant") is False,
        "unexpected cache quantization",
    )
    require(
        shard.get("cache_bytes", math.inf) <= env.get("cache_budget_bytes", 0),
        "cache exceeds planned budget",
    )
    steps = by_kind.get("step", [])
    require(
        len(steps) == 2 and [row.get("step") for row in steps] == [0, 1],
        "two completed ordered optimizer steps are required",
    )
    for row in steps:
        require(finite(row.get("loss")), "non-finite training loss")
        require(
            finite(row.get("gradient_norm")) and row.get("gradient_norm", 0) > 0,
            "missing finite nonzero gradients",
        )
        require(row.get("gradient_tensors") == 610, "incomplete LoRA gradient set")
        require(
            row.get("adapter_sha256_before") != row.get("adapter_sha256_after")
            and len(row.get("adapter_sha256_after", "")) == 64,
            "optimizer did not demonstrably change the adapter",
        )
        seconds, tokens, speed = row.get("seconds"), row.get("tokens"), row.get("tokens_per_second")
        require(
            finite(seconds)
            and seconds > 0
            and type(tokens) is int
            and 0 < tokens <= maximum_length
            and finite(speed)
            and math.isclose(speed, tokens / seconds, rel_tol=1e-9),
            "invalid measured throughput",
        )
        require(
            0
            < row.get("peak_allocated", 0)
            <= row.get("peak_reserved", 0)
            <= env.get("vram_bytes", 0),
            "invalid VRAM measurement",
        )
    if len(steps) == 2:
        require(
            steps[1].get("adapter_sha256_before") == steps[0].get("adapter_sha256_after"),
            "adapter digest chain is broken",
        )
    quality = {}
    for kind in ("heldout_before", "heldout_after"):
        rows = by_kind.get(kind, [])
        require(len(rows) == 1 and finite(rows[0].get("loss")), "missing finite " + kind)
        if rows:
            quality[kind] = rows[0].get("loss")
    reload_rows = by_kind.get("reload", [])
    require(
        len(reload_rows) == 1
        and reload_rows[0].get("equal") is True
        and reload_rows[0].get("max_abs_error") == 0,
        "reload logits are not exactly equal",
    )
    require(
        len(reload_rows) == 1
        and reload_rows[0].get("parameter_digest_matches") is True
        and bool(steps)
        and reload_rows[0].get("parameter_sha256") == steps[-1].get("adapter_sha256_after"),
        "reload adapter parameters differ from trained parameters",
    )
    require(bool(events) and events[-1].get("event") == "complete", "pilot did not complete")
    return {
        "pilot_checks_passed": not failures,
        "failures": failures,
        "source_parameters": index.get("total_params"),
        "steps": steps,
        "heldout_probe": quality,
        "ram_32gb_hardware_validated": False,
        "scope": "two updates; one developer-authored heldout text; no general quality verdict",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    lines = Path(args.events).read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line]
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    with Path(args.gpu).open(newline="") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if len(rows) != 1:
        raise ValueError("exactly one physical GPU must be recorded")
    memory = int(rows[0]["memory.total [MiB]"].split()[0])
    result = verdict(events, index, memory)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Console().print(result)
    raise SystemExit(0 if result["pilot_checks_passed"] else 1)
