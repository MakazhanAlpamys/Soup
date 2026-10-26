"""Continue a verified #2162 two-update gate into 10 warmups + 50 timed updates."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rich.console import Console


def validate_times(values: list[float]) -> None:
    if len(values) != 50 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("timing requires exactly 50 finite positive step durations")


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timing-out", required=True, type=Path)
    own, remaining = parser.parse_known_args()
    if own.timing_out.exists():
        raise FileExistsError("preserve the earlier timing result")
    import stream_projection_exactness as gate

    sys.argv = [sys.argv[0], *remaining]

    def measure(model: Any, optimizer: Any, ids: Any, labels: Any, state: dict) -> None:
        import torch

        if state["optimizer_updates"] != 2 or "reference_sha256" not in state:
            raise RuntimeError("two exact verified warm-up updates are required")
        cap = state["cap_bytes"]
        row = dict(status="warming", warmup_updates=10, measured_updates=50,
                   verified_initial_updates=2, initial_reference_sha256=state["reference_sha256"],
                   shape=state["shape"], arm=state["arm"], cap_bytes=cap,
                   native=state["native"], plan=state["plan"],
                   step_times_s=[], losses=[], clock="time.perf_counter",
                   optimizer="PagedAdamW8bit")
        own.timing_out.parent.mkdir(parents=True, exist_ok=True)

        def save() -> None:
            own.timing_out.write_text(json.dumps(row, indent=2)+"\n", encoding="utf-8")

        def step() -> float:
            optimizer.zero_grad(set_to_none=True)
            output = model(input_ids=ids, labels=labels)
            output.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            value = float(output.loss.detach().item())
            if not math.isfinite(value):
                raise RuntimeError("nonfinite training loss")
            return value

        try:
            save()
            for _ in range(8):
                step()
            torch.cuda.synchronize()
            row["status"] = "measuring"
            row["window_start_unix_s"] = time.time()
            save()
            for _ in range(50):
                started = time.perf_counter()
                value = step()
                row["step_times_s"].append(time.perf_counter()-started)
                row["losses"].append(value)
            row["window_end_unix_s"] = time.time()
            validate_times(row["step_times_s"])
            row.update(peak_reserved=torch.cuda.max_memory_reserved(),
                       peak_allocated=torch.cuda.max_memory_allocated(),
                       median_step_s=statistics.median(row["step_times_s"]))
            if row["peak_allocated"] > cap or row["peak_reserved"] > cap:
                raise RuntimeError("combined training peak exceeds shared 8 GiB cap")
            row["status"] = "ok"
            save()
            state["continuation_optimizer_updates"] = 58
            state["timing_receipt"] = str(own.timing_out)
            Console().print({"arm": row["arm"], "timed_updates_completed": 50})
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            save()
            raise

    return gate.main(after_exactness=measure)


if __name__ == "__main__":
    raise SystemExit(main())
