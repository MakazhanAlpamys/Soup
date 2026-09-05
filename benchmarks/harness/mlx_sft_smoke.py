#!/usr/bin/env python3
"""End-to-end MLX SFT smoke run through Soup's own trainer wrapper (#23).

Drives ``MLXSFTTrainerWrapper`` rather than ``mlx_lm`` directly: the point of
#23 is that the *wrapper's* training loop had never run against a real MLX
runtime, so calling mlx-lm here would test the wrong thing.

Asserts MLX dispatch BEFORE the timer starts. #363 was ``backend: mlx`` never
reaching the MLX trainer, which means a plausible number can be the transformers
path wearing a disguise; the assert is what stops this harness publishing one.

Fixture is generated in-process — no external data file — so a run on another
machine is directly comparable to `benchmarks/run-m1-8gb-mlx-sft.md`.

Requires Apple Silicon and ``pip install -e ".[mlx]"``.

Attaches Soup's Rich display and a local experiment tracker, and refuses a run
with no bridge metrics. The database stays beside the temporary artifacts;
the bridge also emits its normal process-local SSE events. Timing includes
display/tracker overhead, unlike the original published M1 measurements.

Usage:
    python mlx_sft_smoke.py [model-id] [rows] [epochs]

The exact commands behind every row of `benchmarks/run-m1-8gb-mlx-sft.md`,
so the table reproduces without reading the thread (8 GB M1, mlx 0.32.2 /
mlx-lm 0.31.3, 48 rows / 1 epoch each):

    python mlx_sft_smoke.py mlx-community/Qwen2.5-0.5B-Instruct-4bit  48 1
    python mlx_sft_smoke.py mlx-community/Llama-3.2-3B-Instruct-4bit  48 1
    python mlx_sft_smoke.py mlx-community/Qwen2.5-7B-Instruct-4bit    48 1
    python mlx_sft_smoke.py mlx-community/Llama-3.1-8B-Instruct-4bit  48 1

The last is the model the shipped `llama3.1-8b-sft-mlx` recipe names; it peaks
at 5.154 GB and completes. The first is the smallest verified-good fixture
(282 MB) and is what a CI job should use.

Known-bad: mlx-community/TinyLlama-1.1B-Chat-v1.0-4bit ships the legacy
``weights.NN.safetensors`` naming, which mlx-lm's ``model*.safetensors`` glob
does not match. It fails with "No safetensors found".
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"

_PAIRS = [
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is 12 x 12?", "12 x 12 = 144."),
    ("Name a primary colour.", "Red is a primary colour."),
    ("What language is this repo written in?", "Python."),
    ("Give me a two-word greeting.", "Hello there."),
    ("What is the boiling point of water at sea level?", "100 degrees Celsius."),
    ("Which planet is closest to the Sun?", "Mercury."),
    ("What is the square root of 81?", "9."),
]


def build_rows(n: int) -> list[dict]:
    """Deliberately trivial and repetitive: this is a smoke signal, not a benchmark."""
    out = []
    for i in range(n):
        q, a = _PAIRS[i % len(_PAIRS)]
        out.append({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]})
    return out


class _Tee:
    """Write to both the real stdout and a buffer, so redirecting does not
    silence mlx-lm's live progress output."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self):
        # Rich must still recognise a terminal through the capture wrapper.
        return getattr(self._streams[0], "isatty", lambda: False)()


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_TRAINED_TOKENS_RE = re.compile(r"Trained Tokens (\d+)")


def _final_trained_tokens(train_stdout: str) -> int | None:
    """mlx-lm's OWN cumulative trained-token count, taken from its last report.

    Read from mlx-lm's printed output rather than recomputed, because
    reproducing its tokenisation (chat template, truncation, masking) here
    would drift silently the first time any of those change upstream. The
    wrapper's result dict does not carry the count, so stdout is the only
    place it surfaces.

    Rich's Live/FileProxy wraps progress lines and inserts ANSI controls.
    Normalise both before matching, or an intact earlier report can silently
    win over the wrapped final counter at some terminal widths.

    Returns None if the counter is absent; the caller reports unavailable.
    """
    flat = re.sub(r"\s+", " ", _ANSI_RE.sub("", train_stdout))
    matches = _TRAINED_TOKENS_RE.findall(flat)
    return int(matches[-1]) if matches else None


def host_mem() -> tuple[str, str]:
    """macOS free percentage and swap, so pressure during the run is on record."""
    try:
        mp = subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=20).stdout
        free = [ln for ln in mp.splitlines() if "free percentage" in ln]
        swap = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        return (free[0].strip() if free else "?"), swap
    except Exception as exc:  # noqa: BLE001 — diagnostics must never kill the run
        return f"unavailable: {exc!r}", "unavailable"


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    rows_n = int(sys.argv[2]) if len(sys.argv) > 2 else 48
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    import mlx.core as mx
    from rich.console import Console

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.experiment.tracker import ExperimentTracker
    from soup_cli.monitoring.display import TrainingDisplay
    from soup_cli.trainer.mlx_routing import resolve_trainer

    tmp = Path(tempfile.mkdtemp(prefix="mlx_smoke_"))
    rows = build_rows(rows_n)
    data_path = tmp / "train.jsonl"
    data_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )
    out = tmp / "out"

    cfg = load_config_from_string(f"""
base: {model}
task: sft
backend: mlx
data:
  train: {data_path}
  format: chatml
  max_length: 512
training:
  epochs: {epochs}
  lr: 1e-4
  batch_size: 1
  lora:
    r: 8
    alpha: 16
  logging_steps: 5
output: {out}
""")

    # Dispatch first (#363): a transformers-path number would be a lie.
    resolved = resolve_trainer(cfg)
    cls = resolved[0] if isinstance(resolved, tuple) else resolved
    if cls is None or "MLX" not in cls.__name__:
        sys.exit(f"NOT the MLX path: resolve_trainer returned {cls!r} — refusing to measure")
    print(f"dispatch      : {cls.__name__}  <- verified before measuring")
    print(f"model         : {model}")
    print(f"rows / epochs : {rows_n} / {epochs}")
    free, swap = host_mem()
    print(f"host before   : {free} | {swap}")

    trainer = cls(cfg)

    mx.reset_peak_memory()
    t0 = time.time()
    trainer.setup({"train": rows, "val": []})
    print(f"load          : {time.time() - t0:.1f}s   "
          f"mlx peak after load: {mx.get_peak_memory() / 1024**3:.3f} GB   "
          f"(includes first-time Hub download)")

    # mlx-lm prints its progress to stdout; tee it so the run stays readable
    # AND the trained-token counter is recoverable for the throughput line.
    buffer = io.StringIO()
    display = TrainingDisplay(cfg, device_name="Apple Silicon (MLX)")
    # A benchmark must not populate the user's normal experiment database.
    with contextlib.closing(ExperimentTracker(db_path=tmp / "experiments.db")) as tracker:
        run_id = tracker.start_run(
            cfg.model_dump(), device="mlx", device_name="Apple Silicon (MLX)", gpu_info={},
        )
        try:
            mx.reset_peak_memory()
            t0 = time.time()
            with contextlib.redirect_stdout(_Tee(sys.stdout, buffer)):
                result = trainer.train(display=display, tracker=tracker, run_id=run_id)
            train_s = time.time() - t0
            metrics = tracker.get_metrics(run_id)
            if not metrics or display.current_step <= 0:
                raise RuntimeError("MLX bridge received no display/tracker metrics")
            tracker.finish_run(
                run_id, initial_loss=result["initial_loss"], final_loss=result["final_loss"],
                total_steps=result["total_steps"], duration_secs=result["duration_secs"],
                output_dir=result["output_dir"],
            )
        except BaseException:
            # Include Ctrl-C; the wrapper stops its display in its own finally.
            tracker.fail_run(run_id)
            raise
        Console().print(
            f"bridge        : {len(metrics)} metric reports saved to {tracker.db_path}",
            markup=False, highlight=False, soft_wrap=True,
        )
    train_stdout = buffer.getvalue()
    print(f"train         : {train_s:.1f}s   "
          f"mlx peak during train: {mx.get_peak_memory() / 1024**3:.3f} GB")

    # Whole-run throughput, computed here rather than eyeballed from mlx-lm's
    # per-report stdout. Those printed `Tokens/sec` values are INSTANTANEOUS --
    # in one 48-iteration run they ranged 19.2 to 254.3 -- so quoting a late one
    # beside a whole-run wall clock silently mixes two different measurements.
    # trained_tokens is mlx-lm's own cumulative counter for the run.
    trained = _final_trained_tokens(train_stdout)
    if trained is not None and train_s > 0:
        print(f"throughput    : {trained / train_s:.1f} tok/s  "
              f"({trained} trained tokens / {train_s:.1f}s, whole-run average)")
    else:
        print("throughput    : unavailable — no trained-token count in the result")
    print(f"result        : {result}")
    free, swap = host_mem()
    print(f"host after    : {free} | {swap}")

    adapter = out / "adapters.safetensors"
    if not adapter.exists():
        sys.exit("adapter file MISSING — training reported success but wrote nothing")
    print(f"adapter file  : {adapter.stat().st_size / 1024:.0f} KB")
    print(f"adapter config: {(out / 'adapter_config.json').exists()}")

    # The criterion that matters: the adapter must LOAD, not merely exist.
    t0 = time.time()
    from mlx_lm import load

    load(model, adapter_path=str(out))
    print(f"reload w/ adapter: OK in {time.time() - t0:.1f}s")
    print(f"\nartifacts: {tmp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
