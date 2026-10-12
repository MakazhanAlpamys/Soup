"""Harness for the #1693 measurement: peak VRAM with and without
``training.stream_loss_chunk``.

Nothing in this file has been run on a GPU by its author, who has none. It
exists so that the table #1693 asks for (512 / 1024 / 2048 tokens, same model
and stack, with and without the option) can be produced by whoever has a card,
and so that the pre-flight's charge for the option can be checked against it.

Requirements
------------
- CUDA GPU. The unchunked arm is expected to run out of memory at the longer
  lengths on a small card; that is recorded as ``OOM``, not as a failure.
- The base model, by Hub id or local path (``--base``).

Usage
-----
    python benchmarks/harness/issue1693_loss_chunk_vram.py --base <id-or-path>
    python benchmarks/harness/issue1693_loss_chunk_vram.py --base <id-or-path> \\
        --quantization 4bit --seqs 512,1024,2048 --chunk 128

Protocol
--------
The real ``soup train`` setup and one training step per arm, batch 1, LoRA r=8
on ``q_proj`` / ``v_proj``, ``stream_buffers: 2``, the rewind log on (its
default). "Real peak" is ``torch.cuda.max_memory_allocated()`` over the step.
"Predicted" is the number the pre-flight itself computed during that setup,
read from ``estimate_stream_peak_vram`` as it was called, so the two columns
are about the same run. ``stream_vram_override`` is set high so that the
pre-flight never refuses an arm this harness is trying to measure.

As in ``issue395_second_stack_vram.py``: ``data.max_length`` truncates, it does
not pad, so the rows overshoot 3x and the realised length is read off the
collated batch and printed. A row whose realised length is not the requested
one is void.

Reading the table
-----------------
``pred/real`` below 1.0 is an under-prediction, the direction the pre-flight
must not err in. The chunked rows are the ones nobody has measured: their
logits term is the shipped 14 bytes per element charged on one chunk, plus two
``[batch, seq, hidden]`` tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import tempfile

import yaml

#: Bytes the pre-flight is told are free, so that it measures instead of refusing.
ASSUME_FREE_BYTES = 1 << 40


def rows(seq: int) -> list[dict]:
    # Overshoot 3x; truncation lands it on the target. See the module docstring.
    body = "word " * (seq * 3)
    return [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": body}]}
        for _ in range(4)
    ]


def config_doc(
    base: str, seq: int, chunk: int | None, quantization: str, out_dir: pathlib.Path
) -> dict:
    """One arm's ``soup.yaml`` as a dict; ``chunk=None`` is the unchunked arm."""
    training = {
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "epochs": 1,
        "quantization": quantization,
        "stream_layers": True,
        "stream_vram_override": ASSUME_FREE_BYTES,
        "logging_steps": 1,
        "save_steps": 100000,
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
    }
    if chunk is not None:
        training["stream_loss_chunk"] = chunk
    return {
        "base": base,
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {
            "train": str(out_dir / "d.jsonl"),
            "max_length": seq,
            "chat_template": "chatml",
        },
        "output": str(out_dir / "out"),
        "training": training,
    }


def table_row(seq: int, realised: int | None, chunk: int | None, predicted: int, peak: int | None):
    """One printed line. ``peak=None`` is an arm that ran out of memory."""
    arm = "off" if chunk is None else str(chunk)
    length = "-" if realised is None else str(realised)
    if peak is None:
        return f"{seq:>6} {length:>8} {arm:>6} | {predicted / 1e9:>10.3f}GB {'OOM':>12} {'-':>9}"
    verdict = "" if predicted >= peak else "  <- UNDER-PREDICTS"
    return (
        f"{seq:>6} {length:>8} {arm:>6} | {predicted / 1e9:>10.3f}GB "
        f"{peak / 1e9:>10.3f}GB {predicted / peak:>9.4f}{verdict}"
    )


HEADER = (
    f"{'seq':>6} {'realised':>8} {'chunk':>6} | "
    f"{'predicted':>12} {'real peak':>12} {'pred/real':>9}"
)


def run_one(base: str, seq: int, chunk: int | None, quantization: str):
    """``(predicted_bytes, real_peak_bytes or None on OOM, realised_seq_len)``."""
    import torch

    from soup_cli.config.loader import load_config_from_string
    from soup_cli.trainer.sft import SFTTrainerWrapper
    from soup_cli.utils import layer_stream

    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "d.jsonl").write_text("\n".join(json.dumps(r) for r in rows(seq)), encoding="utf-8")
    (tmp / "out").mkdir(exist_ok=True)

    estimate = layer_stream.estimate_stream_peak_vram
    seen: dict = {}

    def recording(**kwargs):
        seen["predicted"] = estimate(**kwargs)
        return seen["predicted"]

    gc.collect()
    torch.cuda.empty_cache()
    wrapper = None
    layer_stream.estimate_stream_peak_vram = recording
    try:
        cfg = load_config_from_string(
            yaml.safe_dump(config_doc(base, seq, chunk, quantization, tmp))
        )
        wrapper = SFTTrainerWrapper(cfg, device="cuda")
        wrapper.setup({"train": rows(seq)})
        batch = next(iter(wrapper.trainer.get_train_dataloader()))
        realised = int(batch["input_ids"].shape[-1])
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        wrapper.trainer.args.max_steps = 1
        try:
            wrapper.trainer.train()
            peak = torch.cuda.max_memory_allocated()
        except torch.cuda.OutOfMemoryError:
            peak = None
    finally:
        layer_stream.estimate_stream_peak_vram = estimate
        if wrapper is not None:
            try:
                wrapper._close_stream_runtime()
            except Exception:  # noqa: BLE001 - teardown must not mask the measurement
                pass
        del wrapper
        gc.collect()
        torch.cuda.empty_cache()
    return seen["predicted"], peak, realised


def sweep(base: str, seqs: list[int], chunk: int, quantization: str) -> None:
    import torch

    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"# {base} | quantization {quantization} | chunk {chunk}")
    print(HEADER)
    for seq in seqs:
        for arm in (None, chunk):
            predicted, peak, realised = run_one(base, seq, arm, quantization)
            print(table_row(seq, realised, arm, predicted, peak))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base", required=True, help="model id or local path")
    parser.add_argument("--seqs", default="512,1024,2048", help="comma-separated token counts")
    parser.add_argument("--chunk", type=int, default=128, help="training.stream_loss_chunk")
    parser.add_argument("--quantization", default="none", choices=["none", "4bit"])
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA required: every figure in this table is a device peak.")
    sweep(args.base, [int(seq) for seq in args.seqs.split(",")], args.chunk, args.quantization)
