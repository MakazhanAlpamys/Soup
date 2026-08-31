"""Harness for the #395 second-stack VRAM record.

Reproduces every figure in ``benchmarks/gate-395-second-stack-vram.md`` against
a released Soup, so the record can be re-measured rather than taken on trust.

Requirements
------------
- CUDA GPU required. A 24 GB card is enough: the largest arm is
  Qwen2.5-0.5B at seq 6144, which peaks at 11.85 GB.
- Downloads ``HuggingFaceTB/SmolLM2-135M`` and ``Qwen/Qwen2.5-0.5B``.
- Original environment: A10G 23 GB (AWS ``g5.xlarge``), Ubuntu 22.04,
  torch 2.13.0+cu130, transformers 5.16.1, trl 0.29.1, peft 0.20.0,
  Python 3.10. Soup 0.73.3.
- Needs ``jinja2>=3.1.0``; the ``[dev]`` extra may resolve lower, and the
  chat-template path raises without it.
- Whole run is a few minutes.

Usage
-----
    python benchmarks/harness/issue395_second_stack_vram.py            # sweep
    python benchmarks/harness/issue395_second_stack_vram.py --measure  # sect. 1
    python benchmarks/harness/issue395_second_stack_vram.py --sdpa     # sect. 4

Protocol
--------
Real ``soup train`` setup + one step, bf16, batch 1, ``quantization: none``,
LoRA r=8, ``stream_buffers: 2``. "Real peak" is
``torch.cuda.max_memory_allocated()`` over the step.

The trap this harness exists to avoid
-------------------------------------
``data.max_length`` **truncates, it does not pad**. A row that tokenizes short
runs at its own length regardless of the configured maximum, so the real peak
stops responding to ``seq`` while the prediction keeps climbing — which reads as
a large, clean over-prediction that grows with sequence and is entirely an
artifact. The first sweep for this record was wrong that way.

Two guards, both load-bearing: rows overshoot 3x so truncation lands on the
target, and the realised length is read off the collated batch
(``input_ids.shape[-1]``) and printed in its own column. If that column does not
track the requested seq, the row is void.
"""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import sys
import tempfile

import torch
import yaml

from soup_cli.config.loader import load_config_from_string
from soup_cli.utils.layer_stream import (
    LOGITS_BYTES_PER_ELEMENT,
    LOGITS_LOSS_BYTES_PER_ELEMENT,
    estimate_logits_bytes,
    estimate_stream_peak_vram,
    measure_logits_loss_bytes_per_element,
)

#: Shapes as the published grid records them (tests/test_v07203.py).
SMOL = dict(
    pool=14160384, extras=56624256, adapter=921600, vocab=49152,
    hidden=576, intermediate=1536, n_layers=30, model="HuggingFaceTB/SmolLM2-135M",
)
QWEN = dict(
    pool=59649536, extras=272271104, adapter=1081344, vocab=151936,
    hidden=896, intermediate=4864, n_layers=24, model="Qwen/Qwen2.5-0.5B",
)


def predict(shape: dict, seq: int, batch: int = 1) -> int:
    """The pre-flight's own number, mapped exactly as ``_predict`` does."""
    return estimate_stream_peak_vram(
        layer_bytes=shape["pool"] // 2,
        buffers=2,
        extras_bytes=shape["extras"],
        adapter_params=shape["adapter"],
        vocab_size=shape["vocab"],
        hidden_size=shape["hidden"],
        intermediate_size=shape["intermediate"],
        n_layers=shape["n_layers"],
        seq_len=seq,
        batch_size=batch,
    )


def _rows(seq: int) -> list[dict]:
    # Overshoot 3x; truncation lands it on the target. See module docstring.
    body = "word " * (seq * 3)
    return [
        {"messages": [{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": body}]}
        for _ in range(4)
    ]


def _config(shape: dict, seq: int, out_dir: pathlib.Path):
    doc = {
        "base": shape["model"],
        "task": "sft",
        "backend": "transformers",
        "modality": "text",
        "data": {
            "train": str(out_dir / "d.jsonl"),
            "max_length": seq,
            "chat_template": "chatml",
        },
        "output": str(out_dir / "out"),
        "training": {
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs": 1,
            "quantization": "none",
            "stream_layers": True,
            "logging_steps": 1,
            "save_steps": 100000,
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        },
    }
    return load_config_from_string(yaml.safe_dump(doc))


def run_one(shape: dict, seq: int) -> tuple[int, int]:
    """Return ``(real_peak_bytes, realised_seq_len)`` for one streamed step."""
    from soup_cli.trainer.sft import SFTTrainerWrapper

    tmp = pathlib.Path(tempfile.mkdtemp())
    (tmp / "d.jsonl").write_text(
        "\n".join(json.dumps(r) for r in _rows(seq)), encoding="utf-8"
    )
    (tmp / "out").mkdir(exist_ok=True)

    gc.collect()
    torch.cuda.empty_cache()
    wrapper = SFTTrainerWrapper(_config(shape, seq, tmp), device="cuda")
    wrapper.setup({"train": _rows(seq)})

    # Assert the shape rather than assume it - this is the guard.
    batch = next(iter(wrapper.trainer.get_train_dataloader()))
    realised = int(batch["input_ids"].shape[-1])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    wrapper.trainer.args.max_steps = 1
    wrapper.trainer.train()
    peak = torch.cuda.max_memory_allocated()

    try:
        wrapper._close_stream_runtime()
    except Exception:  # noqa: BLE001 - teardown must not mask the measurement
        pass
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return peak, realised


def sweep() -> None:
    """Sections 'The headline result' and 3."""
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    for shape, seqs in (
        (SMOL, (2048, 3072, 4096, 4352, 5120, 6144)),
        (QWEN, (2048, 4096, 5120, 6144)),
    ):
        print(f"\n## {shape['model']}")
        print(f"{'seq':>6} {'actual':>7} | {'predicted':>12} {'real peak':>12} "
              f"{'pred/real':>9} | {'14 x E':>11} {'non-logits':>10} {'over-est':>8}")
        for seq in seqs:
            peak, realised = run_one(shape, seq)
            pred = predict(shape, seq)
            elements = seq * shape["vocab"]
            at14 = estimate_logits_bytes(
                vocab_size=shape["vocab"], seq_len=seq, batch_size=1
            )
            non_modelled = pred - at14
            non_true = peak - LOGITS_LOSS_BYTES_PER_ELEMENT * elements
            flag = "" if at14 > peak else "  <- retained copy would FIT"
            print(f"{seq:>6} {realised:>7} | {pred/1e9:>10.4f}GB {peak/1e9:>10.4f}GB "
                  f"{pred/peak:>9.4f} | {at14/1e9:>9.3f}GB {non_modelled/1e9:>8.3f}GB "
                  f"{non_modelled/non_true:>8.3f}x{flag}")


def measure() -> None:
    """Section 1: the loss term, measured rather than back-solved."""
    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"shipped LOGITS_BYTES_PER_ELEMENT      = {LOGITS_BYTES_PER_ELEMENT}")
    print(f"shipped LOGITS_LOSS_BYTES_PER_ELEMENT = {LOGITS_LOSS_BYTES_PER_ELEMENT}")
    vals = [measure_logits_loss_bytes_per_element() for _ in range(3)]
    for i, v in enumerate(vals, 1):
        print(f"  repeat {i}: {v!r}")
    if all(v is not None for v in vals):
        print(f"  spread: {max(vals) - min(vals):.2e}")
    for vocab in (8192, 32768, 49152, 151936):
        print(f"  vocab {vocab:>6}: {measure_logits_loss_bytes_per_element(vocab_size=vocab)}")


def sdpa() -> None:
    """Section 4: the negative result, with its positive control.

    The math arm IS the control - it shows the hypothesis' cost is real and
    would be catastrophic, which is what makes 'the default never selects it'
    a finding rather than an absence of evidence.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    print(f"# torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"{'seq':>6} | {'flash':>12} | {'mem_efficient':>13} | {'math':>12} | {'default':>12}")
    for seq in (2048, 3072, 4096, 5120, 6144):
        query = torch.randn(1, 9, seq, 64, device="cuda", dtype=torch.bfloat16)
        out = {}
        for name, backend in (
            ("flash", SDPBackend.FLASH_ATTENTION),
            ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION),
            ("math", SDPBackend.MATH),
            ("default", None),
        ):
            try:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                if backend is None:
                    res = torch.nn.functional.scaled_dot_product_attention(
                        query, query, query, is_causal=True
                    )
                else:
                    with sdpa_kernel(backend):
                        res = torch.nn.functional.scaled_dot_product_attention(
                            query, query, query, is_causal=True
                        )
                torch.cuda.synchronize()
                out[name] = f"{(torch.cuda.max_memory_allocated() - base) / 1e6:.1f} MB"
                del res
            except Exception as exc:  # noqa: BLE001 - an unavailable backend is data
                out[name] = f"n/a ({type(exc).__name__})"
            gc.collect()
            torch.cuda.empty_cache()
        del query
        print(f"{seq:>6} | {out['flash']:>12} | {out['mem_efficient']:>13} | "
              f"{out['math']:>12} | {out['default']:>12}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true", help="loss term only")
    parser.add_argument("--sdpa", action="store_true", help="SDPA backends only")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("CUDA required: every figure in this record is a device peak.")
    if args.measure:
        measure()
    elif args.sdpa:
        sdpa()
    else:
        sweep()
