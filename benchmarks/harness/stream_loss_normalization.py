"""Paired FP32 training benchmark for shared chunk-loss normalization.

Run from the repository root with the dev environment installed::

    python benchmarks/harness/stream_loss_normalization.py --output /tmp/loss-ab.json

A tiny randomly initialized Llama learns synthetic modular sequences. This is
an end-to-end CPU training/parity check, not a pretrained-model quality eval or
a measurement of disk streaming. Both arms use the production ChunkedHeadLoss;
only _chunk_terms is replaced with its pre-optimization reference in the old arm.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from rich.console import Console


def legacy_chunk_terms(
    head: Any, piece: Any, target: Any, ignore_index: int,
) -> tuple[Any, Any, Any, Any]:
    """Reference from main at 7fd7df5, before sharing the FP32 normalization."""
    import torch
    from torch.nn import functional

    logits = head(piece)
    with torch.no_grad():
        log_probs = functional.log_softmax(logits, dim=-1)
        entropy = -(torch.exp(log_probs) * log_probs).sum(-1)
        labelled = target != ignore_index
        correct = ((logits.argmax(dim=-1) == target) & labelled).sum()
        at_label = log_probs.gather(-1, target.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        token_loss = (-at_label).float() * labelled
        del log_probs
    wide = logits.float()
    total = functional.cross_entropy(
        wide.view(-1, wide.shape[-1]),
        target.reshape(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    return total, entropy, token_loss, correct


def run_arm(
    seed: int, steps: int, legacy: bool,
) -> tuple[dict[str, float], list[float], list[Any], dict[str, Any], dict[str, Any]]:
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from soup_cli.utils import stream_loss_chunk as chunk_loss

    torch.manual_seed(seed)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=8192, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.0,
        tie_word_embeddings=False, use_cache=False,
    ))
    model.lm_head.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=0.003,
    )
    # Disjoint starting offsets for training and held-out examples. Mask the prompt in
    # one row and padding in the other, exercising both metric and loss masks.
    def batches(count: int, data_seed: int, heldout: bool = False) -> tuple[Any, Any]:
        generator = torch.Generator().manual_seed(data_seed)
        low, high = (48, 64) if heldout else (0, 48)
        starts = torch.randint(low, high, (count, 2, 1), generator=generator)
        strides = torch.randint(1, 5, (count, 2, 1), generator=generator)
        tokens = (starts + strides * torch.arange(128)) % 64
        labels = tokens.clone()
        labels[:, 0, :32] = -100
        labels[:, 1, -8:] = -100
        return tokens, labels

    inputs, labels = batches(steps + 10, seed + 1000)
    eval_inputs, eval_labels = batches(8, seed + 10000, heldout=True)
    hook = chunk_loss.ChunkedHeadLoss(model, 64)
    optimized = chunk_loss._chunk_terms
    if legacy:
        chunk_loss._chunk_terms = legacy_chunk_terms
    durations, curve, metrics = [], [], []
    first_grad = {}
    try:
        model.train()
        for index, (tokens, targets) in enumerate(zip(inputs, labels, strict=True)):
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=tokens, labels=targets).loss
            loss.backward()
            if index == 0:
                first_grad = {
                    name: param.grad.clone() for name, param in model.named_parameters()
                    if param.grad is not None
                }
            optimizer.step()
            elapsed = time.perf_counter() - start
            if index >= 10:
                durations.append(elapsed)
            curve.append(loss.item())
            step = hook.take_step()
            metrics.append(torch.cat((
                step.token_entropy.flatten(), step.token_loss.flatten(),
                step.correct.reshape(1).float(),
            )))
    finally:
        chunk_loss._chunk_terms = optimized
        hook.remove()
    model.eval()
    with torch.no_grad():
        heldout = [model(input_ids=tokens, labels=targets) for tokens, targets in
                   zip(eval_inputs, eval_labels, strict=True)]
        eval_loss = statistics.mean(output.loss.item() for output in heldout)
        correct = total = 0
        for output, targets in zip(heldout, eval_labels, strict=True):
            target = chunk_loss.shift_for_next_token(targets)
            mask = target != -100
            correct += ((output.logits.argmax(-1) == target) & mask).sum().item()
            total += mask.sum().item()
    return {
        "median_step_ms": 1000 * statistics.median(durations),
        "timed_training_ms": 1000 * sum(durations),
        "initial_loss": curve[0], "final_loss": curve[-1],
        "heldout_loss": eval_loss, "heldout_accuracy": correct / total,
    }, curve, metrics, first_grad, model.state_dict()


def main() -> None:
    import torch
    import transformers

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1:
        parser.error("repeats and steps must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pairs = []
    for repeat in range(args.repeats):
        seed = 1234 + repeat
        arms = {}
        # Alternate order to avoid systematically giving one arm warm caches.
        for legacy in ((True, False) if repeat % 2 == 0 else (False, True)):
            arms[legacy] = run_arm(seed, args.steps, legacy)
        old, new = arms[True], arms[False]
        deltas = {
            "train_loss_max_abs": max(abs(a - b) for a, b in zip(old[1], new[1], strict=True)),
            "metrics_max_abs": max((a - b).abs().max().item() for a, b in
                                   zip(old[2], new[2], strict=True)),
            "first_gradient_max_abs": max((old[3][key] - new[3][key]).abs().max().item()
                                          for key in old[3]),
            "final_weight_max_abs": max((old[4][key] - new[4][key]).abs().max().item()
                                        for key in old[4]),
            "heldout_loss_abs": abs(old[0]["heldout_loss"] - new[0]["heldout_loss"]),
            "heldout_accuracy_abs": abs(old[0]["heldout_accuracy"] - new[0]["heldout_accuracy"]),
        }
        pairs.append({
            "seed": seed, "order": "old,new" if repeat % 2 == 0 else "new,old",
            "old": old[0], "new": new[0], "deltas": deltas,
            "speedup": old[0]["timed_training_ms"] / new[0]["timed_training_ms"],
        })
    report = {
        "environment": {"platform": platform.platform(), "processor": platform.machine(),
                        "python": platform.python_version(), "torch": torch.__version__,
                        "transformers": transformers.__version__, "threads": 1},
        "config": {"timed_steps": args.steps, "warmup_steps": 10, "dtype": "float32",
                   "device": "cpu", "batch": 2, "sequence": 128, "chunk": 64,
                   "vocab": 8192, "hidden": 64, "layers": 2},
        "pairs": pairs,
        "median_speedup": statistics.median(pair["speedup"] for pair in pairs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Console().print_json(data=report)
    # A finite parity gate; timing remains evidence, never a flaky CI assertion.
    if any(not math.isfinite(value) or value > 1e-6
           for pair in pairs for value in pair["deltas"].values()):
        raise SystemExit("Training parity exceeded the 1e-6 absolute tolerance")


if __name__ == "__main__":
    main()
