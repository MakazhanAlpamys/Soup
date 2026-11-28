"""Frozen, paired causal-loss evaluation; no post-hoc quality threshold."""

import hashlib
import json
import math
import re
from pathlib import Path


def evaluate_examples(model, examples, make_batch, emit, stage):
    import torch

    emit("evaluation_" + stage + "_start")
    rows = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            inputs = make_batch(example["text"])
            ids = inputs["input_ids"].cpu().tolist()[0]
            loss = float(model(**inputs).loss)
            if len(ids) < 2 or not math.isfinite(loss):
                raise RuntimeError("invalid evaluation sequence or non-finite loss")
            row = {
                "id": example["id"],
                "loss": loss,
                "predicted_tokens": len(ids) - 1,
                "token_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode("ascii")
                ).hexdigest(),
            }
            rows.append(row)
            emit("evaluation_" + stage, **row)
    return rows


def load_evaluation(protocol_path, data_path):
    raw = Path(data_path).read_bytes()
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    if hashlib.sha256(raw).hexdigest() != protocol["evaluation_data_sha256"]:
        raise ValueError("evaluation data digest differs from the frozen protocol")
    data = json.loads(raw)
    examples = data["examples"]
    ids = [row["id"] for row in examples]
    if (
        len(ids) != 8
        or len(set(ids)) != 8
        or ids != protocol["example_ids"]
        or protocol["max_length"] != 64
    ):
        raise ValueError("evaluation selection differs from the eight-example protocol")
    for row in examples:
        text = row["text"]
        if (
            not isinstance(text, str)
            or not text.strip()
            or hashlib.sha256(text.encode("utf-8")).hexdigest() != row["text_sha256"]
        ):
            raise ValueError("evaluation text digest mismatch")
    return protocol, examples


def paired_summary(before, after, *, expected_ids):
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or [r["id"] for r in before] != expected_ids
        or [r["id"] for r in after] != expected_ids
    ):
        raise ValueError("missing, duplicate or reordered paired evaluation examples")
    for left, right in zip(before, after):
        for row in (left, right):
            if (
                type(row["loss"]) not in (float, int)
                or not math.isfinite(row["loss"])
                or row["loss"] < 0
                or type(row["predicted_tokens"]) is not int
                or row["predicted_tokens"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", row["token_sha256"])
            ):
                raise ValueError("invalid loss, token count or token digest")
        if (
            left["predicted_tokens"] != right["predicted_tokens"]
            or left["token_sha256"] != right["token_sha256"]
        ):
            raise ValueError("paired evaluation used different token IDs/counts")
    tokens = sum(row["predicted_tokens"] for row in before)
    means = [
        sum(row["loss"] * row["predicted_tokens"] for row in rows) / tokens
        for rows in (before, after)
    ]

    def perplexity(value):
        try:
            return math.exp(value)
        except OverflowError:
            return None  # NLL is retained even if perplexity exceeds float64.

    return {
        "examples": len(expected_ids),
        "predicted_tokens": tokens,
        "before_loss": means[0],
        "after_loss": means[1],
        "loss_delta": means[1] - means[0],
        "before_perplexity": perplexity(means[0]),
        "after_perplexity": perplexity(means[1]),
        "paired_loss_deltas": [b["loss"] - a["loss"] for a, b in zip(before, after)],
        "quality_gain_claimed": False,
        "scope": "eight fixed short test fragments; no generalization/convergence verdict",
    }
