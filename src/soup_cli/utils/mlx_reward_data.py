"""Pure text/length and tail-preserving batch contracts for native reward training."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

TOKENIZER_REFUSAL = "MLX reward: fast tokenizer with offsets and EOS required."


@dataclass(frozen=True)
class RewardPair:
    index: int
    chosen: list[int]
    rejected: list[int]


def check_tokenizer(tokenizer: Any) -> None:
    if (
        not getattr(tokenizer, "is_fast", False)
        or type(getattr(tokenizer, "eos_token_id", None)) is not int
    ):
        raise ValueError(TOKENIZER_REFUSAL)


def encode_completion(
    tokenizer: Any,
    prompt: str,
    completion: str,
    max_length: int,
) -> tuple[list[int], bool, int]:
    """Return EOS-terminated ids, content-lost flag and untruncated token count.

    Offsets overlap non-whitespace completion characters: a BPE token may start
    in the separator preceding the answer. Special tokens' zero offsets never
    count. Callers distinguish malformed content from truncation loss.
    """
    check_tokenizer(tokenizer)
    text = prompt + " " + completion if prompt else completion
    answer_start = len(prompt) + 1 if prompt else 0
    content_positions = {
        answer_start + i for i, char in enumerate(completion) if not char.isspace()
    }
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    eos = tokenizer.eos_token_id
    content = [
        token != eos and any(pos in content_positions for pos in range(start, end))
        for token, (start, end) in zip(ids, offsets, strict=True)
    ]
    if not any(content):
        raise ValueError("no non-EOS content tokens")
    if not ids or ids[-1] != eos:
        ids.append(eos)
        content.append(False)
    count = len(ids)
    if count > max_length:
        ids = ids[: max_length - 1] + [eos]
        content = content[: max_length - 1] + [False]
    return ids, not any(content), count


def prepare_pairs(
    rows: Any,
    tokenizer: Any,
    max_length: int,
    split: str,
) -> tuple[list[RewardPair], dict]:
    """Validate both sides before filtering; preserve normalized source indices."""
    pairs = []
    stats = {"retained": 0, "skipped_lost": 0, "skipped_collapsed": 0, "first_skip": None}
    for index, row in enumerate(rows):
        prefix = f"reward_model {split} row {index}"
        if not isinstance(row, Mapping):
            raise ValueError(
                f"{prefix}: expected a prompt/chosen/rejected mapping. Supply a preference row."
            )
        values = {"prompt": row.get("prompt", "")}
        for side in ("chosen", "rejected"):
            if side not in row:
                raise ValueError(f"{prefix} {side}: required field is missing. Supply this answer.")
            values[side] = row[side]
        for field, value in values.items():
            if not isinstance(value, str):
                raise ValueError(
                    f"{prefix} {field}: expected a string. "
                    "Supply string prompt/chosen/rejected fields."
                )
            if field != "prompt" and not value.strip():
                raise ValueError(f"{prefix} {field}: blank answer. Supply a nonblank answer.")
        encoded = {}
        for side in ("chosen", "rejected"):
            try:
                encoded[side] = encode_completion(
                    tokenizer,
                    values["prompt"],
                    values[side],
                    max_length,
                )
            except ValueError as exc:
                if str(exc) != "no non-EOS content tokens":
                    raise
                raise ValueError(
                    f"{prefix} {side}: no non-EOS content tokens before truncation. "
                    "Supply a meaningful answer."
                ) from exc
        lost = next((side for side, data in encoded.items() if data[1]), None)
        diagnostic = None
        if lost:
            stats["skipped_lost"] += 1
            diagnostic = (
                f"{prefix} {lost}: completion lost at max_length={max_length}. "
                "Raise data.max_length or shorten the prompt."
            )
        elif encoded["chosen"][0] == encoded["rejected"][0]:
            stats["skipped_collapsed"] += 1
            diagnostic = (
                f"{prefix} chosen/rejected: identical after tokenization/truncation. "
                "Supply distinguishable answers; if truncated, raise data.max_length "
                "or shorten the prompt."
            )
        if diagnostic:
            if stats["first_skip"] is None:
                stats["first_skip"] = diagnostic
        else:
            pairs.append(RewardPair(index, encoded["chosen"][0], encoded["rejected"][0]))
    if not pairs:
        raise ValueError(
            f"reward_model {split}: no usable preference rows remain. "
            "Correct the reported rows or length setting."
        )
    stats["retained"] = len(pairs)
    return pairs, stats


def plan_windows(
    count: int,
    batch_size: int,
    accumulation: int,
    epochs: int,
    seed: int,
) -> list[list[list[list[int]]]]:
    """Epoch -> optimizer window -> microbatch -> pair indices, including tails."""
    if any(
        type(value) is not int or value < 1
        for value in (
            count,
            batch_size,
            accumulation,
            epochs,
        )
    ):
        raise ValueError("reward_model: planner dimensions must be positive integers.")
    rng = random.Random(seed)
    result = []
    for _ in range(epochs):
        indices = list(range(count))
        rng.shuffle(indices)
        batches = [indices[start : start + batch_size] for start in range(0, count, batch_size)]
        result.append(
            [
                batches[start : start + accumulation]
                for start in range(0, len(batches), accumulation)
            ]
        )
    return result


def padded_arrays(sequences: list[list[int]], pad_id: int) -> tuple[Any, Any]:
    """Validity derives from sequence lengths, including real EOS == PAD tokens."""
    import mlx.core as mx

    width = max(map(len, sequences))
    tokens = [ids + [pad_id] * (width - len(ids)) for ids in sequences]
    masks = [[True] * len(ids) + [False] * (width - len(ids)) for ids in sequences]
    return mx.array(tokens), mx.array(masks)
