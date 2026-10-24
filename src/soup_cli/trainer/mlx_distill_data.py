"""Verifiable shared token IDs and the existing native text/masking path for KD."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any


def _tokenizer_description(tokenizer: Any) -> dict[str, Any]:
    from transformers import PreTrainedTokenizerFast

    underlying, wrapper = tokenizer, None
    cls = type(tokenizer)
    if (cls.__module__, cls.__name__) == ("mlx_lm.tokenizer_utils", "TokenizerWrapper"):
        # Import MLX only for its real wrapper, preserving portable HF checks.
        from mlx_lm.tokenizer_utils import TokenizerWrapper

        if type(tokenizer) is not TokenizerWrapper or tokenizer._chat_template is not None:
            raise ValueError("MLX distillation cannot verify custom tokenizer renderers")
        underlying = tokenizer._tokenizer
        # MLX-LM 0.31 uses a fixed keyword; 0.32 stores its remapping.
        wrapper = {"thinking_keyword": getattr(tokenizer, "_thinking_kwarg", "enable_thinking"),
                   "thinking_default": tokenizer.has_thinking}
    if type(underlying) is not PreTrainedTokenizerFast or any(
        name in vars(obj) for obj in (tokenizer, underlying) for name in (
            "encode", "__call__", "apply_chat_template", "get_chat_template",
            "encode_plus", "_encode_plus", "batch_encode_plus", "_batch_encode_plus",
        )
    ):
        raise ValueError("MLX distillation cannot verify custom tokenizer encoding or rendering")
    try:
        pipeline = json.loads(tokenizer.backend_tokenizer.to_str())
        vocab = tokenizer.get_vocab()
        special = tokenizer.special_tokens_map
        eos_ids = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        description = {
            "pipeline": pipeline,
            "vocab": vocab,
            "special_tokens": special,
            "ids": {name: getattr(tokenizer, name, None) for name in (
                "bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id",
            )},
            "eos_token_ids": sorted(eos_ids or []),
            "chat_template": tokenizer.chat_template,
            # These live in Python, outside the serialized Rust pipeline.
            "encoding_settings": {name: getattr(underlying, name, None) for name in (
                "split_special_tokens", "add_bos_token", "add_eos_token", "add_prefix_space",
                "padding_side", "truncation_side", "model_max_length", "model_input_names",
                "clean_up_tokenization_spaces",
            )},
            "wrapper_rendering": wrapper,
        }
        json.dumps(description, allow_nan=False)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("MLX distillation cannot verify the tokenizer pipeline") from exc
    if not vocab or any(type(i) is not int or i < 0 for i in vocab.values()):
        raise ValueError("MLX distillation tokenizer has invalid token IDs")
    return description


def compatibility_identity(
    student: Any, teacher: Any, student_vocab: int, teacher_vocab: int,
) -> dict[str, Any]:
    """Compare complete pipelines/mappings, not a finite set of probe encodings.

    Matching padded output vocabularies are allowed; every named token must fit.
    Return a compact fingerprint rather than embedding two full vocabularies in
    an adapter's metadata. Different model depths/widths are not tokenizer identity.
    """
    left, right = _tokenizer_description(student), _tokenizer_description(teacher)
    if left != right:
        raise ValueError(
            "MLX distillation requires identical tokenizer pipelines, IDs and templates"
        )
    if (
        type(student_vocab) is not int or type(teacher_vocab) is not int
        or student_vocab != teacher_vocab or student_vocab <= max(left["vocab"].values())
    ):
        raise ValueError("MLX distillation output vocabulary dimensions do not match token IDs")
    canonical = json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "method": "complete-fast-tokenizer-pipeline-mapping-and-effective-settings",
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "named_tokens": len(left["vocab"]),
        "output_vocabulary_size": student_vocab,
    }


@dataclass(frozen=True)
class TokenExample:
    tokens: tuple[int, ...]
    labels: tuple[int, ...]
    active_count: int


def build_examples(
    rows: list[dict[str, Any]], tokenizer: Any, *, max_length: int,
    responses_only: bool, split: str,
) -> list[TokenExample]:
    """Use native tokenization and Soup's checked per-assistant response masks."""
    from mlx_lm.tuner.datasets import create_dataset

    from soup_cli.trainer.mlx_masking import build_response_mask

    examples = []
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"MLX distillation {split} row {position}: expected a row mapping")
        if row.get("chat_template_kwargs"):
            raise ValueError(
                f"MLX distillation {split} row {position}: chat_template_kwargs is unsupported"
            )
        try:
            messages = row.get("messages")
            if messages is None and "prompt" in row and "completion" in row:
                messages = [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            if responses_only and messages is not None:
                tokens, mask = build_response_mask(messages, tokenizer, tools=row.get("tools"))
            else:
                native = create_dataset([row], tokenizer, SimpleNamespace(mask_prompt=False))
                tokens, _ = native.process(row)
                mask = [1] * len(tokens)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"MLX distillation {split} row {position}: {exc}") from exc
        tokens, mask = list(tokens[:max_length]), mask[:max_length]
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(
                f"MLX distillation {split} row {position}: invalid tokenizer token IDs"
            )
        labels = tuple(token if active else -100 for token, active in zip(tokens, mask))
        count = sum(label != -100 for label in labels[1:])
        if count == 0:
            raise ValueError(
                f"MLX distillation {split} row {position}: no causal-loss target "
                f"after data.max_length={max_length}"
            )
        examples.append(TokenExample(tuple(tokens), labels, count))
    return examples


def collate(examples: list[TokenExample], *, pad_id: int, max_length: int) -> tuple:
    """Pad IDs separately from labels: pad=EOS must not mask real EOS targets."""
    import mlx.core as mx

    width = min(max_length, ((max(len(row.tokens) for row in examples) + 31) // 32) * 32)
    tokens, labels, active = [], [], []
    for i, row in enumerate(examples):
        tokens.append([*row.tokens, *([pad_id] * (width - len(row.tokens)))])
        labels.append([*row.labels, *([-100] * (width - len(row.labels)))])
        active.extend(
            i * (width - 1) + j for j, value in enumerate(labels[-1][1:]) if value != -100
        )
    return mx.array(tokens), mx.array(labels), mx.array(active, dtype=mx.int32)
