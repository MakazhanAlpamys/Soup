"""Reconstruct saved Transformers adapter vocabularies before PEFT attachment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _adapter_file(adapter: str, filename: str) -> str | None:
    if os.path.isdir(adapter):
        path = Path(adapter) / filename
        return str(path) if path.is_file() else None
    from transformers.utils.hub import cached_file

    return cached_file(adapter, filename, _raise_exceptions_for_missing_entries=False)


def _weight_shapes(adapter: str) -> dict[str, tuple[int, ...]] | None:
    filename = _adapter_file(adapter, "adapter_model.safetensors")
    if filename is not None:
        from safetensors import safe_open

        # Read shapes without materializing a second copy of full embeddings.
        with safe_open(filename, framework="pt", device="cpu") as weights:
            return {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    filename = _adapter_file(adapter, "adapter_model.bin")
    if filename is None:
        return None
    import torch

    weights = torch.load(filename, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in weights.items()
    ):
        raise ValueError("adapter weights must be a tensor state dictionary")
    return {name: tuple(value.shape) for name, value in weights.items()}


def prepare_adapter_vocab(
    model: Any,
    adapter: str,
    base: str,
    *,
    tokenizer: Any = None,
    base_tokenizer: Any = None,
    trust_remote_code: bool = False,
    revision_kwargs: dict[str, str] | None = None,
    quantization: str | None = None,
) -> Any:
    """Select saved token IDs, validate full embedding state and resize before PEFT.

    ``tokenizer`` is a preloaded adapter tokenizer (merge's existing preflight).
    Evaluation supplies its ``base_tokenizer`` as the legacy fallback. Missing
    embedding rows are never recreated from random initialization for expansion.
    Ordinary LoRA weights can use the saved tokenizer when the base has none,
    provided its token IDs fit the existing embeddings.
    """
    from transformers import AutoTokenizer

    if tokenizer is None and _adapter_file(adapter, "tokenizer_config.json") is not None:
        tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=trust_remote_code)
    saved_tokenizer = tokenizer is not None
    shapes = _weight_shapes(adapter)
    if shapes is None:
        # PEFT reports missing weights at attachment.
        return tokenizer if saved_tokenizer else base_tokenizer
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    saved_modules: set[str] = set()
    row_counts: set[int] = set()
    for name, module in model.named_modules():
        if module is not embedding and module is not head:
            continue
        for suffix in ("weight", "base_layer.weight", "original_module.weight"):
            shape = shapes.get(f"base_model.model.{name}.{suffix}")
            if shape is not None:
                if len(shape) != 2 or shape[0] < 1 or shape[1:] != tuple(module.weight.shape[1:]):
                    raise ValueError(
                        "saved adapter embedding/head shape is incompatible with the base"
                    )
                saved_modules.add(name)
                row_counts.add(shape[0])
    if len(row_counts) > 1:
        raise ValueError("saved adapter embedding and head shapes disagree")
    original_rows = embedding.weight.shape[0]
    target_rows = next(iter(row_counts), original_rows)
    if base_tokenizer is None:
        try:
            base_tokenizer = AutoTokenizer.from_pretrained(
                base, trust_remote_code=trust_remote_code, **(revision_kwargs or {}),
            )
        except (OSError, ValueError):
            # Legacy merge needs only the adapter tokenizer for ordinary LoRA weights.
            if not row_counts and saved_tokenizer and len(tokenizer) <= original_rows and all(
                token_id < original_rows for token_id in tokenizer.get_vocab().values()
            ):
                return tokenizer
            raise
    if tokenizer is None:
        tokenizer = base_tokenizer
    vocabulary = tokenizer.get_vocab()
    for token, token_id in base_tokenizer.get_vocab().items():
        if vocabulary.get(token) != token_id:
            raise ValueError("adapter tokenizer changes an existing base token ID")
    expanded = target_rows != original_rows or len(tokenizer) > len(base_tokenizer)
    if expanded:
        if not saved_tokenizer:
            raise ValueError("expanded adapter requires its saved tokenizer")
        for module, label in ((embedding, "embedding"), (head, "head")):
            names = [name for name, candidate in model.named_modules() if candidate is module]
            if not any(name in saved_modules for name in names):
                raise ValueError(f"expanded adapter is missing its saved {label} weights")
        if target_rows < len(tokenizer) or any(
            token_id >= target_rows for token_id in vocabulary.values()
        ):
            raise ValueError("saved tokenizer does not fit the adapter embedding/head shape")
        if quantization not in (None, "none") or getattr(model, "is_quantized", False):
            raise ValueError(
                "expanded adapter vocabulary requires an unquantized base (#3351); "
                "quantized reconstruction is not supported"
            )
        model.resize_token_embeddings(target_rows)
    return tokenizer
