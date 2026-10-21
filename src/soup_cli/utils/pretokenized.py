"""Shared persistent cache gate; no training-stack imports."""

import json
import os
from typing import Optional, Tuple

from rich.console import Console


def maybe_load_pretokenized(
    dcfg, base: str, console_obj: Console, tcfg=None, *, task: str,
) -> Optional[Tuple[object, object]]:
    """v0.53.7 #86 — short-circuit tokenization when caller pre-tokenized via
    ``soup data preprocess``.

    Returns ``(train_ds, eval_ds)`` when the pre-tokenized path is configured
    and valid, otherwise ``None`` (caller falls back to the normal tokenize
    pipeline).

    Cache-hash gate: when ``<tokenized_path>/metadata.json`` exists, its
    ``cache_key`` is cross-checked against the current
    ``(dataset, base, max_length, format, chat_template, mask_mode, task)``
    config via :func:`make_preprocess_cache_key`: every input that changes which
    rows are cached or what a cached row looks like (``PREPROCESS_KEY_FIELDS``).
    Mismatch raises ``ValueError`` with
    the keyword ``"cache hash mismatch"`` so users know to re-run
    ``soup data preprocess``. Missing ``metadata.json`` falls back to
    "trusted" mode with a yellow advisory.
    """
    if dcfg.format != "pre_tokenized" or not dcfg.tokenized_path:
        return None

    from soup_cli.data.chat_templates import resolve_chat_template
    from soup_cli.utils.data_pipeline import (
        load_pretokenized_dataset,
        make_preprocess_cache_key,
        preprocess_dataset_key_diff,
        preprocess_dataset_key_input,
        preprocess_mask_mode,
    )

    tokenized_path = dcfg.tokenized_path
    metadata_path = os.path.join(tokenized_path, "metadata.json")
    if os.path.isfile(metadata_path):
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"pre_tokenized metadata.json is unreadable: {exc}"
            ) from exc
        if not isinstance(metadata, dict):
            raise ValueError("pre_tokenized metadata.json must contain an object")
        stored_key = metadata.get("cache_key")
        # #1038: preprocess hashed the SOURCE format (chatml, alpaca, ...), which a
        # ``pre_tokenized`` config cannot restate -- ``dcfg.format`` is always
        # ``pre_tokenized`` here, so hashing it rejected every real cache. Use the
        # format preprocess recorded as an input to the recomputed key: it is not
        # trusted on its own, since editing it without the key still mismatches.
        # Metadata without the field (older hand-written caches) keeps the old input.
        source_format = metadata.get("format")
        if not isinstance(source_format, str) or not source_format:
            source_format = dcfg.format
        dataset_key = preprocess_dataset_key_input(dcfg)
        current_key = make_preprocess_cache_key(
            dataset_path=dataset_key,
            tokenizer_name=base,
            max_length=dcfg.max_length,
            format_name=source_format,
            # #1067: unlike the format, the template is restated in this config. It
            # has to match, since training saves the tokenizer with this template.
            chat_template=resolve_chat_template(dcfg.chat_template),
            mask_mode=preprocess_mask_mode(dcfg, tcfg),
            task=task,
        )
        if stored_key != current_key:
            # A cache without the field was written before that input joined the
            # key, so name the reason rather than leaving two hashes to compare
            # by eye. Oldest gap first: a cache missing both predates #1067, and
            # saying so places it further back than naming #1054 alone would.
            if "chat_template" not in metadata:
                predates = "the cache predates chat_template keying (#1067); "
            elif "mask_mode" not in metadata:
                predates = "the cache predates loss-mask keying (#1054); "
            elif "key_schema" not in metadata:
                predates = "the cache predates row-set keying (#1127); "
            else:
                changed = preprocess_dataset_key_diff(
                    metadata.get("dataset_key"), dataset_key
                )
                predates = (
                    "the cache was built with a different "
                    + ", ".join(f"data.{name}" for name in changed)
                    + "; "
                    if changed
                    else ""
                )
            raise ValueError(
                "pre_tokenized cache hash mismatch: was generated with "
                f"{stored_key!r}, current config implies {current_key!r}; "
                f"{predates}re-run `soup data preprocess`"
            )
    else:
        console_obj.print(
            "[yellow]pre-tokenized cache has no metadata.json — assuming "
            "compatible; consider re-running `soup data preprocess`[/]"
        )

    console_obj.print(
        f"[dim]v0.53.7: skipping tokenization, loading pre-tokenized "
        f"Arrow shards from {tokenized_path}[/dim]"
    )
    arrow_ds = load_pretokenized_dataset(tokenized_path)
    # ``load_from_disk`` returns either a Dataset (single split) or a
    # DatasetDict (multiple splits). Handle both shapes.
    if hasattr(arrow_ds, "keys") and "train" in arrow_ds:
        train_ds = arrow_ds["train"]
        eval_ds = arrow_ds.get("val") or arrow_ds.get("validation")
    else:
        train_ds = arrow_ds
        eval_ds = None
    return train_ds, eval_ds
