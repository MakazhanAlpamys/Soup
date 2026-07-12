"""Speculative decoding auto-pairing (v0.30.0).

Maps a target model to a known-good draft model. Draft+target overhead is
only worth it at ~30B+ — below that, pair returns None.
"""

from __future__ import annotations

import re
from typing import Optional

# Draft models picked for vocab / tokenizer match with the target family.
# Rule of thumb: draft 10-50x smaller than target, same architecture family.
_DRAFT_PAIRS: dict[str, str] = {
    # Llama 3 family (share Llama-3 tokenizer)
    "meta-llama/llama-3.1-70b": "meta-llama/Llama-3.2-1B",
    "meta-llama/llama-3.1-70b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-3.3-70b": "meta-llama/Llama-3.2-1B",
    "meta-llama/llama-3.3-70b-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-3.1-405b": "meta-llama/Llama-3.2-3B",
    "meta-llama/llama-3.1-405b-instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # Llama 4
    "meta-llama/llama-4-scout-17b-16e": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct": "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/llama-4-maverick-17b-128e": "meta-llama/Llama-3.2-1B-Instruct",
    # Qwen 2.5 / 3
    "qwen/qwen2.5-72b": "Qwen/Qwen2.5-0.5B",
    "qwen/qwen2.5-72b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen/qwen2.5-32b": "Qwen/Qwen2.5-0.5B",
    "qwen/qwen2.5-32b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen/qwen3-32b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen/qwen3-14b": "Qwen/Qwen2.5-0.5B-Instruct",
    # Mistral / Mixtral
    "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/mistral-large-instruct-2407": "mistralai/Mistral-7B-Instruct-v0.3",
    # DeepSeek
    "deepseek-ai/deepseek-v3": "deepseek-ai/DeepSeek-V3-0324",
    "deepseek-ai/deepseek-r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # Gemma 2/3 (large)
    "google/gemma-2-27b": "google/gemma-2-2b",
    "google/gemma-2-27b-it": "google/gemma-2-2b-it",
    "google/gemma-3-27b": "google/gemma-3-4b",
    "google/gemma-3-27b-it": "google/gemma-3-4b-it",
}


_HF_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-./]*$")


def _is_valid_hf_id(name: str) -> bool:
    if not name or len(name) > 200:
        return False
    if "\x00" in name:
        return False
    # Block URLs
    if name.startswith(("http://", "https://", "file://")):
        return False
    return bool(_HF_ID_RE.match(name))


def _lookup_local_draft(target: str) -> Optional[str]:
    """Draft trained locally by ``soup draft distill`` (v0.71.33), or None."""
    from soup_cli.utils.draft import lookup_draft

    return lookup_draft(target)


def pick_draft_model(target: str) -> Optional[str]:
    """Pick a known-good draft model for the target.

    A draft the user trained themselves via ``soup draft distill`` (recorded in
    the local registry, v0.71.33) wins over the built-in pairing table — it was
    distilled from *this* target, so its acceptance rate is strictly better
    informed than a generic same-family pick.

    Returns None for unknown / too-small targets. Target names are normalised
    to lowercase for matching. URLs and null-byte names are rejected for
    defence-in-depth (the caller loads the returned value via
    ``AutoModelForCausalLM.from_pretrained``).
    """
    if not _is_valid_hf_id(target):
        return None
    key = target.strip().lower()

    # This runs on `soup serve` startup: a corrupt or unreadable registry must
    # degrade to the static map, never take the server down.
    try:
        local = _lookup_local_draft(key)
    except Exception:  # noqa: BLE001 — registry problems are never fatal here
        local = None
    if local:
        return local

    # Strip trailing path segments (e.g. revisions) not present in map
    return _DRAFT_PAIRS.get(key)
