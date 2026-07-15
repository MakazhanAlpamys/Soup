"""v0.71.36 — the one embedding kernel (Data Moat II).

``transformers.AutoModel`` + attention-masked mean-pool + L2-normalize.
This is exactly what sentence-transformers does for MiniLM-class models
(their own model card documents the equivalence), so Soup needs NO new
dependency — ``transformers`` already ships in the ``[train]`` extra.

Torch is imported lazily inside :func:`embed_texts`; this module must stay
importable on the light core.
"""

from __future__ import annotations

import json
from typing import Optional

# Models whose pooling is verified pure-mean. Short-circuits the hub fetch.
# all-mpnet-base-v2 already ships as the `ra-dit-retriever` recipe base.
POOLING_ALLOWLIST: dict[str, str] = {
    "sentence-transformers/all-minilm-l6-v2": "mean",
    "sentence-transformers/all-minilm-l12-v2": "mean",
    "sentence-transformers/all-mpnet-base-v2": "mean",
}

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_MAX_MODEL_ID_CHARS = 256

# Keys in 1_Pooling/config.json that are NOT plain mean pooling. If any is
# true we refuse — mean-pooling such a model emits wrong vectors silently.
_NON_MEAN_POOLING_KEYS = (
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_max_tokens", "max"),
    ("pooling_mode_weightedmean_tokens", "weightedmean"),
    ("pooling_mode_lasttoken", "lasttoken"),
)


def _require_model_id(model_id: object) -> str:
    if isinstance(model_id, bool) or not isinstance(model_id, str):
        raise TypeError(
            f"model_id must be str, got {type(model_id).__name__}"
        )
    cleaned = model_id.strip()
    if not cleaned:
        raise ValueError("model_id must be a non-empty string")
    if "\x00" in cleaned:
        raise ValueError("model_id must not contain null bytes")
    if len(cleaned) > _MAX_MODEL_ID_CHARS:
        raise ValueError(
            f"model_id too long (max {_MAX_MODEL_ID_CHARS} chars)"
        )
    return cleaned


def _fetch_pooling_config(model_id: str) -> Optional[dict]:
    """Read ``1_Pooling/config.json`` from the hub repo.

    Returns None when the file is absent or unreadable — the caller then
    REFUSES rather than assuming mean pooling.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id, filename="1_Pooling/config.json"
        )
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_pooling(model_id: str) -> str:
    """Return ``"mean"`` for a verified mean-pooled model, else raise.

    Never guesses. An unverifiable model is refused, because silently
    mean-pooling a CLS model produces wrong vectors and every downstream
    number is then quietly garbage.
    """
    cleaned = _require_model_id(model_id)
    allow = POOLING_ALLOWLIST.get(cleaned.lower())
    if allow is not None:
        return allow

    config = _fetch_pooling_config(cleaned)
    if config is None:
        raise ValueError(
            f"cannot verify pooling for {cleaned!r}: no 1_Pooling/config.json "
            "and not in the verified allowlist. Soup refuses rather than "
            "assume mean-pooling (wrong vectors would be silent). Use "
            f"{DEFAULT_EMBED_MODEL} or another allowlisted model."
        )
    for key, label in _NON_MEAN_POOLING_KEYS:
        if config.get(key):
            raise ValueError(
                f"{cleaned!r} uses {label!r} pooling; Soup's embedding kernel "
                "implements mean-pooling only. Refusing rather than emitting "
                "wrong vectors."
            )
    if not config.get("pooling_mode_mean_tokens"):
        raise ValueError(
            f"{cleaned!r} does not declare mean-token pooling; refusing."
        )
    return "mean"
