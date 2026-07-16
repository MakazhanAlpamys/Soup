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
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
    REFUSES rather than assuming mean pooling, so every failure here is
    fail-CLOSED. The breadth of the ``except`` is therefore safe, but a
    hub outage and a genuinely-absent file are indistinguishable to the
    user, so the reason is logged at debug level before returning.
    """
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=model_id, filename="1_Pooling/config.json"
        )
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 — fail-closed; caller refuses
        logger.debug(
            "could not fetch 1_Pooling/config.json for %r: %s: %s",
            model_id,
            type(exc).__name__,
            exc,
        )
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


# ---------------------------------------------------------------------------
# Batched encode
# ---------------------------------------------------------------------------

_MAX_ROWS = 200_000
_MAX_CHARS_PER_ROW = 8_192
_MAX_BATCH_SIZE = 512
_MAX_SEQ_TOKENS = 512


def _mean_pool(hidden, attention_mask):
    """Attention-masked mean over the token axis. Padding contributes 0.

    An unmasked ``hidden.mean(dim=1)`` averages padded positions too, which
    silently drags every vector toward the pad embedding.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)  # never divide by zero
    return summed / counts


def _l2_normalize(vectors):
    """Row-wise L2 normalize so cosine == dot product. Zero rows stay zero."""
    import numpy as np

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (vectors / norms).astype(np.float32)


def _validate_texts(texts: object) -> list:
    """Coerce + bound-check the input rows.

    A bare ``str`` is rejected: it is a sequence of characters, so accepting
    it would silently embed one row per letter.
    """
    if isinstance(texts, (str, bytes)) or not hasattr(texts, "__len__"):
        raise TypeError("texts must be a sequence of str")
    items = list(texts)
    if not items:
        raise ValueError("texts must contain at least one text")
    if len(items) > _MAX_ROWS:
        raise ValueError(
            f"too many texts ({len(items)}); cap is {_MAX_ROWS}. Sample the "
            "dataset first (`soup data sample`) — Soup refuses rather than "
            "silently subsampling."
        )
    for idx, item in enumerate(items):
        if not isinstance(item, str):
            raise TypeError(
                f"texts[{idx}] must be str, got {type(item).__name__}"
            )
    return [item[:_MAX_CHARS_PER_ROW] for item in items]


def _require_batch_size(batch_size: object) -> int:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            f"batch_size must be int, got {type(batch_size).__name__}"
        )
    if batch_size < 1 or batch_size > _MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be in [1, {_MAX_BATCH_SIZE}], got {batch_size}"
        )
    return batch_size


def embed_texts(
    texts,
    *,
    model_id: str = DEFAULT_EMBED_MODEL,
    device: str = "auto",
    batch_size: int = 32,
):
    """Embed ``texts`` -> an ``(n, d)`` float32 array with L2-normalized rows.

    Torch / transformers / numpy are imported lazily so this module stays
    importable on the light core (a ``pip install soup-cli`` without the
    ``[train]`` extra).
    """
    items = _validate_texts(texts)
    batch = _require_batch_size(batch_size)
    # Refuse an unverified model BEFORE any download starts.
    resolve_pooling(model_id)

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from soup_cli.utils.live_eval import resolve_device

    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(dev)
    model.eval()

    chunks = []
    with torch.no_grad():
        for start in range(0, len(items), batch):
            encoded = tokenizer(
                items[start: start + batch],
                padding=True,
                truncation=True,
                max_length=_MAX_SEQ_TOKENS,
                return_tensors="pt",
            ).to(dev)
            out = model(**encoded)
            pooled = _mean_pool(out.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.float().cpu().numpy())
    return _l2_normalize(np.vstack(chunks))
