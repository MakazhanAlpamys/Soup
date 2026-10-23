"""The preprocess cache key of every schema generation, shared by the key tests (#1127).

``make_preprocess_cache_key`` hashes a ``\\x1f``-joined blob whose segments were
appended one issue at a time. Each ``TestCacheKey`` class used to write that blob
out by hand, so every new key input was a merge surprise in three files. They now
rebuild historical keys here, and ``test_issue1127_preprocess_cache_key_fields``
checks that the newest generation matches the production function's own
parameters, so a new key input cannot land without this table seeing it.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Tuple

_V2_SEGMENTS = ("dataset_path", "tokenizer_name", "max_length", "format_name")
_V4_SEGMENTS = _V2_SEGMENTS + ("chat_template",)
_V6_SEGMENTS = _V4_SEGMENTS + ("mask_mode",)
_V7_SEGMENTS = _V6_SEGMENTS + ("task",)

# Segments each generation hashed, in writer order. "" is the pre-schema blob,
# which had no schema token in front.
SEGMENTS_BY_SCHEMA: Mapping[str, Tuple[str, ...]] = {
    "": _V2_SEGMENTS,
    "v2": _V2_SEGMENTS,  # #785
    "v3": _V2_SEGMENTS,  # #791
    "v4": _V4_SEGMENTS,  # #1067
    "v5": _V4_SEGMENTS,  # #876
    "v6": _V6_SEGMENTS,  # #1054
    "v7": _V7_SEGMENTS,  # #1127
}

CURRENT_SCHEMA = "v7"


def blob_key(schema: str, **inputs: object) -> str:
    """Hash ``inputs`` exactly as the ``schema`` generation of the writer did.

    ``chat_template`` of ``None`` is written as the empty string, as the writer
    does. Every segment the generation had must be given; extra inputs are
    ignored, so one argument dict can be replayed through every generation.
    """
    segments = SEGMENTS_BY_SCHEMA[schema]
    values = []
    for name in segments:
        value = inputs[name]
        values.append("" if value is None else str(value))
    prefix = f"{schema}\x1f" if schema else ""
    blob = prefix + "\x1f".join(values)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
