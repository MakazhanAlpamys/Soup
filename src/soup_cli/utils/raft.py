"""v0.71.10 #199 — live RAFT (Retrieval-Augmented Fine-Tuning) data builders.

RAFT (Stanford 2024) trains a model to answer a query given a *golden* document
mixed with *distractor* documents, so the model learns to attend to the relevant
context and ignore noise. This module ships the pure-Python composition + answer
-only loss-mask tokenisation consumed by the live RAFT trainer
(:mod:`soup_cli.trainer.raft`).

Design:

* ``build_raft_prompt`` combines ``golden_doc`` + ``distractor_docs`` (shuffled
  for distractor robustness — reproducibly when ``shuffle_seed`` is set),
  labels every document with a deterministic ``[doc-N]`` id, and returns the
  prompt + answer + the golden document's id. The ``[doc-N]`` ids let the
  answer cite the supporting document verbatim (composes with the v0.62.0
  citation-faithful kernel — #202).
* ``tokenize_raft_example`` builds ``{input_ids, attention_mask, labels,
  loss_weights}`` with the prompt span masked to ``-100``. When
  ``citation_faithful`` is set it boosts ``loss_weights`` on bracketed citation
  spans inside the answer (the span-mask from #199 / #202).

No heavy imports — everything operates on python lists / strings so it runs
inside a ``datasets.Dataset.map`` without torch. ``return_offsets_mapping``
needs a fast tokenizer; when unavailable the citation boost degrades to a flat
answer mask (logged at DEBUG, never raises).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Caps mirror the v0.62.0 ``_convert_raft`` limits.
_MAX_DOCS = 65  # golden + up to 64 distractors
_MAX_FIELD_LEN = 65_536
_DEFAULT_CITATION_BOOST = 5.0
_MAX_CITATION_BOOST = 100.0

#: Instruction prepended to every RAFT prompt. The bracket-citation hint pairs
#: with the v0.62.0 citation-faithful span mask (#202).
RAFT_INSTRUCTION = (
    "Answer the question using only the relevant document(s) below. "
    "Cite the supporting document id in [brackets]."
)


@dataclass(frozen=True)
class RaftComposed:
    """One composed RAFT example: prompt + answer + the golden doc id."""

    prompt: str
    answer: str
    golden_doc_id: str
    doc_ids: Tuple[str, ...]


def _doc_id(index: int) -> str:
    return f"doc-{index}"


def _check_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"RAFT {name!r} must be a string, got {type(value).__name__}")
    if not value:
        raise ValueError(f"RAFT {name!r} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"RAFT {name!r} must not contain null bytes")
    if len(value) > _MAX_FIELD_LEN:
        raise ValueError(f"RAFT {name!r} must be <= {_MAX_FIELD_LEN} chars")
    return value


def build_raft_prompt(
    row: Mapping[str, Any],
    *,
    shuffle_seed: Optional[int] = None,
    row_index: int = 0,
) -> RaftComposed:
    """Compose a RAFT row into ``(prompt, answer, golden_doc_id, doc_ids)``.

    ``row`` is a ``{query, golden_doc, distractor_docs, answer}`` mapping (the
    output of ``data.formats._convert_raft``). The golden doc and distractors
    are concatenated and shuffled deterministically — when ``shuffle_seed`` is
    ``None`` a fixed seed (0) is used so the order is still reproducible (the
    documents ARE shuffled for distractor robustness; the knob only controls
    *which* reproducible permutation). Each document is labelled ``[doc-N]`` in
    its shuffled position, and the golden document's id is returned so callers
    (citation scorer / span-mask) know the ground-truth citation.

    Raises ``ValueError`` on malformed rows (mirrors ``_convert_raft`` policy).
    """
    if not isinstance(row, Mapping):
        raise ValueError(f"RAFT row must be a mapping, got {type(row).__name__}")
    if isinstance(shuffle_seed, bool) or (
        shuffle_seed is not None and not isinstance(shuffle_seed, int)
    ):
        raise TypeError("shuffle_seed must be int or None")
    if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index < 0:
        raise ValueError("row_index must be a non-negative int")

    query = _check_str("query", row.get("query"))
    golden_doc = _check_str("golden_doc", row.get("golden_doc"))
    answer = _check_str("answer", row.get("answer"))
    raw_distractors = row.get("distractor_docs", []) or []
    if not isinstance(raw_distractors, list):
        raise ValueError("RAFT 'distractor_docs' must be a list")
    distractors = [
        _check_str(f"distractor_docs[{i}]", d) for i, d in enumerate(raw_distractors)
    ]
    if 1 + len(distractors) > _MAX_DOCS:
        raise ValueError(f"RAFT example has > {_MAX_DOCS} documents")

    # (text, is_golden) pairs; shuffle deterministically.
    docs: List[Tuple[str, bool]] = [(golden_doc, True)]
    docs.extend((d, False) for d in distractors)
    seed_base = 0 if shuffle_seed is None else int(shuffle_seed)
    rng = random.Random(seed_base * 1_000_003 + row_index)
    rng.shuffle(docs)

    doc_lines: List[str] = []
    doc_ids: List[str] = []
    golden_id = _doc_id(0)
    for index, (text, is_golden) in enumerate(docs):
        did = _doc_id(index)
        doc_ids.append(did)
        if is_golden:
            golden_id = did
        doc_lines.append(f"[{did}] {text}")

    prompt = (
        f"{RAFT_INSTRUCTION}\n\n"
        f"Question: {query}\n\n"
        f"Documents:\n" + "\n".join(doc_lines) + "\n\nAnswer:"
    )
    return RaftComposed(
        prompt=prompt,
        answer=answer,
        golden_doc_id=golden_id,
        doc_ids=tuple(doc_ids),
    )


def citation_span_token_weights(
    answer: str,
    offsets: List[Tuple[int, int]],
    *,
    style: str = "bracket",
    boost: float = _DEFAULT_CITATION_BOOST,
) -> List[float]:
    """Per-token loss weights for ``answer`` tokens, boosting citation spans.

    ``offsets`` is the per-token ``(start_char, end_char)`` mapping for the
    answer (from a fast tokenizer's ``return_offsets_mapping``). Tokens whose
    char span overlaps any ``[doc-id]`` (per ``style``) citation get ``boost``;
    every other token gets ``1.0``. Returns one weight per offset entry.
    """
    if isinstance(boost, bool) or not isinstance(boost, (int, float)):
        raise TypeError("boost must be a number")
    boost_f = float(boost)
    if boost_f < 1.0 or boost_f > _MAX_CITATION_BOOST:
        raise ValueError(f"boost must be in [1.0, {_MAX_CITATION_BOOST}]")

    # Local import keeps utils/raft import-light + avoids a circular import.
    from soup_cli.utils.citation_faithful import citation_spans

    spans = citation_spans(answer, style=style)
    weights: List[float] = []
    for start, end in offsets:
        # A zero-width offset (special token) never overlaps a citation span.
        overlaps = any(start < s_end and end > s_start for (s_start, s_end) in spans)
        weights.append(boost_f if overlaps else 1.0)
    return weights


def tokenize_raft_example(
    tokenizer: Any,
    composed: RaftComposed,
    *,
    max_length: int,
    citation_faithful: bool = False,
    citation_style: str = "bracket",
    citation_boost: float = _DEFAULT_CITATION_BOOST,
) -> dict:
    """Tokenise a composed RAFT example into a pre-tokenised training row.

    Returns ``{input_ids, attention_mask, labels, loss_weights}`` (python
    lists). The prompt span is masked with ``-100`` in ``labels`` and ``0.0``
    in ``loss_weights`` so only the answer contributes to the loss. When
    ``citation_faithful`` is set, ``loss_weights`` on bracketed citation spans
    inside the answer are boosted to ``citation_boost``.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 8:
        raise ValueError("max_length must be an int >= 8")
    if not isinstance(composed, RaftComposed):
        raise TypeError("composed must be a RaftComposed")

    prompt_text = _render_prompt(tokenizer, composed.prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

    answer_ids: List[int]
    answer_weights: List[float]
    if citation_faithful:
        answer_ids, answer_weights = _tokenize_answer_with_citation(
            tokenizer, composed.answer, style=citation_style, boost=citation_boost
        )
    else:
        answer_ids = tokenizer(composed.answer, add_special_tokens=False)["input_ids"]
        answer_weights = [1.0] * len(answer_ids)

    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        answer_ids = answer_ids + [eos]
        answer_weights = answer_weights + [1.0]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    loss_weights = ([0.0] * len(prompt_ids) + answer_weights)[:max_length]
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "loss_weights": loss_weights,
    }


def _tokenize_answer_with_citation(
    tokenizer: Any, answer: str, *, style: str, boost: float
) -> Tuple[List[int], List[float]]:
    """Tokenise the answer and compute per-token citation-boost weights.

    Uses ``return_offsets_mapping`` (fast tokenizer). When offsets are
    unavailable (slow tokenizer), degrades to a flat ``1.0`` answer mask and
    logs at DEBUG — never raises.
    """
    try:
        enc = tokenizer(
            answer, add_special_tokens=False, return_offsets_mapping=True
        )
        answer_ids = list(enc["input_ids"])
        offsets = enc.get("offset_mapping")
    except (NotImplementedError, ValueError, TypeError):
        offsets = None
        answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    if not offsets or len(offsets) != len(answer_ids):
        logger.debug(
            "citation span boost skipped: tokenizer offsets unavailable "
            "(slow tokenizer?); using flat answer mask"
        )
        return answer_ids, [1.0] * len(answer_ids)
    weights = citation_span_token_weights(
        answer, [tuple(o) for o in offsets], style=style, boost=boost
    )
    return answer_ids, weights


def render_raft_prompt(tokenizer: Any, prompt: str) -> str:
    """Render a single user turn through the tokenizer's chat template.

    Falls back to the raw prompt when the tokenizer has no chat template or the
    template raises. Public so other modules (e.g. ``steering._capture_attn_heads``)
    can reuse it without importing a private symbol across modules (code-review L3).
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 — malformed template → raw prompt
            return prompt
    return prompt


# Back-compat private alias (internal callers + pre-review imports).
_render_prompt = render_raft_prompt
