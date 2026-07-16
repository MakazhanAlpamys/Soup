"""v0.71.36 — topic map (Data Moat II): BERTopic-lite.

PURE numpy: k-means over embedding vectors + class-based TF-IDF labels.
No torch, no I/O — the caller supplies the vectors (from
:func:`soup_cli.utils.embed.embed_texts`), so every decision here is
testable on CPU with hand-built input.

Honest framing, repeated in the docs: labels are **emergent unsupervised
term clusters**, not a classification against a fixed ontology. "82% code"
means "82% of rows fell in a cluster whose top c-TF-IDF terms look like
code". There is also NO join to ``soup eval coverage`` — that command
compares an eval suite's *scorer mix* to a task taxonomy, a different axis
entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_MAX_TOPIC_ROWS = 200_000
# 'auto' never proposes more than this, so the coverage table stays one
# screen. An explicit --clusters N is the operator's call and is not capped.
_MAX_K = 25


@dataclass(frozen=True)
class Topic:
    """One cluster: its label, its share of the corpus, and its members."""

    label: str
    terms: tuple
    size: int
    fraction: float
    member_indices: tuple


@dataclass(frozen=True)
class TopicReport:
    topics: tuple
    n_rows: int
    n_clusters: int
    warnings: tuple


def resolve_k(n_rows: int, requested) -> int:
    """Pick the cluster count.

    An explicit int is honoured (clamped to ``[1, n_rows]``). ``"auto"``
    uses the standard sqrt heuristic, capped at :data:`_MAX_K`.
    """
    if isinstance(n_rows, bool) or not isinstance(n_rows, int):
        raise TypeError(f"n_rows must be int, got {type(n_rows).__name__}")
    if n_rows < 0:
        raise ValueError("n_rows must be >= 0")
    if isinstance(requested, str):
        if requested.strip().lower() != "auto":
            raise ValueError(
                f"clusters must be an int or 'auto', got {requested!r}"
            )
        if n_rows < 4:
            return 1
        return max(2, min(_MAX_K, round(math.sqrt(n_rows / 2))))
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError(
            f"clusters must be int or 'auto', got {type(requested).__name__}"
        )
    if requested < 1:
        raise ValueError(f"clusters must be >= 1, got {requested}")
    return max(1, min(requested, n_rows)) if n_rows else 1


def kmeans(vectors, *, k: int, seed: int, max_iter: int = 50):
    """Deterministic k-means++ over ``vectors``; returns integer labels."""
    import numpy as np

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"vectors must be 2-D (n, d), got shape {arr.shape}")
    n_rows = arr.shape[0]
    if n_rows > _MAX_TOPIC_ROWS:
        raise ValueError(
            f"too many rows ({n_rows}); cap is {_MAX_TOPIC_ROWS}. Sample "
            "first — Soup refuses rather than silently subsampling."
        )
    if n_rows == 0:
        return np.zeros((0,), dtype=np.int64)

    k_eff = max(1, min(int(k), n_rows))
    rng = np.random.default_rng(seed)

    # k-means++ init: seed centers far apart so the result is stable.
    centers = [arr[rng.integers(n_rows)]]
    for _ in range(1, k_eff):
        dists = np.min(
            np.stack([((arr - c) ** 2).sum(axis=1) for c in centers]), axis=0
        )
        total = float(dists.sum())
        if total <= 0.0:
            # Every point coincides with a center (e.g. all-identical rows).
            centers.append(arr[rng.integers(n_rows)])
            continue
        centers.append(arr[rng.choice(n_rows, p=dists / total)])
    cent = np.stack(centers)

    labels = np.full((n_rows,), -1, dtype=np.int64)
    for _ in range(max_iter):
        dists = ((arr[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int64)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k_eff):
            members = arr[labels == idx]
            if len(members):
                cent[idx] = members.mean(axis=0)
    return labels


def ctfidf_labels(token_docs, labels, *, k: int, top_n: int = 3) -> list:
    """Class-based TF-IDF: each CLUSTER is one document.

    A term frequent in one cluster and rare across clusters scores high; a
    term frequent in EVERY cluster ("the", "code") scores low. That
    contrast is what separates c-TF-IDF from plain TF-IDF, which would
    just surface the corpus's most common words in every label.
    """
    from collections import Counter

    docs = list(token_docs)
    label_list = list(labels)
    if len(docs) != len(label_list):
        raise ValueError(
            "token_docs and labels must be the same length; got "
            f"{len(docs)} and {len(label_list)}"
        )
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be an int >= 1")

    per_class: list = [Counter() for _ in range(k)]
    for tokens, label in zip(docs, label_list):
        idx = int(label)
        if 0 <= idx < k:
            per_class[idx].update(tokens)

    # Cluster frequency: how many clusters contain the term at all.
    cluster_freq: Counter = Counter()
    for counter in per_class:
        for term in counter:
            cluster_freq[term] += 1

    n_classes = max(1, sum(1 for counter in per_class if counter))
    out: list = []
    for counter in per_class:
        if not counter:
            out.append(())
            continue
        total = sum(counter.values())
        scored = []
        for term, freq in counter.items():
            term_freq = freq / total
            idf = math.log(1.0 + n_classes / cluster_freq[term])
            scored.append((term, term_freq * idf))
        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        out.append(tuple(term for term, _ in scored[:top_n]))
    return out


def build_topic_report(
    rows, labels, *, k: int, min_fraction: float = 0.02
) -> TopicReport:
    """Assemble the coverage table + gap warnings from cluster labels."""
    from soup_cli.utils._eval_text import row_text, tokenize

    row_list = list(rows)
    label_list = [int(x) for x in labels]
    if len(row_list) != len(label_list):
        raise ValueError(
            "rows and labels must be the same length; got "
            f"{len(row_list)} and {len(label_list)}"
        )
    n_rows = len(row_list)
    if n_rows == 0:
        return TopicReport(topics=(), n_rows=0, n_clusters=0, warnings=())

    token_docs = [tokenize(row_text(row)) for row in row_list]
    label_terms = ctfidf_labels(token_docs, label_list, k=k, top_n=3)

    topics = []
    warnings = []
    for idx in range(k):
        members = tuple(
            i for i, label in enumerate(label_list) if label == idx
        )
        if not members:
            continue
        fraction = len(members) / n_rows
        terms = label_terms[idx]
        label = " / ".join(terms) if terms else f"cluster-{idx}"
        topics.append(
            Topic(
                label=label,
                terms=terms,
                size=len(members),
                fraction=fraction,
                member_indices=members,
            )
        )
        if fraction < min_fraction:
            warnings.append(
                f"topic {label!r} is thin: {fraction * 100:.1f}% of rows "
                f"({len(members)}/{n_rows}) — likely under-represented"
            )
    topics.sort(key=lambda topic: -topic.fraction)
    return TopicReport(
        topics=tuple(topics),
        n_rows=n_rows,
        n_clusters=len(topics),
        warnings=tuple(warnings),
    )
