"""Pure-Python inverted-index Okapi BM25 implementation.

Self-contained with zero third-party dependencies, matching reference
rank_bm25.BM25Okapi scoring and deterministic tie-breaking.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence


class PureBM25:
    """Inverted-index Okapi BM25 ranking algorithm."""

    def __init__(
        self,
        tokenized_docs: Sequence[Sequence[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        eps: float = 0.25,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.eps = eps
        self.n_docs = 0
        self.avgdl = 0.0
        self.doc_lens: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

        if tokenized_docs is not None:
            self._fit(tokenized_docs)

    def _fit(self, tokenized_docs: Sequence[Sequence[str]]) -> None:
        self.n_docs = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = (sum(self.doc_lens) / self.n_docs) if self.n_docs > 0 else 0.0

        df: Counter[str] = Counter()
        for doc_id, doc in enumerate(tokenized_docs):
            term_counts = Counter(doc)
            for term, count in term_counts.items():
                self.postings[term].append((doc_id, count))
                df[term] += 1

        idf_sum = 0.0
        neg_terms: list[str] = []
        for term, freq in df.items():
            val = math.log(self.n_docs - freq + 0.5) - math.log(freq + 0.5)
            self.idf[term] = val
            idf_sum += val
            if val < 0:
                neg_terms.append(term)

        avg_idf = (idf_sum / len(self.idf)) if self.idf else 0.0
        eps_floor = self.eps * avg_idf
        for term in neg_terms:
            self.idf[term] = eps_floor

    def score(self, query_tokens: Sequence[str]) -> list[float]:
        """Compute BM25 scores for all documents given tokenized query."""
        scores = [0.0] * self.n_docs
        if not self.n_docs or not self.avgdl:
            return scores

        for q in query_tokens:
            q_idf = self.idf.get(q)
            if q_idf is None:
                continue
            for doc_id, tf in self.postings.get(q, ()):
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                if denom > 0:
                    scores[doc_id] += q_idf * (tf * (self.k1 + 1.0) / denom)

        return scores

    def rank(self, query_tokens: Sequence[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Return top-k (doc_id, score) pairs with deterministic tie-breaking."""
        if top_k < 1 or self.n_docs == 0:
            return []
        scores = self.score(query_tokens)
        ranked_indices = sorted(range(self.n_docs), key=lambda i: (-scores[i], i))
        return [(idx, scores[idx]) for idx in ranked_indices[:top_k]]

    def to_dict(self) -> dict[str, Any]:
        """Serialize index parameters, IDF, and postings to a dictionary."""
        return {
            "k1": self.k1,
            "b": self.b,
            "eps": self.eps,
            "n_docs": self.n_docs,
            "avgdl": self.avgdl,
            "doc_lens": self.doc_lens,
            "idf": self.idf,
            "postings": {k: v for k, v in self.postings.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PureBM25:
        """Reconstruct index from a serialized dictionary."""
        instance = cls(k1=data["k1"], b=data["b"], eps=data["eps"])
        instance.n_docs = data["n_docs"]
        instance.avgdl = data["avgdl"]
        instance.doc_lens = data["doc_lens"]
        instance.idf = data["idf"]
        instance.postings = defaultdict(
            list, {k: [tuple(p) for p in v] for k, v in data["postings"].items()}
        )
        return instance
