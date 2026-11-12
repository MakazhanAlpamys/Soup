"""v0.52.0 Part B — Classifier / reranker / cross_encoder task helpers.

Three new task strings build on the existing v0.16.0 embedding trainer:

* ``classifier`` — sequence classification head (single-label / multi-label).
* ``reranker`` — pointwise scoring head for retrieval reranking.
* ``cross_encoder`` — paired-input scoring (e.g. MS-MARCO-style).

Schema-only release: validators here are reused by the SoupConfig
cross-validator, while the live trainer wrappers ship in v0.52.1
(mirrors v0.50.0 stub-then-live pattern).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Mapping, Optional, Sequence

CLASSIFIER_TASKS: frozenset[str] = frozenset(
    {"classifier", "reranker", "cross_encoder"}
)

_CLASSIFIER_KIND: frozenset[str] = frozenset({"single_label", "multi_label"})

_MAX_LABELS: int = 1024
_MAX_LABEL_LEN: int = 128


@dataclass(frozen=True)
class ClassifierKindSpec:
    """Metadata for a classifier task. Frozen so callers cannot mutate."""

    name: str
    description: str
    paired_input: bool
    live_wired: bool


_CLASSIFIER_METADATA: Mapping[str, ClassifierKindSpec] = MappingProxyType({
    "classifier": ClassifierKindSpec(
        name="classifier",
        description="Sequence classification head (single/multi-label)",
        paired_input=False,
        live_wired=False,
    ),
    "reranker": ClassifierKindSpec(
        name="reranker",
        description="Pointwise scoring head for retrieval reranking",
        paired_input=False,
        live_wired=False,
    ),
    "cross_encoder": ClassifierKindSpec(
        name="cross_encoder",
        description="Paired-input scoring head (query/document)",
        paired_input=True,
        live_wired=False,
    ),
})


def is_classifier_task(task: object) -> bool:
    """Return True iff ``task`` is one of classifier/reranker/cross_encoder."""
    if isinstance(task, bool):
        return False
    if not isinstance(task, str):
        return False
    return task in CLASSIFIER_TASKS


def get_classifier_spec(task: str) -> ClassifierKindSpec:
    """Return the frozen :class:`ClassifierKindSpec` for ``task`` or raise."""
    if not is_classifier_task(task):
        supported = ", ".join(sorted(CLASSIFIER_TASKS))
        raise ValueError(
            f"task {task!r} is not a classifier task. Supported: {supported}"
        )
    return _CLASSIFIER_METADATA[task]


def validate_num_labels(value: object) -> int:
    """Validate a ``num_labels`` integer (1..1024). Rejects bool."""
    if isinstance(value, bool):
        raise TypeError(f"num_labels must not be bool, got {value!r}")
    if not isinstance(value, int):
        raise TypeError(
            f"num_labels must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"num_labels must be >= 1, got {value}")
    if value > _MAX_LABELS:
        raise ValueError(
            f"num_labels must be <= {_MAX_LABELS}, got {value}"
        )
    return value


def validate_label_names(value: object) -> list[str]:
    """Validate an optional label-name list. Returns a defensive copy."""
    if not isinstance(value, list):
        raise TypeError(
            f"label_names must be a list, got {type(value).__name__}"
        )
    if len(value) > _MAX_LABELS:
        raise ValueError(
            f"label_names too long (max {_MAX_LABELS} entries)"
        )
    seen: set[str] = set()
    result: list[str] = []
    for entry in value:
        if isinstance(entry, bool):
            raise TypeError("label_names entries must not be bool")
        if not isinstance(entry, str):
            raise TypeError(
                f"label_names entries must be str, got {type(entry).__name__}"
            )
        if not entry:
            raise ValueError("label_names entries must be non-empty")
        if "\x00" in entry:
            raise ValueError("label_names entries must not contain null bytes")
        if len(entry) > _MAX_LABEL_LEN:
            raise ValueError(
                f"label_names entry too long (max {_MAX_LABEL_LEN} chars)"
            )
        if entry in seen:
            raise ValueError(f"label_names entries must be unique: {entry!r}")
        seen.add(entry)
        result.append(entry)
    return result


def validate_classifier_compat(*, task: str, backend: str, modality: str) -> None:
    """Schema-time gate for the three classifier tasks.

    Rejects:
    - non-string / bool args (defence-in-depth).
    - non-classifier task.
    - ``backend == 'mlx'`` (no MLX classifier path in v0.52.0).
    - ``modality != 'text'`` (vision/audio classifier deferred).
    """
    for name, value in (("task", task), ("backend", backend), ("modality", modality)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if task not in CLASSIFIER_TASKS:
        supported = ", ".join(sorted(CLASSIFIER_TASKS))
        raise ValueError(
            f"validate_classifier_compat called with task={task!r}; "
            f"expected one of {supported}"
        )
    if backend == "mlx":
        raise ValueError(
            f"task={task!r} is not supported on backend=mlx in v0.52.0"
        )
    if modality != "text":
        raise ValueError(
            f"task={task!r} requires modality='text'; got modality={modality!r}"
        )


# ---------------------------------------------------------------------------
# Ranking metrics for ``reranker`` / ``cross_encoder`` (#2618)
# ---------------------------------------------------------------------------

#: Tasks whose validation rows can be ranked per query.
RANKING_TASKS: frozenset[str] = frozenset({"reranker", "cross_encoder"})
#: Optional row key that groups the candidates of one query.
QUERY_ID_KEY: str = "query_id"
#: ``recall@k`` is reported for each of these cut-offs.
RANKING_RECALL_KS: tuple[int, ...] = (1, 5, 10)
#: Cut-off for ``mrr@10`` and ``ndcg@10``.
RANKING_CUTOFF: int = 10

_MAX_QUERY_ID_LEN: int = 512


def validate_query_id(value: object) -> str | int:
    """Validate one row's ``query_id``: a non-empty str or an int (not bool).

    Ids are compared exactly as given, so ``1`` and ``"1"`` are two queries.
    """
    if isinstance(value, bool):
        raise TypeError(f"query_id must not be bool, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if not value:
            raise ValueError("query_id must be a non-empty string")
        if "\x00" in value:
            raise ValueError("query_id must not contain null bytes")
        if len(value) > _MAX_QUERY_ID_LEN:
            raise ValueError(f"query_id too long (max {_MAX_QUERY_ID_LEN} chars)")
        return value
    raise TypeError(f"query_id must be str or int, got {type(value).__name__}")


def ranking_score_skip_reason(num_labels: int, multi_label: bool) -> Optional[str]:
    """Why the candidates of a head with this shape cannot be ranked, or None.

    * ``num_labels: 2`` (single-label): ranked by ``logit[1] - logit[0]``, which
      orders documents like the softmax probability of the positive class that
      cross-entropy trains.
    * ``num_labels: 1`` with ``classifier_kind: multi_label``: ranked by the one
      logit, which ``BCEWithLogitsLoss`` trains (same order as its sigmoid).
    """
    if num_labels == 2 and not multi_label:
        return None
    if num_labels == 1 and multi_label:
        return None
    if num_labels == 1:
        return (
            "num_labels: 1 accepts only label 0 with the default single-label kind, "
            "so there is no positive candidate to rank; use classifier_kind: multi_label"
        )
    if num_labels == 2:
        return (
            "num_labels: 2 with classifier_kind: multi_label has no single positive "
            "class to rank by"
        )
    return f"num_labels: {num_labels} has no defined positive class to rank by"


def _as_rows(values: object) -> list:
    return values.tolist() if hasattr(values, "tolist") else list(values)  # type: ignore[arg-type]


def ranking_scores(logits: object, num_labels: int, multi_label: bool) -> list[float]:
    """One ranking score per row from the model's logits (see the rule above)."""
    reason = ranking_score_skip_reason(num_labels, multi_label)
    if reason is not None:
        raise ValueError(reason)
    rows = _as_rows(logits)
    if num_labels == 2:
        return [float(row[1]) - float(row[0]) for row in rows]
    return [float(row[0]) if isinstance(row, (list, tuple)) else float(row) for row in rows]


def ranking_relevance(label_ids: object, num_labels: int, multi_label: bool) -> list[int]:
    """1 for a relevant (positive) candidate, 0 otherwise, from the encoded labels."""
    reason = ranking_score_skip_reason(num_labels, multi_label)
    if reason is not None:
        raise ValueError(reason)
    rows = _as_rows(label_ids)
    if num_labels == 2:
        return [1 if int(label) == 1 else 0 for label in rows]
    return [
        1 if float(label[0] if isinstance(label, (list, tuple)) else label) >= 0.5 else 0
        for label in rows
    ]


def ranking_metrics(
    scores: Sequence[float],
    relevance: Sequence[int],
    groups: Sequence[Optional[Hashable]],
    *,
    recall_ks: Sequence[int] = RANKING_RECALL_KS,
    cutoff: int = RANKING_CUTOFF,
) -> dict[str, float]:
    """``mrr@10``, ``recall@{1,5,10}`` and ``ndcg@10`` averaged over queries.

    ``groups[i]`` is the query of row ``i``; ``None`` leaves the row out (no
    ``query_id``, or a query whose rows are not all in the validation rows).
    Candidates are ranked by descending score, ties broken by row order, so the
    result is deterministic. A query with no relevant candidate cannot be
    scored: it is skipped and counted in ``ranking_queries_no_positive``
    instead of pulling the means towards 0. ``recall@k`` is the share of a
    query's relevant candidates found in its top ``k``; nDCG uses binary gains.
    """
    if not (len(scores) == len(relevance) == len(groups)):
        raise ValueError(
            "scores, relevance and groups must have the same length, got "
            f"{len(scores)}, {len(relevance)} and {len(groups)}"
        )
    by_query: dict[Hashable, list[tuple[float, int, int]]] = {}
    for index, (score, rel, group) in enumerate(zip(scores, relevance, groups)):
        if group is None:
            continue
        safe = float(score) if math.isfinite(float(score)) else -math.inf
        by_query.setdefault(group, []).append((safe, index, int(rel)))

    totals = {"mrr": 0.0, "ndcg": 0.0, **{f"recall@{k}": 0.0 for k in recall_ks}}
    scored = 0
    no_positive = 0
    for candidates in by_query.values():
        relevant_total = sum(rel for _, _, rel in candidates)
        if relevant_total == 0:
            no_positive += 1
            continue
        scored += 1
        ranked = [rel for _, _, rel in sorted(candidates, key=lambda c: (-c[0], c[1]))]
        first_hit = next((pos for pos, rel in enumerate(ranked, start=1) if rel), None)
        if first_hit is not None and first_hit <= cutoff:
            totals["mrr"] += 1.0 / first_hit
        for k in recall_ks:
            totals[f"recall@{k}"] += sum(ranked[:k]) / relevant_total
        dcg = sum(rel / math.log2(pos + 1) for pos, rel in enumerate(ranked[:cutoff], start=1))
        ideal = sum(1.0 / math.log2(pos + 1) for pos in range(1, min(relevant_total, cutoff) + 1))
        totals["ndcg"] += dcg / ideal

    result: dict[str, float] = {
        "ranking_queries": float(scored),
        "ranking_queries_no_positive": float(no_positive),
    }
    if scored:
        result[f"mrr@{cutoff}"] = totals["mrr"] / scored
        for k in recall_ks:
            result[f"recall@{k}"] = totals[f"recall@{k}"] / scored
        result[f"ndcg@{cutoff}"] = totals["ndcg"] / scored
    return result


def ranking_query_groups(
    train_rows: Sequence[dict], val_rows: Sequence[dict]
) -> tuple[list[Optional[Hashable]], dict[str, int]]:
    """Group validation rows by ``query_id`` and drop incomplete queries.

    A query whose ``query_id`` also appears in the training rows has only part
    of its candidates in validation (``data.val_split`` cuts the file at a row
    index, and a dataset's own validation split can do the same). Its list
    would be incomplete, so its rows map to ``None`` and it is counted in
    ``queries_split``. This needs no config key: the two id sets decide it.
    """
    def key(row: dict) -> Optional[Hashable]:
        value = row.get(QUERY_ID_KEY) if isinstance(row, dict) else None
        return None if value is None else validate_query_id(value)

    val_keys = [key(row) for row in val_rows]
    train_keys = {k for k in (key(row) for row in train_rows) if k is not None}
    split = {k for k in val_keys if k is not None and k in train_keys}
    groups = [None if (k is None or k in split) else k for k in val_keys]
    counts = {
        "rows_without_query_id": sum(1 for k in val_keys if k is None),
        "queries_split": len(split),
    }
    return groups, counts


def build_classifier_trainer(
    config: object, **kwargs: object
) -> object:
    """Live classifier / reranker / cross_encoder trainer factory (v0.53.2 #132).

    Returns a :class:`ClassifierTrainerWrapper`. Lazy import keeps the heavy
    transformers/peft surface out of schema-only import paths.
    """
    from soup_cli.trainer.classifier import ClassifierTrainerWrapper

    return ClassifierTrainerWrapper(config, **kwargs)  # type: ignore[arg-type]
