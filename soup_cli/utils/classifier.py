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

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

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


def build_classifier_trainer() -> None:
    """Live classifier trainer factory — deferred to v0.52.1."""
    raise NotImplementedError(
        "Classifier / reranker / cross_encoder trainer live wiring deferred "
        "to v0.52.1. Schema accepts the task but no trainer wrapper is "
        "registered yet."
    )
