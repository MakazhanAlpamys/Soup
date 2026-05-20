"""v0.62.0 Part B — RA-DIT (Retrieval-Augmented Dual Instruction Tuning).

Meta 2023 recipe. Two-stage:

* ``retriever`` — train a sentence-transformer (contrastive triplet loss)
  on (query, golden_doc, distractor) triples.  Composes with the existing
  v0.16 embedding trainer.
* ``generator`` — RAFT-style SFT on the query + golden_doc + distractor
  bundle (uses v0.62.0 Part A ``data.format='raft'``).

Schema-only release. Both stages share the existing trainer wrappers;
v0.62.0 ships the ``ra_dit_stage`` schema field + cross-validator so a
``soup.yaml`` can lock both stages in a hub-shareable recipe. Live
orchestration that chains the two stages in a single ``soup train`` call
is deferred to v0.62.1 (mirrors the v0.50.0 / v0.52.0 / v0.61.0
stub-then-live pattern).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

SUPPORTED_RA_DIT_STAGES: frozenset[str] = frozenset({"retriever", "generator"})

_MAX_STAGE_LEN: int = 32
_MAX_RETRIEVER_MODEL_LEN: int = 512


@dataclass(frozen=True)
class RaDitStageSpec:
    """Metadata for a single RA-DIT training stage. Frozen post-construction."""

    name: str
    description: str
    base_task: str
    live_wired: bool


_RA_DIT_STAGE_METADATA: Mapping[str, RaDitStageSpec] = MappingProxyType({
    "retriever": RaDitStageSpec(
        name="retriever",
        description=(
            "Stage 1 — train a sentence-transformer with contrastive loss "
            "on (anchor, positive, negative) triples. Composes with the "
            "v0.16 embedding trainer."
        ),
        base_task="embedding",
        live_wired=False,
    ),
    "generator": RaDitStageSpec(
        name="generator",
        description=(
            "Stage 2 — SFT the generator on RAFT-style rows "
            "{query, golden_doc, distractor_docs, answer}. Uses "
            "v0.62.0 Part A `data.format='raft'`."
        ),
        base_task="sft",
        live_wired=False,
    ),
})


def validate_ra_dit_stage(value: object) -> str:
    """Normalise + validate an RA-DIT stage name.

    Returns the canonical (lowercase) form. Mirrors v0.51.0
    ``validate_hub_name`` / v0.61.0 ``validate_unlearn_method`` policy:
    bool-rejected, null-byte-rejected, oversize-rejected, case-insensitive
    normalisation, unknown rejected with friendly actionable message.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"ra_dit_stage must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"ra_dit_stage must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("ra_dit_stage must be non-empty")
    if "\x00" in value:
        raise ValueError("ra_dit_stage must not contain null bytes")
    if len(value) > _MAX_STAGE_LEN:
        raise ValueError(
            f"ra_dit_stage must be <= {_MAX_STAGE_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_RA_DIT_STAGES:
        supported = ", ".join(sorted(SUPPORTED_RA_DIT_STAGES))
        raise ValueError(
            f"unknown ra_dit_stage {value!r}; supported: {supported}"
        )
    return canonical


def get_ra_dit_stage_spec(name: str) -> RaDitStageSpec:
    """Return the frozen :class:`RaDitStageSpec` for ``name`` or raise."""
    canonical = validate_ra_dit_stage(name)
    return _RA_DIT_STAGE_METADATA[canonical]


def validate_ra_dit_retriever_model(value: object) -> Optional[str]:
    """Validate the operator-supplied retriever-model HF repo id or local path.

    Bool / non-string rejected. ``None`` passes through unchanged so
    callers can rely on Pydantic ``Optional[str]`` semantics. Length cap
    matches the v0.40.5 ``reward_model`` policy (512 chars).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(
            f"ra_dit_retriever_model must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"ra_dit_retriever_model must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("ra_dit_retriever_model must be non-empty")
    if "\x00" in value:
        raise ValueError(
            "ra_dit_retriever_model must not contain null bytes"
        )
    if len(value) > _MAX_RETRIEVER_MODEL_LEN:
        raise ValueError(
            f"ra_dit_retriever_model must be <= {_MAX_RETRIEVER_MODEL_LEN} chars"
        )
    return value


def validate_ra_dit_compat(*, stage: str, task: str) -> None:
    """Schema-time gate: each stage must pair with the right base task.

    * ``retriever`` -> ``task='embedding'`` (contrastive trainer).
    * ``generator`` -> ``task='sft'`` (RAFT-style SFT).

    Raises ``ValueError`` on mismatch with a friendly actionable message.
    """
    for name, value in (("stage", stage), ("task", task)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str):
            raise TypeError(
                f"{name} must be str, got {type(value).__name__}"
            )
        if not value:
            raise ValueError(f"{name} must be non-empty")
        if "\x00" in value:
            raise ValueError(f"{name} must not contain null bytes")
    canonical_stage = validate_ra_dit_stage(stage)
    expected_task = _RA_DIT_STAGE_METADATA[canonical_stage].base_task
    if task != expected_task:
        raise ValueError(
            f"training.ra_dit_stage={canonical_stage!r} requires "
            f"task={expected_task!r}; got task={task!r}."
        )
