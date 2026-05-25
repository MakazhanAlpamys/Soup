"""Universal Logit Distillation (ULD) — v0.70.0 Part B.

Cross-tokenizer knowledge distillation extending v0.53.2 DistillTrainer
to teacher/student pairs with **different vocabularies** (e.g. Llama →
Mistral, Llama → Qwen, Qwen → Llama). The existing v0.53.2 path
assumes column-wise logit alignment, which fails the moment student
and teacher have different vocab sizes.

Two strategies ship in v0.70.0 (schema-only; live projection module
wired in v0.70.1):

- ``wasserstein``: Boizard et al. 2024 ULD — uses 1D Wasserstein
  distance between sorted teacher / student logit distributions. No
  alignment required; works across arbitrary vocab boundaries.
  (arXiv 2402.12030)

- ``topk_align``: Top-K projection — pick top-K teacher logits, map
  via BPE-overlap heuristic to student token ids, distil only on the
  aligned subset. Requires ``top_k`` to be set.

Live wiring deferred to v0.70.1 — mirrors v0.27.0 MII / v0.50.0 GRPO
Plus / v0.61.0 unlearning / v0.62.0 RAG stub-then-live pattern.

Security:
- Closed allowlist (frozenset); arbitrary strategy rejected.
- Bool / null-byte / non-string / oversize rejection on every validator.
- Vocab-size bounds: [1, 262144] (covers multilingual SentencePiece +
  GPT-OSS 200K vocab; matches v0.42 token-cap policy).
- ``top_k`` mutually exclusive with non-topk_align strategies (silent
  no-op footgun rejection mirroring v0.52 distill / classifier policy).
- No top-level torch import — lazy import inside ``build_uld_projection``.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

_MAX_STRATEGY_NAME_LEN = 32
# 262144 covers multilingual SentencePiece (e.g. NLLB) + GPT-OSS 200K.
_MAX_VOCAB_SIZE = 262144

SUPPORTED_ULD_STRATEGIES: frozenset[str] = frozenset({"wasserstein", "topk_align"})


@dataclass(frozen=True)
class ULDStrategySpec:
    """Static metadata for a ULD strategy."""

    name: str
    description: str
    requires_top_k: bool


_STRATEGY_METADATA = types.MappingProxyType({
    "wasserstein": ULDStrategySpec(
        name="wasserstein",
        description=(
            "1D Wasserstein distance on sorted logit distributions. "
            "No alignment required across vocabularies."
        ),
        requires_top_k=False,
    ),
    "topk_align": ULDStrategySpec(
        name="topk_align",
        description=(
            "Top-K teacher logit alignment via BPE overlap. Requires "
            "top_k to be set."
        ),
        requires_top_k=True,
    ),
})


def validate_uld_strategy(name: object) -> str:
    """Validate and normalise a ULD strategy name.

    Returns the canonical (lower-cased) name. Raises ``ValueError`` on
    any failure.
    """
    if isinstance(name, bool):
        raise ValueError("uld_strategy must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"uld_strategy must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("uld_strategy must be a non-empty string")
    if "\x00" in name:
        raise ValueError("uld_strategy must not contain null bytes")
    if len(name) > _MAX_STRATEGY_NAME_LEN:
        raise ValueError(
            f"uld_strategy exceeds {_MAX_STRATEGY_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_ULD_STRATEGIES:
        raise ValueError(
            f"uld_strategy={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_ULD_STRATEGIES)}"
        )
    return normalised


def get_strategy_spec(name: str) -> ULDStrategySpec:
    """Return the :class:`ULDStrategySpec` for ``name``."""
    normalised = validate_uld_strategy(name)
    return _STRATEGY_METADATA[normalised]


def validate_uld_projection_dim(value: object) -> int:
    """Validate a projection dimensionality (vocab size).

    Bounds: ``[1, _MAX_VOCAB_SIZE=262144]``. Bool rejected per project
    bool-as-int policy.
    """
    if isinstance(value, bool):
        raise ValueError("dim must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"dim must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"dim must be >= 1, got {value}")
    if value > _MAX_VOCAB_SIZE:
        raise ValueError(
            f"dim={value} exceeds {_MAX_VOCAB_SIZE} cap"
        )
    return value


def validate_uld_top_k(value: object) -> int:
    """Validate ``top_k`` for the topk_align strategy.

    Bounds: ``[1, _MAX_VOCAB_SIZE=262144]``. Bool rejected.
    """
    if isinstance(value, bool):
        raise ValueError("uld_top_k must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"uld_top_k must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"uld_top_k must be >= 1, got {value}")
    if value > _MAX_VOCAB_SIZE:
        raise ValueError(
            f"uld_top_k={value} exceeds {_MAX_VOCAB_SIZE} cap"
        )
    return value


@dataclass(frozen=True)
class ULDConfig:
    """Frozen ULD configuration.

    - ``strategy``: one of :data:`SUPPORTED_ULD_STRATEGIES`.
    - ``student_vocab_size`` / ``teacher_vocab_size``: positive ints,
      bounded by ``_MAX_VOCAB_SIZE``.
    - ``top_k``: required when ``strategy='topk_align'``, rejected
      otherwise (silent no-op footgun rejection).
    """

    strategy: str
    student_vocab_size: int
    teacher_vocab_size: int
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        normalised = validate_uld_strategy(self.strategy)
        if normalised != self.strategy:
            object.__setattr__(self, "strategy", normalised)
        validate_uld_projection_dim(self.student_vocab_size)
        validate_uld_projection_dim(self.teacher_vocab_size)
        spec = _STRATEGY_METADATA[normalised]
        if spec.requires_top_k:
            if self.top_k is None:
                raise ValueError(
                    f"uld_strategy='{normalised}' requires top_k to be set"
                )
            validate_uld_top_k(self.top_k)
        elif self.top_k is not None:
            raise ValueError(
                f"top_k is only valid when uld_strategy='topk_align'; "
                f"got uld_strategy='{normalised}'"
            )


def build_uld_projection(config):
    """Build the projection module that bridges teacher / student vocabs.

    Deferred to v0.70.1. Validates the config type at the public boundary
    so misconfigured callers fail fast (mirrors v0.50.0 ``apply_variant_loss``
    / v0.62.0 ``apply_steering`` / v0.67.0 ``apply_bank_to_serve`` policy).
    """
    if not isinstance(config, ULDConfig):
        raise TypeError(
            f"config must be ULDConfig, got {type(config).__name__}"
        )
    raise NotImplementedError(
        f"Live ULD projection for strategy={config.strategy!r} is deferred "
        "to v0.70.1. v0.70.0 ships the schema + validators only."
    )
