"""v0.69.0 Part D — Persona-Hub diversity sampler.

Samples prompt × persona × style combinations to drive synthetic-data diversity
(per Tencent's Persona-Hub paper). Ships with a small bundled persona list so
``soup data persona-mix`` works offline; operators wanting the full 200k-persona
HF dataset can pass ``--personas <jsonl>`` from a downloaded copy.

The topic-diversity metric is a pure-Python token-entropy heuristic — same
approach as the v0.55.0 ``eval_design`` TF-IDF baseline. The full
sentence-transformer embedding-entropy variant is the operator-installable
``[data-pro]`` extras path; this module does not import any embedding library
to keep the CLI startup fast.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence, Tuple

# 12 bundled personas covering broad demographic + occupational diversity.
# Operators wanting the full Persona-Hub corpus pass a JSONL via the CLI.
_BUNDLED_PERSONAS: Tuple[str, ...] = (
    "a curious high school student",
    "a software engineer with 10 years of experience",
    "a non-native English speaker learning to code",
    "a research scientist in molecular biology",
    "a small business owner planning their first product launch",
    "a freelance journalist on a tight deadline",
    "a retired teacher revisiting old hobbies",
    "a graduate student preparing a thesis defense",
    "a parent helping a child with homework",
    "a healthcare professional asking patient-care questions",
    "a hobbyist game developer trying a new engine",
    "a senior engineer onboarding a junior teammate",
)

_BUNDLED_STYLES: Tuple[str, ...] = (
    "formal",
    "casual",
    "step-by-step",
    "concise",
    "playful",
)

_MAX_PERSONA_LEN = 1024
_MAX_STYLE_LEN = 128
_MAX_LIST_LEN = 100_000
_MAX_SAMPLES = 1_000_000
_MAX_DIVERSITY_TEXT_LEN = 65_536


# -----------------------------------------------------------------------------
# Validators
# -----------------------------------------------------------------------------


def validate_persona(persona: object) -> str:
    """Validate a persona string."""
    if isinstance(persona, bool) or not isinstance(persona, str):
        raise TypeError(
            f"persona must be str, got {type(persona).__name__}"
        )
    if not persona:
        raise ValueError("persona must be non-empty")
    if "\x00" in persona:
        raise ValueError("persona must not contain null bytes")
    if len(persona) > _MAX_PERSONA_LEN:
        raise ValueError(f"persona must be <= {_MAX_PERSONA_LEN} chars")
    return persona


def validate_style(style: object) -> str:
    """Validate a style string."""
    if isinstance(style, bool) or not isinstance(style, str):
        raise TypeError(f"style must be str, got {type(style).__name__}")
    if not style:
        raise ValueError("style must be non-empty")
    if "\x00" in style:
        raise ValueError("style must not contain null bytes")
    if len(style) > _MAX_STYLE_LEN:
        raise ValueError(f"style must be <= {_MAX_STYLE_LEN} chars")
    return style


def list_bundled_personas() -> Tuple[str, ...]:
    """Return the bundled persona tuple. Immutable by virtue of being a tuple."""
    return _BUNDLED_PERSONAS


def list_bundled_styles() -> Tuple[str, ...]:
    """Return the bundled style tuple."""
    return _BUNDLED_STYLES


# -----------------------------------------------------------------------------
# Plan dataclass + factory
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaPlan:
    """Frozen plan for a persona-mix sample run."""

    prompts: Tuple[str, ...]
    personas: Tuple[str, ...]
    styles: Tuple[str, ...]
    n: int
    seed: int

    def __post_init__(self) -> None:
        for field_name in ("prompts", "personas", "styles"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise TypeError(
                    f"PersonaPlan.{field_name} must be a tuple "
                    "(frozen=True does not make List immutable)"
                )
        if isinstance(self.n, bool):
            raise TypeError("PersonaPlan.n must be int, not bool")
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError("PersonaPlan.n must be a positive integer")
        if isinstance(self.seed, bool):
            raise TypeError("PersonaPlan.seed must be int, not bool")
        if not isinstance(self.seed, int):
            raise TypeError("PersonaPlan.seed must be an integer")


def _check_list(
    name: str, values: Sequence[Any], item_validator: Any
) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise TypeError(f"{name} must be a list of strings")
    materialised = list(values)
    if not materialised:
        raise ValueError(f"{name} must be non-empty")
    if len(materialised) > _MAX_LIST_LEN:
        raise ValueError(f"{name} exceeds {_MAX_LIST_LEN} entries")
    return tuple(item_validator(v) for v in materialised)


def build_persona_plan(
    *,
    prompts: Sequence[str],
    personas: Sequence[str],
    styles: Sequence[str],
    n: int,
    seed: int,
) -> PersonaPlan:
    """Validate inputs + return a frozen ``PersonaPlan``."""
    if isinstance(n, bool):
        raise TypeError("n must be int, not bool")
    if not isinstance(n, int):
        raise TypeError("n must be int")
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > _MAX_SAMPLES:
        raise ValueError(f"n exceeds {_MAX_SAMPLES}")
    if isinstance(seed, bool):
        raise TypeError("seed must be int, not bool")
    if not isinstance(seed, int):
        raise TypeError("seed must be int")
    return PersonaPlan(
        prompts=_check_list("prompts", prompts, validate_persona),
        personas=_check_list("personas", personas, validate_persona),
        styles=_check_list("styles", styles, validate_style),
        n=n,
        seed=seed,
    )


# -----------------------------------------------------------------------------
# Sampler
# -----------------------------------------------------------------------------


def sample_persona_matrix(
    *,
    prompts: Sequence[str],
    personas: Sequence[str],
    styles: Sequence[str],
    n: int,
    seed: int,
) -> List[Mapping[str, str]]:
    """Sample ``n`` rows from the prompt × persona × style matrix.

    Deterministic given the same ``(inputs, seed)`` — uses a seeded ``Random``
    instance so concurrent samplers in the same process do not affect each
    other.
    """
    plan = build_persona_plan(
        prompts=prompts,
        personas=personas,
        styles=styles,
        n=n,
        seed=seed,
    )
    rng = random.Random(plan.seed)
    rows: List[Mapping[str, str]] = []
    for _ in range(plan.n):
        rows.append(
            {
                "prompt": rng.choice(plan.prompts),
                "persona": rng.choice(plan.personas),
                "style": rng.choice(plan.styles),
            }
        )
    return rows


# -----------------------------------------------------------------------------
# Topic diversity
# -----------------------------------------------------------------------------


def _extract_text(row: Mapping[str, Any]) -> str:
    """Extract concatenated text from a row across common fields."""
    parts: List[str] = []
    for key in ("text", "content", "output", "prompt", "instruction", "response"):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
    if not parts:
        return ""
    joined = " ".join(parts)
    if len(joined) > _MAX_DIVERSITY_TEXT_LEN:
        joined = joined[:_MAX_DIVERSITY_TEXT_LEN]
    return joined


def compute_topic_diversity(rows: Any) -> float:
    """Return a topic-diversity score in [0, 1].

    Heuristic: pool the whitespace tokens across all rows, compute Shannon
    entropy in bits, and normalise by ``log2(unique_tokens)`` so the score
    is comparable across corpora. Returns ``0.0`` on empty input.
    """
    if isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a list of mappings")
    if not hasattr(rows, "__iter__"):
        raise TypeError("rows must be iterable")
    tokens: List[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        text = _extract_text(row)
        if not text:
            continue
        tokens.extend(text.lower().split())
    if not tokens:
        return 0.0
    counter = Counter(tokens)
    total = sum(counter.values())
    unique = len(counter)
    if unique <= 1:
        return 0.0
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counter.values()
    )
    max_entropy = math.log2(unique)
    if max_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy / max_entropy))


__all__ = [
    "PersonaPlan",
    "build_persona_plan",
    "compute_topic_diversity",
    "list_bundled_personas",
    "list_bundled_styles",
    "sample_persona_matrix",
    "validate_persona",
    "validate_style",
]
