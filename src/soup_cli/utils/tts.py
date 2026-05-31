"""v0.52.0 Part A — TTS (text-to-speech) fine-tuning schema helpers.

Schema-only support for ``task='tts'`` paired with ``modality='audio_out'``.
Five upstream TTS model families are recognised: orpheus / sesame_csm /
llasa / spark / oute. Each has a stable name string + per-family ``emotion``
allowlist so trainer wiring (deferred to v0.52.1) can route correctly.

Mirrors v0.50.0 stub-then-live pattern: this module exposes pure validators
and a frozen ``TTSFamilySpec`` dataclass; the live ``TTSTrainerWrapper``
lands in v0.52.1.

Security:
- Pure schema-time validation; no filesystem touch.
- All validators raise ``ValueError`` / ``TypeError`` with actionable
  messages.
- Bool rejected before int / str checks (project bool-as-int policy —
  matches v0.30.0 ``Candidate``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# Closed allowlist — wrapped via MappingProxyType so callers cannot mutate
# the registry at runtime (mirrors v0.36.0 ``_REGISTRY`` / v0.51.0 ``hubs``
# policy).
SUPPORTED_TTS_FAMILIES: frozenset[str] = frozenset(
    {"orpheus", "sesame_csm", "llasa", "spark", "oute"}
)

_MAX_TTS_FAMILY_LEN: int = 32
_MAX_EMOTION_LEN: int = 32


@dataclass(frozen=True)
class TTSFamilySpec:
    """Metadata for a TTS family. Frozen so callers cannot mutate."""

    name: str
    description: str
    supports_emotion: bool
    live_wired: bool


_TTS_FAMILY_METADATA: Mapping[str, TTSFamilySpec] = MappingProxyType({
    "orpheus": TTSFamilySpec(
        name="orpheus",
        description="Orpheus emotional TTS (canopylabs)",
        supports_emotion=True,
        live_wired=False,
    ),
    "sesame_csm": TTSFamilySpec(
        name="sesame_csm",
        description="Sesame CSM conversational speech",
        supports_emotion=False,
        live_wired=False,
    ),
    "llasa": TTSFamilySpec(
        name="llasa",
        description="Llasa-TTS (HKUSTAudio)",
        supports_emotion=False,
        live_wired=False,
    ),
    "spark": TTSFamilySpec(
        name="spark",
        description="Spark-TTS (SparkAudio)",
        supports_emotion=False,
        live_wired=False,
    ),
    "oute": TTSFamilySpec(
        name="oute",
        description="Oute-TTS (outeai)",
        supports_emotion=True,
        live_wired=False,
    ),
})

# Per-family emotion allowlists. Orpheus + Oute support emotion conditioning;
# the others ignore the tag. Closed allowlist keeps trainer dispatch
# deterministic.
ORPHEUS_EMOTIONS: frozenset[str] = frozenset({
    "neutral", "happy", "sad", "angry", "excited", "calm", "whisper", "laugh",
})

# Oute supports a tighter set focused on prosody / register.
OUTE_EMOTIONS: frozenset[str] = frozenset({
    "neutral", "happy", "sad", "angry", "calm", "excited",
})

_FAMILY_EMOTIONS: Mapping[str, frozenset[str]] = MappingProxyType({
    "orpheus": ORPHEUS_EMOTIONS,
    "oute": OUTE_EMOTIONS,
})


def validate_tts_family(name: object) -> str:
    """Validate a TTS family name and return the canonical (lowercase) form."""
    if isinstance(name, bool):
        raise TypeError(f"tts_family must not be bool, got {name!r}")
    if not isinstance(name, str):
        raise TypeError(
            f"tts_family must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("tts_family must be non-empty")
    if "\x00" in name:
        raise ValueError("tts_family must not contain null bytes")
    if len(name) > _MAX_TTS_FAMILY_LEN:
        raise ValueError(
            f"tts_family too long (max {_MAX_TTS_FAMILY_LEN} chars)"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_TTS_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_TTS_FAMILIES))
        raise ValueError(
            f"tts_family {name!r} not supported. Supported: {supported}"
        )
    return canonical


def get_tts_family_spec(name: str) -> TTSFamilySpec:
    """Return the frozen :class:`TTSFamilySpec` for ``name`` or raise."""
    canonical = validate_tts_family(name)
    return _TTS_FAMILY_METADATA[canonical]


def family_supports_emotion(name: str) -> bool:
    """Whether the named family supports an ``emotion`` tag."""
    canonical = validate_tts_family(name)
    return _TTS_FAMILY_METADATA[canonical].supports_emotion


def validate_emotion_tag(emotion: object, *, family: str) -> str:
    """Validate an ``emotion`` tag for ``family`` (currently Orpheus only).

    Raises if the family does not support emotion conditioning or the tag
    is not in the per-family allowlist. ``family`` is canonicalised first
    so callers don't have to.
    """
    canonical_family = validate_tts_family(family)
    if isinstance(emotion, bool):
        raise TypeError(f"emotion must not be bool, got {emotion!r}")
    if not isinstance(emotion, str):
        raise TypeError(
            f"emotion must be str, got {type(emotion).__name__}"
        )
    if not emotion:
        raise ValueError("emotion must be non-empty")
    if "\x00" in emotion:
        raise ValueError("emotion must not contain null bytes")
    if len(emotion) > _MAX_EMOTION_LEN:
        raise ValueError(
            f"emotion too long (max {_MAX_EMOTION_LEN} chars)"
        )
    spec = _TTS_FAMILY_METADATA[canonical_family]
    if not spec.supports_emotion:
        raise ValueError(
            f"tts_family={canonical_family!r} does not support emotion "
            "conditioning"
        )
    canonical = emotion.lower()
    # Per-family allowlist — data-driven so future emotion-supporting
    # families cannot silently bypass the check.
    family_allowlist = _FAMILY_EMOTIONS.get(canonical_family)
    if family_allowlist is not None and canonical not in family_allowlist:
        allowed = ", ".join(sorted(family_allowlist))
        raise ValueError(
            f"emotion {emotion!r} not in {canonical_family} allowlist. "
            f"Allowed: {allowed}"
        )
    return canonical


def validate_tts_compat(*, task: str, modality: str, backend: str) -> None:
    """Schema-time gate for ``task='tts'``.

    Rejects:
    - non-string / bool args (defence-in-depth — sister-function bool
      guards align with v0.30.0 ``Candidate`` policy).
    - non-TTS task (intended for ``task == 'tts'``).
    - ``modality != 'audio_out'`` (TTS is audio-output by definition).
    - ``backend == 'mlx'`` (no MLX TTS path in v0.52.0).
    """
    for name, value in (("task", task), ("modality", modality), ("backend", backend)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if task != "tts":
        raise ValueError(
            f"validate_tts_compat called with task={task!r} (expected 'tts')"
        )
    if modality != "audio_out":
        raise ValueError(
            f"task='tts' requires modality='audio_out'; got modality={modality!r}"
        )
    if backend == "mlx":
        raise ValueError(
            "task='tts' is not supported on backend=mlx in v0.52.0"
        )


def build_tts_trainer() -> None:
    """Live TTS trainer factory — deferred to v0.52.1.

    Mirrors v0.50.0 ``build_prm_trainer`` / v0.49.0 ``apply_longlora_forward_override``
    stub-then-live pattern.
    """
    raise NotImplementedError(
        "TTS trainer (task='tts') live wiring deferred to v0.52.1. "
        "Schema accepts the value but no trainer wrapper is registered yet."
    )
