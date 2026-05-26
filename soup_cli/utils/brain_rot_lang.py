"""Per-language brain-rot heuristic bundles (closes issue #234).

Extends v0.69.0 Part E ``score_triviality`` + ``score_popularity_signal``
to non-English corpora. The Shannon-entropy + punctuation-run signals in
``brain_rot.py`` stay language-agnostic; the keyword half is what goes
silent on non-English text. This module ships a small closed-allowlist
registry of per-language token + phrase bundles so the keyword half
keeps working on es / fr / de / ru without a heavyweight dependency.

Design (Option A from the issue — preferred over the operator-supplied
JSON form because it composes with the project's bundled-resources
policy from v0.65.0 behavior batteries + v0.68.0 local-rl fixtures):

- ``BrainRotLangBundle`` is a frozen dataclass with token / phrase tuples.
- ``_LANG_BUNDLES`` is a ``MappingProxyType`` so callers can't mutate it
  at runtime (matches v0.51.0 hubs / v0.60.0 license_matrix policy).
- ``SUPPORTED_LANGS`` is a frozenset of canonical ISO 639-1 codes.
- ``get_lang_bundle`` is the lookup path used by the scorers: unknown
  codes silently fall back to the English bundle (no exception — the
  detector should keep working even on weakly-resourced languages).
- ``validate_lang_code`` is the strict CLI-boundary validator that
  rejects unknown codes loudly (matches v0.41.0 ``validate_optimizer_name``
  policy).

The token + phrase lists are intentionally small seed sets — operators
who want richer per-language coverage can either extend them upstream
or pass a custom bundle via a v0.69.x follow-up flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

_MAX_LANG_CODE_LEN = 64


@dataclass(frozen=True)
class BrainRotLangBundle:
    """Frozen per-language low-effort + clickbait token bundle.

    ``code`` is the ISO 639-1 identifier. ``low_effort_tokens`` are
    whitespace-tokenised lowercase strings matched against the per-token
    body (with surrounding punctuation stripped). ``clickbait_phrases``
    are lowercase substrings matched against the lowercased input text.
    """

    code: str
    low_effort_tokens: Tuple[str, ...]
    clickbait_phrases: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("code must be a non-empty string")
        if "\x00" in self.code:
            raise ValueError("code must not contain null bytes")
        if not isinstance(self.low_effort_tokens, tuple):
            raise TypeError("low_effort_tokens must be a tuple")
        if not isinstance(self.clickbait_phrases, tuple):
            raise TypeError("clickbait_phrases must be a tuple")
        for entries, name in (
            (self.low_effort_tokens, "low_effort_tokens"),
            (self.clickbait_phrases, "clickbait_phrases"),
        ):
            for entry in entries:
                if not isinstance(entry, str):
                    raise TypeError(f"{name} entries must be strings")
                if not entry:
                    raise ValueError(f"{name} entries must be non-empty")
                if "\x00" in entry:
                    raise ValueError(
                        f"{name} entries must not contain null bytes"
                    )


# ---------------------------------------------------------------------------
# Bundles. Each list is a small seed set (≈ 8-14 tokens + 6-10 phrases).
# Phrases are lowercased upfront so the substring scan in
# ``score_popularity_signal`` matches reliably against ``text.lower()``.
# ---------------------------------------------------------------------------


_EN_BUNDLE = BrainRotLangBundle(
    code="en",
    low_effort_tokens=(
        "lol", "omg", "lmao", "rofl", "smh", "tbh", "idk",
        "wtf", "ikr", "tldr", "fml", "yolo",
    ),
    clickbait_phrases=(
        "you won't believe",
        "you wont believe",
        "won't believe what happened",
        "top 10",
        "top ten",
        "click here",
        "this one weird trick",
        "what happened next",
        "the rest is history",
        "shocked the world",
        "doctors hate",
        "gone wrong",
        "gone viral",
    ),
)

_ES_BUNDLE = BrainRotLangBundle(
    code="es",
    low_effort_tokens=(
        "jaja", "jajaja", "jeje", "jiji", "lol", "omg",
        "xd", "wtf", "ojalá", "ay",
    ),
    clickbait_phrases=(
        "no creerás",
        "no vas a creer",
        "haz clic aquí",
        "click aquí",
        "lo que pasó después",
        "lo que pasó a continuación",
        "top 10",
        "top diez",
        "no podrás creer",
        "te dejará sin palabras",
        "te sorprenderá",
        "los médicos odian",
    ),
)

_FR_BUNDLE = BrainRotLangBundle(
    code="fr",
    low_effort_tokens=(
        "mdr", "ptdr", "lol", "omg", "xd", "jpp", "tkt",
        "wtf", "rofl", "lmao",
    ),
    clickbait_phrases=(
        "vous n'allez pas le croire",
        "vous n'allez pas y croire",
        "n'allez pas le croire",
        "cliquez ici",
        "ce qui se passe ensuite",
        "top 10",
        "top dix",
        "ce qu'il s'est passé",
        "ce qu'il s'est passé ensuite",
        "incroyable mais vrai",
        "les médecins détestent",
        "vous serez choqué",
    ),
)

_DE_BUNDLE = BrainRotLangBundle(
    code="de",
    low_effort_tokens=(
        "krass", "omg", "lol", "wtf", "xd", "hä", "alter",
        "boah", "ehh", "rofl", "lmao",
    ),
    clickbait_phrases=(
        "du wirst es nicht glauben",
        "du wirst nicht glauben",
        "ihr werdet nicht glauben",
        "hier klicken",
        "klick hier",
        "top 10",
        "top zehn",
        "was als nächstes passiert",
        "was dann geschah",
        "schockierend",
        "ärzte hassen",
        "der eine geheime trick",
    ),
)

_RU_BUNDLE = BrainRotLangBundle(
    code="ru",
    low_effort_tokens=(
        "ааа", "ххх", "лол", "ржу", "кек", "ыыы", "омг",
        "пиздец", "жесть", "капец",
    ),
    clickbait_phrases=(
        "вы не поверите",
        "вы не поверите что",
        "не поверите что",
        "нажмите здесь",
        "кликните здесь",
        "топ 10",
        "топ десять",
        "что произошло дальше",
        "что случилось дальше",
        "шокировало весь мир",
        "врачи ненавидят",
        "один странный трюк",
    ),
)


_LANG_BUNDLES: Mapping[str, BrainRotLangBundle] = MappingProxyType(
    {
        "en": _EN_BUNDLE,
        "es": _ES_BUNDLE,
        "fr": _FR_BUNDLE,
        "de": _DE_BUNDLE,
        "ru": _RU_BUNDLE,
    }
)


SUPPORTED_LANGS: frozenset = frozenset(_LANG_BUNDLES.keys())


# ---------------------------------------------------------------------------
# Public lookup + validation surface
# ---------------------------------------------------------------------------


def _check_lang_arg_shape(value: object) -> None:
    """Common shape check for ``code``-typed inputs."""
    if isinstance(value, bool):
        raise TypeError("lang must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"lang must be str, got {type(value).__name__}")
    if "\x00" in value:
        raise ValueError("lang must not contain null bytes")
    if len(value) > _MAX_LANG_CODE_LEN:
        raise ValueError(
            f"lang must be <= {_MAX_LANG_CODE_LEN} chars"
        )


def get_lang_bundle(lang: object) -> BrainRotLangBundle:
    """Return the bundle for ``lang`` (silent fallback to English).

    ``lang=None`` is accepted and resolves to the English bundle so callers
    can pass it through from optional kwargs without branching. Unknown
    codes fall back silently — the brain-rot detector should keep
    producing a result rather than raising on weakly-resourced languages.
    For a strict surface (e.g. the CLI), use :func:`validate_lang_code`.
    """
    if lang is None:
        return _EN_BUNDLE
    _check_lang_arg_shape(lang)
    canonical = lang.lower()  # type: ignore[union-attr]
    return _LANG_BUNDLES.get(canonical, _EN_BUNDLE)


def validate_lang_code(value: object) -> str:
    """Strict validator (CLI boundary): return canonical lower-case code.

    Accepts every entry in :data:`SUPPORTED_LANGS` plus the literal
    ``"auto"`` sentinel (which the scorer surface resolves via
    ``langdetect`` per v0.53.10 #113 ``[data-pro]`` extras). Unknown
    codes raise ``ValueError`` so the CLI fails fast on typos.
    """
    _check_lang_arg_shape(value)
    canonical = value.lower()  # type: ignore[union-attr]
    if not canonical:
        raise ValueError("lang must be non-empty")
    if canonical == "auto":
        return canonical
    if canonical not in SUPPORTED_LANGS:
        raise ValueError(
            f"lang {canonical!r} not in supported set "
            f"{sorted(SUPPORTED_LANGS) + ['auto']}"
        )
    return canonical


__all__ = [
    "BrainRotLangBundle",
    "SUPPORTED_LANGS",
    "get_lang_bundle",
    "validate_lang_code",
]
