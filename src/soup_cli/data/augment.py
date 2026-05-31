"""Data augmentation strategies (Part F of v0.25.0).

Three built-in strategies:
  - rephrase: rewrite each example preserving meaning
  - translate: translate examples to target languages
  - style: rewrite examples in different tonal styles

Each strategy takes an LLM-like provider with a ``generate(prompt, max_tokens)``
method and returns a list of augmented dict rows in the same format as input.
"""

from __future__ import annotations

from typing import Callable, Protocol

MAX_AUGMENT_COUNT = 10
DEFAULT_LANGUAGES = ("ru", "zh", "es")
DEFAULT_STYLES = ("formal", "casual", "technical")


class Provider(Protocol):
    def generate(self, prompt: str, max_tokens: int = 512) -> str: ...


def _validate_count(count: int) -> None:
    if count < 1 or count > MAX_AUGMENT_COUNT:
        raise ValueError(
            f"count must be between 1 and {MAX_AUGMENT_COUNT}, got {count}"
        )


def _text_fields(row: dict) -> dict:
    """Return a copy of ``row`` containing only string fields safe to rewrite."""
    return {k: v for k, v in row.items() if isinstance(v, str)}


def _apply_rewrite(
    row: dict,
    rewrite: Callable[[str], str],
) -> dict:
    new_row = dict(row)
    for k, v in row.items():
        if isinstance(v, str) and v:
            new_row[k] = rewrite(v)
    return new_row


def augment_rephrase(
    examples: list[dict],
    provider: Provider,
    count: int = 2,
) -> list[dict]:
    """Rephrase each example ``count`` times preserving meaning.

    Returns ``len(examples) * count`` rows.
    """
    _validate_count(count)
    augmented: list[dict] = []
    for row in examples:
        for i in range(count):
            def _rewrite(text: str, _i: int = i) -> str:
                prompt = (
                    f"Rewrite the following text preserving its meaning but using "
                    f"different wording (variant {_i + 1}):\n\n{text}"
                )
                return provider.generate(prompt)
            augmented.append(_apply_rewrite(row, _rewrite))
    return augmented


def augment_translate(
    examples: list[dict],
    provider: Provider,
    languages: list[str] | None = None,
) -> list[dict]:
    """Translate each example into every target language."""
    if languages is None:
        langs = list(DEFAULT_LANGUAGES)
    else:
        langs = list(languages)
    if not langs:
        raise ValueError("augment_translate requires at least one language")
    if len(langs) > MAX_AUGMENT_COUNT:
        raise ValueError(
            f"too many languages: {len(langs)} > {MAX_AUGMENT_COUNT}"
        )
    augmented: list[dict] = []
    for row in examples:
        for lang in langs:
            def _rewrite(text: str, _lang: str = lang) -> str:
                prompt = (
                    f"Translate the following text into {_lang}, preserving the "
                    f"meaning exactly. Do not add commentary:\n\n{text}"
                )
                return provider.generate(prompt)
            augmented.append(_apply_rewrite(row, _rewrite))
    return augmented


def augment_style(
    examples: list[dict],
    provider: Provider,
    styles: list[str] | None = None,
) -> list[dict]:
    """Rewrite each example in multiple tonal styles."""
    if styles is None:
        target_styles = list(DEFAULT_STYLES)
    else:
        target_styles = list(styles)
    if not target_styles:
        raise ValueError("augment_style requires at least one style")
    if len(target_styles) > MAX_AUGMENT_COUNT:
        raise ValueError(
            f"too many styles: {len(target_styles)} > {MAX_AUGMENT_COUNT}"
        )
    augmented: list[dict] = []
    for row in examples:
        for style in target_styles:
            def _rewrite(text: str, _style: str = style) -> str:
                prompt = (
                    f"Rewrite the following in a {_style} tone, preserving "
                    f"meaning:\n\n{text}"
                )
                return provider.generate(prompt)
            augmented.append(_apply_rewrite(row, _rewrite))
    return augmented


STRATEGIES: dict[str, Callable] = {
    "rephrase": augment_rephrase,
    "translate": augment_translate,
    "style": augment_style,
}
