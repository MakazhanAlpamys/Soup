"""Regression tests for long-row memorization detection (#1437)."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import pytest

from soup_cli.utils.diagnose.memorization import score_memorization, split_prefix

_LIVE_GENERATION_WORDS = 48
_LIVE_GENERATION_TOKENS = 64
_ROW_LENGTHS = (60, 100, 150, 300, 1000)
_SYLLABLES = (
    "ka",
    "lo",
    "mi",
    "ren",
    "tu",
    "sha",
    "vo",
    "del",
    "pin",
    "zar",
    "qui",
    "bel",
)
_VOCAB = [a + b + c for a in _SYLLABLES for b in _SYLLABLES for c in _SYLLABLES]
_WEIGHTS = [1.0 / (rank + 1) ** 0.8 for rank in range(len(_VOCAB))]


class _CharTokenizer:
    """Small reversible tokenizer distinct from whitespace tokenization."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [chr(token_id) for token_id in ids]


def _training_text(word_count: int, *, stem: str = "term") -> str:
    return " ".join(f"{stem}{index:04d}x" for index in range(word_count))


def _capped_continuation(
    text: str,
    *,
    tokenizer: Optional[_CharTokenizer] = None,
) -> tuple[str, str]:
    prefix, suffix = split_prefix(text, tokenizer=tokenizer)
    if tokenizer is None:
        completion = " ".join(suffix.split()[:_LIVE_GENERATION_WORDS])
    else:
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        completion = tokenizer.decode(
            suffix_ids[:_LIVE_GENERATION_TOKENS],
            skip_special_tokens=True,
        )
    return prefix, completion


def _natural_text(word_count: int, seed: int) -> str:
    return " ".join(random.Random(seed).choices(_VOCAB, weights=_WEIGHTS, k=word_count))


def _flagged(text: str, completion: str) -> bool:
    prefix, _ = split_prefix(text)
    score = score_memorization(
        [{"text": text}],
        lambda prompt: completion if prompt == prefix else "",
    )
    return score.verdict == "MAJOR"


@pytest.mark.parametrize("word_count", _ROW_LENGTHS)
def test_capped_verbatim_continuation_flags_every_row_length(word_count: int) -> None:
    text = _training_text(word_count)
    prefix, completion = _capped_continuation(text)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: completion if prompt == prefix else "",
    )

    assert score.verdict == "MAJOR"
    assert "echo_rate=1.000" in score.evidence


@pytest.mark.parametrize("word_count", _ROW_LENGTHS)
def test_tokenizer_capped_verbatim_continuation_flags_every_row_length(
    word_count: int,
) -> None:
    tokenizer = _CharTokenizer()
    text = _training_text(word_count)
    prefix, completion = _capped_continuation(text, tokenizer=tokenizer)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: completion if prompt == prefix else "",
        tokenizer=tokenizer,
    )

    assert score.verdict == "MAJOR"
    assert "echo_rate=1.000" in score.evidence


def test_tokenizer_path_detects_subword_echo_whitespace_misses() -> None:
    tokenizer = _CharTokenizer()
    text = "x" * 400
    prefix, completion = _capped_continuation(text, tokenizer=tokenizer)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: completion if prompt == prefix else "",
        tokenizer=tokenizer,
    )

    assert score.verdict == "MAJOR"
    assert "tokenizer_aware=True" in score.evidence


@pytest.mark.parametrize("word_count", _ROW_LENGTHS)
@pytest.mark.parametrize("tokenizer_aware", [False, True])
def test_same_length_unrelated_continuation_stays_ok(
    word_count: int,
    tokenizer_aware: bool,
) -> None:
    tokenizer = _CharTokenizer() if tokenizer_aware else None
    text = _training_text(word_count)
    prefix, completion = _capped_continuation(text, tokenizer=tokenizer)
    completion_length = len(completion.split())
    unrelated = _training_text(completion_length, stem="noise")

    score = score_memorization(
        [{"text": text}],
        lambda prompt: unrelated if prompt == prefix else "",
        tokenizer=tokenizer,
    )

    assert score.verdict == "OK"
    assert "echo_rate=0.000" in score.evidence


def test_repeated_suffix_word_does_not_count_as_verbatim_echo() -> None:
    text = _training_text(300)
    prefix, suffix = split_prefix(text)
    repeated_word = suffix.split()[0]
    repeated_completion = " ".join([repeated_word] * _LIVE_GENERATION_WORDS)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: repeated_completion if prompt == prefix else "",
    )

    assert score.verdict == "OK"
    assert "echo_rate=0.000" in score.evidence


@pytest.mark.parametrize("word_count", (300, 1000, 2000, 4000))
def test_verbatim_with_repeated_words_is_flagged(word_count: int) -> None:
    text = _natural_text(word_count, seed=1)
    _, suffix = split_prefix(text)

    assert _flagged(text, " ".join(suffix.split()[:_LIVE_GENERATION_WORDS]))


@pytest.mark.parametrize("word_count", (300, 1000, 2000, 4000))
def test_unrelated_text_from_the_same_vocabulary_is_not_flagged(
    word_count: int,
) -> None:
    text = _natural_text(word_count, seed=1)
    unrelated = " ".join(_natural_text(_LIVE_GENERATION_WORDS, seed=2).split())

    assert not _flagged(text, unrelated)


@pytest.mark.parametrize("word_count", (1000, 4000))
def test_repetition_loop_is_not_memorization(word_count: int) -> None:
    text = _natural_text(word_count, seed=1)
    _, suffix = split_prefix(text)
    common = Counter(suffix.split()).most_common(1)[0][0]

    assert not _flagged(text, " ".join([common] * _LIVE_GENERATION_WORDS))


def test_one_word_completion_is_not_memorization() -> None:
    text = _natural_text(1000, seed=1)
    _, suffix = split_prefix(text)

    assert not _flagged(text, suffix.split()[0])


def test_run_live_diagnose_flags_regurgitation_on_250_word_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from soup_cli.utils.diagnose import live

    rows = []
    completions_by_prefix = {}
    for row_index in range(12):
        text = _training_text(250, stem=f"row{row_index}term")
        prefix, completion = _capped_continuation(text)
        completions_by_prefix[prefix] = completion
        rows.append(
            {
                "prompt": f"question {row_index}",
                "response": f"answer {row_index}",
                "text": text,
            }
        )

    dataset_path = tmp_path / "training.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def generate(prompt: str) -> str:
        return completions_by_prefix.get(prompt, "ordinary response")

    def generate_many(prompt: str, count: int) -> list[str]:
        return [f"{prompt} response {index}" for index in range(count)]

    def load_pair(
        base: str,
        adapter: Optional[str] = None,
        **kwargs: object,
    ) -> dict[str, object]:
        del base, adapter, kwargs
        return {
            "base_gen": generate,
            "adapter_gen": generate,
            "base_multi": generate_many,
            "adapter_multi": generate_many,
        }

    monkeypatch.setattr(live, "load_adapter_pair", load_pair)

    report = live.run_live_diagnose(
        run_id="long-row-regression",
        base="stub-model",
        dataset_path=str(dataset_path),
    )

    assert report.scores["memorization"].verdict == "MAJOR"
    assert report.scores["memorization"].score == 0.0
