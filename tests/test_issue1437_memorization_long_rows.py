"""Regression tests for long-row memorization detection (#1437)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from soup_cli.utils.diagnose.memorization import score_memorization, split_prefix

_LIVE_GENERATION_BUDGET = 64
_ROW_LENGTHS = (60, 100, 150, 300, 1000)


class _WordTokenizer:
    """Small reversible word tokenizer for the tokenizer-aware probe path."""

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = []
        for token in text.split():
            if token not in self._token_to_id:
                token_id = len(self._token_to_id) + 1
                self._token_to_id[token] = token_id
                self._id_to_token[token_id] = token
            ids.append(self._token_to_id[token])
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(self._id_to_token[token_id] for token_id in ids)

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._id_to_token[token_id] for token_id in ids]


def _training_text(word_count: int, *, stem: str = "term") -> str:
    return " ".join(f"{stem}{index:04d}x" for index in range(word_count))


def _capped_continuation(
    text: str,
    *,
    tokenizer: Optional[_WordTokenizer] = None,
) -> tuple[str, str]:
    prefix, suffix = split_prefix(text, tokenizer=tokenizer)
    if tokenizer is None:
        completion = " ".join(suffix.split()[:_LIVE_GENERATION_BUDGET])
    else:
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        completion = tokenizer.decode(
            suffix_ids[:_LIVE_GENERATION_BUDGET],
            skip_special_tokens=True,
        )
    return prefix, completion


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
    tokenizer = _WordTokenizer()
    text = _training_text(word_count)
    prefix, completion = _capped_continuation(text, tokenizer=tokenizer)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: completion if prompt == prefix else "",
        tokenizer=tokenizer,
    )

    assert score.verdict == "MAJOR"
    assert "echo_rate=1.000" in score.evidence


@pytest.mark.parametrize("word_count", _ROW_LENGTHS)
@pytest.mark.parametrize("tokenizer_aware", [False, True])
def test_same_length_unrelated_continuation_stays_ok(
    word_count: int,
    tokenizer_aware: bool,
) -> None:
    tokenizer = _WordTokenizer() if tokenizer_aware else None
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
    repeated_completion = " ".join([repeated_word] * _LIVE_GENERATION_BUDGET)

    score = score_memorization(
        [{"text": text}],
        lambda prompt: repeated_completion if prompt == prefix else "",
    )

    assert score.verdict == "OK"
    assert "echo_rate=0.000" in score.evidence


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
