"""The email PII baseline must stay linear on long near-miss local parts."""

from __future__ import annotations

import random
import re
import statistics
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score

_BASELINE_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PATHOLOGICAL_RUNS = ("1-", "1.", "a.", "a-", "a+")


def _email_pattern() -> re.Pattern[str]:
    return dict(data_score._PII_PATTERNS)["email"]


def _email_snippets(text: str) -> list[str]:
    return [hit["snippet"] for hit in data_score.detect_pii(text) if hit["kind"] == "email"]


def test_email_detection_growth_is_subquadratic(monkeypatch: pytest.MonkeyPatch) -> None:
    """An 8x longer row must not consume 64x the regex time."""
    monkeypatch.setattr(data_score, "_presidio_pii", lambda _text: None)
    n = 5_000

    for run in _PATHOLOGICAL_RUNS:
        smaller = (run * ((n + len(run) - 1) // len(run)))[:n]
        larger = (run * ((8 * n + len(run) - 1) // len(run)))[: 8 * n]

        small_samples = []
        large_samples = []
        for _ in range(3):
            start = time.perf_counter()
            small_hits = data_score.detect_pii(smaller)
            small_samples.append(time.perf_counter() - start)

            start = time.perf_counter()
            large_hits = data_score.detect_pii(larger)
            large_samples.append(time.perf_counter() - start)

        small_elapsed = statistics.median(small_samples)
        large_elapsed = statistics.median(large_samples)

        assert not any(hit["kind"] == "email" for hit in small_hits)
        assert not any(hit["kind"] == "email" for hit in large_hits)
        assert large_elapsed < small_elapsed * 32, (
            f"{run!r}: 8N took {large_elapsed:.3f}s vs N {small_elapsed:.3f}s"
        )


def test_email_detection_matches_baseline_row_flags_on_seeded_texts() -> None:
    """Keep row-level decisions stable across a deterministic corpus."""
    rng = random.Random(1329)
    alphabet = "abxyz0123456789 ._+-@\n"
    at_sign = "@"
    samples = [
        "jane.doe" + at_sign + "example.com",
        "--john" + at_sign + "example.com",
        "a" + at_sign + "b.com" + "-x" + at_sign + "y.com",
        "+@example.com",
        "...@example.com",
        "plain text without an address",
    ]

    for _ in range(20_000):
        length = rng.randrange(0, 81)
        sample = "".join(rng.choice(alphabet) for _ in range(length))
        if rng.random() < 0.25:
            sample += rng.choice(("@example.com", "@b.co", ".com", ".org"))
        samples.append(sample)

    current = _email_pattern()
    for sample in samples:
        assert bool(current.search(sample)) == bool(_BASELINE_EMAIL_PATTERN.search(sample)), sample


def test_documented_email_snippet_differences_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The detector may change snippets, but not whether either row is flagged."""
    monkeypatch.setattr(data_score, "_presidio_pii", lambda _text: None)
    current = _email_pattern()

    at_sign = "@"
    leading_punctuation = "--john" + at_sign + "example.com"
    assert _BASELINE_EMAIL_PATTERN.search(leading_punctuation).group() == (
        "john" + at_sign + "example.com"
    )
    assert current.search(leading_punctuation).group() == leading_punctuation
    assert _email_snippets(leading_punctuation) == [leading_punctuation]

    first_address = "a" + at_sign + "b.com"
    second_address = "-x" + at_sign + "y.com"
    glued_addresses = first_address + second_address
    assert [m.group() for m in _BASELINE_EMAIL_PATTERN.finditer(glued_addresses)] == [
        first_address,
        second_address,
    ]
    assert [m.group() for m in current.finditer(glued_addresses)] == [first_address]
    assert bool(current.search(glued_addresses)) == bool(
        _BASELINE_EMAIL_PATTERN.search(glued_addresses)
    )


def test_soup_expect_completes_for_pathological_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public gate completes a <50 KB pathological row within two seconds."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_score, "_presidio_pii", lambda _text: None)

    data_file = tmp_path / "pathological.jsonl"
    text = "1-" * 24_991
    data_file.write_text('{"text": "' + text + '"}\n', encoding="utf-8")
    suite_file = tmp_path / "suite.yaml"
    suite_file.write_text("expectations:\n  - name: expect_no_pii\n", encoding="utf-8")

    start = time.perf_counter()
    result = CliRunner().invoke(app, ["expect", str(data_file), str(suite_file)])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, f"output: {result.output}\nexc: {result.exception!r}"
    assert elapsed < 2.0, f"pathological `soup expect` took {elapsed:.3f}s"
