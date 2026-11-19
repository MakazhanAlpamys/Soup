"""The email PII baseline must stay linear on long near-miss local parts."""

from __future__ import annotations

import random
import re
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from soup_cli.cli import app
from soup_cli.utils import data_score
from tests._growth_timing import assert_grows_linearly, linux_only

_BASELINE_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PATHOLOGICAL_RUNS = ("1-", "1.", "a.", "a-", "a+")


def _email_pattern() -> re.Pattern[str]:
    return dict(data_score._PII_PATTERNS)["email"]


def _email_snippets(text: str) -> list[str]:
    return [
        hit["snippet"]
        for hit in data_score.detect_pii(text)
        if hit["kind"] == "email"
    ]


def _row(run: str, length: int) -> str:
    return (run * ((length + len(run) - 1) // len(run)))[:length]


@linux_only
def test_email_pattern_growth_is_linear() -> None:
    """The pattern alone: an 8x longer row must not take 32x the time (linear 8x, quadratic 64x).

    ``detect_pii`` reads only the first ``_PII_SCAN_CAP`` characters, so a row long enough to
    time reliably is past what it scans. The compiled pattern has no such cap: the helper
    doubles the row until one search lasts long enough to time, then compares it with 8x.
    """
    pattern = _email_pattern()
    for run in _PATHOLOGICAL_RUNS:
        assert_grows_linearly(
            pattern.search, lambda length, run=run: _row(run, length), 500, label=repr(run)
        )


@linux_only
def test_email_detection_growth_is_subquadratic(monkeypatch: pytest.MonkeyPatch) -> None:
    """``detect_pii`` end to end (all four patterns), within the rows it scans in full."""
    monkeypatch.setattr(data_score, "_presidio_pii", lambda _text: None)
    n = 5_000

    for run in _PATHOLOGICAL_RUNS:
        # The growth check goes first: a quadratic pattern fails here in seconds, where
        # scanning the 8n row for the no-hit check below would take it minutes. The cap keeps
        # the larger row inside the 50,000 characters ``detect_pii`` reads (past it the time
        # stops growing and the check would pass for any pattern).
        assert_grows_linearly(
            data_score.detect_pii,
            lambda length, run=run: _row(run, length),
            n // 10,
            label=repr(run),
            cap=data_score._PII_SCAN_CAP,
        )
        for text in (_row(run, n), _row(run, 8 * n)):
            assert not any(hit["kind"] == "email" for hit in data_score.detect_pii(text))


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


def _expect_elapsed(tmp_path: Path, name: str, text: str) -> float:
    """Run the public gate on one row and return the wall-clock time it took."""
    data_file = tmp_path / f"{name}.jsonl"
    data_file.write_text('{"text": "' + text + '"}\n', encoding="utf-8")
    suite_file = tmp_path / f"{name}.yaml"
    suite_file.write_text(
        "expectations:\n  - name: expect_no_pii\n", encoding="utf-8"
    )

    start = time.perf_counter()
    result = CliRunner().invoke(app, ["expect", str(data_file), str(suite_file)])
    elapsed = time.perf_counter() - start

    assert result.exit_code == 0, f"output: {result.output}\nexc: {result.exception!r}"
    return elapsed


def test_soup_expect_completes_for_pathological_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public gate does not go quadratic on a row of near-miss local parts.

    The bound is relative, not a wall-clock one: a fixed two-second bound failed
    on a loaded macOS runner (10.5 s for the linear scan, #2302). A harmless row
    of the same length is timed first, and the pathological row must stay within
    a small factor of it; restoring the old quadratic email pattern fails this by
    orders of magnitude.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_score, "_presidio_pii", lambda _text: None)

    length = 24_991 * 2
    harmless_text = "word " * (length // 5)
    pathological_text = "1-" * (length // 2)

    harmless = min(
        _expect_elapsed(tmp_path, "harmless", harmless_text) for _ in range(2)
    )
    pathological = min(
        _expect_elapsed(tmp_path, "pathological", pathological_text) for _ in range(2)
    )

    assert pathological < 5 * harmless, (
        f"pathological `soup expect` took {pathological:.3f}s against "
        f"{harmless:.3f}s for a harmless row of the same length"
    )
