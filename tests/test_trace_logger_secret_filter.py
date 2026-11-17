"""The trace log's secret filter runs in linear time on a long message.

``soup serve --trace-log`` redacts every request message and answer with ``_SECRET_RE`` inline in
the request thread, and ``re`` holds the GIL, so a pattern that backtracks quadratically stalls the
whole server on one long message. The JWT alternative (``eyJ...``) did: on a text made of ``eyJ``
repeated, every ``eyJ`` was a start and each start consumed the rest of the run and backtracked
(120,000 characters 8 s, 240,000 characters 36 s). It now starts only at the left edge of a
base64url run and its repeats are possessive (no backtracking into a run that cannot continue).

The bounds are generous absolute ones (the broken pattern takes 30 s and more on these inputs, the
fixed one a few milliseconds), so a slow machine does not make them flaky.
"""

from __future__ import annotations

import re
import time

import pytest

from soup_cli.monitoring import trace_logger
from soup_cli.monitoring.trace_logger import TraceLogWriter, _redact_secrets

BOUND_SECONDS = 5.0
SIZE = 240_000


def _repeat(unit: str, size: int = SIZE) -> str:
    return (unit * (size // len(unit) + 1))[:size]


def _took(text: str) -> float:
    started = time.perf_counter()
    _redact_secrets(text)
    return time.perf_counter() - started


# What a long hostile message can look like, one family per alternative of the pattern and the
# shapes that make a repeated alternative expensive (a start at every character, a run that has
# to be scanned and given up).
HOSTILE = {
    "jwt prefixes, no dots": _repeat("eyJ"),
    "jwt prefixes with dots": _repeat("eyJaaaaaaaa."),
    "jwt two parts and a short tail": _repeat("eyJaaaaaaaa.eyJaaaaaaaa.ab "),
    "jwt prefix and one long run": "eyJ" + "a" * (SIZE - 3),
    "jwt header, dot, prefix, long run": "eyJaaaaaaaa.eyJ" + "b" * (SIZE - 15),
    "hf prefixes": _repeat("hf_"),
    "sk prefixes": _repeat("sk-"),
    "sk prefix and a short run": _repeat("sk-abcdefghij "),
    "bearer words": _repeat("Bearer "),
    "bearer and a whitespace run": "Bearer" + " " * (SIZE - 6),
    "github prefixes": _repeat("ghp_"),
    "github pat prefixes": _repeat("github_pat_"),
    "aws key prefixes": _repeat("AKIA"),
    "slack prefixes": _repeat("xoxb-"),
    "pem headers": _repeat("-----BEGIN PRIVATE KEY-----"),
    "pem header and a long body": "-----BEGIN PRIVATE KEY-----" + "A" * (SIZE - 27),
    "assignment name and a whitespace run": "token" + " " * (SIZE - 5),
    "assignment sign and a whitespace run": "token=" + " " * (SIZE - 6),
    "assignment names": _repeat("password: "),
    "assignments with short values": _repeat("api_key=abc "),
    "assignment name and a long value": "token " + "a" * (SIZE - 6),
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_a_240k_character_message_is_filtered_in_a_bounded_time(name):
    text = HOSTILE[name]
    assert len(text) == SIZE

    assert _took(text) < BOUND_SECONDS, name


@pytest.mark.parametrize(
    "unit", ["eyJ", "eyJaaaaaaaa.", "eyJaaaaaaaa.eyJaaaaaaaa.ab "], ids=["no dots", "dots", "tail"]
)
def test_the_jwt_inputs_stay_bounded_at_twice_the_size_too(unit):
    """Linear growth: 480,000 characters is under the same bound as 240,000."""
    assert _took(_repeat(unit, 2 * SIZE)) < BOUND_SECONDS


def test_the_adversarial_jwt_input_is_filtered_through_the_real_writer(tmp_path, monkeypatch):
    """The path a chat request takes: ``record_conversation`` redacts the message inline."""
    monkeypatch.chdir(tmp_path)
    writer = TraceLogWriter("trace.jsonl")
    started = time.perf_counter()

    writer.record_conversation(
        response_id="0de3d93e-f426-4096-8ba2-14f7fb992421",
        model="m",
        messages=[{"role": "user", "content": _repeat("eyJ")}],
        response="ok",
    )

    assert time.perf_counter() - started < BOUND_SECONDS
    assert len((tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_the_filter_is_one_compiled_pattern():
    assert isinstance(trace_logger._SECRET_RE, re.Pattern)


JWT = "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0NTY3ODkwIn0." + "abcdefghijk123-_"


@pytest.mark.parametrize(
    "template",
    [
        "{jwt}",
        "before {jwt} after",
        "authorization: {jwt}",
        'cookie="{jwt}"',
        "token_in_json: {{\"jwt\":\"{jwt}\"}}",
        "(see {jwt}).",
        "line one\n{jwt}\nline three",
        "a.{jwt}",
        "{jwt}.",
        "id={jwt}&x=1",
        "two {jwt} and {jwt}",
    ],
)
def test_a_jwt_is_still_redacted_wherever_it_sits(template):
    text = template.format(jwt=JWT)

    redacted = _redact_secrets(text)

    assert "<redacted>" in redacted
    assert "JhbGciOiJIUzI1NiJ9" not in redacted and "abcdefghijk123" not in redacted


def test_a_jwt_is_redacted_whole_and_the_text_around_it_is_kept():
    assert _redact_secrets(f"see {JWT} now") == "see <redacted> now"
    assert _redact_secrets(f"{JWT}, then {JWT}") == "<redacted>, then <redacted>"


@pytest.mark.parametrize(
    "text",
    [
        "eyJ alone is nothing",
        "eyJhbGciOiJIUzI1NiJ9.onlytwoparts torch.nn.Module.forward v1.2.3",
        "eyJabcde.eyJabcde.abcdefgh",  # parts shorter than 8 characters
        "eyJhbGciOiJIUzI1NiJ9.notajwtpartatall.abcdefghijk",  # the second part is no JWT part
        "Eyjhbgcioijiuzi1nij9.eyjzdwiioiixmjm0nty3odkwin0.abcdefghijk123",  # wrong case
    ],
)
def test_a_near_miss_stays_readable(text):
    assert _redact_secrets(text) == text
