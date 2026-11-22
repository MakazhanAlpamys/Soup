"""The trace log's secret filter runs in linear time on a long message.

``soup serve --trace-log`` redacts every request message and answer with ``_SECRET_RE`` inline in
the request thread, and ``re`` holds the GIL, so a pattern that backtracks quadratically stalls the
whole server on one long message. The JWT alternative (``eyJ...``) did: on a text made of ``eyJ``
repeated, every ``eyJ`` was a start and each start consumed the rest of the run and backtracked
(120,000 characters 8 s, 240,000 characters 36 s). It now starts only at the left edge of a
base64url run (a look-behind) and checks once per run, with a look-ahead, that a second and a third
part follow, so a run that cannot be a token is scanned once instead of once per ``eyJ`` in it. A
token that is glued to word characters (``x<token>``, ``id123<token>``) is redacted together with
them, and the placements the old pattern redacted are pinned below (a separator before the token,
a character of the run before it, a token that ends inside the run).

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
    "glued jwt prefixes": _repeat("xeyJ"),
    "glued jwt prefixes with dots": _repeat("xeyJaaaaaaaa."),
    "glued jwt parts with a short tail": _repeat("id123eyJaaaaaaaa.eyJaaaaaaaa.aaaaaaa "),
    "long first run, then a whole token": "a" * (SIZE - 21) + ".eyJaaaaaaaa.aaaaaaaa",
    "long first run ending in eyJ, then a token": "a" * (SIZE - 24) + "eyJ.eyJaaaaaaaa.aaaaaaaa",
    "jwt prefixes, then a whole token": _repeat("eyJ", SIZE - 21) + ".eyJaaaaaaaa.aaaaaaaa",
    "many first runs, third part too short": _repeat("a" * 40 + ".eyJaaaaaaaa.aaaaaaa "),
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


# --- a JWT directly after other characters -----------------------------------------------------
# The first part of a JWT is base64url, so the character before it can belong to the same run
# (`x<token>`, a URL path `/tok-<token>`, a base64 blob). The pattern that ran in quadratic time
# redacted all of these; a first version of the linear one (a left edge alone) stopped doing so.
# The shapes are written down here, each must be redacted, and the old pattern is kept below as
# the reference for "nothing that was redacted is left in the log".

HEADER, PAYLOAD, SIGNATURE = JWT.split(".")

#: The character before the token is in [A-Za-z0-9_-]: it is part of the token's run.
GLUED = [
    "x{jwt}",
    "0{jwt}",
    "_{jwt}",
    "-{jwt}",
    "https://h/a_{jwt}",
    "https://h/tok-{jwt}",
    "session_{jwt}.txt",
    "id123{jwt}",
    "jwt=x{jwt}",
    "QUJD{jwt}",
]

#: A separator (or nothing) before the token: redacted before and after, kept as a pin.
SEPARATED = [
    "{jwt}",
    "k={jwt}",
    "Bearer:{jwt}",
    "Bearer: {jwt}",
    "bearer {jwt}",
    "/{jwt}",
    ".{jwt}",
    ":{jwt}",
    '"{jwt}"',
    "({jwt})",
    "x\n{jwt}",
    ";{jwt}",
    "a|{jwt}",
    "https://h/?t={jwt}",
    "https://h/#{jwt}",
    '{{"jwt":"{jwt}"}}',
    "token={jwt}",
    "{jwt}-x",
    chr(0xE9) + "{jwt}",
    chr(0x1F600) + "{jwt}",
]


def _no_part_of_the_token_is_left(redacted: str) -> bool:
    return HEADER not in redacted and PAYLOAD not in redacted and SIGNATURE not in redacted


@pytest.mark.parametrize("template", GLUED + SEPARATED)
def test_a_jwt_is_redacted_whatever_precedes_it(template):
    redacted = _redact_secrets(template.format(jwt=JWT))

    assert "<redacted>" in redacted, template
    assert _no_part_of_the_token_is_left(redacted), template


def test_the_characters_glued_to_a_token_go_with_it_and_the_rest_of_the_text_stays():
    assert _redact_secrets("x" + JWT) == "<redacted>"
    assert _redact_secrets("id123" + JWT) == "<redacted>"
    assert _redact_secrets("see https://h/a_" + JWT + " now") == "see https://h/<redacted> now"
    assert _redact_secrets("session_" + JWT + ".txt") == "<redacted>.txt"
    assert _redact_secrets("a=b;" + JWT) == "a=b;<redacted>"
    assert _redact_secrets("xx " + JWT + " yy") == "xx <redacted> yy"


#: The JWT alternative as it was before the left edge (quadratic, but it matched wherever an
#: ``eyJ`` stood). Kept here only as the reference of what must still be redacted.
OLD_JWT_RE = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")

PREFIXES = sorted({template.split("{jwt}")[0] for template in GLUED + SEPARATED})
SUFFIXES = ["", ".", "-x", ".txt", ")", " and more", "\n", "_"]


def test_nothing_the_old_pattern_redacted_is_left_in_the_log():
    checked = 0
    for prefix in PREFIXES:
        for suffix in SUFFIXES:
            text = prefix + JWT + suffix
            old_matches = OLD_JWT_RE.findall(text)
            assert old_matches, (prefix, suffix)  # the reference does match: not a vacuous loop

            redacted = _redact_secrets(text)

            for matched in old_matches:
                assert matched not in redacted, (prefix, suffix)
            assert _no_part_of_the_token_is_left(redacted), (prefix, suffix)
            checked += 1
    assert checked == len(PREFIXES) * len(SUFFIXES) and checked > 150


def test_two_tokens_on_one_line_each_go_whole_whatever_glues_them_to_the_text():
    text = f"a{JWT} b_{JWT}, c-{JWT}."

    assert _redact_secrets(text) == "<redacted> <redacted>, <redacted>."


# --- a JWT after "Bearer" -----------------------------------------------------------------------
# The Bearer alternative used to stop at the first dot and, being the leftmost, won over the JWT
# alternative: payload and signature stayed in the log. The commonest place for a JWT in a request
# log is an Authorization header.


@pytest.mark.parametrize(
    "template",
    [
        "Bearer {jwt}",
        "Authorization: Bearer {jwt}",
        "Authorization:   Bearer    {jwt}",
        'headers={{"Authorization": "Bearer {jwt}"}}',
    ],
)
def test_a_jwt_after_bearer_is_redacted_whole(template):
    redacted = _redact_secrets(template.format(jwt=JWT))

    assert "<redacted>" in redacted, template
    assert _no_part_of_the_token_is_left(redacted), template


def test_a_bearer_jwt_goes_whole_and_the_header_name_stays():
    assert _redact_secrets("Authorization: Bearer " + JWT) == "Authorization: <redacted>"
    assert _redact_secrets("Bearer " + JWT + ".") == "<redacted>."
    assert _redact_secrets("Bearer " + JWT + " and " + JWT) == "<redacted> and <redacted>"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Use Bearer abc12345xyz.", "Use <redacted>."),
        ("Use Bearer abc12345xyz. Then go.", "Use <redacted>. Then go."),
        ("Bearer abc12345xyz.\n", "<redacted>.\n"),
        ("Bearer abc12345xyz..", "<redacted>.."),
        ("Bearer abc12345xyz,more", "<redacted>,more"),
        ("Bearer abc12345xyz", "<redacted>"),
    ],
)
def test_a_bearer_token_keeps_a_full_stop_that_ends_the_sentence(text, expected):
    assert _redact_secrets(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Bearer abcdefgh.Bearer abcdefgh12", "<redacted>.<redacted>"),
        ("Bearer abcdefgh.token=abcdefgh", "<redacted>.<redacted>"),
        ("Bearer abcdefgh.-----BEGIN PRIVATE KEY-----AAAA", "<redacted>.<redacted>"),
        ("Bearer abcdefgh.hf_abcdefgh12", "<redacted>.<redacted>"),
    ],
)
def test_a_bearer_token_does_not_swallow_the_secret_that_follows_it(text, expected):
    """The Bearer-JWT alternative applies only when a whole token follows; any other Bearer
    token stops where it always stopped, so what comes after a dot is still looked at."""
    assert _redact_secrets(text) == expected


# --- a JWT after another token that ends inside the same run -------------------------------------
# `ghp_...` and `hf_...` tokens stop at `_` / `-`, so a JWT glued behind one of them keeps a run
# the first token cut into: the placeholder of the first pass is the left edge of the second.

TOKENS_THAT_END_INSIDE_A_RUN = [
    "ghp_" + "A" * 20 + "_",
    "ghp_" + "A" * 20 + "-",
    "hf_abcdefgh12_",
    "hf_abcdefgh12-",
    "xoxb-" + "A" * 10 + "_",
    "github_pat_" + "A" * 20 + "-",
]


@pytest.mark.parametrize("head", TOKENS_THAT_END_INSIDE_A_RUN)
def test_a_jwt_glued_behind_a_token_that_ends_inside_the_run_is_redacted_too(head):
    redacted = _redact_secrets(head + JWT)

    assert head not in redacted, head
    assert _no_part_of_the_token_is_left(redacted), head
    assert redacted.count("<redacted>") == 2, (head, redacted)


# --- the same shapes stay linear ----------------------------------------------------------------

#: Shapes that make the look-ahead pass and the body search for a start, or that glue a prefix to
#: every part: the cost of each is a few passes over the text, so four times the text takes about
#: four times as long (the measured ladder is in the PR; the bound here is only generous).
LADDER_SIZES = (60_000, 240_000, 960_000)
LADDER = {
    "glued prefixes": lambda size: _repeat("xeyJ", size),
    "glued parts": lambda size: _repeat("xeyJaaaaaaaa.xeyJaaaaaaaa.xaaaaaaaa ", size),
    "a long first run, then a whole token": lambda size: (
        "a" * (size - 21) + ".eyJaaaaaaaa.aaaaaaaa"
    ),
    "a long first run, then a token with a short third part": lambda size: (
        "a" * (size - 20) + ".eyJaaaaaaaa.aaaaaaa"
    ),
    "prefixes, then a whole token": lambda size: (
        _repeat("eyJ", size - 21) + ".eyJaaaaaaaa.aaaaaaaa"
    ),
    "bearer and a dotted chain": lambda size: "Bearer aaaaaaaa" + _repeat(".a", size - 15),
    "bearer and a dotted chain of long parts": lambda size: (
        "Bearer aaaaaaaa" + _repeat(".aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", size)
    ),
    "many bearers with a short value": lambda size: _repeat("Bearer abc.", size),
    "bearer and jwt prefixes": lambda size: "Bearer " + _repeat("eyJ", size - 7),
    "bearer and jwt prefixes, then a whole token": lambda size: (
        "Bearer " + _repeat("eyJ", size - 28) + ".eyJaaaaaaaa.aaaaaaaa"
    ),
    "a token cut short by another token, again and again": lambda size: _repeat(
        "ghp_" + "A" * 20 + "_eyJaaaaaaaa.eyJaaaaaaaa.aaaaaaaa ", size
    ),
}


@pytest.mark.parametrize("name", sorted(LADDER))
def test_the_jwt_and_bearer_shapes_stay_bounded_up_to_960k_characters(name):
    timings = []
    for size in LADDER_SIZES:
        text = LADDER[name](size)
        assert len(text) >= size - 1, (name, size)
        timings.append(_took(text))
        assert timings[-1] < BOUND_SECONDS, (name, size, timings)  # stop at the first size over
