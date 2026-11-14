"""Crash bundle redaction: URLs, credential shapes, key names and the redaction counts.

The bundle is meant to be attached to an issue, so everything stored in it (the error
message, the traceback, the metrics tail and the config) goes through
``soup_cli.utils.redaction`` first. The spelling table below is written down BEFORE the
patterns it covers: every credential shape has one positive row (the secret must be
gone) and a near-miss negative (ordinary text that must come back unchanged).

Credential-looking fixtures are assembled at run time from pieces, so no line of this
file looks like a real key to a scanner.
"""

from __future__ import annotations

import base64
import copy
import faulthandler
import json
import sys
import time
from dataclasses import dataclass

import pytest

from soup_cli.utils import redaction
from soup_cli.utils.crash import build_crash_bundle, redact_secrets, write_crash_bundle
from soup_cli.utils.redaction import (
    REDACTED,
    RedactionReport,
    is_secret_key,
    redact_config,
    redact_text,
    redact_token_shapes,
    redact_url,
)


def _alnum(length: int, seed: str = "aB3dE5gH7jK9mN1pQ3rS5tU7vW9xY2zC4") -> str:
    return (seed * (length // len(seed) + 1))[:length]


def _b64url(raw: str) -> str:
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


HF = "hf" + "_" + _alnum(34)
SK = "sk" + "-" + _alnum(48)
SK_PROJ = "sk" + "-proj-" + _alnum(40)
SK_ANT = "sk" + "-ant-api03-" + _alnum(40) + "-" + _alnum(10)
GH_PAT = "github" + "_pat_" + _alnum(22) + "_" + _alnum(40)
AKIA = "AKIA" + "IOSFODNN7EXAMPLE"
ASIA = "ASIA" + "IOSFODNN7EXAMPLE"
AWS_SECRET = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
GOOGLE = "AIza" + "SyD" + _alnum(32)
SLACK_TAIL = "123456789012-" + _alnum(24)
WANDB = "0123456789abcdef" * 2 + "01234567"
JWT = ".".join([
    _b64url(json.dumps({"alg": "HS256", "typ": "JWT"})),
    _b64url(json.dumps({"sub": "1234567890", "name": "someone"})),
    _b64url("signature-bytes-0123456789"),
])
BEARER_VALUE = _alnum(30)
BASIC_VALUE = base64.b64encode(b"someuser:somepassword").decode()
PEM_BODY = "\n".join(_alnum(64, seed="MIIEvQIBADANBgkqhkiG9w0BAQEFAASC") for _ in range(4))


def _pem(kind: str, body: str = PEM_BODY) -> str:
    return f"-----BEGIN {kind}-----\n{body}\n-----END {kind}-----"


@dataclass(frozen=True)
class Row:
    """One spelling of one credential shape."""

    group: str  # the pattern group that must catch it
    label: str
    positive: str  # contains ``secret``
    secret: str  # must be absent from the redacted text
    negative: str  # near miss: must come back unchanged
    keeps: str = ""  # text that must survive (a key name, an auth scheme)
    exclusive: bool = True  # no other shape group catches the positive


GH_NEGATIVE = "ghp_short ghx_" + _alnum(36) + " ghost_" + _alnum(36)
SLACK_NEGATIVE = "xoxz-" + SLACK_TAIL + " xox-" + SLACK_TAIL + " xoxb-short"
NOT_KEYS = (
    "max_new_tokens=512 tokenizer=Qwen/Qwen2.5-0.5B auth_mode=none author: alice "
    "token_count=12 learning_rate: 2e-5 num_tokens: 7 tokens: 4096 authentication: required "
    "authenticated: yes"
)
URL_NEGATIVE = (
    "mail me@example.com or git@github.com:org/repo.git; note:me@example.com, key:v@x.org. "
    "see https://huggingface.co/api/models?revision=main and ssh://example.org/a/b"
)
# A secret longer than any fixed cap on the length of a value: only the tail is checked, so
# a pass that replaces just the first part of the value cannot hide behind the whole string.
LONG_HEAD = _alnum(700, seed="LmNoPqRsTuVwXyZaBcDeFgHiJkMn")
LONG_TAIL = "TailPart" + _alnum(300, seed="zYxWvUtSrQpOnMlKjIhGfEdCbA9876543")


def _github_rows() -> list[Row]:
    rows = []
    for prefix in ("ghp", "gho", "ghu", "ghs", "ghr"):
        token = prefix + "_" + _alnum(36)
        rows.append(Row(
            "github", prefix + "_", f"cloning with {token} now", token, GH_NEGATIVE,
        ))
    rows.append(Row("github", "github_pat_", f"cloning with {GH_PAT} now", GH_PAT, GH_NEGATIVE))
    return rows


def _slack_rows() -> list[Row]:
    return [
        Row("slack", f"xox{kind}-", f"posting with xox{kind}-{SLACK_TAIL} now",
            f"xox{kind}-{SLACK_TAIL}", SLACK_NEGATIVE)
        for kind in "abprs"
    ]


ROWS: list[Row] = [
    Row("hf", "hf_", f"download failed for {HF} today", HF,
        "cache key hf_short and chf_" + _alnum(30) + " and hf_" + _alnum(19)),
    Row("openai", "sk-", f"api returned 401 for {SK}", SK,
        "task-force-alpha-bravo-charlie-delta and disk-usage-report-for-each-node sk-short"),
    Row("openai", "sk-proj-", f"api returned 401 for {SK_PROJ}", SK_PROJ,
        "task-force-alpha-bravo-charlie-delta and disk-usage-report-for-each-node sk-short"),
    Row("sk_ant", "sk-ant-", f"api returned 401 for {SK_ANT}", SK_ANT,
        "sk-ant-short task-ant-colony-optimisation-algorithm-run"),
    *_github_rows(),
    Row("aws_access_key_id", "AKIA", f"identity {AKIA} was denied", AKIA,
        "AKIA" + _alnum(15, "ABCDEFGH") + " " + AKIA + "X " + AKIA.lower()),
    Row("aws_access_key_id", "ASIA", f"identity {ASIA} was denied", ASIA,
        "ASIA" + _alnum(15, "ABCDEFGH") + " " + ASIA + "X " + ASIA.lower()),
    Row("aws_secret_access_key", "aws_secret_access_key",
        f"aws_secret_access_key = {AWS_SECRET}", AWS_SECRET,
        f"checksum {AWS_SECRET} aws_region = us-east-1", keeps="aws_secret_access_key"),
    Row("google_api_key", "AIza", f"maps key {GOOGLE} was refused", GOOGLE,
        "AIza" + _alnum(10) + " AIzb" + _alnum(35)),
    *_slack_rows(),
    Row("wandb", "WANDB_API_KEY", f"WANDB_API_KEY={WANDB}", WANDB,
        "commit " + WANDB + " WANDB_PROJECT=soup", keeps="WANDB_API_KEY"),
    Row("bearer", "Bearer <x>", f"retrying with Bearer {BEARER_VALUE} again", BEARER_VALUE,
        "Bearer of bad news travels fast; Bearer authentication is required"),
    Row("bearer", "headers repr",
        "{'Accept': 'application/json', 'Authorization': 'Bearer " + BEARER_VALUE + "'}",
        BEARER_VALUE, "{'Accept': 'application/json', 'Authorization': 'Bearer'}",
        keeps="'Accept': 'application/json'", exclusive=False),
    Row("authorization_header", "Basic", f"Authorization: Basic {BASIC_VALUE}", BASIC_VALUE,
        "Basic understanding of the loss curve; the Authorization header is missing",
        keeps="Authorization: Basic "),
    Row("authorization_header", "json Token",
        '{"authorization": "Token ' + _alnum(24) + '"}', _alnum(24),
        '{"authorization": "none"}', keeps='{"authorization": "Token '),
    Row("jwt", "three parts", f"session cookie {JWT} expired", JWT,
        "eyJhbGciOiJIUzI1NiJ9.onlytwoparts torch.nn.Module.forward v1.2.3"),
    Row("pem", "RSA block", f"loaded key\n{_pem('RSA PRIVATE KEY')}\nthen failed",
        PEM_BODY.splitlines()[0],
        f"{_pem('CERTIFICATE')}\n{_pem('PUBLIC KEY')}", keeps="loaded key\n"),
    Row("pem", "OPENSSH block", f"key={_pem('OPENSSH PRIVATE KEY')};", PEM_BODY.splitlines()[1],
        f"{_pem('CERTIFICATE')}"),
    Row("pem", "truncated block (no END line)",
        "-----BEGIN PRIVATE KEY-----\n" + PEM_BODY, PEM_BODY.splitlines()[2],
        "-----BEGIN PUBLIC KEY-----\n" + PEM_BODY),
    Row("assignment", "env line", "HF_TOKEN=abc123plain", "abc123plain", NOT_KEYS,
        keeps="HF_TOKEN="),
    Row("assignment", "export", "export OPENAI_API_KEY=plainvalue123\n", "plainvalue123",
        NOT_KEYS, keeps="export OPENAI_API_KEY="),
    Row("assignment", "yaml", "password: hunter2\nmodel: x", "hunter2", NOT_KEYS,
        keeps="password: "),
    Row("assignment", "yaml secret", "  client_secret: s3cretvalue", "s3cretvalue", NOT_KEYS,
        keeps="client_secret: "),
    Row("assignment", "json pair", '{"api_key": "plainsecretvalue", "model": "x"}',
        "plainsecretvalue", NOT_KEYS, keeps='"model": "x"'),
    Row("assignment", "python repr", "{'access_token': 'plainsecretvalue'}",
        "plainsecretvalue", NOT_KEYS, keeps="{'access_token': '"),
    Row("assignment", "escaped json", '{\\"api_key\\": \\"plainsecretvalue\\"}',
        "plainsecretvalue", NOT_KEYS, keeps='{\\"api_key\\": \\"'),
    Row("assignment", "camelCase key", 'accessToken = "plainsecretvalue"', "plainsecretvalue",
        NOT_KEYS, keeps="accessToken = "),
    # The values below hold none of the words a key name is searched for (token, secret,
    # key, auth, ...): only the KEY NAME can make the pass look at them.
    Row("assignment", "header key", "X-Api-Key: plainvalue-77", "plainvalue-77",
        NOT_KEYS, keeps="X-Api-Key: "),
    Row("assignment", "lower-case header key", "x-api-key: plainvalue-77", "plainvalue-77",
        NOT_KEYS, keeps="x-api-key: "),
    Row("assignment", "hyphenated dict repr", "{'x-api-key': 'plainvalue-77', 'n': 3}",
        "plainvalue-77", NOT_KEYS, keeps="{'x-api-key': '"),
    Row("assignment", "hyphenated cli flag", "run --api-key=plainvalue-77 --x", "plainvalue-77",
        NOT_KEYS, keeps="--api-key="),
    Row("assignment", "dotted key", "client.api.key=plainvalue-77", "plainvalue-77",
        NOT_KEYS, keeps="client.api.key="),
    Row("assignment", "long unquoted value", f"access_token={LONG_HEAD}{LONG_TAIL} next",
        LONG_TAIL[:40], NOT_KEYS, keeps="access_token="),
    Row("assignment", "long double-quoted value",
        '{"refresh_token": "' + LONG_HEAD + LONG_TAIL + '", "n": 1}',
        LONG_TAIL[:40], NOT_KEYS, keeps='"n": 1}'),
    Row("assignment", "long single-quoted value",
        "{'refresh_token': '" + LONG_HEAD + LONG_TAIL + "'}",
        LONG_TAIL[:40], NOT_KEYS, keeps="{'refresh_token': '"),
    Row("assignment", "quoted value with an escaped quote",
        '{"password": "plain\\"tail-77x", "n": 1}', "tail-77x", NOT_KEYS, keeps='"n": 1}'),
    Row("assignment", "quoted value with a backslash", "password='ab\\\\cd-77plain' ok",
        "cd-77plain", NOT_KEYS, keeps="password='"),
    Row("assignment", "unquoted value with a backslash", "password=ab\\cd-77plain next",
        "cd-77plain", NOT_KEYS, keeps="password="),
    Row("url", "long scheme (SQLAlchemy driver)",
        "engine postgresql+psycopg2://admin:plainvalue-77@db.internal:5432/soup failed",
        "plainvalue-77", URL_NEGATIVE, keeps="@db.internal:5432/soup failed"),
    Row("url", "scheme-relative userinfo", "proxy //admin:plainvalue-77@proxy.example/x now",
        "plainvalue-77", URL_NEGATIVE, keeps="@proxy.example/x now"),
    Row("url", "userinfo without a scheme (path)", "mirror admin:plainvalue-77@mirror.example/x",
        "plainvalue-77", URL_NEGATIVE, keeps="@mirror.example/x"),
    Row("url", "userinfo without a scheme (port)", "db admin:plainvalue-77@10.0.0.5:5432 down",
        "plainvalue-77", URL_NEGATIVE, keeps="@10.0.0.5:5432 down"),
]

SHAPE_GROUPS = {shape.name for shape in redaction._SHAPES}
ROW_IDS = [f"{row.group}:{row.label}" for row in ROWS]


# ---------------------------------------------------------------------------
# The spelling table
# ---------------------------------------------------------------------------


def test_every_shape_group_has_a_row_in_the_table():
    assert SHAPE_GROUPS <= {row.group for row in ROWS}
    assert {"pem", "assignment"} <= {row.group for row in ROWS}


def test_table_covers_the_spellings_that_were_listed():
    labels = {row.label for row in ROWS}
    for spelling in ("hf_", "sk-", "sk-proj-", "sk-ant-", "ghp_", "gho_", "ghu_", "ghs_",
                     "ghr_", "github_pat_", "AKIA", "ASIA", "aws_secret_access_key", "AIza",
                     "WANDB_API_KEY", "Bearer <x>", "headers repr", "Basic", "three parts",
                     "RSA block", "env line", "yaml", "json pair"):
        assert spelling in labels, spelling
    for kind in "abprs":
        assert f"xox{kind}-" in labels


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_positive_spelling_is_redacted(row):
    report = RedactionReport()
    out = redact_text(row.positive, report)
    assert row.secret not in out, out
    assert REDACTED in out
    assert report.count >= 1
    assert row.keeps in out, out


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_near_miss_is_left_alone(row):
    report = RedactionReport()
    assert redact_text(row.negative, report) == row.negative
    assert report.count == 0


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_redacting_twice_changes_nothing(row):
    once = redact_text(row.positive)
    report = RedactionReport()
    assert redact_text(once, report) == once
    assert report.count == 0


@pytest.mark.parametrize(
    "row", [row for row in ROWS if row.group in SHAPE_GROUPS and row.exclusive],
    ids=lambda row: f"{row.group}:{row.label}",
)
def test_a_shape_group_is_the_only_thing_catching_its_row(row, monkeypatch):
    assert redact_token_shapes(row.positive) != row.positive
    without = tuple(shape for shape in redaction._SHAPES if shape.name != row.group)
    monkeypatch.setattr(redaction, "_SHAPES", without)
    assert redact_token_shapes(row.positive) == row.positive


def test_pem_pass_is_what_removes_a_private_key_block(monkeypatch):
    text = _pem("RSA PRIVATE KEY")
    monkeypatch.setattr(redaction, "_redact_pem", lambda value, report: value)
    assert PEM_BODY.splitlines()[0] in redact_text(text)


def test_assignment_pass_is_what_removes_a_plain_key_value(monkeypatch):
    monkeypatch.setattr(redaction, "_redact_assignments", lambda value, report: value)
    assert "hunter2" in redact_text("password: hunter2")


def test_pem_block_between_other_text_keeps_the_text():
    out = redact_text(f"before\n{_pem('RSA PRIVATE KEY')}\nafter {_pem('EC PRIVATE KEY')} end")
    assert out == f"before\n{REDACTED}\nafter {REDACTED} end"


def test_redact_secrets_keeps_its_old_contract():
    assert "hf_" not in redact_secrets(f"token={HF} broken")
    assert redact_secrets("everything fine here") == "everything fine here"
    assert redact_secrets(None) is None
    assert redact_secrets(42) == 42


# ---------------------------------------------------------------------------
# URLs: userinfo and sensitive query parameters
# ---------------------------------------------------------------------------

URL_SECRET = "Zq9Xv7Lm2Kp4Rt6W"
SENSITIVE_PARAMS = [
    "token", "Token", "TOKEN", "key", "api_key", "api-key", "apiKey", "API_KEY", "apikey",
    "access_token", "access-token", "accessToken", "id_token", "refresh_token", "secret",
    "client_secret", "signature", "sig", "password", "pwd", "auth", "credential",
    "X-Amz-Signature", "x_amz_signature", "X-Amz-Credential", "x-amz-credential",
    "X-Amz-Security-Token", "x_amz_security_token", "X-Goog-Signature",
]
HARMLESS_PARAMS = [
    "page", "revision", "max_tokens", "tokenizer", "monkey", "sort", "format", "keyword",
    "author", "authors", "limit", "X-Amz-Algorithm", "X-Amz-Date", "X-Amz-Expires",
    "X-Amz-SignedHeaders", "token_count",
]


@pytest.mark.parametrize("name", SENSITIVE_PARAMS)
def test_sensitive_query_parameter_value_is_replaced(name):
    out = redact_text(f"GET https://hub.example/v1/models?revision=main&{name}={URL_SECRET}"
                      "&page=2 failed")
    assert URL_SECRET not in out
    assert f"{name}={REDACTED}" in out
    assert "https://hub.example/v1/models?revision=main&" in out
    assert out.endswith("&page=2 failed")


@pytest.mark.parametrize("name", HARMLESS_PARAMS)
def test_harmless_query_parameter_is_kept(name):
    text = f"GET https://hub.example/v1/models?{name}=value123 failed"
    assert redact_text(text) == text


@pytest.mark.parametrize("url, host_part", [
    (f"https://someuser:{URL_SECRET}@host.example/path?x=1", "@host.example/path?x=1"),
    (f"https://{URL_SECRET}@github.com/org/repo.git", "@github.com/org/repo.git"),
    (f"postgres://admin:{URL_SECRET}@db.internal:5432/soup", "@db.internal:5432/soup"),
    (f"git+https://someuser:{URL_SECRET}@github.com/org/repo.git@v1",
     "@github.com/org/repo.git@v1"),
    (f"s3://AKIDEXAMPLE:{URL_SECRET}@bucket/key", "@bucket/key"),
    (f"postgresql+psycopg2://admin:{URL_SECRET}@db.internal/soup", "@db.internal/soup"),
    (f"mysql+pymysql://admin:{URL_SECRET}@db.internal/soup", "@db.internal/soup"),
    (f"HTTPS://Admin:{URL_SECRET}@Host.Example/Path", "@Host.Example/Path"),
    (f"https://:{URL_SECRET}@host.example/path", "@host.example/path"),
    (f"redis://:{URL_SECRET}@cache.internal:6379/0", "@cache.internal:6379/0"),
    (f"{'x' * 30}://u:{URL_SECRET}@host.example/p", "@host.example/p"),
])
def test_url_userinfo_is_replaced_and_the_host_is_kept(url, host_part):
    out = redact_text(f"fetching {url} failed")
    assert URL_SECRET not in out
    assert f"://{REDACTED}{host_part}" in out
    assert out.startswith("fetching ") and out.endswith(" failed")


@pytest.mark.parametrize("text, expected", [
    (f"//u:{URL_SECRET}@h.example/x", f"//{REDACTED}@h.example/x"),
    (f"proxy=//{URL_SECRET}@h.example/x", f"proxy=//{REDACTED}@h.example/x"),
    (f"u:{URL_SECRET}@h.example/x", f"{REDACTED}@h.example/x"),
    (f"u:{URL_SECRET}@h.example:8080", f"{REDACTED}@h.example:8080"),
    (f"u:{URL_SECRET}@localhost:5432/soup", f"{REDACTED}@localhost:5432/soup"),
    (f"(u:{URL_SECRET}@10.0.0.5:5432)", f"({REDACTED}@10.0.0.5:5432)"),
    (f"a.b-c_d:{URL_SECRET}@h.example/x?y=1", f"{REDACTED}@h.example/x?y=1"),
    (f"<u:{URL_SECRET}@h.example/x>", f"<{REDACTED}@h.example/x>"),
    (f"socks5h:u:{URL_SECRET}@h.example/x", f"socks5h:{REDACTED}@h.example/x"),
    (f"{'x' * 50}://u:{URL_SECRET}@h.example/x", f"{'x' * 50}://{REDACTED}@h.example/x"),
])
def test_userinfo_without_a_scheme_is_replaced(text, expected):
    report = RedactionReport()
    out = redact_text(text, report)
    assert out == expected, out
    assert report.count == 1
    assert redact_text(out) == out


@pytest.mark.parametrize("text", [
    "me@example.com",
    "mail me@example.com or you@example.org.",
    "git@github.com:org/repo.git",
    "ssh git@github.com:22/org/repo.git",
    "note:me@example.com, then retry",
    "key:v@x.org.",
    "12:30@noon.example",
    "see /srv/a:b@host.example/x for details",
    "https://host.example/a:b@c.example/d",
    "user: me@example.com/path",
    "x//me@example.com",
    "https://host.example/path//a@b.example",
    "file:///home/me@laptop/x",
    "a:b@c",
    "name:" + "x" * 600 + "@host.example/p",
])
def test_text_that_only_resembles_userinfo_is_untouched(text):
    report = RedactionReport()
    assert redact_text(text, report) == text
    assert report.count == 0


def test_fragment_parameters_are_masked():
    out = redact_text(f"redirect https://app.example/cb#access_token={URL_SECRET}&state=xyz")
    assert URL_SECRET not in out
    assert f"#access_token={REDACTED}&state=xyz" in out


def test_url_that_urlsplit_refuses_is_still_masked():
    out = redact_text(f"bad http://user:{URL_SECRET}@[::1/path?token={URL_SECRET}&x=1 end")
    assert URL_SECRET not in out
    assert f"token={REDACTED}&x=1" in out


def test_trailing_punctuation_is_not_part_of_the_url():
    out = redact_text(f"(see https://u:{URL_SECRET}@h.example/x?token={URL_SECRET}).")
    assert URL_SECRET not in out
    assert out.endswith(f"token={REDACTED}).")


def test_url_inside_a_repr_keeps_the_surrounding_structure():
    out = redact_text(f"{{'url': 'https://u:{URL_SECRET}@h.example/x?sig={URL_SECRET}', 'n': 3}}")
    assert URL_SECRET not in out
    assert out.endswith(f"?sig={REDACTED}', 'n': 3}}")


@pytest.mark.parametrize("text", [
    "https://huggingface.co/api/models/org/name?revision=main",
    "http://localhost:8000/v1/chat/completions",
    "contact me@example.com about it",
    "C:\\Users\\me\\run\\config.yaml",
    "HTTP://Host.Example/Path?",
    "file:///tmp/data/train.jsonl",
])
def test_url_without_credentials_is_untouched(text):
    report = RedactionReport()
    assert redact_text(text, report) == text
    assert report.count == 0


def test_url_redaction_is_idempotent_and_counts_each_value():
    text = f"https://u:{URL_SECRET}@h.example/x?token={URL_SECRET}&api_key={URL_SECRET}&n=1"
    report = RedactionReport()
    once = redact_text(text, report)
    assert report.count == 3
    again = RedactionReport()
    assert redact_text(once, again) == once
    assert again.count == 0


def test_redact_url_keeps_the_judge_behaviour_by_default():
    url = "https://u:p@h.example:8080/p?q=1#f"
    assert redact_url(url) == "https://h.example:8080/p"
    assert redact_url("http://[::1]:8000/v1") == "http://[::1]:8000/v1"
    assert redact_url(url, keep_query=True) == f"https://{REDACTED}@h.example:8080/p?q=1#f"
    with pytest.raises(ValueError):
        redact_url("http://[::1/p")


def test_the_judge_uses_the_shared_url_helper():
    from soup_cli.eval import judge

    url = f"https://u:{URL_SECRET}@judge.example/v1?api_key={URL_SECRET}"
    assert judge._redact_url(url) == redact_url(url) == "https://judge.example/v1"
    assert judge._redact_url("http://[::1/p") == "<unparseable judge url>"


# ---------------------------------------------------------------------------
# Key names in structured configs
# ---------------------------------------------------------------------------

SECRET_KEYS = [
    "token", "hf_token", "HF_TOKEN", "api_key", "apiKey", "API_KEY", "apikey", "openai_api_key",
    "wandb_api_key", "x-api-key", "--api-key", "access_token", "accessToken", "auth_token",
    "use_auth_token", "client_secret", "secret_key", "password", "db_passwd", "passphrase",
    "private_key", "privateKey", "aws_secret_access_key", "aws_access_key_id", "credentials",
    "Authorization", "authorization_header", "bearer", "cookie", "Set-Cookie", "session_id",
    "sessionToken", "id_token", "admin_tokens", "oauth", "GOOGLE_APPLICATION_CREDENTIALS",
]
HARMLESS_KEYS = [
    "max_new_tokens", "max_tokens", "min_tokens", "num_tokens", "n_tokens", "token_count",
    "tokenizer", "tokenizer_name", "tokenizer_path", "tokenized_path", "tokenization",
    "eos_token", "pad_token", "bos_token", "unk_token", "special_tokens", "new_special_tokens",
    "add_new_tokens", "online_dpo_max_new_tokens", "grpo_judge_max_tokens", "maxNewTokens",
    "numTokens", "tokens_per_second", "pad_token_id", "author", "authors", "auth_required",
    "tokens", "authentication", "authenticated", "authenticator", "authority",
    "auth_enabled", "auth_mode", "require_auth", "learning_rate", "output_dir", "base",
    "max_seq_length", "keyboard", "monkey", "compass", "pass_rate",
]


KEY_SPELLINGS = [
    "api_key", "api-key", "API-KEY", "Api-Key", "apiKey", "ApiKey", "apikey", "x-api-key",
    "X-Api-Key", "client.api.key", "hf_token", "hf-token", "hf.token", "HF-TOKEN",
    "access-token", "accessToken", "x-auth-token", "set-cookie", "private-key", "PASSWORD",
    "db-passwd", "client-secret",
]
# (template, the text kept in front of the value): separator, quoting and spacing variants
KEY_FORMS = [
    ("{k}={v}", "{k}="),
    ("{k} = {v}", "{k} = "),
    ("{k}: {v}", "{k}: "),
    ("{k}:{v}", "{k}:"),
    ("{k}:   {v}", "{k}:   "),
    ('{k}="{v}"', '{k}="'),
    ("{k}='{v}'", "{k}='"),
    ('"{k}": "{v}"', '"{k}": "'),
    ("{{'{k}': '{v}'}}", "{{'{k}': '"),
    ('{{\\"{k}\\": \\"{v}\\"}}', '{{\\"{k}\\": \\"'),
    ("export {k}={v}", "export {k}="),
    ("  {k}: {v}  ", "  {k}: "),
]
PLAIN_VALUE = "plainvalue-77"


@pytest.mark.parametrize("key", KEY_SPELLINGS)
@pytest.mark.parametrize("form, keeps", KEY_FORMS)
def test_every_spelling_of_a_credential_key_is_redacted(key, form, keeps):
    text = form.format(k=key, v=PLAIN_VALUE)
    report = RedactionReport()
    out = redact_text(text, report)
    assert PLAIN_VALUE not in out, out
    assert keeps.format(k=key) in out, out
    assert report.count == 1
    assert redact_text(out) == out


@pytest.mark.parametrize("name", SECRET_KEYS)
def test_secret_key_names_are_recognised(name):
    assert is_secret_key(name), name


@pytest.mark.parametrize("name", SECRET_KEYS)
def test_a_secret_key_name_is_enough_for_the_text_pass(name):
    # the cheap pre-filter in front of the assignment pass must never skip a flagged name
    assert PLAIN_VALUE not in redact_text(f"{name}={PLAIN_VALUE}"), name


@pytest.mark.parametrize("name", HARMLESS_KEYS)
def test_harmless_key_names_are_not_secret(name):
    assert not is_secret_key(name), name


@pytest.mark.parametrize("name", SECRET_KEYS)
def test_a_string_under_a_secret_key_is_replaced(name):
    redacted, report = redact_config({name: "plain-value-9"})
    assert redacted == {name: REDACTED}
    assert report.count == 1 and report.paths == [name]


@pytest.mark.parametrize("name", HARMLESS_KEYS)
def test_a_string_under_a_harmless_key_is_kept(name):
    redacted, report = redact_config({name: "plain-value-9"})
    assert redacted == {name: "plain-value-9"}
    assert report.count == 0 and report.paths == []


def test_nested_config_is_redacted_by_key_name_and_by_text():
    config = {
        "base": "Qwen/Qwen2.5-0.5B",
        "data": {
            "train": f"https://user:pw0rd-1@data.example/train.jsonl?api_key={URL_SECRET}",
            "hf_token": "plainvalue-1",
            "sources": [{"path": "a.jsonl", "api_key": "plainvalue-2"}, {"path": "b.jsonl"}],
        },
        "training": {
            "lr": 2e-5, "max_new_tokens": 512, "tokenizer": "Qwen/Qwen2.5-0.5B",
            "use_auth_token": "plainvalue-3", "eos_token": "<|im_end|>",
        },
        "env": {"WANDB_API_KEY": "plainvalue-4", "auth_required": True, "session_timeout": 30},
    }
    original = copy.deepcopy(config)
    redacted, report = redact_config(config)
    assert config == original  # the input is never modified
    assert redacted["data"]["hf_token"] == REDACTED
    assert redacted["data"]["sources"][0] == {"path": "a.jsonl", "api_key": REDACTED}
    assert redacted["data"]["sources"][1] == {"path": "b.jsonl"}
    assert redacted["data"]["train"] == (
        f"https://{REDACTED}@data.example/train.jsonl?api_key={REDACTED}"
    )
    assert redacted["training"] == {
        "lr": 2e-5, "max_new_tokens": 512, "tokenizer": "Qwen/Qwen2.5-0.5B",
        "use_auth_token": REDACTED, "eos_token": "<|im_end|>",
    }
    assert redacted["env"] == {
        "WANDB_API_KEY": REDACTED, "auth_required": True, "session_timeout": 30,
    }
    assert report.count == 6
    assert report.paths == [
        "data.train", "data.hf_token", "data.sources[0].api_key", "training.use_auth_token",
        "env.WANDB_API_KEY",
    ]
    assert not any("plainvalue" in path or URL_SECRET in path for path in report.paths)


@pytest.mark.usefixtures("perf_watchdog")
def test_many_redacted_values_in_one_config_are_recorded_in_linear_time():
    count = 40_000
    config = {"credentials": [f"value-{index}" for index in range(count)] + ["value-0"]}
    start = time.perf_counter()
    redacted, report = redact_config(config)
    elapsed = time.perf_counter() - start
    assert elapsed < 5, elapsed  # a list scan per path is quadratic: minutes at this size
    assert report.count == count + 1
    assert len(report.paths) == count + 1
    assert report.paths[0] == "credentials[0]" and report.paths[-1] == f"credentials[{count}]"
    assert not _leaked(json.dumps(redacted), "value-0", f"value-{count - 1}")


def test_non_string_values_are_never_touched():
    config = {
        "api_key": 12345, "token": None, "password": False, "secret": 1.5, "auth_required": True,
        "session": [1, 2, 3], "limits": (1, 2),
    }
    redacted, report = redact_config(config)
    assert redacted == config
    assert report.count == 0
    assert isinstance(redacted["limits"], tuple)


def test_everything_below_a_secret_named_dict_or_list_is_replaced():
    config = {"credentials": {"user": "alice", "retries": 3, "tags": ["x", "y"]}}
    redacted, report = redact_config(config)
    assert redacted == {"credentials": {"user": REDACTED, "retries": 3, "tags": [REDACTED] * 2}}
    assert report.paths == ["credentials.user", "credentials.tags[0]", "credentials.tags[1]"]


@pytest.mark.parametrize("value", [
    "", "none", "None", "null", "true", "False", "yes", "off", "bearer", "${OPENAI_API_KEY}",
    "$HF_TOKEN", REDACTED,
])
def test_plainly_harmless_values_under_a_secret_key_stay_readable(value):
    redacted, report = redact_config({"api_key": value, "auth_scheme": value})
    assert redacted == {"api_key": value, "auth_scheme": value}
    assert report.count == 0


def test_redacting_a_config_twice_changes_nothing():
    config = {"a": {"api_key": "plain-1", "u": "https://x:y@h.example/p?token=abc12345"}}
    once, _ = redact_config(config)
    twice, report = redact_config(once)
    assert twice == once and report.count == 0


def test_a_training_config_without_credentials_comes_back_identical():
    config = {
        "base": "meta-llama/Llama-3.2-1B", "task": "sft", "backend": "transformers",
        "data": {"train": "./data/train.jsonl", "format": "alpaca", "max_length": 2048},
        "training": {
            "epochs": 3, "lr": 2e-5, "quantization": "4bit", "max_new_tokens": 256,
            "tokenizer": "meta-llama/Llama-3.2-1B", "new_special_tokens": ["<|tool|>"],
            "lora": {"r": 64, "alpha": 16, "target_modules": ["q_proj", "v_proj"]},
        },
        "output": "./output", "eval": {"judge_api_base": "http://localhost:11434/v1"},
    }
    redacted, report = redact_config(config)
    assert redacted == config and report.count == 0


# ---------------------------------------------------------------------------
# The crash bundle
# ---------------------------------------------------------------------------

BUNDLE_SECRETS = [
    URL_SECRET, BEARER_VALUE, "pw0rd-1", "plain-key-1", "plainvalue-c1", "plainvalue-c2",
    "plainvalue-m1", HF,
]


def _raise_with_headers() -> RuntimeError:
    headers = {"Authorization": "Bearer " + BEARER_VALUE, "X-Api-Key": "plain-key-1"}
    try:
        raise RuntimeError(
            f"request failed url=https://u:pw0rd-1@api.example/v1?token={URL_SECRET} "
            f"headers={headers!r}"
        )
    except RuntimeError as exc:
        return exc


def test_error_message_and_traceback_are_redacted_with_counts():
    exc = _raise_with_headers()
    bundle = build_crash_bundle(error=exc)
    blob = json.dumps(bundle)
    for secret in BUNDLE_SECRETS[:4]:
        assert secret not in blob, secret
    message = bundle["error"]["message"]
    assert f"token={REDACTED}" in message
    assert f"://{REDACTED}@api.example/v1" in message
    assert "'Accept'" not in message and "'Authorization': 'Bearer " in message
    counts = bundle["redaction"]
    assert counts["error_message"] >= 4 and counts["traceback"] >= 4
    assert counts["values_redacted"] == sum(
        value for key, value in counts.items() if key != "values_redacted"
    )


def test_chained_exceptions_are_redacted_in_the_traceback():
    try:
        try:
            raise ValueError(f"inner {HF}")
        except ValueError as inner:
            raise RuntimeError("outer failure") from inner
    except RuntimeError as exc:
        bundle = build_crash_bundle(error=exc)
    trace = "".join(bundle["error"]["traceback"])
    assert "inner" in trace and HF not in trace
    assert bundle["redaction"]["traceback"] >= 1


def test_config_and_metrics_are_redacted_with_counts():
    config = {
        "base": "x", "hf_token": "plainvalue-c1",
        "data": {"train": "t.jsonl", "sources": [{"api_key": "plainvalue-c2"}]},
        "training": {"lr": 1e-4, "max_new_tokens": 128},
    }
    metrics = [{"step": 1, "loss": 0.5, "note": f"saw {HF}", "session_token": "plainvalue-m1"}]
    bundle = build_crash_bundle(error=RuntimeError("x"), config=config, metrics=metrics)
    blob = json.dumps(bundle)
    for secret in ("plainvalue-c1", "plainvalue-c2", "plainvalue-m1", HF):
        assert secret not in blob, secret
    assert bundle["config"]["training"] == {"lr": 1e-4, "max_new_tokens": 128}
    assert bundle["metrics_tail"][0]["loss"] == 0.5
    counts = bundle["redaction"]
    assert counts["config"] == 2 and counts["metrics_tail"] == 2
    assert counts["values_redacted"] == 4
    assert all(isinstance(value, int) for value in counts.values())


LEGACY_BARE_API_KEY = "api_key " + _alnum(24)


def test_shapes_the_bundle_redacted_before_are_still_redacted():
    # the bare-space ``api_key <value>`` form is only caught by the legacy layer
    assert redact_secrets(f"using {LEGACY_BARE_API_KEY} now") == f"using {REDACTED} now"
    bundle = build_crash_bundle(
        error=RuntimeError(f"failed with {LEGACY_BARE_API_KEY}"),
        config={"note": f"x {LEGACY_BARE_API_KEY}"},
        metrics=[{"msg": f"y {LEGACY_BARE_API_KEY}"}],
    )
    assert _alnum(24) not in json.dumps(bundle)
    assert bundle["redaction"]["config"] == 1 and bundle["redaction"]["metrics_tail"] == 1


def test_the_runpod_controller_pattern_tuple_still_exists_in_the_crash_module():
    from soup_cli.utils import crash

    assert len(crash._SECRET_PATTERNS) == 4


def test_environment_values_are_redacted_too(monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", f"https://user:{URL_SECRET}@mirror.example/hub")
    bundle = build_crash_bundle(error=RuntimeError("x"))
    assert bundle["environment"]["env"]["HF_ENDPOINT"] == f"https://{REDACTED}@mirror.example/hub"
    assert URL_SECRET not in json.dumps(bundle)
    assert bundle["redaction"]["environment"] == 1


def test_a_bundle_without_credentials_reports_zero_and_keeps_the_config():
    config = {"base": "x", "training": {"lr": 1e-4, "tokenizer": "x", "max_new_tokens": 8}}
    bundle = build_crash_bundle(error=RuntimeError("CUDA out of memory"), config=config)
    assert bundle["config"] == config
    assert bundle["error"]["kind"] == "oom"
    assert bundle["redaction"]["values_redacted"] == 0


def test_the_written_file_carries_no_credential(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exc = _raise_with_headers()
    bundle = build_crash_bundle(error=exc, config={"data": {"hf_token": "plainvalue-c1"}})
    path = write_crash_bundle(bundle)
    text = path.read_text(encoding="utf-8")
    for secret in BUNDLE_SECRETS:
        assert secret not in text, secret
    assert json.loads(text)["redaction"]["values_redacted"] >= 5


def test_a_truncated_bundle_still_says_how_much_was_redacted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {"blob": "x" * 1_200_000, "hf_token": "plainvalue-c1"}
    bundle = build_crash_bundle(error=RuntimeError("x"), config=config)
    text = write_crash_bundle(bundle).read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["_truncated"] is True
    assert data["redaction"]["config"] == 1
    assert not _leaked(text, "plainvalue-c1")


@pytest.mark.parametrize("row", ROWS, ids=ROW_IDS)
def test_every_spelling_is_gone_from_all_three_inputs(row):
    try:
        raise RuntimeError(row.positive)
    except RuntimeError as exc:
        bundle = build_crash_bundle(
            error=exc, config={"note": row.positive, "nested": [{"text": row.positive}]},
            metrics=[{"msg": row.positive}],
        )
    blob = json.dumps(bundle)
    assert row.secret not in blob
    assert row.secret.replace("\n", "\\n") not in blob
    assert bundle["redaction"]["values_redacted"] >= 4


# ---------------------------------------------------------------------------
# Linear time (model: tests/test_issue1367_quadratic_backtracking.py)
#
# Two things keep a regression here from hanging the run instead of failing it: a size
# ladder (a pattern that is super-linear shows up at 4-64 KiB, long before the 256 KiB and
# 1 MiB inputs), and assertions on big strings that never put the string into pytest's
# failure explanation (``assert secret not in big_text`` makes pytest diff the text with
# difflib, which on a 1 MiB repeated traceback takes far longer than the code under test).
# ---------------------------------------------------------------------------

LADDER = (4 << 10, 16 << 10, 64 << 10, 256 << 10)
LADDER_STEP_LIMIT = 10  # 4x more text: linear is about 4x the time, quadratic about 16x
LADDER_NOISE_FLOOR = 0.02  # seconds; timings below this are noise
STEP_FLOOR = 0.25  # seconds a step may always take, whatever its size
CASE_DEADLINE = 90  # seconds: past this the whole run is aborted with a traceback


@pytest.fixture
def perf_watchdog():
    """A performance case that hangs ends the run with a traceback instead of waiting.

    A regex that went exponential would sit inside ``redact_text`` where no assertion can
    see it; ``faulthandler`` aborts the process from a timer thread, so the case fails fast.
    """
    faulthandler.dump_traceback_later(CASE_DEADLINE, exit=True, file=sys.__stderr__)
    yield
    faulthandler.cancel_dump_traceback_later()


def _time_bound(size: int) -> float:
    """The #1367 test allows 0.5 s for 50 000 characters; the same rate, never tighter."""
    return 0.5 * size / 50_000


def _leaked(text: str, *secrets: str) -> list[str]:
    """The secrets still in ``text``: a short list, so a failing assert stays cheap."""
    return [secret for secret in secrets if secret in text]


def _repeat(unit: str):
    return lambda size: unit * (size // len(unit) + 1)


def _assert_scales_linearly(make_text, label: str) -> str:
    """Time ``redact_text`` up the ladder; fail at the first step that grows too fast."""
    timings: list[float] = []
    text = ""
    for size in LADDER:
        text = make_text(size)
        start = time.perf_counter()
        redact_text(text)
        timings.append(time.perf_counter() - start)
        # every step has its own bound: a case never climbs the whole ladder to fail
        assert timings[-1] < max(_time_bound(size), STEP_FLOOR), (label, size, timings[-1])
        if len(timings) > 1:
            limit = LADDER_STEP_LIMIT * max(timings[-2], LADDER_NOISE_FLOOR)
            assert timings[-1] <= limit, (label, size, [round(t, 4) for t in timings])
    assert timings[-1] < _time_bound(len(text)), (label, timings[-1])
    return text


def _frame_text(size: int) -> str:
    frame = (
        '  File "train.py", line 12, in step\n'
        f"    client.get(url='https://u:{URL_SECRET}@api.example/v1?token={URL_SECRET}', "
        f"headers={{'Authorization': 'Bearer {BEARER_VALUE}', 'hf': '{HF}'}})\n"
    )
    return frame * (size // len(frame) + 1)


@pytest.mark.usefixtures("perf_watchdog")
def test_the_helpers_that_keep_a_failing_perf_test_cheap_work():
    assert _leaked("a b c", "b", "x") == ["b"]
    assert len(_repeat("ab")(11)) >= 11
    assert _assert_scales_linearly(_repeat("hf_"), "hf prefixes").startswith("hf_")


@pytest.mark.usefixtures("perf_watchdog")
def test_one_mib_traceback_with_a_url_and_a_token_in_a_repr_is_fast():
    _assert_scales_linearly(_frame_text, "traceback frames")
    text = _frame_text(1 << 20)
    assert len(text) >= 1 << 20
    report = RedactionReport()
    start = time.perf_counter()
    out = redact_text(text, report)
    elapsed = time.perf_counter() - start
    assert elapsed < _time_bound(len(text)), elapsed
    assert not _leaked(out, URL_SECRET, BEARER_VALUE, HF)
    assert report.count >= 4 * (len(text) // len(_frame_text(1)))


@pytest.mark.usefixtures("perf_watchdog")
def test_a_one_mib_error_message_goes_through_the_bundle_in_time():
    message = ("step failed " + HF + " ") * (1 << 20 >> 6)
    start = time.perf_counter()
    bundle = build_crash_bundle(error=RuntimeError(message))
    elapsed = time.perf_counter() - start
    assert elapsed < _time_bound(len(message) * 2), elapsed
    assert not _leaked(json.dumps(bundle), HF)


ADVERSARIAL = {
    "hf prefixes": _repeat("hf_"),
    "sk prefixes": _repeat("sk-"),
    "sk-ant prefixes": _repeat("sk-ant-"),
    "github prefixes": _repeat("ghp_"),
    "github_pat run": _repeat("github_pat_"),
    "aws key prefixes": _repeat("AKIA"),
    "google prefixes": _repeat("AIza"),
    "slack prefixes": _repeat("xoxb-"),
    "jwt prefixes": _repeat("eyJ"),
    "jwt dots": _repeat("eyJabcdef."),
    "bearer words": _repeat("Bearer "),
    "bearer then spaces": lambda size: "bearer" + " " * size,
    "authorization words": _repeat("authorization: "),
    "authorization then spaces": lambda size: "authorization:" + " " * size,
    "pem begin without end": _repeat("-----BEGIN PRIVATE KEY-----"),
    "pem begin prefixes": _repeat("-----BEGIN "),
    "token assignments": _repeat("token="),
    "short assignments": _repeat("a="),
    "colon assignments": _repeat("a:"),
    "underscore run": _repeat("a_"),
    "key then spaces": lambda size: "api_key" + " " * size + "=",
    "quote run": lambda size: "token: " + "'" * size,
    "double quote run": lambda size: 'password="' + '"' * size,
    "url schemes": _repeat("https://"),
    "url userinfo": _repeat("http://a@"),
    "bare separators": _repeat("://"),
    "one long word": lambda size: "a" * size,
    "dotted words": _repeat("x."),
    "distinct key names": lambda size: " ".join(
        f"k{index}_token=v" for index in range(size // 12 + 1)
    ),
    "unclosed double quote values": _repeat('token="'),
    "unclosed single quote values": _repeat("token='"),
    "one unclosed quoted value": lambda size: 'token="' + "a b " * (size // 4),
    "one huge unquoted value": lambda size: "token=" + "a" * size,
    "hyphenated key names": _repeat("x-api-key: "),
    "colon userinfo": _repeat("a:b"),
    "colon then at": _repeat("a:b@"),
    "colon userinfo hosts": _repeat("a:b@c.d/"),
    "userinfo long host": lambda size: "a:b@" + "c." * (size // 2),
    "userinfo long password": lambda size: "a:" + "b" * size,
    "relative userinfo": _repeat("//a@"),
    "relative userinfo no at": lambda size: "//" + "a" * size,
    "long scheme runs": lambda size: "a+" * (size // 2) + "://",
    "long scheme then userinfo": _repeat("postgresql+psycopg2://u:p@"),
}


@pytest.mark.usefixtures("perf_watchdog")
@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_input_is_handled_in_linear_time(name):
    text = _assert_scales_linearly(ADVERSARIAL[name], name)
    assert len(text) >= LADDER[-1]
