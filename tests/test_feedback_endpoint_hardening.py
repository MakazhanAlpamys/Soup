"""``POST /v1/feedback`` hardening: own origin only, issued ids only, bounded log, private files.

The route stays token-free on a loopback bind (a local UI posts to it), so what it accepts is
bounded instead:

* a request that carries an ``Origin`` is refused unless that origin is the server's own;
* a ``response_id`` the process did not issue is a 404 ``unknown response_id``;
* the feedback log stops at a size cap (507, nothing appended);
* every string field has its own cap, below the 64 KiB body cap;
* the log files are created owner-only and the secret redaction knows more token shapes.
"""

from __future__ import annotations

import json
import os
import uuid
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import soup_cli.commands.serve as serve  # noqa: E402
from soup_cli.monitoring import trace_logger  # noqa: E402
from soup_cli.monitoring.feedback_log import (  # noqa: E402
    MAX_BODY_BYTES,
    MAX_COMMENT_CHARS,
    MAX_EDITED_TEXT_CHARS,
    MAX_LOG_BYTES,
    MAX_USER_REF_CHARS,
    FeedbackLogFullError,
    FeedbackLogWriter,
    IssuedResponseIds,
    build_record,
    validate_feedback,
)
from soup_cli.monitoring.trace_logger import TraceLogWriter  # noqa: E402

MODEL_NAME = "test-model"
OWN_ORIGIN_HOST = "127.0.0.1"


def _fake_generate(model, tokenizer, messages, **kwargs):
    return "hello there", 5, 2, "stop"


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _client(cwd, monkeypatch, *, max_bytes=None, **kwargs) -> TestClient:
    monkeypatch.setattr(serve, "_generate_response", _fake_generate)
    writer_kwargs = {} if max_bytes is None else {"max_bytes": max_bytes}
    app = serve._create_app(
        model_obj=MagicMock(),
        tokenizer=MagicMock(),
        device="cpu",
        model_name=MODEL_NAME,
        max_tokens_default=8,
        trace_log_writer=TraceLogWriter(str(cwd / "trace.jsonl")),
        feedback_log_writer=FeedbackLogWriter(str(cwd / "feedback.jsonl"), **writer_kwargs),
        host=OWN_ORIGIN_HOST,
        **kwargs,
    )
    return TestClient(app, base_url=f"http://{OWN_ORIGIN_HOST}")


def _answer(client) -> str:
    """The response id of a real answer from this server."""
    reply = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert reply.status_code == 200, reply.text
    return reply.json()["response_id"]


def _post(client, response_id, headers=None, **fields):
    body = {"response_id": response_id, "kind": "thumb", "value": 1, **fields}
    return client.post("/v1/feedback", json=body, headers=headers)


def _lines(cwd) -> list[dict]:
    path = cwd / "feedback.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------------------------------------------------------------------------
# A normal post still works
# ---------------------------------------------------------------------------


class TestNormalPost:
    def test_a_post_for_an_answer_this_server_gave_is_stored(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)

        resp = _post(client, response_id)

        assert resp.status_code == 200, resp.text
        (line,) = _lines(cwd)
        assert line["response_id"] == response_id and line["kind"] == "thumb"

    def test_a_streamed_answer_id_is_known_too(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        reply = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        response_id = reply.headers["X-Response-Id"]

        assert _post(client, response_id).status_code == 200

    def test_the_route_stays_open_on_loopback_without_a_token(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        assert _post(client, _answer(client)).status_code == 200


# ---------------------------------------------------------------------------
# F1(a) Origin: the server's own, or none
# ---------------------------------------------------------------------------


class TestOwnOriginOnly:
    @pytest.mark.parametrize(
        "origin",
        [
            "http://evil.example",
            "http://localhost:3000",  # another local port: still another origin
            "http://localhost",  # same host family, different host name than Host
            "http://127.0.0.1:9999",
            "null",
            "http://127.0.0.1/extra",
        ],
    )
    def test_a_foreign_origin_is_refused_and_nothing_is_stored(self, cwd, monkeypatch, origin):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)

        resp = _post(client, response_id, headers={"Origin": origin})

        assert resp.status_code == 403, (origin, resp.text)
        assert resp.json()["detail"].startswith("Origin not allowed")
        assert _lines(cwd) == []

    def test_another_local_port_is_refused_by_the_own_origin_rule(self, cwd, monkeypatch):
        # The generic loopback check lets a page on any localhost port through; this route
        # is stricter, and says why.
        client = _client(cwd, monkeypatch)

        resp = _post(client, _answer(client), headers={"Origin": "http://localhost:3000"})

        assert resp.status_code == 403
        assert "own origin" in resp.json()["detail"]

    def test_the_servers_own_origin_is_accepted(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)

        resp = _post(client, response_id, headers={"Origin": f"http://{OWN_ORIGIN_HOST}"})

        assert resp.status_code == 200, resp.text

    def test_the_origin_is_compared_with_the_port_the_client_used(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)

        own = _post(
            client, response_id,
            headers={"Host": "127.0.0.1:8123", "Origin": "http://127.0.0.1:8123"},
        )
        other_port = _post(
            client, response_id,
            headers={"Host": "127.0.0.1:8123", "Origin": "http://127.0.0.1:8124"},
        )

        assert (own.status_code, other_port.status_code) == (200, 403)

    def test_no_origin_header_is_a_non_browser_client_and_is_allowed(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        assert _post(client, _answer(client)).status_code == 200

    def test_a_foreign_origin_is_refused_before_the_token_is_judged(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token="t0ken-value")
        response_id = _answer(client)

        resp = _post(client, response_id, headers={"Origin": "http://evil.example"})

        assert resp.status_code == 403


# ---------------------------------------------------------------------------
# F1(b) Only ids this server issued
# ---------------------------------------------------------------------------


class TestIssuedIdsOnly:
    def test_an_id_the_server_never_issued_is_a_404(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post(client, str(uuid.uuid4()))

        assert resp.status_code == 404
        assert resp.json() == {"detail": "unknown response_id"}
        assert _lines(cwd) == []

    def test_a_malformed_id_is_still_a_400_naming_the_field(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)

        resp = _post(client, "not-a-uuid")

        assert resp.status_code == 400
        assert resp.json() == {"detail": "Invalid request: response_id"}

    def test_the_loops_own_internal_check_does_not_issue_an_id(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, auth_token="t0ken-value")
        reply = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer t0ken-value", "X-Soup-Internal": "loop-check"},
        )
        assert reply.status_code == 200, reply.text

        resp = _post(
            client, reply.json()["response_id"],
            headers={"Authorization": "Bearer t0ken-value"},
        )

        assert resp.status_code == 404

    def test_a_failed_generation_issues_no_id(self, cwd, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("generation failed")

        monkeypatch.setattr(serve, "_generate_response", boom)
        client = _client(cwd, monkeypatch)
        monkeypatch.setattr(serve, "_generate_response", boom)
        client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})

        assert len(client.app.state.issued_response_ids) == 0

    def test_the_memory_is_bounded_and_forgets_the_oldest_first(self):
        ids = IssuedResponseIds(capacity=3)
        for name in ("a", "b", "c", "d", "e"):
            ids.add(name)

        assert len(ids) == 3
        assert "a" not in ids and "b" not in ids
        assert all(name in ids for name in ("c", "d", "e"))

    def test_the_default_capacity_is_a_named_constant(self):
        from soup_cli.monitoring.feedback_log import MAX_ISSUED_RESPONSE_IDS

        assert MAX_ISSUED_RESPONSE_IDS == 100_000

    def test_a_bad_capacity_is_refused(self):
        for bad in (0, -1, True, "3"):
            with pytest.raises(ValueError, match="capacity"):
                IssuedResponseIds(capacity=bad)

    def test_the_log_is_not_read_per_request(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)
        (cwd / "trace.jsonl").unlink()  # the trace log is gone; the in-memory ids remain

        assert _post(client, response_id).status_code == 200


# ---------------------------------------------------------------------------
# F1(c) A size cap on the feedback log
# ---------------------------------------------------------------------------


class TestLogSizeCap:
    def test_the_default_cap_is_256_mib(self):
        assert MAX_LOG_BYTES == 256 * 1024 * 1024

    def test_a_log_at_the_cap_answers_507_naming_the_file_and_appends_nothing(
        self, cwd, monkeypatch
    ):
        client = _client(cwd, monkeypatch, max_bytes=200)
        response_id = _answer(client)
        log = cwd / "feedback.jsonl"
        log.write_bytes(b"x" * 200)

        resp = _post(client, response_id)

        assert resp.status_code == 507, resp.text
        assert "feedback.jsonl" in resp.json()["detail"]
        assert log.read_bytes() == b"x" * 200

    def test_a_log_just_below_the_cap_still_takes_a_post(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch, max_bytes=200)
        response_id = _answer(client)
        (cwd / "feedback.jsonl").write_bytes(b"x" * 199)

        assert _post(client, response_id).status_code == 200

    def test_the_writer_raises_a_named_error_at_the_cap(self, cwd):
        writer = FeedbackLogWriter(str(cwd / "feedback.jsonl"), max_bytes=10)
        record = build_record(validate_feedback(
            {"response_id": str(uuid.uuid4()), "kind": "thumb", "value": 1}
        ))
        writer.append(record)  # the first line may cross the cap; the next one is refused

        with pytest.raises(FeedbackLogFullError) as excinfo:
            writer.append(record)

        assert excinfo.value.path == writer.path
        assert len((cwd / "feedback.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    @pytest.mark.parametrize("bad", [0, -5, True, "big", 1.5])
    def test_a_bad_cap_is_refused(self, cwd, bad):
        with pytest.raises(ValueError, match="max_bytes"):
            FeedbackLogWriter(str(cwd / "feedback.jsonl"), max_bytes=bad)


# ---------------------------------------------------------------------------
# F1(d) Per-field caps below the body cap
# ---------------------------------------------------------------------------


class TestFieldCaps:
    def test_every_field_cap_is_below_the_body_cap(self):
        for cap in (MAX_EDITED_TEXT_CHARS, MAX_COMMENT_CHARS, MAX_USER_REF_CHARS):
            assert cap < MAX_BODY_BYTES

    def _fields(self, **extra) -> dict:
        return {"response_id": str(uuid.uuid4()), **extra}

    def test_edited_text_at_the_cap_is_accepted_one_more_is_refused_by_name(self):
        ok = validate_feedback(
            self._fields(kind="edit", edited_text="e" * MAX_EDITED_TEXT_CHARS)
        )
        assert len(ok["edited_text"]) == MAX_EDITED_TEXT_CHARS

        with pytest.raises(Exception) as excinfo:
            validate_feedback(
                self._fields(kind="edit", edited_text="e" * (MAX_EDITED_TEXT_CHARS + 1))
            )
        assert excinfo.value.field == "edited_text"

    def test_comment_and_user_ref_are_refused_by_name(self):
        with pytest.raises(Exception) as comment:
            validate_feedback(
                self._fields(kind="comment", comment="c" * (MAX_COMMENT_CHARS + 1))
            )
        with pytest.raises(Exception) as user_ref:
            validate_feedback(
                self._fields(kind="thumb", value=1, user_ref="u" * (MAX_USER_REF_CHARS + 1))
            )
        assert comment.value.field == "comment"
        assert user_ref.value.field == "user_ref"

    def test_the_route_answers_400_naming_edited_text(self, cwd, monkeypatch):
        client = _client(cwd, monkeypatch)
        response_id = _answer(client)

        resp = client.post("/v1/feedback", json={
            "response_id": response_id, "kind": "edit",
            "edited_text": "e" * (MAX_EDITED_TEXT_CHARS + 1),
        })

        assert resp.status_code == 400
        assert resp.json() == {"detail": "Invalid request: edited_text"}
        assert _lines(cwd) == []


# ---------------------------------------------------------------------------
# F3 Redaction of more token shapes, user_ref and conversation_id; owner-only files
# ---------------------------------------------------------------------------

# Built from parts so no secret scanner mistakes this file for a leak.
SECRET_SHAPES = {
    "github classic": "gh" + "p_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "github oauth": "gh" + "o_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8",
    "github fine-grained": "github" + "_pat_" + "11ABCDEFG0abcdefghijkl_mnopqrstuvwxyz0123",
    "aws access key": "AK" + "IA" + "ABCDEFGHIJKLMNOP",
    "aws temporary key": "AS" + "IA" + "ABCDEFGHIJKLMNOP",
    "slack bot": "xo" + "xb-" + "1234567890-abcdefghijkl",
    "slack user": "xo" + "xp-" + "1234567890-abcdefghijkl",
    "jwt": "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0NTY3ODkwIn0." + "abcdefghijk123",
    "pem block": (
        "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA1234567890\nabcdef==\n"
        "-----END " + "RSA PRIVATE KEY-----"
    ),
    "pem without footer": "-----BEGIN " + "PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC",
    "api_key assignment": "api" + "_key=" + "abcd1234efgh5678",
    "api-key header style": "api" + "-key: " + "abcd1234efgh5678",
    "apikey assignment": "api" + "key=" + "abcd1234efgh5678",
    "password assignment": "pass" + "word: " + "hunter2hunter2",
    "secret assignment": "sec" + "ret=" + "s3cr3tvalue99",
}


@pytest.mark.parametrize("name", sorted(SECRET_SHAPES))
class TestRedaction:
    def test_each_shape_is_redacted_in_free_text(self, name):
        secret = SECRET_SHAPES[name]
        key = secret.split("=")[-1].split(": ")[-1].splitlines()[-1]

        redacted = trace_logger.redact_value(f"before {secret} after")

        assert "<redacted>" in redacted, name
        assert key not in redacted, (name, redacted)
        assert redacted.startswith("before ")

    def test_it_is_redacted_in_every_free_text_field_of_a_feedback_record(self, name):
        secret = SECRET_SHAPES[name]
        key = secret.split("=")[-1].split(": ")[-1].splitlines()[-1]

        record = build_record(validate_feedback({
            "response_id": str(uuid.uuid4()), "kind": "edit",
            "edited_text": f"x {secret} y", "comment": f"x {secret} y",
            "user_ref": f"x {secret} y",
        }))

        for field in ("edited_text", "comment", "user_ref"):
            assert key not in record[field], (name, field)


class TestRedactionKeepsOrdinaryText:
    @pytest.mark.parametrize(
        "text",
        [
            "the tokenizer: abcdefgh",
            "token: 5",
            "a secret garden",
            "I wrote a password manager",
            "AKIA is a prefix",
            "eyJ alone is nothing",
        ],
    )
    def test_ordinary_sentences_are_not_touched(self, text):
        assert trace_logger.redact_value(text) == text

    def test_a_header_without_a_footer_cannot_make_the_scan_slow(self):
        import time

        text = ("-----BEGIN " + "PRIVATE KEY-----\n") * 400 + "z" * 20000
        started = time.perf_counter()
        trace_logger.redact_value(text)
        assert time.perf_counter() - started < 5.0


class TestConversationIdRedaction:
    def test_a_secret_in_a_conversation_id_is_redacted_in_the_trace_record(
        self, cwd, monkeypatch
    ):
        client = _client(cwd, monkeypatch)
        secret = SECRET_SHAPES["github classic"]

        client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}],
                  "conversation_id": f"user-{secret}"},
        )

        raw = (cwd / "trace.jsonl").read_text(encoding="utf-8")
        assert secret not in raw and "<redacted>" in raw


class TestOwnerOnlyLogFiles:
    def test_the_helper_creates_a_log_with_mode_0600(self, cwd, monkeypatch):
        seen = []
        real_open = os.open

        def spy(path, flags, mode=0o777, **kwargs):
            seen.append(mode)
            return real_open(path, flags, mode, **kwargs)

        monkeypatch.setattr(os, "open", spy)

        with trace_logger.open_private_append(cwd / "created.log") as handle:
            handle.write(b"x")

        assert seen == [0o600]

    def test_both_writers_go_through_the_helper(self, cwd, monkeypatch):
        calls = []
        real = trace_logger.open_private_append
        monkeypatch.setattr(
            trace_logger, "open_private_append", lambda path: calls.append(str(path)) or real(path)
        )
        import soup_cli.monitoring.feedback_log as feedback_log

        monkeypatch.setattr(feedback_log, "open_private_append", trace_logger.open_private_append)

        TraceLogWriter(str(cwd / "t.jsonl")).record_conversation(
            response_id=str(uuid.uuid4()), model="m",
            messages=[{"role": "user", "content": "hi"}], response="ok",
        )
        FeedbackLogWriter(str(cwd / "f.jsonl")).append(build_record(validate_feedback(
            {"response_id": str(uuid.uuid4()), "kind": "thumb", "value": 1}
        )))

        assert any(path.endswith("t.jsonl") for path in calls)
        assert any(path.endswith("f.jsonl") for path in calls)

    @pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits")
    def test_a_new_feedback_log_is_not_readable_by_others(self, cwd):
        FeedbackLogWriter(str(cwd / "f.jsonl")).append(build_record(validate_feedback(
            {"response_id": str(uuid.uuid4()), "kind": "thumb", "value": 1}
        )))

        assert (cwd / "f.jsonl").stat().st_mode & 0o077 == 0
