"""``soup loop status`` says when the server no longer serves the adapter loop.yaml names (F11).

``loop.yaml`` keeps ``production_name`` across a restart of ``soup serve``; the restarted server
has no adapter loaded and answers with the base model, and ``soup loop status`` still said
``production manual-sft-1`` with nothing about it (review 5, F11). Now the ``server`` row asks
``<SOUP_LOOP_SERVE_ENDPOINT>/health`` which adapter is active and says so: serving it, NOT serving
it (and the way back), serving another, not reachable, or not checked.
"""

from __future__ import annotations

import http.server
import json
import re
import threading

import pytest
from typer.testing import CliRunner

import soup_cli.utils.loop_stages as ls
from soup_cli.commands.loop import app as loop_app
from soup_cli.utils.loop_state import LoopState, write_state
from tests._loopback_http import LoopbackHTTPServer

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _plain(text: str) -> str:
    return " ".join(_ANSI.sub("", text).replace("│", " ").split())


def _state(**kwargs) -> LoopState:
    return LoopState(
        served_model="org/base", eval_suite="e", baseline="b", base_model="org/base",
        status="running", **kwargs,
    )


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SOUP_LOOP_SERVE_ENDPOINT", raising=False)
    monkeypatch.setattr(ls, "_HEALTH_GETTER", None)
    return tmp_path


class TestTheNote:
    def _ask(self, monkeypatch, health, **state_fields):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        asked = []

        def getter(endpoint):
            asked.append(endpoint)
            return health

        monkeypatch.setattr(ls, "_HEALTH_GETTER", getter)
        return ls.server_adapter_note(_state(**state_fields)), asked

    def test_production_is_the_served_model_so_nothing_is_asked(self, cwd, monkeypatch):
        note, asked = self._ask(monkeypatch, {"active_adapter": None})

        assert note is None and asked == []

    def test_the_server_serves_it(self, cwd, monkeypatch):
        note, asked = self._ask(
            monkeypatch, {"status": "ok", "active_adapter": "manual-sft-1"},
            production_name="manual-sft-1",
        )

        assert note == "serving production manual-sft-1"
        assert asked == ["http://127.0.0.1:8000"]

    def test_a_restarted_server_does_not_and_the_note_says_so_and_how_to_go_on(
        self, cwd, monkeypatch
    ):
        note, _ = self._ask(
            monkeypatch, {"status": "ok", "active_adapter": None},
            production_name="manual-sft-1",
        )

        assert note == (
            "NOT serving production manual-sft-1: the server has no active adapter and answers "
            "with the base model (it was restarted, or the adapter was unloaded). Load and "
            "activate it again (POST /v1/adapters/load, then "
            "/v1/adapters/activate/manual-sft-1) or run soup loop rollback"
        )

    def test_another_adapter_is_named(self, cwd, monkeypatch):
        note, _ = self._ask(
            monkeypatch, {"active_adapter": "other"}, production_name="manual-sft-1"
        )

        assert note == "serving other, not production manual-sft-1"

    def test_an_unreachable_server_is_said_too(self, cwd, monkeypatch):
        note, _ = self._ask(monkeypatch, None, production_name="manual-sft-1")

        assert note == (
            "not reachable at http://127.0.0.1:8000: cannot tell whether it serves production "
            "manual-sft-1"
        )

    def test_no_endpoint_in_this_shell_is_said_not_guessed(self, cwd):
        note = ls.server_adapter_note(_state(production_name="manual-sft-1"))

        assert note == (
            "not checked: SOUP_LOOP_SERVE_ENDPOINT is not set here, so it is not known whether "
            "the server still serves production manual-sft-1"
        )

    def test_a_public_endpoint_is_refused_and_never_asked(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "https://example.com")
        asked = []
        monkeypatch.setattr(ls, "_HEALTH_GETTER", lambda endpoint: asked.append(endpoint))

        note = ls.server_adapter_note(_state(production_name="manual-sft-1"))

        assert note.startswith("not checked: SOUP_LOOP_SERVE_ENDPOINT")
        assert asked == []

    def test_a_hostile_active_name_is_cut_and_printable(self, cwd, monkeypatch):
        note, _ = self._ask(
            monkeypatch, {"active_adapter": "x\x1b]0;PWNED\x07" + "y" * 500},
            production_name="manual-sft-1",
        )

        assert "\x1b" not in note and "\x07" not in note and len(note) < 220


class _Health(http.server.BaseHTTPRequestHandler):
    body = b'{"status": "ok", "model": "m", "device": "cpu", "active_adapter": null}'
    status = 200

    def do_GET(self):  # noqa: N802
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.body)))
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, *args):  # silence
        pass


class TestAgainstARealSocket:
    """``_get_health`` itself: a real loopback server, the real capped GET."""

    @pytest.fixture
    def server(self, cwd, monkeypatch):
        handler = type("H", (_Health,), {})
        httpd = LoopbackHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", f"http://127.0.0.1:{httpd.server_port}")
        yield handler
        httpd.shutdown()
        httpd.server_close()

    def test_the_restarted_server_is_reported(self, server):
        note = ls.server_adapter_note(_state(production_name="manual-sft-1"))

        assert note.startswith("NOT serving production manual-sft-1")

    def test_the_server_that_serves_it_is_reported(self, server):
        server.body = b'{"status": "ok", "active_adapter": "manual-sft-1"}'

        assert ls.server_adapter_note(_state(production_name="manual-sft-1")) == (
            "serving production manual-sft-1"
        )

    def test_a_server_error_is_not_reachable(self, server):
        server.status = 500

        assert "not reachable" in ls.server_adapter_note(_state(production_name="manual-sft-1"))

    @pytest.mark.parametrize(
        "body", [b"not json", b"[1]", b"x" * (70 * 1024)], ids=["not-json", "array", "oversize"]
    )
    def test_an_answer_that_is_not_a_small_json_object_is_not_reachable(self, server, body):
        server.body = body

        assert "not reachable" in ls.server_adapter_note(_state(production_name="manual-sft-1"))

    def test_nothing_listening_is_not_reachable(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:9")

        assert "not reachable" in ls.server_adapter_note(_state(production_name="manual-sft-1"))


class TestInStatus:
    def _status(self, cwd, state):
        write_state(state, str(cwd / ".soup" / "loop.yaml"))
        result = CliRunner().invoke(loop_app, ["status"])
        assert result.exit_code == 0, (result.output, repr(result.exception))
        return _plain(result.output)

    def test_the_server_row_names_the_adapter_the_server_lost(self, cwd, monkeypatch):
        monkeypatch.setenv("SOUP_LOOP_SERVE_ENDPOINT", "http://127.0.0.1:8000")
        monkeypatch.setattr(ls, "_HEALTH_GETTER", lambda endpoint: {"active_adapter": None})

        text = self._status(cwd, _state(production_name="manual-sft-1"))

        assert "server NOT serving production manual-sft-1" in text
        assert "soup loop rollback" in text

    def test_a_loop_whose_production_is_the_served_model_has_no_server_row(self, cwd):
        assert " server " not in self._status(cwd, _state())

    def test_status_prints_the_row_even_without_an_endpoint(self, cwd):
        text = self._status(cwd, _state(production_name="manual-sft-1"))

        assert "server not checked: SOUP_LOOP_SERVE_ENDPOINT is not set here" in text


def test_the_capped_get_refuses_a_declared_oversize_reply(cwd, monkeypatch):
    from soup_cli.utils.http_response import ResponseTooLargeError, get_capped

    handler = type("H", (_Health,), {"body": json.dumps({"x": "y" * 2000}).encode()})
    httpd = LoopbackHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(ResponseTooLargeError):
            get_capped(
                f"http://127.0.0.1:{httpd.server_port}/health",
                max_bytes=100, endpoint_label="test /health", trust_env=False,
            )
    finally:
        httpd.shutdown()
        httpd.server_close()
