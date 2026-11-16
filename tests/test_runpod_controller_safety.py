"""Safety properties of the RunPod controller that a second read asked to pin.

What counts as proof that a billed Pod is gone, how a rate limit is waited out, which
deadlines end each wait, how far the Pod list is walked, which spellings of the API key
are removed from provider text, and that a redirect never carries the key anywhere.

The bodies are shaped like RunPod's published v2 examples (documentation of 2026-11-15);
nothing here has been exercised against the live service. Every wait runs on a virtual
clock that only the controller's own sleeps advance, so a deadline that is too long or
missing fails in milliseconds instead of waiting in real time.
"""

import base64
import http.server
import io
import json
import threading
import types
import urllib.error
import urllib.parse
from pathlib import Path

import pytest

from tests._loopback_http import LoopbackHTTPServer
from tests.test_runpod_cloud import _controller as _plain_controller
from tests.test_runpod_rest_v2 import (
    DOC_404,
    DOC_422,
    DOC_429,
    DOC_POD_CREATED,
    DOC_POD_RUNNING,
    KEY,
    PROBLEM,
    _http_error,
    _serve,
)

POD = "7h9k2m4n6p"
_ROOT = Path(__file__).resolve().parents[1]


class _Clock:
    """Monotonic time that only sleeps advance, plus a tiny step per read so that no loop
    can ever stand still."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        self.now += 0.001
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def _controller():
    ns = _plain_controller()
    clock = _Clock()
    ns["time"] = types.SimpleNamespace(monotonic=clock.monotonic, sleep=clock.sleep)
    return ns, clock


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-real-private-key", encoding="utf-8")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    return key


def _limited(url, retry_after=None):
    """A 429 as RunPod documents it: problem+json plus Retry-After in seconds."""
    headers = {"Content-Type": PROBLEM}
    if retry_after is not None:
        headers["Retry-After"] = retry_after
    return urllib.error.HTTPError(
        url, 429, "Too Many Requests", headers, io.BytesIO(json.dumps(DOC_429).encode())
    )


def _page(ids=(), cursor=None):
    return {
        "pods": [{"id": pod_id, "name": "x"} for pod_id in ids],
        "pagination": {"nextCursor": cursor, "hasNextPage": cursor is not None},
    }


# ----------------------------------------------------------------------------- Retry-After --
@pytest.mark.parametrize("value,expected", [
    ("12", 12), ("0", 0), (" 7 ", 7), ("60", 60), ("61", 60), ("600", 60), ("999999", 60),
    ("1000000", None), ("Wed, 21 Oct 2026 07:28:00 GMT", None), ("-5", None), ("1.5", None),
    ("", None), ("٣", None), ("12 seconds", None), ("1e2", None),
])
def test_retry_after_is_an_integer_capped_at_sixty_and_nothing_else(monkeypatch, value, expected):
    ns, _ = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(_limited(url, value)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods/" + POD)
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == expected


def test_a_429_without_the_header_has_no_retry_after_and_keeps_its_text(monkeypatch):
    ns, _ = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(_limited(url)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods/" + POD)
    assert excinfo.value.retry_after is None
    assert str(excinfo.value).startswith("RunPod API GET failed: HTTP 429 (Too Many Requests;")


@pytest.mark.parametrize("retry_after,slept", [("30", 30), ("600", 60), ("junk", 5), (None, 5),
                                                ("0", 5)])
def test_a_429_while_waiting_sleeps_and_goes_on(monkeypatch, retry_after, slept):
    ns, clock = _controller()
    calls = {"n": 0}

    def handler(method, path, body, url):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _limited(url, retry_after)
        return DOC_POD_RUNNING

    _serve(ns, monkeypatch, handler)
    assert ns["_wait_for_ssh"](KEY, POD) == ("195.26.233.3", 34446)
    assert clock.sleeps == [slept, slept]


def test_a_429_that_never_ends_still_ends_at_the_fifteen_minute_deadline(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(_limited(url, "7")))
    with pytest.raises(RuntimeError, match="did not expose SSH within 15 minutes"):
        ns["_wait_for_ssh"](KEY, POD)
    assert max(clock.sleeps) <= 7
    # The last sleep is cut to the time left: 129 whole sleeps of 7 would end at 903.
    assert 899 <= clock.now <= 900.5


def test_the_poll_deadline_is_fifteen_minutes(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda *args: DOC_POD_CREATED)
    with pytest.raises(RuntimeError, match="did not expose SSH within 15 minutes"):
        ns["_wait_for_ssh"](KEY, POD)
    assert 899 <= clock.now <= 901


def test_another_api_error_while_waiting_still_ends_the_wait_at_once(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 500, {"title": "Oops", "status": 500, "detail": "later"})))
    with pytest.raises(RuntimeError, match="HTTP 500"):
        ns["_wait_for_ssh"](KEY, POD)
    assert clock.sleeps == []


@pytest.mark.parametrize("stage", ["delete", "get"])
def test_a_429_during_cleanup_is_waited_out_and_cleanup_goes_on(monkeypatch, stage):
    ns, clock = _controller()
    state = {"limited": False, "deleted": False}

    def handler(method, path, body, url):
        if method == stage.upper() and not state["limited"]:
            state["limited"] = True
            raise _limited(url, "7")
        if method == "DELETE":
            state["deleted"] = True
            return None
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    assert "accepted the DELETE" in ns["_terminate"](KEY, POD)
    assert clock.sleeps == [7]


def test_a_429_that_never_ends_still_ends_cleanup_at_five_minutes(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(_limited(url, "7")))
    with pytest.raises(RuntimeError, match="not confirmed within five minutes"):
        ns["_terminate"](KEY, POD)
    assert max(clock.sleeps) <= 7
    # 43 whole sleeps of 7 would end at 301: the last one is cut to the time left.
    assert 299 <= clock.now <= 300.5


# ---------------------------------------------------------- what proves a Pod is gone ------
def test_a_204_then_a_404_is_confirmed_by_the_delete_without_walking_the_list(monkeypatch):
    ns, _ = _controller()
    log = []

    def handler(method, path, body, url):
        log.append((method, path))
        if method == "DELETE":
            return None
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    evidence = ns["_terminate"](KEY, POD)
    assert evidence == "RunPod accepted the DELETE and no longer finds the Pod"
    assert log == [("DELETE", "/pods/" + POD), ("GET", "/pods/" + POD)]


def test_a_404_to_the_delete_needs_the_pod_to_be_absent_from_the_list(monkeypatch):
    ns, _ = _controller()
    log = []

    def handler(method, path, body, url):
        log.append((method, path))
        if path.startswith("/pods?"):
            return _page(["other"])
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    evidence = ns["_terminate"](KEY, POD)
    assert evidence == "the Pod is not in RunPod's Pod list"
    assert log == [("DELETE", "/pods/" + POD), ("GET", "/pods/" + POD),
                   ("GET", "/pods?limit=100")]


def test_a_404_to_the_delete_is_not_proof_while_the_pod_is_still_listed(monkeypatch):
    ns, clock = _controller()
    deletes = []

    def handler(method, path, body, url):
        if method == "DELETE":
            deletes.append(path)
        if path.startswith("/pods?"):
            return _page([POD])
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="not confirmed within five minutes"):
        ns["_terminate"](KEY, POD)
    assert len(deletes) > 10
    assert 299 <= clock.now <= 301


def test_the_pod_is_found_on_a_later_page_of_the_list(monkeypatch):
    ns, clock = _controller()

    def handler(method, path, body, url):
        if path == "/pods?limit=100":
            return _page(["a"], cursor="c1")
        if path == "/pods?limit=100&cursor=c1":
            return _page([POD])
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="not confirmed within five minutes"):
        ns["_terminate"](KEY, POD)
    assert 299 <= clock.now <= 301


@pytest.mark.parametrize("failure", ["500", "shape", "transport"])
def test_a_list_that_cannot_be_read_is_not_proof_of_absence(monkeypatch, failure):
    ns, clock = _controller()

    def handler(method, path, body, url):
        if path.startswith("/pods?"):
            if failure == "500":
                raise _http_error(url, 500, {"title": "x", "status": 500, "detail": "y"})
            if failure == "transport":
                raise TimeoutError("reset")
            return [{"id": "other"}]
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="not confirmed within five minutes"):
        ns["_terminate"](KEY, POD)
    assert 299 <= clock.now <= 301


def test_a_terminated_record_is_proof_even_when_the_delete_answered_404(monkeypatch):
    ns, _ = _controller()
    log = []

    def handler(method, path, body, url):
        log.append((method, path))
        if method == "DELETE":
            raise _http_error(url, 404, DOC_404)
        return {**DOC_POD_RUNNING, "status": "TERMINATED"}

    _serve(ns, monkeypatch, handler)
    assert ns["_terminate"](KEY, POD) == "RunPod reports the Pod as TERMINATED"
    assert all(not path.startswith("/pods?") for _, path in log)


def test_a_204_remembered_from_an_earlier_round_still_counts(monkeypatch):
    ns, _ = _controller()
    state = {"round": 0}

    def handler(method, path, body, url):
        if method == "DELETE":
            state["round"] += 1
            if state["round"] == 1:
                return None
            raise _http_error(url, 404, DOC_404)
        if state["round"] == 1:
            return {**DOC_POD_RUNNING, "status": "RUNNING"}
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    assert "accepted the DELETE" in ns["_terminate"](KEY, POD)


def test_the_cleanup_deadline_is_five_minutes(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: DOC_POD_RUNNING if m == "GET" else None)
    with pytest.raises(RuntimeError, match="not confirmed within five minutes"):
        ns["_terminate"](KEY, POD)
    assert 299 <= clock.now <= 301


def test_a_delete_404_is_a_pod_not_found_error_not_a_silent_none(monkeypatch):
    ns, _ = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 404, DOC_404)))
    assert ns["_request_json"](KEY, "/pods/" + POD) is None
    with pytest.raises(ns["_PodNotFoundError"]):
        ns["_request_json"](KEY, "/pods/" + POD, method="DELETE")


def test_main_says_which_evidence_proved_the_deletion(credentials, monkeypatch):
    ns, _ = _controller()
    notices = []
    state = {"deleted": False}

    def handler(method, path, body, url):
        if method == "POST":
            return DOC_POD_CREATED
        if method == "DELETE":
            state["deleted"] = True
            return None
        if state["deleted"]:
            raise _http_error(url, 404, DOC_404)
        return {**DOC_POD_CREATED, "status": "ERROR"}

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert "RunPod termination evidence: RunPod accepted the DELETE and no longer finds the Pod." \
        in notices
    assert notices.index("RunPod termination verified.") == len(notices) - 1


# ------------------------------------------------- a failed wait still ends in the DELETE ----
@pytest.mark.parametrize("how", ["error-status", "429-past-deadline", "never-ready"])
def test_main_deletes_the_pod_when_the_wait_itself_fails(credentials, monkeypatch, how):
    ns, clock = _controller()
    notices, log = [], []
    state = {"deleted": False}

    def handler(method, path, body, url):
        log.append((method, path))
        if method == "POST":
            return DOC_POD_CREATED
        if method == "DELETE":
            state["deleted"] = True
            return None
        if state["deleted"]:
            raise _http_error(url, 404, DOC_404)
        if how == "error-status":
            return {**DOC_POD_CREATED, "status": "ERROR"}
        if how == "429-past-deadline":
            raise _limited(url, "60")
        return DOC_POD_CREATED

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert ("DELETE", "/pods/" + POD) in log
    assert "RunPod termination verified." in notices
    assert "CRITICAL" not in str(notices) and KEY not in str(notices)
    expected = {
        "error-status": "ERROR state before SSH startup",
        "429-past-deadline": "did not expose SSH within 15 minutes",
        "never-ready": "did not expose SSH within 15 minutes",
    }[how]
    assert any(expected in line for line in notices)


# --------------------------------------------------------------- the CRITICAL notice -------
def test_an_unconfirmed_cleanup_names_the_pod_its_bill_and_where_to_look(credentials, monkeypatch):
    ns, _ = _controller()
    notices, state = [], {"name": None}

    def handler(method, path, body, url):
        if method == "POST":
            state["name"] = body["name"]
            return {**DOC_POD_CREATED, "name": body["name"]}
        if method == "DELETE":
            raise _http_error(url, 500, {"title": "x", "status": 500, "detail": "y"})
        return {**DOC_POD_CREATED, "status": "ERROR"}

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    critical = [line for line in notices if line.startswith("CRITICAL")]
    assert len(critical) == 1
    text = critical[0]
    assert POD in text and state["name"] in text
    assert "may still be running and billed" in text
    assert "RunPod console" in text
    assert "http" not in text.lower()


def test_an_unconfirmed_cleanup_without_a_known_pod_says_so_and_names_the_run(
    credentials, monkeypatch
):
    ns, clock = _controller()
    notices, log, state = [], [], {"name": None}

    def handler(method, path, body, url):
        log.append((method, path.split("?")[0]))
        if method == "POST":
            state["name"] = body["name"]
            raise _http_error(url, 503, b"upstream", content_type="text/plain")
        raise _http_error(url, 500, {"title": "x", "status": 500, "detail": "y"})

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    critical = [line for line in notices if line.startswith("CRITICAL")]
    assert len(critical) == 1
    assert "id is unknown" in critical[0] and state["name"] in critical[0]
    assert "may still be running and billed" in critical[0]
    assert ("DELETE", "/pods/" + POD) not in log and all(m != "DELETE" for m, _ in log)
    assert 59 <= clock.now <= 61


@pytest.mark.parametrize("kind", ["404", "429", "500", "transport"])
def test_the_reconcile_retries_a_failing_list_until_its_sixty_second_deadline(
    credentials, monkeypatch, kind
):
    ns, clock = _controller()
    notices, log = [], []

    def handler(method, path, body, url):
        log.append((method, path.split("?")[0]))
        if method == "POST":
            raise _http_error(url, 503, b"upstream", content_type="text/plain")
        if kind == "404":
            raise _http_error(url, 404, b"<html>", content_type="text/html")
        if kind == "429":
            raise _limited(url, "7")
        if kind == "500":
            raise _http_error(url, 500, {"title": "x", "status": 500, "detail": "y"})
        raise TimeoutError("reset")

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    lists = [entry for entry in log if entry == ("GET", "/pods")]
    assert len(lists) >= 2
    assert any(line.startswith("CRITICAL") for line in notices)
    assert all(method != "DELETE" for method, _ in log)
    if kind == "429":
        assert clock.sleeps[0] == 7
    assert 59 <= clock.now <= 61


def test_a_transient_list_failure_is_retried_and_the_pod_is_then_found(credentials, monkeypatch):
    ns, _ = _controller()
    notices, state = [], {"name": None, "lists": 0, "deleted": False}

    def handler(method, path, body, url):
        if method == "POST":
            state["name"] = body["name"]
            raise _http_error(url, 503, b"upstream", content_type="text/plain")
        if path.startswith("/pods?"):
            state["lists"] += 1
            if state["lists"] == 1:
                raise TimeoutError("reset")
            return {"pods": [{"id": "owned", "name": state["name"]}],
                    "pagination": {"nextCursor": None, "hasNextPage": False}}
        if method == "DELETE":
            state["deleted"] = True
            return None
        raise _http_error(url, 404, DOC_404)

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert state["lists"] == 2 and state["deleted"]
    assert "RunPod termination verified." in notices


def test_a_list_in_the_wrong_shape_is_refused_at_once_not_retried(monkeypatch):
    ns, clock = _controller()
    calls = []

    def handler(method, path, body, url):
        calls.append(path)
        return [{"id": "x", "name": "soup-x"}]

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="could not reconcile the launch"):
        ns["_recover_created"](KEY, "soup-x")
    assert len(calls) == 1 and clock.sleeps == []


def test_the_reconcile_deadline_is_sixty_seconds(monkeypatch):
    ns, clock = _controller()
    _serve(ns, monkeypatch, lambda *args: _page([]))
    with pytest.raises(RuntimeError, match="launch result is unknown"):
        ns["_recover_created"](KEY, "soup-x")
    assert 59 <= clock.now <= 61


# ------------------------------------------------------------------- the Pod list walk -----
def test_the_page_cap_is_twenty_and_the_walk_never_goes_past_it(monkeypatch):
    ns, _ = _controller()
    assert ns["_MAX_LIST_PAGES"] == 20 and ns["_LIST_PAGE"] == 100
    seen = []

    def handler(method, path, body, url):
        seen.append(path)
        if len(seen) > ns["_MAX_LIST_PAGES"] + 1:
            raise AssertionError("the walk went past the page cap")
        return _page([], cursor=f"c{len(seen)}")

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="could not reconcile the launch"):
        list(ns["_pod_pages"](KEY))
    assert len(seen) == 20


def test_a_cursor_of_512_characters_is_followed_and_513_is_refused(monkeypatch):
    ns, _ = _controller()
    seen = []

    def handler(method, path, body, url):
        seen.append(path)
        return _page([], cursor="c" * 512) if len(seen) == 1 else _page([POD])

    _serve(ns, monkeypatch, handler)
    pages = list(ns["_pod_pages"](KEY))
    assert len(pages) == 2 and seen[1] == "/pods?limit=100&cursor=" + "c" * 512

    seen.clear()
    _serve(ns, monkeypatch, lambda m, p, b, u: seen.append(p) or _page([], cursor="c" * 513))
    with pytest.raises(RuntimeError, match="invalid page cursor"):
        list(ns["_pod_pages"](KEY))
    assert len(seen) == 1


@pytest.mark.parametrize("cursor", [br'"\ud800"', br'"\udc00"', br'"ok\ud83d"'])
def test_a_cursor_with_a_lone_surrogate_is_a_named_error(monkeypatch, cursor):
    ns, _ = _controller()
    body = (b'{"pods": [], "pagination": {"nextCursor": ' + cursor
            + b', "hasNextPage": true}}')
    _serve(ns, monkeypatch, lambda *args: body)
    with pytest.raises(RuntimeError, match="invalid page cursor"):
        list(ns["_pod_pages"](KEY))


def test_the_deadline_is_checked_between_pages_of_one_pass(monkeypatch):
    ns, clock = _controller()
    requests = []

    def handler(method, path, body, url):
        requests.append(path)
        clock.now += 25
        return _page([], cursor=f"c{len(requests)}")

    _serve(ns, monkeypatch, handler)
    with pytest.raises(RuntimeError, match="launch result is unknown"):
        ns["_recover_created"](KEY, "soup-x")
    assert len(requests) == 3


# ------------------------------------------------------------- the key and provider text ---
SPELLED = "Key/With+Special=Chars-AbCdEf123"
_ENCODED = base64.b64encode(SPELLED.encode()).decode()
SPELLINGS = {
    "as is": SPELLED,
    "quoted without safe": urllib.parse.quote(SPELLED, safe=""),
    "quoted": urllib.parse.quote(SPELLED),
    "base64": _ENCODED,
    "base64 unpadded": _ENCODED.rstrip("="),
    "lower case": SPELLED.lower(),
}


def _failure(ns, monkeypatch, body, key=KEY):
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 400, body)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](key, "/pods", method="POST", payload={})
    return str(excinfo.value)


@pytest.mark.parametrize("spelling", list(SPELLINGS))
@pytest.mark.parametrize("field", ["title", "detail", "errors"])
def test_the_key_is_removed_in_every_spelling_and_field(monkeypatch, spelling, field):
    ns, _ = _controller()
    echoed = SPELLINGS[spelling]
    body = {"title": "Bad", "detail": "Bad", "errors": ["Bad"]}
    body[field] = [f"x {echoed} y"] if field == "errors" else f"x {echoed} y"
    text = _failure(ns, monkeypatch, body, key=SPELLED)
    assert echoed not in text
    assert "<redacted>" in text


@pytest.mark.parametrize("control", ["\x07", "\x1b", "​", "\x00", "\x9b"])
def test_a_key_echoed_with_a_control_character_inside_is_still_removed(monkeypatch, control):
    ns, _ = _controller()
    echoed = SPELLED[:7] + control + SPELLED[7:]
    text = _failure(ns, monkeypatch, {"title": "Bad", "detail": f"seen {echoed}"}, key=SPELLED)
    assert SPELLED not in text and "<redacted>" in text
    assert control not in text


def test_a_key_echoed_in_two_spellings_at_once_leaves_neither(monkeypatch):
    ns, _ = _controller()
    detail = f"a {SPELLED} b {_ENCODED} c {SPELLED.lower()}"
    text = _failure(ns, monkeypatch, {"title": "Bad", "detail": detail}, key=SPELLED)
    assert SPELLED not in text and _ENCODED not in text and SPELLED.lower() not in text


def test_the_stderr_tail_cleaner_is_unchanged_without_a_key():
    ns, _ = _controller()
    assert ns["_clean_text"]("a\x07b ghp_x hf_" + "a" * 30) == "ab ghp_x <redacted>"


# --------------------------------------------------------- headers, redirects, wire ---------
def _capture(ns, monkeypatch):
    seen = []

    class Opener:
        def open(self, request, timeout):
            seen.append(request)
            return io.BytesIO(b'{"id": "x"}')

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    return seen


def test_the_key_header_is_never_one_that_a_redirect_could_resend(monkeypatch):
    ns, _ = _controller()
    seen = _capture(ns, monkeypatch)
    ns["_request_json"](KEY, "/pods/" + POD)
    request = seen[0]
    assert "Bearer " + KEY not in request.headers.values()
    assert request.unredirected_hdrs["Authorization"] == "Bearer " + KEY


def test_the_request_names_only_the_client_and_its_version(monkeypatch):
    ns, _ = _controller()
    seen = _capture(ns, monkeypatch)
    ns["_request_json"](KEY, "/pods/" + POD)
    assert seen[0].get_header("User-agent") == "soup-cli/0.76.0"


class _Quiet(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        return None


def _start(handler):
    server = LoopbackHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _stop(*servers):
    for server in servers:
        server.shutdown()
        server.server_close()


def _loopback_env(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy",
                 "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")


def test_on_the_wire_the_request_carries_the_key_and_only_the_client_name(monkeypatch):
    _loopback_env(monkeypatch)
    seen = []

    class Answer(_Quiet):
        def do_GET(self):  # noqa: N802 - http.server API
            seen.append(dict(self.headers))
            body = b'{"id": "x"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = _start(Answer)
    try:
        ns = _plain_controller()
        ns["_API_BASE"] = f"http://127.0.0.1:{server.server_port}"
        assert ns["_request_json"](KEY, "/pods/" + POD) == {"id": "x"}
    finally:
        _stop(server)
    assert len(seen) == 1
    assert seen[0]["Authorization"] == "Bearer " + KEY
    assert seen[0]["User-Agent"] == "soup-cli/0.76.0"


def test_a_redirect_to_another_port_is_never_followed_and_nothing_is_resent(monkeypatch):
    _loopback_env(monkeypatch)
    to_target, to_redirector = [], []

    class Target(_Quiet):
        def do_GET(self):  # noqa: N802 - http.server API
            to_target.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    target = _start(Target)

    class Redirector(_Quiet):
        def do_GET(self):  # noqa: N802 - http.server API
            to_redirector.append(dict(self.headers))
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_port}/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

    redirector = _start(Redirector)
    try:
        ns = _plain_controller()
        ns["_API_BASE"] = f"http://127.0.0.1:{redirector.server_port}"
        with pytest.raises(ns["_ApiRetiredError"], match="HTTP 302"):
            ns["_request_json"](KEY, "/pods/" + POD)
    finally:
        _stop(redirector, target)
    assert len(to_redirector) == 1
    assert to_target == []


# ------------------------------------------------------------------ size limits ------------
def _padded(size):
    body = b'{"id": "x"}'
    return body + b" " * (size - len(body))


def test_a_response_of_exactly_the_cap_is_read_and_one_byte_more_is_refused(monkeypatch):
    ns, _ = _controller()
    cap = ns["_MAX_RESPONSE"]
    assert cap == 1024 * 1024
    _serve(ns, monkeypatch, lambda *args: _padded(cap))
    assert ns["_request_json"](KEY, "/pods/" + POD) == {"id": "x"}
    _serve(ns, monkeypatch, lambda *args: _padded(cap + 1))
    with pytest.raises(RuntimeError, match="response exceeds 1 MB"):
        ns["_request_json"](KEY, "/pods/" + POD)


def test_a_response_is_read_with_a_bound_never_whole(monkeypatch):
    ns, _ = _controller()
    sizes = []

    class Body(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    class Opener:
        def open(self, request, timeout):
            return Body(b'{"id": "x"}')

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    ns["_request_json"](KEY, "/pods/" + POD)
    assert sizes == [ns["_MAX_RESPONSE"] + 1]


def test_an_error_body_is_read_only_up_to_its_cap(monkeypatch):
    ns, _ = _controller()
    assert ns["_MAX_ERROR_BODY"] == 64 * 1024
    document = json.dumps(DOC_422).encode()
    late = b" " * (ns["_MAX_ERROR_BODY"] + 10) + document
    assert _failure(ns, monkeypatch, late) == "RunPod API POST failed: HTTP 400"
    early = document + b" " * (ns["_MAX_ERROR_BODY"] + 10)
    assert "Request validation failed." in _failure(ns, monkeypatch, early)


# ------------------------------------------------------------- docs and the disk notice ----
def test_main_prints_one_line_naming_the_disk_size_before_creating(credentials, monkeypatch):
    ns, _ = _controller()
    notices = []
    state = {"deleted": False}

    def handler(method, path, body, url):
        if method == "POST":
            notices.append("<POST>")
            assert body["disk"] == 80
            return DOC_POD_CREATED
        if method == "DELETE":
            state["deleted"] = True
            return None
        if state["deleted"]:
            raise _http_error(url, 404, DOC_404)
        return {**DOC_POD_CREATED, "status": "ERROR"}

    _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    ns["main"]()
    disk = [line for line in notices if "container disk" in line]
    assert len(disk) == 1 and "80 GB" in disk[0]
    assert "/workspace" in disk[0] and "model cache" in disk[0]
    assert notices.index(disk[0]) < notices.index("<POST>")


def test_the_docs_say_the_disk_is_shared_and_ephemeral_and_where_a_list_404_goes():
    docs = (_ROOT / "docs" / "backends-and-ops.md").read_text(encoding="utf-8")
    section = docs.split("### RunPod", 1)[1].split("### Lambda Cloud", 1)[0]
    flat = " ".join(section.split())
    assert "shares the 80 GB container disk" in flat
    assert "wiped on restart" in flat
    assert "ephemeral" in flat
    assert "request for the Pod list or the Pod create" not in flat
    assert "only when the create itself is answered" in flat


def test_the_fragment_does_not_promise_a_clean_exit_for_a_list_404():
    text = (_ROOT / "changelog.d" / "0.77.2" / "2877.fixed.md").read_text(encoding="utf-8")
    assert "Pod list or create" not in text
    assert "no Pod created or cleaned up" in text
