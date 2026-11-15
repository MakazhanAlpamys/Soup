"""The RunPod controller names a retired or moved REST API instead of a bare HTTP status.

RunPod retired its REST API v1 on 2026-11-15. What its retired address answers is not
published, so these tests pin the part that is certain: an answer that says "this route
is not served" (404 on the Pod collection, 410, or a redirect, which the controller never
follows) is reported once, in words, and never as a Pod that needs cleaning up.
"""

import io
import json
import urllib.error
from pathlib import Path

import pytest

from tests.test_runpod_cloud import _controller
from tests.test_runpod_controller import _clock

_ROOT = Path(__file__).resolve().parents[1]
_NOT_SERVED = [404, 410, 301, 302, 307, 308]


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-real-private-key", encoding="utf-8")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", "test-provider-secret")
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    return key


def _answering(ns, monkeypatch, status, body=b"provider-secret", content_type=None):
    """Make every request the controller sends get the HTTP ``status``; return the
    list of (method, path) it sent."""
    sent = []
    headers = {"Content-Type": content_type} if content_type else {}

    class Opener:
        def open(self, request, timeout):
            sent.append((request.method, request.full_url.removeprefix(ns["_API_BASE"])))
            raise urllib.error.HTTPError(
                request.full_url, status, "msg", headers, io.BytesIO(body)
            )

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    return sent


@pytest.mark.parametrize("status", _NOT_SERVED)
@pytest.mark.parametrize("method", ["POST", "GET"])
def test_a_route_that_is_not_served_is_named_not_reported_as_a_bare_status(
    monkeypatch, status, method
):
    ns = _controller()
    _answering(ns, monkeypatch, status)
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"]("test-provider-secret", "/pods", method=method, payload=None)
    text = str(excinfo.value)
    assert isinstance(excinfo.value, ns["_ApiRetiredError"])
    assert "soup train --cloud runpod --cloud-submit" in text
    assert "retired" in text and "REST API v2" in text
    assert "pip install -U soup-cli" in text
    assert "RunPod console" in text
    assert f"HTTP {status}" in text
    assert "provider-secret" not in text and "msg" not in text


@pytest.mark.parametrize("status", [410, 301, 302, 307, 308])
@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_gone_or_moved_is_named_on_a_pod_path_too(monkeypatch, status, method):
    ns = _controller()
    _answering(ns, monkeypatch, status)
    with pytest.raises(ns["_ApiRetiredError"], match="retired"):
        ns["_request_json"]("test-provider-secret", "/pods/pod-1", method=method)


def test_404_on_a_single_pod_is_still_an_absent_pod_not_a_retired_api(monkeypatch):
    ns = _controller()
    _answering(ns, monkeypatch, 404)
    assert ns["_request_json"]("test-provider-secret", "/pods/pod-1") is None
    # A DELETE that is answered 404 is reported as such: it is not, by itself, proof
    # that the Pod is gone (see test_runpod_controller_safety.py).
    with pytest.raises(ns["_PodNotFoundError"]):
        ns["_request_json"]("test-provider-secret", "/pods/pod-1", method="DELETE")


_NOT_FOUND = {"title": "Not Found", "status": 404, "detail": "gpu type not found"}


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.parametrize("content_type", ["application/problem+json",
                                          "application/problem+json; charset=utf-8"])
def test_a_problem_json_404_or_410_to_the_create_is_an_ordinary_rejection_with_its_text(
    monkeypatch, status, content_type
):
    ns = _controller()
    body = json.dumps({**_NOT_FOUND, "status": status}).encode()
    _answering(ns, monkeypatch, status, body=body, content_type=content_type)
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"]("test-provider-secret", "/pods", method="POST", payload={})
    assert not isinstance(excinfo.value, ns["_ApiRetiredError"])
    assert str(excinfo.value) == (
        f"RunPod API POST failed: HTTP {status} (Not Found; gpu type not found)"
    )
    assert excinfo.value.create_rejected is True
    assert "retired" not in str(excinfo.value)


@pytest.mark.parametrize("content_type", ["application/json", "text/html", "text/plain", ""])
@pytest.mark.parametrize("status", [404, 410])
def test_a_404_or_410_that_is_not_problem_json_is_still_the_named_retirement(
    monkeypatch, status, content_type
):
    ns = _controller()
    _answering(ns, monkeypatch, status, body=json.dumps(_NOT_FOUND).encode(),
               content_type=content_type)
    with pytest.raises(ns["_ApiRetiredError"], match="retired"):
        ns["_request_json"]("test-provider-secret", "/pods", method="POST", payload={})


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_the_named_retirement_whatever_its_content_type(monkeypatch, status):
    ns = _controller()
    _answering(ns, monkeypatch, status, body=json.dumps(_NOT_FOUND).encode(),
               content_type="application/problem+json")
    with pytest.raises(ns["_ApiRetiredError"], match="retired"):
        ns["_request_json"]("test-provider-secret", "/pods", method="POST", payload={})


@pytest.mark.parametrize("method,path", [("GET", "/pods?limit=100"), ("GET", "/pods/pod-1"),
                                         ("DELETE", "/pods/pod-1")])
def test_a_problem_json_410_is_an_ordinary_answer_on_every_route(monkeypatch, method, path):
    ns = _controller()
    _answering(ns, monkeypatch, 410, body=json.dumps({**_NOT_FOUND, "status": 410}).encode(),
               content_type="application/problem+json")
    with pytest.raises(RuntimeError, match=r"HTTP 410 \(Not Found; gpu type not found\)") as err:
        ns["_request_json"]("test-provider-secret", path, method=method)
    assert not isinstance(err.value, ns["_ApiRetiredError"])


def test_a_problem_json_404_on_the_create_stops_the_run_with_the_provider_text(
    credentials, monkeypatch
):
    ns = _controller()
    _clock(ns, 100)
    sent = _answering(ns, monkeypatch, 404, body=json.dumps(_NOT_FOUND).encode(),
                      content_type="application/problem+json")
    notices = []
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert sent == [("POST", "/pods")]
    assert "RunPod API POST failed: HTTP 404 (Not Found; gpu type not found)" in notices
    assert "RunPod create request rejected; no Pod to clean up." in notices
    assert not any("retired" in line for line in notices)
    assert "CRITICAL" not in str(notices)


@pytest.mark.parametrize("status", [400, 401, 403, 422, 429, 500, 503])
def test_other_statuses_keep_their_plain_report(monkeypatch, status):
    ns = _controller()
    _answering(ns, monkeypatch, status)
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"]("test-provider-secret", "/pods", method="POST", payload={})
    assert str(excinfo.value) == f"RunPod API POST failed: HTTP {status}"
    assert not isinstance(excinfo.value, ns["_ApiRetiredError"])


@pytest.mark.parametrize("status", _NOT_SERVED)
def test_create_stops_with_one_message_and_no_cleanup_attempt(credentials, monkeypatch, status):
    ns = _controller()
    _clock(ns, 100)
    sent = _answering(ns, monkeypatch, status)
    notices = []
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    # One POST: nothing is listed, deleted or polled for a Pod that cannot exist.
    assert sent == [("POST", "/pods")]
    retired = [line for line in notices if "retired" in line]
    assert len(retired) == 1
    assert "soup train --cloud runpod --cloud-submit" in retired[0]
    assert "RunPod create request rejected; no Pod to clean up." in notices
    assert "CRITICAL" not in str(notices)
    assert "termination verified" not in str(notices)
    assert "provider-secret" not in str(notices)


def test_the_docs_say_that_v1_is_retired_and_what_to_do():
    docs = (_ROOT / "docs" / "backends-and-ops.md").read_text(encoding="utf-8")
    section = docs.split("### RunPod", 1)[1].split("### Lambda Cloud", 1)[0]
    assert "retired" in section and "2026-11-15" in section
    assert "pip install -U soup-cli" in section
