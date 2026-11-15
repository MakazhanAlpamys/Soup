"""The RunPod controller names a retired or moved REST API instead of a bare HTTP status.

RunPod retired its REST API v1 on 2026-11-15. What its retired address answers is not
published, so these tests pin the part that is certain: an answer that says "this route
is not served" (404 on the Pod collection, 410, or a redirect, which the controller never
follows) is reported once, in words, and never as a Pod that needs cleaning up.
"""

import io
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


def _answering(ns, monkeypatch, status, body=b"provider-secret"):
    """Make every request the controller sends get the HTTP ``status``; return the
    list of (method, path) it sent."""
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append((request.method, request.full_url.removeprefix(ns["_API_BASE"])))
            raise urllib.error.HTTPError(request.full_url, status, "msg", {}, io.BytesIO(body))

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
    assert "retired" in text and "2026-11-15" in text
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


@pytest.mark.parametrize("method", ["GET", "DELETE"])
def test_404_on_a_single_pod_is_still_an_absent_pod_not_a_retired_api(monkeypatch, method):
    ns = _controller()
    _answering(ns, monkeypatch, 404)
    assert ns["_request_json"]("test-provider-secret", "/pods/pod-1", method=method) is None


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
