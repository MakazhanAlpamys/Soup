"""The RunPod controller against the shapes RunPod publishes for its REST API v2.

Every body below is copied from the examples in RunPod's published v2 reference
(https://docs.runpod.io/api-reference-v2/openapi.json, read 2026-11-15) or follows its
schema. They come from the documentation, not from a live call: nothing in this file has
been exercised against the service.
"""

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

from tests.test_runpod_cloud import _controller as _plain_controller
from tests.test_runpod_controller import _clock

KEY = "test-provider-secret"


def _controller():
    """The rendered controller on a fast fake clock, so that a regression that makes a
    poll loop run to its deadline fails in milliseconds instead of waiting 15 real
    minutes. A test that checks sleeps or deadlines installs its own clock afterwards."""
    ns = _plain_controller()
    _clock(ns)
    return ns

# --- from the documentation of 2026-11-15 (Get a pod, 200 "pod" example) -------------
DOC_POD_RUNNING = {
    "id": "7h9k2m4n6p",
    "name": "pytorch-training",
    "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "args": "",
    "disk": 50,
    "ports": ["8888/http", "22/tcp"],
    "env": {"MODEL_NAME": "llama-3"},
    "registry": None,
    "status": "RUNNING",
    "actions": ["stop", "restart", "terminate"],
    "mounts": {"persistent": {"size": 20, "path": "/workspace"}},
    "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "vcpuCount": 16, "memory": 64},
    "cloud": "SECURE",
    "dataCenterId": "US-KS-2",
    "cudaVersion": "12.8",
    "ssh": {
        "proxy": {
            "host": "ssh.runpod.io", "port": 22, "username": "7h9k2m4n6p-64411eb2",
            "command": "ssh 7h9k2m4n6p-64411eb2@ssh.runpod.io",
        },
        "direct": {
            "host": "195.26.233.3", "port": 34446, "username": "root",
            "command": "ssh root@195.26.233.3 -p 34446",
        },
    },
    "template": "9x4m2p7v",
    "cost": 0.44,
    "locked": False,
    "globalNetworking": {"enabled": False},
    "runtime": {
        "uptime": 3600,
        "ports": [{"private": 22, "public": 34446, "type": "tcp", "ip": "195.26.233.3"}],
    },
    "createdAt": "2026-06-01T12:00:00Z",
    "startedAt": "2026-06-01T12:02:00Z",
}
# --- Create a pod, 201 "pod" example: just created, nothing assigned yet --------------
DOC_POD_CREATED = {
    **DOC_POD_RUNNING,
    "status": "PROVISIONING",
    "actions": ["start", "terminate"],
    "ssh": {"proxy": DOC_POD_RUNNING["ssh"]["proxy"], "direct": None},
    "runtime": None,
    "startedAt": None,
}
# --- List pods, 200 "pods" example (a listing carries no live mapping) ----------------
DOC_LIST_PAGE = {
    "pods": [{**DOC_POD_CREATED, "status": "RUNNING"}],
    "pagination": {"nextCursor": None, "hasNextPage": False},
}
# --- Error bodies: the documented application/problem+json examples -------------------
DOC_401 = {"title": "Unauthorized", "status": 401, "detail": "missing bearer token"}
DOC_403 = {"title": "Forbidden", "status": 403, "detail": "access denied"}
DOC_404 = {"title": "Not Found", "status": 404, "detail": "resource not found"}
DOC_422 = {
    "title": "Unprocessable Entity", "status": 422, "detail": "Request validation failed.",
    "errors": ["$: additional properties 'bogus' not allowed"],
}
DOC_429 = {
    "title": "Too Many Requests", "status": 429,
    "detail": "rate limit exceeded for the minute window",
}
PROBLEM = "application/problem+json"


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = tmp_path / "id_test"
    key.write_text("not-a-real-private-key", encoding="utf-8")
    Path(str(key) + ".pub").write_text("ssh-ed25519 AAAATEST fixture", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", KEY)
    monkeypatch.setenv("RUNPOD_SSH_PRIVATE_KEY", str(key))
    return key


def _http_error(url, status, body, content_type=PROBLEM):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    headers = {"Content-Type": content_type} if content_type else {}
    return urllib.error.HTTPError(url, status, "reason", headers, io.BytesIO(raw))


def _serve(ns, monkeypatch, handler):
    """Route every request through ``handler(method, path, body)``, which returns the
    response body (bytes or JSON-able) or raises; return the log of requests."""
    log = []

    class Opener:
        def open(self, request, timeout):
            path = request.full_url.removeprefix(ns["_API_BASE"])
            body = json.loads(request.data) if request.data else None
            log.append((request.method, path, request.get_header("Authorization"), timeout))
            result = handler(request.method, path, body, request.full_url)
            if result is None:
                return io.BytesIO(b"")
            return io.BytesIO(result if isinstance(result, bytes) else json.dumps(result).encode())

    monkeypatch.setattr(ns["urllib"].request, "build_opener", lambda *handlers: Opener())
    return log


def _ssh_record(status, *, host="195.26.233.3", port=34446, via="direct"):
    record = {**DOC_POD_CREATED, "status": status}
    if via in {"direct", "both"}:
        record["ssh"] = {
            "proxy": DOC_POD_RUNNING["ssh"]["proxy"],
            "direct": {"host": host, "port": port, "username": "root", "command": "ssh ..."},
        }
    if via in {"runtime", "both"}:
        record["runtime"] = {
            "uptime": 5, "ports": [{"private": 22, "public": port, "type": "tcp", "ip": host}],
        }
    return record


# --------------------------------------------------------------------------- the base --
def test_the_api_base_is_the_documented_v2_server():
    ns = _controller()
    assert ns["_API_BASE"] == "https://api.runpod.io/v2"


def test_no_v1_address_or_field_name_is_left_in_the_controller():
    source = (Path(__file__).resolve().parents[1] / "src/soup_cli/cloud/_runpod_controller.py")
    text = source.read_text(encoding="utf-8")
    for old in ("rest.runpod.io", "desiredStatus", "publicIp", "portMappings", "gpuTypeIds",
                "dockerStartCmd", "dockerEntrypoint", "containerDiskInGb", "supportPublicIp",
                "cloudType", "imageName"):
        assert old not in text.replace("REST API v1", ""), old


# ------------------------------------------------------------------------ create body --
def test_the_create_body_is_the_documented_create_pod_request():
    ns = _controller()
    body = ns["_launch_payload"]("ssh-ed25519 AAAATEST fixture", "soup-run")
    assert body == {
        "name": "soup-run",
        "image": "runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404",
        "cloud": "SECURE",
        "gpu": {"id": "NVIDIA A100-SXM4-80GB", "count": 1},
        "disk": 80,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": "ssh-ed25519 AAAATEST fixture"},
        "entrypoint": ["/bin/bash", "-lc"],
        "cmd": [ns["_BOOT"]],
    }
    # CreatePodRequest refuses unknown properties (422): nothing outside its schema.
    allowed = {"name", "image", "cloud", "gpu", "disk", "ports", "env", "entrypoint", "cmd"}
    assert set(body) <= allowed
    assert set(body["gpu"]) <= {"id", "count"}


def test_the_boot_command_installs_the_public_key_from_the_environment_it_is_given():
    ns = _controller()
    assert '"$PUBLIC_KEY"' in ns["_BOOT"] and "exec /usr/sbin/sshd -D -e" in ns["_BOOT"]


@pytest.mark.parametrize("gpu", ["a100", "h100", "a40", "rtx3090", "rtx4090"])
def test_each_gpu_name_reaches_the_gpu_id_field(gpu):
    from soup_cli.cloud import runpod

    text = runpod.render_runpod_stub(
        "base: public/model\ndata: {train: public/dataset}\noutput: out\n",
        gpu=gpu, output_dir="out", soup_version="0.76.0",
    )
    namespace = {"__name__": "test_controller"}
    exec(compile(text, "soup_runpod_app.py", "exec"), namespace)
    body = namespace["_launch_payload"]("ssh-ed25519 AAAATEST fixture", "soup-run")
    assert body["gpu"]["id"] == runpod._GPU_IDS[gpu]
    assert body["gpu"]["count"] == 1


def test_the_create_request_is_a_post_to_the_pods_collection_with_a_bearer_token(
    monkeypatch,
):
    ns = _controller()
    log = _serve(ns, monkeypatch, lambda *args: DOC_POD_CREATED)
    payload = ns["_launch_payload"]("ssh-ed25519 AAAATEST fixture", "soup-run")
    created = ns["_request_json"](KEY, "/pods", method="POST", payload=payload)
    assert created["id"] == "7h9k2m4n6p"
    assert log == [("POST", "/pods", "Bearer " + KEY, 60)]


# ----------------------------------------------------------------------------- listing --
def _list_handler(pages):
    """Serve ``pages`` (a list of response bodies) in order for GET /pods."""
    seen = []

    def handler(method, path, body, url):
        assert method == "GET" and path.startswith("/pods?"), (method, path)
        seen.append(path)
        return pages[len(seen) - 1]

    return handler, seen


def test_a_listing_in_the_documented_shape_is_read_by_name(monkeypatch):
    ns = _controller()
    page = {
        "pods": [{"id": "other", "name": "other"}, {"id": "owned", "name": "soup-x"}],
        "pagination": {"nextCursor": None, "hasNextPage": False},
    }
    handler, seen = _list_handler([page])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    assert ns["_recover_created"](KEY, "soup-x") == ["owned"]
    assert seen == ["/pods?limit=100"]


@pytest.mark.parametrize("bare", [[{"id": "owned", "name": "soup-x"}], {"pods": "no"}, {}])
def test_a_listing_that_is_not_the_documented_shape_is_not_trusted(monkeypatch, bare):
    ns = _controller()
    handler, _ = _list_handler([bare])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    with pytest.raises(RuntimeError, match="could not reconcile the launch"):
        ns["_recover_created"](KEY, "soup-x")


def test_the_listing_follows_the_cursor_until_the_pod_is_found(monkeypatch):
    ns = _controller()
    first = {
        "pods": [{"id": "a", "name": "a"}],
        "pagination": {"nextCursor": "Y3JlYXRlZEF0PTE3NDg3ODA0MDA", "hasNextPage": True},
    }
    second = {
        "pods": [{"id": "owned", "name": "soup-x"}],
        "pagination": {"nextCursor": None, "hasNextPage": False},
    }
    handler, seen = _list_handler([first, second])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    assert ns["_recover_created"](KEY, "soup-x") == ["owned"]
    assert seen == ["/pods?limit=100", "/pods?limit=100&cursor=Y3JlYXRlZEF0PTE3NDg3ODA0MDA"]


def test_the_cursor_is_sent_percent_encoded_and_never_as_a_new_parameter(monkeypatch):
    ns = _controller()
    first = {
        "pods": [], "pagination": {"nextCursor": "a&b=c d/é", "hasNextPage": True},
    }
    second = {"pods": [{"id": "owned", "name": "soup-x"}],
              "pagination": {"nextCursor": None, "hasNextPage": False}}
    handler, seen = _list_handler([first, second])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    ns["_recover_created"](KEY, "soup-x")
    assert seen[1] == "/pods?limit=100&cursor=a%26b%3Dc%20d%2F%C3%A9"


@pytest.mark.parametrize("paging", [
    {"nextCursor": None, "hasNextPage": True},
    {"nextCursor": "", "hasNextPage": True},
    {"nextCursor": 7, "hasNextPage": True},
    "next",
    None,
])
def test_a_page_that_promises_more_without_a_usable_cursor_is_an_error(monkeypatch, paging):
    ns = _controller()
    handler, _ = _list_handler([{"pods": [], "pagination": paging}])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    with pytest.raises(RuntimeError, match="could not reconcile the launch"):
        ns["_recover_created"](KEY, "soup-x")


def test_the_listing_walk_is_bounded(monkeypatch):
    ns = _controller()
    pages = {"count": 0}

    def handler(method, path, body, url):
        pages["count"] += 1
        if pages["count"] > ns["_MAX_LIST_PAGES"] + 1:
            # A walk that ignored its cap fails here instead of running on.
            raise AssertionError("the walk went past the page cap")
        return {"pods": [], "pagination": {"nextCursor": f"c{pages['count']}",
                                           "hasNextPage": True}}

    _serve(ns, monkeypatch, handler)
    _clock(ns)
    with pytest.raises(RuntimeError, match="could not reconcile the launch"):
        ns["_recover_created"](KEY, "soup-x")
    assert pages["count"] == ns["_MAX_LIST_PAGES"] == 20


@pytest.mark.parametrize("pod_id", ["../x", "x" * 65, 17, None, "", "a b"])
def test_a_listed_pod_with_an_unsafe_id_is_never_returned_for_deletion(monkeypatch, pod_id):
    ns = _controller()
    page = {
        "pods": [{"id": pod_id, "name": "soup-x"}],
        "pagination": {"nextCursor": None, "hasNextPage": False},
    }
    handler, _ = _list_handler([page])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    with pytest.raises(RuntimeError, match="invalid Pod ID"):
        ns["_recover_created"](KEY, "soup-x")


def test_only_the_owned_name_is_returned_when_pods_share_other_traits(monkeypatch):
    ns = _controller()
    page = {
        "pods": [{"id": "x1", "name": "soup-x-extra"}, {"id": "x2", "name": "SOUP-X"},
                 {"id": "x3", "name": "soup-x"}],
        "pagination": {"nextCursor": None, "hasNextPage": False},
    }
    handler, _ = _list_handler([page])
    _serve(ns, monkeypatch, handler)
    _clock(ns)
    assert ns["_recover_created"](KEY, "soup-x") == ["x3"]


# ------------------------------------------------------------------------- the poll -----
@pytest.mark.parametrize("via", ["direct", "runtime", "both"])
@pytest.mark.parametrize("status", ["STARTING", "RUNNING"])
def test_the_ssh_address_is_read_from_the_documented_fields(status, via):
    ns = _controller()
    ns["_request_json"] = lambda *args: _ssh_record(status, via=via)
    assert ns["_wait_for_ssh"](KEY, "7h9k2m4n6p") == ("195.26.233.3", 34446)


def test_the_documented_running_pod_yields_its_direct_address():
    ns = _controller()
    ns["_request_json"] = lambda *args: DOC_POD_RUNNING
    assert ns["_wait_for_ssh"](KEY, "7h9k2m4n6p") == ("195.26.233.3", 34446)


def test_the_direct_endpoint_wins_and_the_runtime_mapping_is_the_fallback():
    ns = _controller()
    record = _ssh_record("RUNNING", via="both")
    record["runtime"]["ports"][0].update(ip="8.8.8.8", public=2222)
    ns["_request_json"] = lambda *args: record
    assert ns["_wait_for_ssh"](KEY, "7h9k2m4n6p") == ("195.26.233.3", 34446)
    record["ssh"]["direct"] = None
    assert ns["_wait_for_ssh"](KEY, "7h9k2m4n6p") == ("8.8.8.8", 2222)


@pytest.mark.parametrize("record", [
    {**DOC_POD_CREATED},
    {**DOC_POD_CREATED, "status": "STARTING"},
    _ssh_record("PROVISIONING"),
    _ssh_record("RUNNING", port=None),
    {**_ssh_record("RUNNING", via="runtime"),
     "runtime": {"ports": [{"private": 8888, "public": 1, "type": "http", "ip": "8.8.8.8"}]}},
    {**_ssh_record("RUNNING", via="runtime"),
     "runtime": {"ports": [{"private": 22, "public": None, "type": "tcp", "ip": None}]}},
    {**DOC_POD_CREATED, "status": "SOMETHING_NEW"},
    # What v1 sent: not accepted as an address.
    {"desiredStatus": "RUNNING", "publicIp": "8.8.8.8", "portMappings": {"22": 2222}},
])
def test_the_poll_keeps_waiting_until_an_address_is_published(record):
    ns = _controller()
    ns["_request_json"] = lambda *args: record
    sleeps = _clock(ns, 300)
    with pytest.raises(RuntimeError, match="did not expose SSH within 15 minutes"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")
    assert sleeps


def test_a_pod_in_error_stops_the_poll_with_its_own_message():
    ns = _controller()
    ns["_request_json"] = lambda *args: {**DOC_POD_CREATED, "status": "ERROR"}
    sleeps = _clock(ns)
    with pytest.raises(RuntimeError, match="ERROR state before SSH startup"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")
    assert sleeps == []


@pytest.mark.parametrize("status", ["EXITED", "TERMINATED"])
def test_a_stopped_or_deleted_pod_stops_the_poll_even_when_it_still_lists_an_address(status):
    ns = _controller()
    ns["_request_json"] = lambda *args: _ssh_record(status, via="both")
    with pytest.raises(RuntimeError, match="unavailable before SSH startup"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")


@pytest.mark.parametrize("record", [None, [], "RUNNING"])
def test_a_missing_or_malformed_record_stops_the_poll(record):
    ns = _controller()
    ns["_request_json"] = lambda *args: record
    with pytest.raises(RuntimeError, match="unavailable before SSH startup"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "::1",
                                  "2001:4860:4860::8888"])
@pytest.mark.parametrize("via", ["direct", "runtime"])
def test_an_address_that_is_not_a_global_ipv4_is_refused(host, via):
    ns = _controller()
    ns["_request_json"] = lambda *args: _ssh_record("RUNNING", host=host, via=via)
    with pytest.raises(RuntimeError, match="did not return a public SSH address"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")


@pytest.mark.parametrize("host", ["ssh.runpod.io", "example.invalid", "8.8.8.8:22", 134744072])
def test_a_host_that_is_not_an_ip_literal_is_never_connected_to(host):
    ns = _controller()
    ns["_request_json"] = lambda *args: _ssh_record("RUNNING", host=host)
    with pytest.raises(RuntimeError, match="did not return a public SSH address"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")


@pytest.mark.parametrize("port", [0, 65536, -1, True, "34446", 1.5])
def test_a_port_outside_the_tcp_range_is_not_an_address(port):
    ns = _controller()
    ns["_request_json"] = lambda *args: _ssh_record("RUNNING", port=port)
    _clock(ns, 300)
    with pytest.raises(RuntimeError, match="15 minutes"):
        ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")


def test_the_poll_reads_the_single_pod_path():
    ns = _controller()
    seen = []
    ns["_request_json"] = lambda key, path, **kw: seen.append((key, path)) or DOC_POD_RUNNING
    ns["_wait_for_ssh"](KEY, "7h9k2m4n6p")
    assert seen == [(KEY, "/pods/7h9k2m4n6p")]


# ---------------------------------------------------------------------------- cleanup --
def test_delete_is_confirmed_by_a_404_after_a_204(monkeypatch):
    ns = _controller()

    def handler(method, path, body, url):
        if method == "DELETE":
            return None
        raise _http_error(url, 404, DOC_404)

    log = _serve(ns, monkeypatch, handler)
    ns["_terminate"](KEY, "7h9k2m4n6p")
    assert [(m, p) for m, p, *_ in log] == [
        ("DELETE", "/pods/7h9k2m4n6p"), ("GET", "/pods/7h9k2m4n6p"),
    ]


def test_a_terminated_pod_record_confirms_deletion():
    ns = _controller()
    calls = []

    def request(key, path, *, method="GET", **kw):
        calls.append(method)
        return None if method == "DELETE" else {**DOC_POD_RUNNING, "status": "TERMINATED"}

    ns["_request_json"] = request
    sleeps = _clock(ns)
    ns["_terminate"](KEY, "7h9k2m4n6p")
    assert calls == ["DELETE", "GET"] and sleeps == []


@pytest.mark.parametrize("status", ["RUNNING", "EXITED", "ERROR", "STARTING", "PROVISIONING"])
def test_a_pod_that_still_exists_is_deleted_again_until_it_is_gone(status):
    ns = _controller()
    reads = iter([{**DOC_POD_RUNNING, "status": status}, None])
    calls = []

    def request(key, path, *, method="GET", **kw):
        calls.append(method)
        return None if method == "DELETE" else next(reads)

    ns["_request_json"] = request
    sleeps = _clock(ns)
    ns["_terminate"](KEY, "7h9k2m4n6p")
    assert calls == ["DELETE", "GET", "DELETE", "GET"] and sleeps == [5]


# ----------------------------------------------------------------------------- errors --
@pytest.mark.parametrize("status,body", [(401, DOC_401), (403, DOC_403), (422, DOC_422),
                                         (429, DOC_429)])
def test_a_documented_problem_body_is_shown_with_the_status(monkeypatch, status, body):
    ns = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, status, body)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods", method="POST", payload={})
    text = str(excinfo.value)
    assert text.startswith(f"RunPod API POST failed: HTTP {status} (")
    assert body["title"] in text and body["detail"] in text
    for line in body.get("errors", []):
        assert line in text


@pytest.mark.parametrize("content_type", ["application/json", "text/html", "", "text/plain"])
def test_a_body_that_is_not_problem_json_is_never_shown(monkeypatch, content_type):
    ns = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 400, DOC_422, content_type=content_type)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods", method="POST", payload={})
    assert str(excinfo.value) == "RunPod API POST failed: HTTP 400"


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b"null", b'"text"', b"",
                                 b'{"title": 5, "detail": ["x"], "errors": 7}'])
def test_a_malformed_problem_body_falls_back_to_the_bare_status(monkeypatch, raw):
    ns = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 400, raw)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods", method="POST", payload={})
    assert str(excinfo.value) == "RunPod API POST failed: HTTP 400"


def _problem_text(ns, monkeypatch, body):
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 400, body)))
    with pytest.raises(RuntimeError) as excinfo:
        ns["_request_json"](KEY, "/pods", method="POST", payload={})
    return str(excinfo.value)


def test_problem_text_is_cleaned_and_redacted_and_shows_three_errors(monkeypatch):
    ns = _controller()
    text = _problem_text(ns, monkeypatch, {
        "title": "Bad\x1b]0;owned\x07 Request",
        "detail": f"key {KEY} was rejected for hf_" + "a" * 30,
        "errors": ["one\r\ntwo", "[bold]three[/bold]", "four", "five-never-shown"],
    })
    assert KEY not in text and "hf_" + "a" * 30 not in text
    assert "<redacted>" in text
    assert not any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in text)
    assert "one two" in text and "four" in text
    assert "five-never-shown" not in text
    assert "[bold]three[/bold]" in text  # literal, not markup


def test_problem_text_is_capped(monkeypatch):
    ns = _controller()
    text = _problem_text(ns, monkeypatch, {
        "title": "Bad Request", "status": 400, "detail": "y" * 5000,
    })
    assert text.startswith("RunPod API POST failed: HTTP 400 (Bad Request; yyy")
    assert len(text) <= len("RunPod API POST failed: HTTP 400 ()") + ns["_PROBLEM_CHARS"] <= 500


def test_a_404_on_a_single_pod_is_absence_with_the_documented_body(monkeypatch):
    ns = _controller()
    _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 404, DOC_404)))
    assert ns["_request_json"](KEY, "/pods/7h9k2m4n6p") is None
    with pytest.raises(ns["_PodNotFoundError"]):
        ns["_request_json"](KEY, "/pods/7h9k2m4n6p", method="DELETE")


def test_a_204_delete_has_no_body_and_returns_nothing(monkeypatch):
    ns = _controller()
    _serve(ns, monkeypatch, lambda *args: None)
    assert ns["_request_json"](KEY, "/pods/7h9k2m4n6p", method="DELETE") is None


# --------------------------------------------------------------------- the whole run ---
def test_a_full_run_against_the_documented_shapes_keeps_the_key_out_of_every_notice(
    credentials, monkeypatch
):
    ns = _controller()
    clock = _clock(ns)
    notices = []
    state = {"polls": 0, "deleted": False, "name": None}

    def handler(method, path, body, url):
        if method == "POST":
            assert path == "/pods"
            state["name"] = body["name"]
            return {**DOC_POD_CREATED, "name": body["name"]}
        if method == "GET" and path == "/pods/7h9k2m4n6p":
            if state["deleted"]:
                raise _http_error(url, 404, DOC_404)
            state["polls"] += 1
            return DOC_POD_CREATED if state["polls"] == 1 else DOC_POD_RUNNING
        if method == "DELETE" and path == "/pods/7h9k2m4n6p":
            state["deleted"] = True
            return None
        raise AssertionError((method, path))

    log = _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    ns["_train_and_copy"] = lambda ip, port, private: notices.append(f"train {ip}:{port}")
    assert ns["main"]() == 0
    assert [(m, p) for m, p, *_ in log] == [
        ("POST", "/pods"), ("GET", "/pods/7h9k2m4n6p"), ("GET", "/pods/7h9k2m4n6p"),
        ("DELETE", "/pods/7h9k2m4n6p"), ("GET", "/pods/7h9k2m4n6p"),
    ]
    assert {auth for _, _, auth, _ in log} == {"Bearer " + KEY}
    assert "train 195.26.233.3:34446" in notices
    assert "RunPod termination verified." in notices
    assert clock == [5]
    assert KEY not in str(notices) and "not-a-real-private-key" not in str(log)


def test_a_rejected_create_shows_the_documented_reason_and_cleans_nothing_up(
    credentials, monkeypatch
):
    ns = _controller()
    _clock(ns, 100)
    notices = []
    log = _serve(ns, monkeypatch, lambda m, p, b, url: (_ for _ in ()).throw(
        _http_error(url, 422, DOC_422)))
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert [(m, p) for m, p, *_ in log] == [("POST", "/pods")]
    assert any("HTTP 422 (Unprocessable Entity" in line for line in notices)
    assert any("additional properties 'bogus' not allowed" in line for line in notices)
    assert "RunPod create request rejected; no Pod to clean up." in notices
    assert "CRITICAL" not in str(notices) and KEY not in str(notices)


def test_a_lost_create_is_found_again_in_a_v2_listing_and_deleted(credentials, monkeypatch):
    ns = _controller()
    _clock(ns)
    notices, state = [], {"name": None, "deleted": False}

    def handler(method, path, body, url):
        if method == "POST":
            state["name"] = body["name"]
            raise _http_error(url, 503, b"upstream", content_type="text/plain")
        if method == "GET" and path.startswith("/pods?"):
            return {"pods": [{"id": "owned", "name": state["name"]},
                             {"id": "other", "name": "other"}],
                    "pagination": {"nextCursor": None, "hasNextPage": False}}
        if method == "DELETE":
            assert path == "/pods/owned"
            state["deleted"] = True
            return None
        assert path == "/pods/owned" and state["deleted"]
        raise _http_error(url, 404, DOC_404)

    log = _serve(ns, monkeypatch, handler)
    ns["_notice"] = notices.append
    assert ns["main"]() == 1
    assert [(m, p) for m, p, *_ in log] == [
        ("POST", "/pods"), ("GET", "/pods?limit=100"),
        ("DELETE", "/pods/owned"), ("GET", "/pods/owned"),
    ]
    assert "RunPod termination verified." in notices


def test_the_docs_describe_v2_and_say_it_was_not_run_against_the_service():
    docs = (Path(__file__).resolve().parents[1] / "docs/backends-and-ops.md").read_text(
        encoding="utf-8")
    section = docs.split("### RunPod", 1)[1].split("### Lambda Cloud", 1)[0]
    assert "https://api.runpod.io/v2" in section
    assert "rest.runpod.io" not in section
    flat = " ".join(section.split())
    assert re.search(r"implemented against the published v2 reference; not exercised against "
                     r"the live service", flat)
