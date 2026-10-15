"""Tests for Web UI Chat Upgrade — SSE proxy, SSRF validation, param bounds.

No test in this file reaches the network. ``chat_send`` proxies an allowed request
with ``httpx.stream``, and three tests used to let that call go out for real: to
``localhost:8000``, to ``api.example.com`` and to ``127.0.0.2:8000``. The last one
took 75.3 s in each of two green ``test (macos-latest, 3.12)`` logs of 2026-10-13
(jobs 115319900581 and 115293582800). The likely reason, not read off a runner: macOS
puts 127.0.0.1 alone on ``lo0``, so a connect to another 127.x address is not refused
and waits out the default TCP connect timeout, 75 s, well inside the 120 s the
handler gives ``httpx``.

So the "allowed" tests answer the proxied call from ``upstream``, an in-process
``httpx.MockTransport``, and say which URL the request was forwarded to. The
endpoint check itself is never patched. ``refused_network`` holds for every test
here: a name lookup or a socket connect raises at once, and a test that tried one
fails with where it was going.
"""

import contextlib
import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

_UPSTREAM_REPLY = 'data: {"choices":[{"delta":{"content":"pong"}}]}\n\ndata: [DONE]\n\n'

# What the browser receives when the proxied reply above went through `chat_send`.
_PROXIED_REPLY = 'data: {"delta": "pong"}\n\ndata: {"done": true}\n\n'


def _auth_headers():
    """Return auth headers with the current UI token."""
    from soup_cli.ui.app import get_auth_token
    return {"Authorization": f"Bearer {get_auth_token()}"}


class _NetworkRefusedError(OSError):
    """Raised in place of a name lookup or a socket connect."""


@contextlib.contextmanager
def _network_refused():
    """Refuse every name lookup and every socket connect; fail on exit if any was tried.

    An ``OSError``, so a client library reports it as the connection error it
    stands for and ``chat_send`` takes its real "Connection failed" branch. That
    branch swallows the exception, which is why the attempts are also counted and
    checked when the block is over. Code that asks for a refusal on purpose clears
    the list itself. ``socket.getfqdn`` swallows it too, and ``gethostbyaddr`` is
    the call it makes.

    One exemption. ``socket.socketpair`` on Windows is a listening socket and a
    ``connect`` to it on 127.0.0.1, and every event loop makes one for its
    self-pipe. The test client starts a loop per request, so a connect made inside
    ``socketpair`` goes through. It cannot leave the process. The exemption is
    per thread and ends when ``socketpair`` returns. Nowhere else does it exist:
    Linux and macOS build the pair without a connect.

    This is a helper rather than only a fixture so that the guard itself can be
    tested (``TestThisFileIsOffline``).
    """
    attempts = []
    plumbing = threading.local()
    real_socketpair = socket.socketpair
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(kind, target):
        attempts.append((kind, target))
        raise _NetworkRefusedError(f"{kind}({target!r}) refused: tests/test_ui_chat.py is offline")

    def socketpair(*args, **kwargs):
        plumbing.depth = getattr(plumbing, "depth", 0) + 1
        try:
            return real_socketpair(*args, **kwargs)
        finally:
            plumbing.depth -= 1

    def connect(sock, address):
        if getattr(plumbing, "depth", 0):
            return real_connect(sock, address)
        return refuse("connect", address)

    def connect_ex(sock, address):
        if getattr(plumbing, "depth", 0):
            return real_connect_ex(sock, address)
        return refuse("connect_ex", address)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(socket, "socketpair", socketpair)
        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
        monkeypatch.setattr(
            socket, "create_connection",
            lambda address, *_args, **_kwargs: refuse("create_connection", address),
        )
        for lookup in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
            monkeypatch.setattr(
                socket, lookup,
                lambda host, *_args, _kind=lookup, **_kwargs: refuse(_kind, host),
            )
        yield attempts
    assert attempts == [], f"this test tried to reach the network: {attempts}"


@pytest.fixture(autouse=True)
def refused_network():
    """``_network_refused`` for every test in this file; yields the attempts list."""
    with _network_refused() as attempts:
        yield attempts


class _Upstream:
    """The model server ``chat_send`` proxies to, living in this process.

    It replaces ``httpx.stream``, the one outbound call the handler makes, and
    answers from an ``httpx.MockTransport``: the handler still reads a real
    ``httpx.Response``, and no socket is opened. A request the endpoint check
    refuses never gets this far, so ``urls`` is how a test tells "allowed and
    forwarded there" from "merely not a 400".
    """

    def __init__(self, httpx):
        self._httpx = httpx
        self.urls = []

    @contextlib.contextmanager
    def stream(self, method, url, *, json, headers, timeout):
        self.urls.append(url)
        transport = self._httpx.MockTransport(
            lambda request: self._httpx.Response(200, text=_UPSTREAM_REPLY)
        )
        with self._httpx.Client(transport=transport, timeout=timeout) as client:
            with client.stream(method, url, json=json, headers=headers) as response:
                yield response


@pytest.fixture
def upstream(monkeypatch):
    httpx = pytest.importorskip("httpx")
    fake = _Upstream(httpx)
    monkeypatch.setattr(httpx, "stream", fake.stream)
    return fake


class TestChatEndpointExists:
    """Test /api/chat/send endpoint registration."""

    def test_chat_send_endpoint_exists(self):
        """Route /api/chat/send should be registered."""
        try:
            import fastapi  # noqa: F401
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        app = create_app()
        routes = [route.path for route in app.routes]
        assert "/api/chat/send" in routes


class TestChatSSRFProtection:
    """Test SSRF protection on chat endpoint."""

    def test_rejects_non_localhost_http(self):
        """Chat endpoint should reject non-localhost HTTP URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://evil.com:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_allows_localhost_http(self, upstream):
        """Chat endpoint should allow localhost HTTP URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
            },
            headers=_auth_headers(),
        )
        # Not 400 (SSRF rejection): the request went on to the model server.
        assert response.status_code == 200, response.text
        assert upstream.urls == ["http://localhost:8000/v1/chat/completions"]
        assert response.text == _PROXIED_REPLY

    def test_allows_https_remote(self, upstream):
        """Chat endpoint should allow HTTPS remote URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "https://api.example.com",
            },
            headers=_auth_headers(),
        )
        # Not 400 (SSRF rejection): the request went on to the remote server.
        assert response.status_code == 200, response.text
        assert upstream.urls == ["https://api.example.com/v1/chat/completions"]
        assert response.text == _PROXIED_REPLY


class TestChatParamBounds:
    """Test parameter validation on chat endpoint."""

    def test_max_tokens_capped(self):
        """max_tokens should be capped at 16384."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "max_tokens": 99999,
            },
            headers=_auth_headers(),
        )
        # Should be rejected (exceeds cap) — 422 from Pydantic or 400
        assert response.status_code in (400, 422)

    def test_temperature_bounded(self):
        """temperature should be bounded 0.0-2.0."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "temperature": 5.0,
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400 or response.status_code == 422

    def test_top_p_bounded(self):
        """top_p should be bounded 0.0-1.0."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
                "top_p": 2.0,
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400 or response.status_code == 422


class TestChatInvalidScheme:
    """Test rejection of invalid URL schemes."""

    def test_rejects_ftp_scheme(self):
        """Chat endpoint should reject ftp:// URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "ftp://evil.com/model",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_rejects_ipv6_mapped_loopback_alias(self):
        """Chat endpoint should reject IPv6-mapped private addresses."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://192.168.1.1:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_allows_127_loopback_range(self, upstream):
        """Chat endpoint should allow 127.x.x.x loopback addresses."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://127.0.0.2:8000",
            },
            headers=_auth_headers(),
        )
        # 127.0.0.2 is loopback — NOT rejected as SSRF, and forwarded to that host
        assert response.status_code == 200, response.text
        assert upstream.urls == ["http://127.0.0.2:8000/v1/chat/completions"]
        assert response.text == _PROXIED_REPLY

    def test_rejects_file_scheme(self):
        """Chat endpoint should reject file:// URLs."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "file:///etc/passwd",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400


class TestChatAuth:
    """Test chat endpoint auth requirements."""

    def test_requires_auth(self):
        """Chat send (POST) should require Bearer auth."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://localhost:8000",
            },
        )
        assert response.status_code == 401


class TestChatValidation:
    """Test chat request validation."""

    def test_empty_messages_rejected(self):
        """Empty messages array should return 400."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [],
                "endpoint": "http://localhost:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 400

    def test_returns_event_stream(self):
        """Chat endpoint should return text/event-stream content type."""
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        # Mock httpx to avoid real connection
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = iter([
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: [DONE]',
        ])
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *a: None

        with patch("httpx.stream", return_value=mock_response):
            with client.stream(
                "POST",
                "/api/chat/send",
                json={
                    "messages": [{"role": "user", "content": "hi"}],
                    "endpoint": "http://localhost:8000",
                },
                headers=_auth_headers(),
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]


class TestThisFileIsOffline:
    """``refused_network`` has to stop what a real client does, or it proves nothing."""

    def test_a_lookup_or_a_connect_is_refused(self, refused_network):
        with pytest.raises(_NetworkRefusedError):
            socket.getaddrinfo("api.example.com", 443)
        with pytest.raises(_NetworkRefusedError):
            socket.gethostbyname("localhost")
        with pytest.raises(_NetworkRefusedError):
            socket.create_connection(("127.0.0.2", 8000), timeout=1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(_NetworkRefusedError):
                sock.connect(("127.0.0.2", 8000))
            with pytest.raises(_NetworkRefusedError):
                sock.connect_ex(("127.0.0.2", 8000))
        finally:
            sock.close()
        assert refused_network == [
            ("getaddrinfo", "api.example.com"),
            ("gethostbyname", "localhost"),
            ("create_connection", ("127.0.0.2", 8000)),
            ("connect", ("127.0.0.2", 8000)),
            ("connect_ex", ("127.0.0.2", 8000)),
        ]
        refused_network.clear()  # asked for on purpose

    def test_control_an_unpatched_proxy_call_is_stopped_at_the_socket(self, refused_network):
        """Without ``upstream`` the real ``httpx.stream`` runs, and gets nowhere.

        This is the request that waited 75 s on macOS. If it ever passes the guard,
        the list is empty and this test says so.
        """
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            pytest.skip("FastAPI not installed")

        from soup_cli.ui.app import create_app

        client = TestClient(create_app())
        response = client.post(
            "/api/chat/send",
            json={
                "messages": [{"role": "user", "content": "hi"}],
                "endpoint": "http://127.0.0.2:8000",
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200, response.text
        assert response.text == 'data: {"error": "Connection failed"}\n\n'
        assert refused_network != []
        refused_network.clear()  # asked for on purpose

    def test_the_event_loop_can_still_build_its_self_pipe(self):
        """The one exemption: a ``socketpair`` is local plumbing, also on Windows."""
        left, right = socket.socketpair()
        try:
            left.sendall(b"x")
            assert right.recv(1) == b"x"
        finally:
            left.close()
            right.close()

    # What follows tests the guard itself: its teardown, its scope and its edges.
    # Every target below is local, so a guard that let a call through would fail
    # the test and still not leave the machine.

    def test_the_guard_applies_to_a_test_that_does_not_ask_for_it(self, request):
        """``autouse``: the tests above that never name the fixture are covered too."""
        assert "refused_network" in request.fixturenames

    @pytest.mark.parametrize(
        ("lookup", "args"),
        [
            ("getaddrinfo", ("localhost", 443)),
            ("gethostbyname", ("localhost",)),
            ("gethostbyname_ex", ("localhost",)),
            ("gethostbyaddr", ("127.0.0.1",)),
        ],
    )
    def test_every_name_lookup_is_refused(self, refused_network, lookup, args):
        with pytest.raises(_NetworkRefusedError):
            getattr(socket, lookup)(*args)
        assert refused_network == [(lookup, args[0])]
        refused_network.clear()  # asked for on purpose

    def test_getfqdn_is_refused_through_gethostbyaddr(self, refused_network):
        """The call behind ``HTTPServer.server_bind``, which swallows the refusal.

        ``getfqdn`` falls back to the name it was given, so only the attempt log
        shows that it went to the resolver.
        """
        assert socket.getfqdn("127.0.0.1") == "127.0.0.1"
        assert refused_network == [("gethostbyaddr", "127.0.0.1")]
        refused_network.clear()  # asked for on purpose

    def test_a_refusal_that_the_code_swallowed_still_fails_the_block(self):
        """The reason for the attempt list: the code under test may catch the OSError."""
        with pytest.raises(AssertionError, match=r"tried to reach the network.*getaddrinfo"):
            with _network_refused():
                with contextlib.suppress(OSError):
                    socket.getaddrinfo("localhost", 443)

    def test_a_block_that_stays_offline_passes(self):
        with _network_refused() as attempts:
            pass
        assert attempts == []

    def test_the_exemption_ends_when_socketpair_returns(self, refused_network):
        """A connect after a ``socketpair`` is refused, on Windows and everywhere else.

        The target is a listener of this test's own, so a connect that got through
        would succeed at once instead of waiting on a refused port.
        """
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            address = listener.getsockname()
            left, right = socket.socketpair()
            left.close()
            right.close()
            with socket.socket() as client:
                with pytest.raises(_NetworkRefusedError):
                    client.connect(address)
                with pytest.raises(_NetworkRefusedError):
                    client.connect_ex(address)
        assert refused_network == [("connect", address), ("connect_ex", address)]
        refused_network.clear()  # asked for on purpose

    def test_the_exemption_ends_when_socketpair_raises(self):
        """The same, for a ``socketpair`` that fails (no free port, out of descriptors)."""

        def failing_socketpair(*_args, **_kwargs):
            raise OSError("no socketpair")

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(socket, "socketpair", failing_socketpair)
            with _network_refused() as attempts:
                with pytest.raises(OSError, match="no socketpair"):
                    socket.socketpair()
                with socket.socket() as client:
                    with pytest.raises(_NetworkRefusedError):
                        client.connect(("127.0.0.2", 8000))
                assert attempts == [("connect", ("127.0.0.2", 8000))]
                attempts.clear()  # asked for on purpose

    def test_the_exemption_belongs_to_the_thread_inside_socketpair(self):
        """A second thread's connect is refused while the first is inside ``socketpair``.

        ``blocked_socketpair`` stands in for the real one and holds the worker
        there until the main thread has made its attempt: two events, no sleeps.
        It is installed before the guard, because the guard wraps whatever
        ``socket.socketpair`` is when it starts.
        """
        inside, release = threading.Event(), threading.Event()

        def blocked_socketpair(*_args, **_kwargs):
            inside.set()
            release.wait(timeout=10)
            return None, None

        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(socket, "socketpair", blocked_socketpair)
            with _network_refused() as attempts:
                worker = threading.Thread(target=socket.socketpair)
                worker.start()
                try:
                    assert inside.wait(timeout=10), "the worker never entered socketpair"
                    with socket.socket() as client:
                        with pytest.raises(_NetworkRefusedError):
                            client.connect(("127.0.0.2", 8000))
                finally:
                    release.set()
                    worker.join(timeout=10)
                assert attempts == [("connect", ("127.0.0.2", 8000))]
                attempts.clear()  # asked for on purpose
