"""``tests/_loopback_http.py``: the stub server binds without a name lookup, and is the only one.

Why the helper exists is in its docstring: on ``macos-latest`` the first stock
``ThreadingHTTPServer`` a process builds waits about 35 s in ``socket.getfqdn``. These
tests pin the two things that keep that wait out of the suite. The helper must not
look a name up, and no test file may build the stock class beside it.
"""

from __future__ import annotations

import ast
import http.server
import socket
import threading
import urllib.request
from pathlib import Path

import pytest

from tests._loopback_http import LoopbackHTTPServer

_TESTS_DIR = Path(__file__).resolve().parent
_HOME = _TESTS_DIR / "_loopback_http.py"

#: The standard-library classes whose ``server_bind`` calls ``socket.getfqdn``.
_STOCK_SERVERS = frozenset({"HTTPServer", "ThreadingHTTPServer"})

#: The only files that may name them: the helper subclasses one, and this file
#: builds one as the control below.
_MAY_NAME_THE_STOCK_CLASS = frozenset({_HOME.name, Path(__file__).name})

_LOOKUPS = ("getfqdn", "gethostbyaddr")


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server API
        body = b"hello"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        return


@pytest.fixture
def lookups(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Every reverse lookup made while the test runs, answered without the resolver."""
    asked: list[tuple[str, str]] = []

    def answer(kind: str):
        def lookup(name: str = "", *_args):
            asked.append((kind, name))
            return name if kind == "getfqdn" else (name, [], [name])

        return lookup

    for kind in _LOOKUPS:
        monkeypatch.setattr(socket, kind, answer(kind))
    return asked


def _stock_class_mentions(source: str) -> list[int]:
    """Lines of ``source`` whose CODE names a stock server class.

    A name, an attribute or an import alias: that covers building one, subclassing
    one and importing one. A comment or a string that mentions the class is not code.
    """
    lines = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in _STOCK_SERVERS:
            lines.append(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr in _STOCK_SERVERS:
            lines.append(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.extend(
                node.lineno for alias in node.names
                if alias.name.rpartition(".")[2] in _STOCK_SERVERS
            )
    return sorted(set(lines))


class TestTheHelperLooksNothingUp:
    def test_binding_makes_no_reverse_lookup(self, lookups):
        server = LoopbackHTTPServer(("127.0.0.1", 0), _Hello)
        try:
            assert lookups == []
            assert server.server_name == "127.0.0.1"
            assert server.server_port == server.server_address[1] != 0
        finally:
            server.server_close()

    def test_control_the_stock_class_does(self, lookups):
        """The recorder sees the lookup the helper removes.

        If this one fails, ``HTTPServer.server_bind`` no longer calls ``getfqdn`` on
        this Python, and the helper has nothing left to remove.
        """
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hello)
        try:
            assert lookups == [("getfqdn", "127.0.0.1")]
        finally:
            server.server_close()

    def test_it_still_serves_a_request(self):
        server = LoopbackHTTPServer(("127.0.0.1", 0), _Hello)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            # No proxy from the environment between the test and its own server.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            url = f"http://127.0.0.1:{server.server_address[1]}/"
            with opener.open(url, timeout=10) as response:
                assert (response.status, response.read()) == (200, b"hello")
        finally:
            server.shutdown()
            server.server_close()

    def test_it_is_the_threading_server(self):
        assert issubclass(LoopbackHTTPServer, http.server.ThreadingHTTPServer)
        assert LoopbackHTTPServer.daemon_threads is True


class TestNoTestFileBuildsTheStockServer:
    def test_only_the_helper_names_the_stock_classes(self):
        offenders = []
        for path in sorted(_TESTS_DIR.rglob("*.py")):
            if path.name in _MAY_NAME_THE_STOCK_CLASS:
                continue
            source = path.read_text(encoding="utf-8")
            if "HTTPServer" not in source:  # cheap: nearly every file stops here
                continue
            offenders += [
                f"{path.relative_to(_TESTS_DIR).as_posix()}:{line}"
                for line in _stock_class_mentions(source)
            ]
        assert offenders == [], (
            "build stub servers from tests._loopback_http.LoopbackHTTPServer; the stock "
            "class waits about 35 s in socket.getfqdn on macos-latest (see that module): "
            f"{offenders}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            "from http.server import ThreadingHTTPServer\n",
            "from http.server import HTTPServer as Server\n",
            "import http.server\nserver = http.server.ThreadingHTTPServer(('::1', 0), None)\n",
            "import http.server\nserver = http.server.HTTPServer(('::1', 0), None)\n",
            "import http.server\nclass Mine(http.server.ThreadingHTTPServer):\n    pass\n",
            "import http.server as hs\nfactory = hs.HTTPServer\n",
        ],
    )
    def test_the_scan_sees_each_way_of_reaching_one(self, source):
        assert _stock_class_mentions(source) != []

    @pytest.mark.parametrize(
        "source",
        [
            "from tests._loopback_http import LoopbackHTTPServer\n"
            "server = LoopbackHTTPServer(('127.0.0.1', 0), None)\n",
            "# a ThreadingHTTPServer would wait in getfqdn here\n",
            "NOTE = 'HTTPServer.server_bind calls socket.getfqdn'\n",
            "import http.server\nclass Handler(http.server.BaseHTTPRequestHandler):\n    pass\n",
        ],
    )
    def test_the_scan_leaves_the_helper_and_plain_mentions_alone(self, source):
        assert _stock_class_mentions(source) == []
