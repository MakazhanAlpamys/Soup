"""THE stub HTTP server for tests that need a real socket to talk to. One definition.

``http.server.HTTPServer.server_bind`` ends with ``self.server_name =
socket.getfqdn(host)``: a reverse lookup of the address the server was just bound
to. Nothing in a test reads ``server_name`` (the standard library uses it for the
CGI ``SERVER_NAME`` variable only), and on GitHub's ``macos-latest`` runners the
first such lookup in a process takes about 35 seconds.

Read off two green ``test (macos-latest, 3.12)`` logs of 2026-10-13 (jobs
115319900581 and 115293582800, three xdist workers each): in each of the five
workers that built a stock ``ThreadingHTTPServer``, the first test to do so took
35.1 to 35.5 s and no later one more than 2.4 s, whichever of the four files that
then used the class came first on that worker. In ``test_issue1221_*`` the test just
before it sends the same client request to a closed port, with no server, and took
0.0 s, so the wait is on the server's side, where ``getfqdn`` is the only lookup.
That is by elimination: nobody has timed the call itself on a runner.

``LoopbackHTTPServer`` is ``ThreadingHTTPServer`` without that lookup. Use it for
every stub server; ``tests/test_loopback_http.py`` fails when a test file builds the
stock class instead.
"""

from __future__ import annotations

import http.server
import socketserver


class LoopbackHTTPServer(http.server.ThreadingHTTPServer):
    """``ThreadingHTTPServer`` whose ``server_name`` is the bound host, not its FQDN."""

    def server_bind(self) -> None:
        # Deliberately skips ``HTTPServer.server_bind``: all it adds to the
        # ``TCPServer`` one is the two attributes below and the ``getfqdn`` call.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port
