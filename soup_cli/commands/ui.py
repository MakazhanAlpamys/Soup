"""soup ui — local web interface for managing experiments and training."""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()


def ui(
    port: int = typer.Option(
        7860,
        "--port",
        "-p",
        help="Port to serve on",
    ),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Host to bind to",
    ),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Don't open browser automatically",
    ),
    show_token: bool = typer.Option(
        False,
        "--show-token",
        help="Print the auth token and exit (for scripting)",
    ),
    public: bool = typer.Option(
        False,
        "--public",
        help=(
            "Bind to 0.0.0.0 so the UI is reachable from a phone on the "
            "same LAN. Prints a phone-scannable URL + ASCII QR at startup. "
            "Loopback HTTP is auto-upgraded to require a Bearer token in "
            "the query string. v0.53.9 #95."
        ),
    ),
    auth_token: Optional[str] = typer.Option(
        None,
        "--auth-token",
        help=(
            "Override the auto-generated Bearer token (16-128 urlsafe "
            "base64 chars). Useful for stable phone bookmarks. "
            "When omitted, a fresh token is generated each startup."
        ),
    ),
):
    """Launch the Soup Web UI.

    A Bearer auth token is auto-generated at startup and printed to the console.
    Mutating API endpoints (POST/DELETE) require 'Authorization: Bearer <token>'.

    `--public` exposes the server on 0.0.0.0 for phone-on-LAN access. The
    auth token is embedded in a phone-scannable URL + ASCII QR code so a
    phone scan lands authenticated.
    """
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        console.print(
            "[red]FastAPI/uvicorn not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[ui]'[/]"
        )
        raise typer.Exit(1)

    from soup_cli.ui.app import create_app, get_auth_token, set_auth_token

    # v0.53.9 #95 — operator-supplied auth token override.
    # `set_auth_token` validates via `qr_url.validate_token`, so we don't
    # double-validate here.
    if auth_token is not None:
        try:
            set_auth_token(auth_token)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]--auth-token:[/] {exc}")
            raise typer.Exit(1) from exc

    # --public binds to 0.0.0.0 unless the operator already set a custom host.
    if public and host == "127.0.0.1":
        host = "0.0.0.0"

    token = get_auth_token()

    if show_token:
        console.print(token)
        raise typer.Exit()

    app = create_app(host=host, port=port)

    url = f"http://{host}:{port}"

    panel_body = (
        f"URL:    [bold]{url}[/]\n"
        f"Token:  [bold]{token}[/]\n\n"
        f"Mutating API endpoints require:\n"
        f"  [dim]Authorization: Bearer {token}[/]\n\n"
        f"Pages:\n"
        f"  [bold]Dashboard[/]      - View experiments, loss charts, system info\n"
        f"  [bold]New Training[/]   - Create config from templates, start training\n"
        f"  [bold]Data Explorer[/]  - Browse and inspect datasets\n"
        f"  [bold]Model Chat[/]     - Chat with a running inference server\n\n"
        f"Press [bold]Ctrl+C[/] to stop."
    )

    console.print(
        Panel(
            panel_body,
            title="[bold green]Soup Web UI[/]",
        )
    )

    # v0.53.9 #95 — phone-visible URL + QR for --public.
    if public:
        import socket

        from soup_cli.utils.qr_url import render_qr_ascii

        # Derive the LAN IP a phone on the same Wi-Fi can reach. Falls
        # back to 127.0.0.1 if DNS resolution fails — operator gets a
        # warning explaining the QR is loopback-only in that case.
        lan_ip: str = "127.0.0.1"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                # Connect to a non-routed address; OS picks the egress
                # interface and we read its bound IP. No packets sent.
                sock.settimeout(0.1)
                sock.connect(("10.255.255.255", 1))
                lan_ip = sock.getsockname()[0]
        except (OSError, socket.error):
            lan_ip = "127.0.0.1"

        # Plain HTTP over LAN is intentional: this is local-network only
        # and the Bearer token gates mutating endpoints. We assemble the
        # URL directly because `build_phone_url` rejects LAN HTTP by
        # design (its threat model is stricter than ours here).
        phone_url = f"http://{lan_ip}:{port}/?token={token}"
        if lan_ip == "127.0.0.1":
            console.print(
                "[yellow]--public:[/] Could not detect LAN IP; "
                f"falling back to loopback. Phone scan will only "
                f"work from the same host: {phone_url}"
            )
        else:
            console.print(
                f"\n[bold green]Phone URL:[/] {phone_url}\n"
                "[dim]Scan the QR code below from your phone camera:[/]\n"
            )
        qr_ascii = render_qr_ascii(phone_url)
        if qr_ascii is None:
            console.print(
                "[dim]Install 'qrcode' to print an ASCII QR: "
                "pip install qrcode[/]"
            )
        else:
            console.print(qr_ascii)

    # Open browser
    if not no_browser:
        import threading
        import webbrowser

        def _open():
            import time

            time.sleep(1)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")
