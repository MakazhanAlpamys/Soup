"""soup ui — local web interface for managing experiments and training."""

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
):
    """Launch the Soup Web UI for managing experiments and training."""
    try:
        import uvicorn  # noqa: F401
        from fastapi import FastAPI  # noqa: F401
    except ImportError:
        console.print(
            "[red]FastAPI/uvicorn not installed.[/]\n"
            "Install with: [bold]pip install 'soup-cli[ui]'[/]"
        )
        raise typer.Exit(1)

    from soup_cli.ui.app import create_app, get_auth_token

    app = create_app(host=host, port=port)
    token = get_auth_token()

    url = f"http://{host}:{port}"

    console.print(
        Panel(
            f"URL:    [bold]{url}[/]\n"
            f"Token:  [bold]{token}[/]\n\n"
            f"Mutating API endpoints require:\n"
            f"  [dim]Authorization: Bearer {token}[/]\n\n"
            f"Pages:\n"
            f"  [bold]Dashboard[/]      - View experiments, loss charts, system info\n"
            f"  [bold]New Training[/]   - Create config from templates, start training\n"
            f"  [bold]Data Explorer[/]  - Browse and inspect datasets\n"
            f"  [bold]Model Chat[/]     - Chat with a running inference server\n\n"
            f"Press [bold]Ctrl+C[/] to stop.",
            title="[bold green]Soup Web UI[/]",
        )
    )

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
