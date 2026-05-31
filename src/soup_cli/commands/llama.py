"""v0.44.0 Part D — `soup llama <subcommand> [args...]` proxy.

Forwards to a llama.cpp binary on PATH (`llama-cli`, `llama-mtmd-cli`,
`llama-gguf-split`, `llama-server`, `llama-quantize`). Closed allowlist;
no shell.
"""

from __future__ import annotations

import os
import subprocess  # noqa: S404 — list-args invocation only
from typing import Callable

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from soup_cli.utils.llama_proxy import (
    build_argv,
    known_subcommands,
    resolve,
)

console = Console()

# A standalone Typer sub-app so `soup llama --help` lists the subcommands.
app = typer.Typer(
    name="llama",
    help="Proxy to llama.cpp binaries (llama-cli / llama-server / etc).",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None and not ctx.args:
        # Show the supported subcommands.
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Subcommand")
        table.add_column("Binary")
        for name, binary in known_subcommands().items():
            table.add_row(escape(name), escape(binary))
        console.print(table)


# Env vars that the llama.cpp binaries legitimately consume. We deliberately
# DROP everything else (HF_TOKEN / OPENAI_API_KEY / ANTHROPIC_API_KEY / etc)
# so the wrapped binary can't exfiltrate Soup-issued credentials.
_LLAMA_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "USERPROFILE",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TERM",
        "COLORTERM",
        # llama.cpp-recognised toggles.
        "LLAMA_CPP_HOME",
        "GGML_CUDA",
        "GGML_METAL",
        "OMP_NUM_THREADS",
    }
)


def _filtered_env() -> dict:
    """Return a minimal env for the child binary."""
    return {k: v for k, v in os.environ.items() if k in _LLAMA_ENV_ALLOWLIST}


def _run(subcommand: str, raw_args: list[str]) -> None:
    try:
        invocation = resolve(subcommand, raw_args)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{escape(str(exc))}[/]")
        raise typer.Exit(code=2) from exc
    argv = build_argv(invocation)
    try:
        # Inherit stdio so the user gets full llama.cpp output streams.
        # Env filtered to the allowlist above to avoid leaking secrets.
        result = subprocess.run(  # noqa: S603 — list args, no shell
            argv,
            check=False,
            env=_filtered_env(),
        )
    except OSError as exc:
        console.print(
            f"[red]Failed to launch {escape(invocation.binary)}: "
            f"{escape(type(exc).__name__)}[/]"
        )
        raise typer.Exit(code=1) from exc
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


def _make_proxy(subcommand: str) -> Callable[..., None]:
    def _proxy(
        ctx: typer.Context,
        args: list[str] = typer.Argument(
            None,
            help=f"Args forwarded to {known_subcommands()[subcommand]}.",
        ),
    ) -> None:
        _run(subcommand, list(args or []) + list(ctx.args or []))

    _proxy.__name__ = f"_{subcommand.replace('-', '_')}_proxy"
    _proxy.__doc__ = (
        f"Forward args to llama.cpp binary {known_subcommands()[subcommand]}."
    )
    return _proxy


for _sub in known_subcommands():
    app.command(
        name=_sub,
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )(_make_proxy(_sub))
