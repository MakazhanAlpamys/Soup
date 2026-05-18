"""soup attest — in-toto + SLSA-3 attestation CLI (v0.59.0 Part B)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.attest import (
    AttestationStatement,
    render_attestation,
    sign_attestation,
    write_attestation,
)

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="In-toto + SLSA-3 attestations per Soup Can stage (v0.59.0).",
)


@app.command("emit")
def emit_cmd(
    stage: str = typer.Option(
        ..., "--stage",
        help="Stage: extract / train / eval / export / publish.",
    ),
    subject_name: str = typer.Option(..., "--subject", help="Artefact name."),
    subject_sha: str = typer.Option(..., "--sha", help="64-hex SHA-256 of the artefact."),
    builder_id: str = typer.Option(
        "soup-cli", "--builder",
        help="Builder identity (default: soup-cli).",
    ),
    invocation: Optional[str] = typer.Option(
        None, "--invocation",
        help="Free-form invocation marker (e.g. command line).",
    ),
    sign_backend: str = typer.Option(
        "unsigned", "--sign",
        help="Signature backend: unsigned (default; sigstore/ed25519 in v0.59.1).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file path (cwd-contained).",
    ),
) -> None:
    """Emit a per-stage in-toto/SLSA-3 attestation."""
    try:
        st = AttestationStatement(
            stage=stage,
            subject_name=subject_name,
            subject_sha256=subject_sha,
            builder_id=builder_id,
            invocation={"command": invocation or ""},
            materials=(),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid attestation: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    text = render_attestation(st)

    try:
        sig = sign_attestation(text.encode("utf-8"), backend=sign_backend)
    except NotImplementedError as exc:
        console.print(f"[yellow]Signing deferred: {escape(str(exc))}[/]")
        sig = {"signature": "", "backend": "unsigned"}
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Sign failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    if output is None:
        console.print(text)
        console.print(f"[dim]signature backend: {escape(sig['backend'])}[/]")
        return

    try:
        written = write_attestation(st, output)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Write failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)
    console.print(
        f"[green]Wrote attestation[/] -> {escape(written)} "
        f"[dim](signature: {escape(sig['backend'])})[/]"
    )
