"""soup bom — CycloneDX ML-BOM + SPDX AI emitter (v0.59.0 Part A)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape

from soup_cli.utils.bom import BomEntry, render_bom, write_bom

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    help="Emit CycloneDX ML-BOM + SPDX AI BOMs from registry entries (v0.59.0).",
)


@app.command("emit")
def emit_cmd(
    name: str = typer.Option(..., "--name", help="Model / adapter name."),
    version: str = typer.Option("0.1.0", "--version", help="Model version string."),
    base_model: str = typer.Option(
        ..., "--base-model", help="HF repo id of the base model.",
    ),
    base_sha: str = typer.Option(..., "--base-sha", help="SHA-256 of the base model."),
    config_sha: str = typer.Option(
        ..., "--config-sha", help="SHA-256 of the resolved soup.yaml config.",
    ),
    data_sha: Optional[str] = typer.Option(
        None, "--data-sha", help="SHA-256 of the training dataset.",
    ),
    task: str = typer.Option("sft", "--task", help="Training task (sft / dpo / grpo / ...)."),
    license_id: Optional[str] = typer.Option(
        None, "--license", help="SPDX license id (e.g. apache-2.0, mit).",
    ),
    fmt: str = typer.Option(
        "cyclonedx", "--format", "-f",
        help="Output BOM format: cyclonedx | spdx | both.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help=("Output file path (cwd-contained). When --format=both, "
              "this is the prefix and Soup writes <prefix>.cdx.json + "
              "<prefix>.spdx.json."),
    ),
) -> None:
    """Emit a CycloneDX + SPDX BOM from CLI-supplied SHAs."""
    fmt_lc = fmt.lower()
    if fmt_lc not in {"cyclonedx", "spdx", "both"}:
        console.print(
            f"[red]Unsupported --format: {escape(fmt)} "
            "(use cyclonedx | spdx | both)[/]"
        )
        raise typer.Exit(2)

    try:
        entry = BomEntry(
            name=name,
            version=version,
            base_model=base_model,
            base_sha=base_sha,
            config_sha=config_sha,
            data_sha=data_sha,
            task=task,
            license=license_id,
            parents=(),
            artifacts=(),
            created_at=datetime.now(tz=timezone.utc).isoformat(),
        )
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid BOM input: {escape(str(exc))}[/]")
        raise typer.Exit(2)

    if fmt_lc == "both":
        if output is None:
            console.print(
                "[red]--format=both requires --output prefix (writes "
                "<prefix>.cdx.json + <prefix>.spdx.json)[/]"
            )
            raise typer.Exit(2)
        try:
            cdx_path = write_bom(entry, "cyclonedx", output + ".cdx.json")
            spdx_path = write_bom(entry, "spdx", output + ".spdx.json")
        except (TypeError, ValueError) as exc:
            console.print(f"[red]Write failed: {escape(str(exc))}[/]")
            raise typer.Exit(2)
        console.print(
            f"[green]Wrote CycloneDX BOM[/] -> {escape(cdx_path)}\n"
            f"[green]Wrote SPDX BOM[/] -> {escape(spdx_path)}"
        )
        return

    if output is None:
        # Print to stdout.
        console.print(render_bom(entry, fmt_lc))
        return

    try:
        written = write_bom(entry, fmt_lc, output)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Write failed: {escape(str(exc))}[/]")
        raise typer.Exit(2)
    console.print(
        f"[green]Wrote BOM ({fmt_lc})[/] -> {escape(written)}"
    )
