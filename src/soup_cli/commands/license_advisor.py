"""`soup license-advisor` — pick a license-clean base for a deploy target.

v0.64.0 Part F. Composes with v0.60 Part E ``license_matrix.check_license_compat``
which gates ``soup adapters merge``.
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from soup_cli.utils.license_advisor import (
    advise_license_for_target,
    flag_downstream_risk,
)
from soup_cli.utils.terminal import for_terminal, strip_control

console = Console()


def license_advisor_cmd(
    target: str = typer.Option(
        ...,
        "--target",
        "-t",
        help="Deploy target: b2c | defense | embedded.",
    ),
    license_id: Optional[str] = typer.Option(
        None,
        "--license",
        help="Optional: per-license risk check (e.g. apache-2.0, llama-3).",
    ),
    monthly_active_users: int = typer.Option(
        0,
        "--monthly-active-users",
        "--mau",
        help="Expected MAU for the per-license risk check (default 0).",
    ),
) -> None:
    """Recommend a license-clean base + flag downstream risk (v0.64.0)."""
    try:
        rec = advise_license_for_target(target)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(2) from exc

    table = Table(title=f"License advisor — target: {rec.target}")
    table.add_column("Category")
    table.add_column("Licenses")
    table.add_row(
        "[green]Recommended[/]",
        ", ".join(escape(lic) for lic in rec.recommended_licenses),
    )
    table.add_row(
        "[red]Forbidden[/]",
        ", ".join(escape(lic) for lic in rec.forbidden_licenses),
    )
    console.print(table)
    console.print(
        Panel(escape(rec.reason), title="Reason", border_style="dim")
    )

    if license_id:
        try:
            risk = flag_downstream_risk(
                license_id=license_id,
                target=target,
                monthly_active_users=monthly_active_users,
            )
        except (TypeError, ValueError) as exc:
            # Only type names, fixed text and numbers: the license id and the target were
            # validated before, and a quoted value is a repr().
            console.print(f"[red]{escape(str(exc))}[/]")
            raise typer.Exit(2) from exc

        border = {
            "ok": "green",
            "warn": "yellow",
            "block": "red",
        }.get(risk.severity, "yellow")
        # The reason quotes the license with repr(); the title is the license as typed.
        console.print(
            Panel(
                Text.from_markup(
                    f"[bold]severity: {escape(risk.severity)}[/]\n"
                    f"{for_terminal(risk.reason)}",
                    emoji=False,
                ),
                title=Text(f"Risk check: {strip_control(license_id)}"),
                border_style=border,
            )
        )
        if risk.severity == "block":
            raise typer.Exit(3)


__all__ = ["license_advisor_cmd"]
