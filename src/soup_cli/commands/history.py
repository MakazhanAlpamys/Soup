"""soup history — DAG-style lineage view for a named artifact."""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.markup import escape
from rich.text import Text
from rich.tree import Tree

from soup_cli.registry.store import RegistryStore, validate_name
from soup_cli.utils.registry import load_registry
from soup_cli.utils.terminal import for_terminal

console = Console()


def history(
    name: str = typer.Argument(
        ..., help="Registry entry name to trace (e.g. 'medical-chat')",
    ),
) -> None:
    """Show the lineage tree for all entries with a given name."""
    try:
        validate_name(name)
    except ValueError as exc:
        # validate_name quotes the name as typed, so strip, escape and keep ":name:" literal.
        console.print(f"[red]Invalid name: {for_terminal(exc)}[/]", emoji=False)
        raise typer.Exit(1) from exc

    with RegistryStore() as store:
        entries = store.list_by_name(name)
        if not entries:
            console.print(
                f"[red]No registry entries named '{escape(name)}'.[/]"
            )
            # v0.40.1 Part D / N6 — disambiguate model registry vs dataset
            # registry. Look up <name> in dataset registry; if found, point
            # the user at `soup data registry`.
            if _name_exists_in_dataset_registry(name):
                console.print(
                    f"[dim]A dataset named '{escape(name)}' exists in the "
                    "local dataset registry. Run `soup data registry` to see its path.[/]"
                )
            console.print(
                "[dim]`soup history` queries the *model* registry only.[/]"
            )
            raise typer.Exit(1)

        tree = Tree(f"[bold cyan]{escape(name)}[/]")
        for entry in entries:
            tag_str = ", ".join(entry.get("tags", [])) or "-"
            node = tree.add(
                f"[cyan]{escape(entry['id'])}[/] "
                f"([magenta]{escape(tag_str)}[/]) "
                f"[dim]{escape(entry['created_at'][:16])}[/]"
            )
            ancestors = store.get_ancestors(entry["id"], max_depth=5)
            if ancestors:
                anc_node = node.add("[dim]ancestors[/]")
                for anc in ancestors:
                    anc_node.add(
                        f"[cyan]{escape(anc['id'])}[/] "
                        f"[yellow]({escape(anc.get('relation', ''))})[/]"
                    )
            descendants = store.get_descendants(entry["id"], max_depth=5)
            if descendants:
                desc_node = node.add("[dim]descendants[/]")
                for desc in descendants:
                    desc_node.add(
                        f"[cyan]{escape(desc['id'])}[/] "
                        f"[yellow]({escape(desc.get('relation', ''))})[/]"
                    )
            # v0.71.4 #173 — surface attached branch pointers as a distinct
            # lineage edge so `soup adapters branch --attach-to-registry`
            # snapshots show up in the DAG view.
            branch_refs = [
                art for art in store.get_artifacts(entry["id"])
                if art.get("kind") == "branch_ref"
            ]
            if branch_refs:
                br_node = node.add("[dim]branches[/]")
                for art in branch_refs:
                    label = os.path.basename(art.get("path", "")) or "branch"
                    # The label is a file name, which is not validated like a name or a tag.
                    br_node.add(
                        Text.from_markup(
                            f"[green]branch[/] [cyan]{for_terminal(label)}[/]", emoji=False
                        )
                    )

        console.print(tree)


def _name_exists_in_dataset_registry(name: str) -> bool:
    """Look for an optional dataset hint without masking the model lookup result."""
    try:
        return name in load_registry()
    except (OSError, RuntimeError, ValueError):
        return False
