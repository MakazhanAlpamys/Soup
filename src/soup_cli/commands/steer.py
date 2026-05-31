"""v0.62.0 Part C — `soup steer` CLI command group.

Three subcommands:

* ``soup steer train`` — fit a control vector from contrastive pairs.
* ``soup steer apply`` — apply a stored vector at decode time (also exposed
  via ``soup serve --steer <name>``).
* ``soup steer list`` — list locally-stored steering vectors.

Schema + CLI surface ship in v0.62.0; the live forward-hook + per-method
fitting kernels land in v0.62.1 (mirrors v0.50.0 / v0.52.0 / v0.61.0
stub-then-live cadence).
"""

from __future__ import annotations

from typing import Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="steer",
    help=(
        "Activation steering (CAA / ITI / RepE) - inference-time "
        "intervention without retraining (v0.62.0)."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _validate_pairs_path(pairs: str) -> str:
    """Containment-check the pairs JSONL path; reject pre-placed symlinks.

    Delegates to the shared :func:`enforce_under_cwd_and_no_symlink`
    helper (centralised in v0.53.1) so the TOCTOU policy stays
    single-source-of-truth (review M1 fix). Adds the 4096-char length
    cap separately because the shared helper does not enforce one.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    if isinstance(pairs, str) and len(pairs) > 4096:
        raise typer.BadParameter("--pairs must be <= 4096 chars")
    try:
        return enforce_under_cwd_and_no_symlink(pairs, "--pairs")
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command(name="train")
def train_steer(
    base: str = typer.Option(
        ..., "--base", "-b",
        help="Base model HF id or local path.",
    ),
    method: str = typer.Option(
        "caa", "--method", "-m",
        help="Steering method: caa / iti / repe.",
    ),
    name: str = typer.Option(
        ..., "--name", "-n",
        help="Identifier for the trained vector (e.g. 'safety-v1').",
    ),
    pairs: str = typer.Option(
        ..., "--pairs", "-p",
        help="Path to JSONL of contrastive (positive, negative) prompt pairs.",
    ),
    layer: Optional[int] = typer.Option(
        None, "--layer", "-l",
        help="MLP layer index to extract the residual-stream vector from.",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help=(
            "Validate inputs + print the resolved plan; skip the "
            "deferred-live training (v0.62.1)."
        ),
    ),
    registry_id: Optional[str] = typer.Option(
        None, "--registry-id",
        help="Optional Registry entry id to attach the trained vector to.",
    ),
) -> None:
    """Train a steering vector from contrastive prompt pairs."""
    from soup_cli.utils.steering import (
        build_steering_vector,
        get_steering_method_spec,
        validate_steering_method,
        validate_steering_name,
    )

    # Validate method + name + pairs path up front so a typo fails fast
    # with a clear message before we attempt the deferred-live call.
    try:
        canonical_method = validate_steering_method(method)
        canonical_name = validate_steering_name(name)
        pairs_path = _validate_pairs_path(pairs)
    except (TypeError, ValueError, typer.BadParameter) as exc:
        console.print(f"[red]Invalid steer-train input:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    # `--base` accepts arbitrary HF ids / local paths but Typer does not
    # cap length. Match the v0.40.5 `reward_model` policy (<=512 chars +
    # null-byte rejection) so a multi-MB --base cannot bloat the Rich
    # panel render (security review L1).
    if not isinstance(base, str) or not base:
        console.print("[red]Invalid --base:[/] must be non-empty string")
        raise typer.Exit(2)
    if "\x00" in base:
        console.print("[red]Invalid --base:[/] null bytes not allowed")
        raise typer.Exit(2)
    if len(base) > 512:
        console.print("[red]Invalid --base:[/] >512 chars")
        raise typer.Exit(2)

    # Validate --registry-id early so the v0.62.1 attach path doesn't
    # see junk. Mirrors v0.61.0 Part C policy on `--registry-id`.
    if registry_id is not None:
        if not isinstance(registry_id, str) or not registry_id:
            console.print(
                "[red]Invalid --registry-id:[/] must be non-empty"
            )
            raise typer.Exit(2)
        if "\x00" in registry_id:
            console.print(
                "[red]Invalid --registry-id:[/] null bytes not allowed"
            )
            raise typer.Exit(2)
        if len(registry_id) > 256:
            console.print("[red]Invalid --registry-id:[/] >256 chars")
            raise typer.Exit(2)

    if layer is not None and (layer < 0 or layer > 2048):
        console.print(
            f"[red]Invalid --layer:[/] must satisfy 0 <= layer <= 2048, got {layer}"
        )
        raise typer.Exit(2)

    spec = get_steering_method_spec(canonical_method)
    panel_body = (
        f"Base: {escape(base)}\n"
        f"Method: {escape(canonical_method)}\n"
        f"Name: {escape(canonical_name)}\n"
        f"Pairs: {escape(pairs_path)}\n"
        f"Layer: {layer if layer is not None else 'auto'}\n"
        f"Description: {escape(spec.description)}"
    )
    console.print(
        Panel(
            panel_body,
            title=f"soup steer train ({escape(canonical_method)})",
            border_style="cyan",
        )
    )

    if plan_only:
        console.print(
            "[yellow]Plan-only mode:[/] live training deferred to v0.62.1. "
            "Inputs validated and plan rendered."
        )
        return

    try:
        build_steering_vector(
            method=canonical_method,
            name=canonical_name,
            pairs_path=pairs_path,
            layer=layer,
        )
    except NotImplementedError as exc:
        console.print(
            Panel(
                f"[yellow]Live steer-train deferred to v0.62.1.[/]\n\n{escape(str(exc))}",
                title="Deferred",
                border_style="yellow",
            )
        )
        # Match v0.61.0 Part C policy: exit code 3 distinguishes
        # "deferred / not yet shipped" from exit code 2 = "validation
        # rejection".
        raise typer.Exit(3) from exc


@app.command(name="apply")
def apply_steer(
    name: str = typer.Option(
        ..., "--name", "-n",
        help="Identifier of a stored steering vector.",
    ),
    strength: float = typer.Option(
        1.0, "--strength", "-s",
        help="Steering strength multiplier (|s| <= 10.0).",
    ),
) -> None:
    """Apply a stored steering vector at decode time (preview-only in v0.62.0)."""
    from soup_cli.utils.steering import (
        validate_steering_name,
        validate_steering_strength,
    )

    try:
        canonical_name = validate_steering_name(name)
        canonical_strength = validate_steering_strength(strength)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]Invalid steer-apply input:[/] {escape(str(exc))}")
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            (
                f"Vector: {escape(canonical_name)}\n"
                f"Strength: {canonical_strength}"
            ),
            title="soup steer apply",
            border_style="cyan",
        )
    )
    console.print(
        "[yellow]Apply path deferred to v0.62.1.[/] Use `soup serve --steer "
        f"{escape(canonical_name)}` once the live decode hook ships."
    )
    raise typer.Exit(3)


@app.command(name="list")
def list_steers() -> None:
    """List locally-stored steering vectors (Registry artifact kind ``steering_vector``)."""
    try:
        from soup_cli.registry.store import RegistryStore
    except ImportError as exc:  # pragma: no cover - registry import optional
        console.print(f"[red]Registry unavailable:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    # Use the store as a context manager so the SQLite connection closes
    # cleanly on every invocation (review M2). The store returns dicts
    # from `.list()` / `.get_artifacts()` — NOT objects (review H1).
    table = Table(title="Steering vectors", border_style="cyan")
    table.add_column("Entry")
    table.add_column("Path")
    rows = 0
    with RegistryStore() as store:
        for entry in store.list():
            for art in store.get_artifacts(entry["id"]):
                if art.get("kind") == "steering_vector":
                    table.add_row(
                        escape(str(entry.get("name", ""))),
                        escape(str(art.get("path", ""))),
                    )
                    rows += 1
    if rows == 0:
        console.print(
            "[dim]No steering_vector artifacts registered. "
            "Use `soup steer train ...` (v0.62.1+).[/]"
        )
        return
    console.print(table)
