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
from rich.text import Text

from soup_cli.utils.terminal import strip_control

console = Console()


def _print_error(label: str, exc: BaseException) -> None:
    # A Text is not parsed as markup and has no emoji codes, so the message is shown as typed.
    console.print(Text.assemble((label, "red"), " ", strip_control(exc)))

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
        help="Decoder layer index to extract the residual-stream vector from "
        "(default: the middle layer).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o",
        help="Directory to write the steering vector to (default: "
        "./steering/<name>). Cwd-contained.",
    ),
    device: Optional[str] = typer.Option(
        None, "--device",
        help="torch device (cpu / cuda). Defaults to CUDA when available.",
    ),
    top_k: int = typer.Option(
        8, "--top-k", "-k",
        help="ITI only: number of attention heads to intervene on (1-256).",
    ),
    plan_only: bool = typer.Option(
        False, "--plan-only",
        help="Validate inputs + print the resolved plan; skip the live fit.",
    ),
    registry_id: Optional[str] = typer.Option(
        None, "--registry-id",
        help=(
            "Registry entry id to attach the trained vector file to, as a "
            "steering_vector artifact. Exits 1 when the attach fails."
        ),
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
        _print_error("Invalid steer-train input:", exc)
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
    if top_k < 1 or top_k > 256:
        console.print(
            f"[red]Invalid --top-k:[/] must satisfy 1 <= top_k <= 256, got {top_k}"
        )
        raise typer.Exit(2)

    spec = get_steering_method_spec(canonical_method)
    panel_body = Text(
        f"Base: {strip_control(base)}\n"
        f"Method: {canonical_method}\n"
        f"Name: {canonical_name}\n"
        f"Pairs: {strip_control(pairs_path)}\n"
        f"Layer: {layer if layer is not None else 'auto (middle)'}\n"
        f"Description: {spec.description}"
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
            "[green]Plan-only mode — inputs validated, plan rendered. "
            "Drop --plan-only to fit the vector.[/]"
        )
        return

    try:
        artifact = build_steering_vector(
            method=canonical_method,
            name=canonical_name,
            pairs_path=pairs_path,
            base=base,
            layer=layer,
            device=device,
            output_dir=output,
            top_k=top_k,
        )
    except (TypeError, ValueError, FileNotFoundError) as exc:
        _print_error("Steer-train failed:", exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                ("Method:", "bold"),
                f" {artifact.method}\n",
                ("Name:", "bold"),
                f" {artifact.name}\n",
                ("Layer:", "bold"),
                f" {artifact.layer}\n",
                ("Intervention:", "bold"),
                f" {artifact.intervention_point}\n",
                ("Hidden dim:", "bold"),
                f" {artifact.hidden_dim}\n",
                ("Pairs:", "bold"),
                f" {artifact.num_pairs}\n",
                ("Saved:", "bold"),
                f" {strip_control(artifact.output_dir)}",
            ),
            title="Steering vector trained",
            border_style="green",
        )
    )

    if registry_id is not None:
        _attach_vector(registry_id, artifact.output_dir)


def _attach_vector(registry_id: str, output_dir: str) -> None:
    """Attach the trained vector file to a registry entry as ``steering_vector``.

    The registry stores one hashed file per artifact and refuses a directory,
    so the row is the vector file; ``resolve_steering_dir`` loads the directory
    around it. A requested attach is part of the command's success: when it
    does not happen the command exits 1, with the vector already on disk.
    """
    from soup_cli.utils.steering import steering_vector_path

    vector_path = steering_vector_path(output_dir)
    try:
        from soup_cli.registry.attach import attach_artifact

        attach_artifact(registry_id, path=vector_path, kind="steering_vector")
    except Exception as exc:  # noqa: BLE001
        console.print(
            Text.assemble(
                ("Error:", "red"), " could not attach to registry: ", strip_control(exc)
            )
        )
        raise typer.Exit(1) from exc
    console.print(
        Text.assemble(
            ("Attached", "green"),
            " steering_vector to registry entry ",
            (strip_control(registry_id), "bold"),
            " ",
            (f"({strip_control(vector_path)})", "dim"),
        )
    )


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
    """Resolve + load a stored steering vector and print its metadata.

    The live decode-time intervention is applied by ``soup serve --steer
    <name> --steer-strength <s>`` (which installs the forward hook on the
    running model). This subcommand resolves the vector by name and confirms it
    loads cleanly so operators can verify an artifact before serving.
    """
    from soup_cli.utils.steering import (
        load_steering_artifact,
        resolve_steering_dir,
        validate_steering_name,
        validate_steering_strength,
    )

    try:
        canonical_name = validate_steering_name(name)
        canonical_strength = validate_steering_strength(strength)
    except (TypeError, ValueError) as exc:
        _print_error("Invalid steer-apply input:", exc)
        raise typer.Exit(2) from exc

    try:
        steer_dir = resolve_steering_dir(canonical_name)
        loaded = load_steering_artifact(steer_dir)
    except (TypeError, ValueError, OSError) as exc:
        _print_error("Cannot load steering vector:", exc)
        raise typer.Exit(2) from exc

    console.print(
        Panel(
            Text.assemble(
                ("Vector:", "bold"),
                f" {canonical_name}\n",
                ("Method:", "bold"),
                f" {loaded.method}\n",
                ("Layer:", "bold"),
                f" {loaded.layer}\n",
                ("Intervention:", "bold"),
                f" {loaded.intervention_point}\n",
                ("Dim:", "bold"),
                f" {len(loaded.vector)}\n",
                ("Strength:", "bold"),
                f" {canonical_strength}\n",
                ("Dir:", "bold"),
                f" {strip_control(steer_dir)}",
            ),
            title="soup steer apply",
            border_style="cyan",
        )
    )
    console.print(
        "[green]Vector loaded.[/] Apply it at decode time with "
        f"`soup serve --steer {escape(canonical_name)} "
        f"--steer-strength {canonical_strength}`."
    )


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
                        Text(strip_control(entry.get("name", ""))),
                        Text(strip_control(art.get("path", ""))),
                    )
                    rows += 1
    if rows == 0:
        console.print(
            "[dim]No steering_vector artifacts registered. "
            "Use `soup steer train ...` (v0.62.1+).[/]"
        )
        return
    console.print(table)
