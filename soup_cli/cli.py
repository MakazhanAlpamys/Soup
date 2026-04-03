"""Main CLI entry point — all commands registered here."""

import sys

import typer
from rich.console import Console

from soup_cli import __version__
from soup_cli.commands import (
    adapters,
    chat,
    data,
    deploy,
    diff,
    eval,
    export,
    generate,
    infer,
    init,
    merge,
    migrate,
    profile,
    push,
    recipes,
    runs,
    serve,
    sweep,
    train,
    ui,
)
from soup_cli.commands import doctor as doctor_cmd
from soup_cli.commands import quickstart as quickstart_cmd
from soup_cli.utils.constants import GITHUB_URL

console = Console()

# Global verbose flag — set via callback, read by error handler
_verbose = False

app = typer.Typer(
    name="soup",
    help=(
        "Fine-tune LLMs in one command. No SSH, no config hell.\n\n"
        f"[dim]GitHub: {GITHUB_URL}[/]"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command()(init.init)
app.command()(train.train)
app.command()(chat.chat)
app.command()(push.push)
app.command(name="export")(export.export)
app.command()(merge.merge)
app.add_typer(
    data.app, name="data",
    help="Dataset tools: inspect, convert, merge, dedup, validate, stats.",
)
app.add_typer(
    deploy.app, name="deploy",
    help="Deploy models: Ollama integration (deploy, list, remove).",
)
app.add_typer(runs.app, name="runs", help="Experiment tracking: list, show, compare runs.")
app.add_typer(
    eval.app, name="eval",
    help="Evaluate models: benchmarks, custom evals, LLM judge, leaderboard.",
)
app.command()(migrate.migrate)
app.add_typer(
    adapters.app, name="adapters",
    help="Adapter management: list, info, compare LoRA adapters.",
)
app.add_typer(
    recipes.app, name="recipes",
    help="Ready-made configs: list, show, use, search recipes for popular models.",
)
app.command()(serve.serve)
app.command()(sweep.sweep)
app.command(name="diff")(diff.diff)
app.command()(infer.infer)
app.command()(profile.profile)
app.command()(doctor_cmd.doctor)
app.command()(quickstart_cmd.quickstart)
app.command()(ui.ui)

# Register data generate as a subcommand of data
data.app.command(name="generate")(generate.generate)


@app.command()
def version(
    full: bool = typer.Option(False, "--full", "-f", help="Show system info and extras"),
):
    """Show Soup CLI version."""
    if not full:
        console.print(f"[bold green]soup[/] v{__version__}")
        console.print(f"[dim]{GITHUB_URL}[/]")
        return

    import platform

    parts = [f"[bold green]soup[/] v{__version__}"]
    parts.append(f"Python {platform.python_version()}")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            parts.append(f"CUDA {torch.version.cuda}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            parts.append("MPS")
        else:
            parts.append("CPU only")
    except ImportError:
        parts.append("no torch")

    # Installed extras
    extras = []
    for name, label in [
        ("fastapi", "serve"),
        ("vllm", "serve-fast"),
        ("datasketch", "data"),
        ("lm_eval", "eval"),
        ("deepspeed", "deepspeed"),
        ("wandb", "wandb"),
    ]:
        try:
            __import__(name)
            extras.append(label)
        except ImportError:
            pass

    if extras:
        parts.append(f"extras: {', '.join(extras)}")

    console.print(" | ".join(parts))
    console.print(f"[dim]GitHub: [link={GITHUB_URL}]{GITHUB_URL}[/link][/]")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Show full traceback on errors",
    ),
):
    """Soup — fine-tune LLMs in one command."""
    global _verbose
    _verbose = verbose


def run():
    """Entry point with friendly error handling."""
    try:
        app()
    except SystemExit:
        raise
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        sys.exit(130)
    except Exception as exc:
        from soup_cli.utils.errors import format_friendly_error

        format_friendly_error(exc, verbose=_verbose)
        sys.exit(1)


# When invoked via `soup` entry point, use run() for error handling.
# When invoked via `python -m soup_cli`, __main__.py calls run() directly.
if __name__ == "__main__":
    run()
