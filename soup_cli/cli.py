"""Main CLI entry point — all commands registered here."""

import sys

# UTF-8 stdio bootstrap (v0.40.1 Part A) — must run before any Rich console
# is constructed. On Windows, reconfigures sys.stdout/stderr to UTF-8 so β /
# ✓ / box-drawing chars don't crash with UnicodeEncodeError on cp1251/cp1252.
# POSIX: no-op.
from soup_cli.utils.encoding import force_utf8_stdio

force_utf8_stdio()
_utf8_bootstrap_done = True

import typer  # noqa: E402
from rich.console import Console  # noqa: E402

from soup_cli import __version__  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    adapters,
    autopilot,
    bench,
    can,
    chat,
    cost,
    data,
    deploy,
    diff,
    eval,
    export,
    generate,
    history,
    infer,
    init,
    merge,
    migrate,
    profile,
    push,
    recipes,
    registry,
    runs,
    serve,
    sweep,
    train,
    ui,
)

# v0.44.0 — Live monitoring + standalone CLI wrappers.
from soup_cli.commands import (  # noqa: E402
    delinearize_llama4 as delinearize_llama4_cmd,
)
from soup_cli.commands import doctor as doctor_cmd  # noqa: E402
from soup_cli.commands import fetch as fetch_cmd  # noqa: E402
from soup_cli.commands import llama as llama_cmd  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    merge_sharded_fsdp_weights as merge_sharded_fsdp_weights_cmd,
)
from soup_cli.commands import monitor as monitor_cmd  # noqa: E402
from soup_cli.commands import quantize as quantize_cmd  # noqa: E402
from soup_cli.commands import quickstart as quickstart_cmd  # noqa: E402
from soup_cli.commands import (  # noqa: E402
    tui as tui_cmd,
)
from soup_cli.commands import (  # noqa: E402
    why as why_cmd,
)
from soup_cli.utils.constants import GITHUB_URL  # noqa: E402

console = Console()

# Global verbose flag — set via callback, read by error handler
_verbose = False
# Global log level (resolved string), set by main() callback
_log_level = "normal"

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
app.command()(cost.cost)
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
app.command()(bench.bench)
app.command()(doctor_cmd.doctor)
app.command()(quickstart_cmd.quickstart)
app.command()(ui.ui)
app.command(name="autopilot")(autopilot.autopilot_cmd)
app.add_typer(
    registry.app, name="registry",
    help="Model Registry: push, list, show, diff, search, promote, delete.",
)
app.command(name="history")(history.history)
app.command(name="why")(why_cmd.why)
app.command(name="tui")(tui_cmd.tui)
app.add_typer(
    can.app, name="can",
    help="Soup Cans: pack/inspect/verify/fork shareable .can artifacts.",
)

# v0.44.0 — register Live Dashboard & UX commands.
app.command(name="monitor")(monitor_cmd.monitor)
app.command(name="fetch")(fetch_cmd.fetch)
app.command(name="quantize")(quantize_cmd.quantize)
app.command(name="merge-sharded-fsdp-weights")(
    merge_sharded_fsdp_weights_cmd.merge_sharded_fsdp_weights
)
app.command(name="delinearize-llama4")(delinearize_llama4_cmd.delinearize_llama4)
app.add_typer(
    llama_cmd.app,
    name="llama",
    help="Proxy to llama.cpp binaries (cli / mtmd-cli / gguf-split / server).",
)

# v0.45.0 Part A — Plugin system CLI.
from soup_cli.commands import plugins as plugins_cmd  # noqa: E402

app.add_typer(
    plugins_cmd.app,
    name="plugins",
    help="List, enable, disable Soup plugins (v0.45.0).",
)

# v0.46.0 Part B — Agent Forge.
from soup_cli.commands import agent as agent_cmd  # noqa: E402

app.add_typer(
    agent_cmd.app,
    name="agent",
    help="Agent Forge: spec -> tool-calling dataset / train / eval (v0.46.0).",
)

# Register data generate as a subcommand of data
data.app.command(name="generate")(generate.generate)


@app.command()
def version(
    full: bool = typer.Option(False, "--full", "-f", help="Show system info and extras"),
    json_output: bool = typer.Option(False, "--json", help="Output in JSON format"),
):
    """Show Soup CLI version."""
    import json
    import platform

    if json_output:
        info = {
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.system().lower(),
        }

        if full:
            for lib in ["torch", "transformers", "peft", "trl", "datasets", "accelerate"]:
                try:
                    mod = __import__(lib)
                    if hasattr(mod, "__version__"):
                        info[lib] = mod.__version__
                except ImportError:
                    pass
            for name in ["fastapi", "vllm", "datasketch", "lm_eval", "deepspeed", "wandb"]:
                try:
                    mod = __import__(name)
                    if hasattr(mod, "__version__"):
                        info[name] = mod.__version__
                    elif hasattr(mod, "version"):
                        info[name] = mod.version
                    else:
                        info[name] = "installed"
                except ImportError:
                    pass

        console.print(json.dumps(info), highlight=False)
        return

    if not full:
        console.print(f"[bold green]soup[/] v{__version__}")
        console.print(f"[dim]{GITHUB_URL}[/]")
        return

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
    log_level: str = typer.Option(
        "normal",
        "--log-level",
        help="Logging tier: quiet | normal | verbose | debug",
    ),
):
    """Soup — fine-tune LLMs in one command."""
    global _verbose, _log_level
    _verbose = verbose
    from soup_cli.utils.log_level import (
        apply_logging_level,
        parse_log_level,
        setup_logging,
    )

    try:
        tier = parse_log_level(log_level)
    except ValueError as exc:
        console.print(f"[red]error:[/] {exc}")
        raise typer.Exit(code=2) from exc
    _log_level = tier.value
    setup_logging(tier)
    # v0.40.2 N1/G2: also push the tier into the root logger so third-party
    # libraries (transformers / peft / trl) respect QUIET / DEBUG. Without
    # this, all four levels were producing nearly-identical output.
    apply_logging_level(tier)


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
