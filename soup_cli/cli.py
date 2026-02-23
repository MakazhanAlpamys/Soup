"""Main CLI entry point — all commands registered here."""

import typer
from rich.console import Console

from soup_cli import __version__
from soup_cli.commands import chat, data, init, push, train

console = Console()

app = typer.Typer(
    name="soup",
    help="Fine-tune LLMs in one command. No SSH, no config hell.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Register sub-commands
app.command()(init.init)
app.command()(train.train)
app.command()(chat.chat)
app.command()(push.push)
app.add_typer(data.app, name="data", help="Dataset tools: inspect, convert, validate.")


@app.command()
def version():
    """Show Soup CLI version."""
    console.print(f"[bold green]soup[/] v{__version__}")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Soup — fine-tune LLMs in one command."""
    pass
