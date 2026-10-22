"""Typer's runner with Click's filesystem isolation across supported releases."""

from click.testing import CliRunner as ClickCliRunner
from typer.testing import CliRunner as TyperCliRunner


class CliRunner(TyperCliRunner):
    """Keep Typer's invocation; reuse Click's independent filesystem helper."""

    isolated_filesystem = ClickCliRunner.isolated_filesystem
