"""Load and validate soup.yaml configs."""

from pathlib import Path

import yaml
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config.schema import SoupConfig

console = Console()


def load_config(path: Path) -> SoupConfig:
    """Load a soup.yaml file and return validated SoupConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if raw is None:
        console.print("[red]Config file is empty[/]")
        raise SystemExit(1)

    try:
        config = SoupConfig(**raw)
    except ValidationError as e:
        console.print("[red bold]Config validation error:[/]\n")
        for err in e.errors():
            loc = " → ".join(str(part) for part in err["loc"])
            console.print(f"  [red]{loc}:[/] {err['msg']}")
        raise SystemExit(1)

    return config
