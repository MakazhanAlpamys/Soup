"""Load and validate soup.yaml configs."""

from pathlib import Path

import yaml
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config.schema import SoupConfig
from soup_cli.config.unknown_keys import find_unknown_config_keys, format_unknown_keys

console = Console()

#: What to do about a key no model declares (#627).
#:
#: ``"warn"``  -- report and continue. Non-breaking: a config written for a
#:               newer Soup still runs on an older one.
#: ``"error"`` -- refuse to load.
#:
#: Detection is identical either way; this is the only difference between the
#: options argued on #627, kept as one switch so the decision is a one-line
#: change rather than a rewrite.
#:
#: The decision was warn-then-forbid, and the release that flips this to
#: ``"error"`` is named by
#: :data:`~soup_cli.config.unknown_keys.UNKNOWN_KEY_REJECTION_VERSION` -- by
#: reference, not by number, because the warning must state that version in
#: exactly one place. ``TestTheDeadline`` fails the moment the declared
#: ``__version__`` reaches it while this still reads ``"warn"``, so the flip
#: cannot be forgotten and the message cannot outlive its own promise.
UNKNOWN_KEY_SEVERITY = "warn"


def _report_unknown_keys(raw: dict) -> "str | None":
    """Return an error string when unknown keys must stop the load.

    Silence is the thing being fixed, so a finding is always surfaced: under
    ``"warn"`` it is printed and ``None`` is returned; under ``"error"`` the
    message is handed back for the caller to raise in its own contract
    (``SystemExit`` for the CLI, ``ValueError`` for the API/UI).
    """
    unknown = find_unknown_config_keys(raw)
    if not unknown:
        return None
    # One report per load with every finding in it, whatever the severity --
    # a config carrying four typos should produce one panel, not four.
    warning = UNKNOWN_KEY_SEVERITY != "error"
    message = format_unknown_keys(unknown, include_deadline=warning)
    if not warning:
        return message
    console.print(f"[yellow]Warning:[/] {message}")
    console.print(
        "[dim]An unapplied key is ignored, not defaulted -- the run proceeds as "
        "if you had not written it.[/]"
    )
    return None


def load_config(path: "Path | str") -> SoupConfig:
    """Load a soup.yaml file and return validated SoupConfig."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    if raw is None:
        console.print("[red]Config file is empty[/]")
        raise SystemExit(1)

    unknown_error = _report_unknown_keys(raw)
    if unknown_error is not None:
        console.print("[red bold]Config validation error:[/]\n")
        console.print(f"  [red]{unknown_error}[/]")
        raise SystemExit(1)

    try:
        config = SoupConfig(**raw)
    except ValidationError as e:
        console.print("[red bold]Config validation error:[/]\n")
        for err in e.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            console.print(f"  [red]{loc}:[/] {err['msg']}")
        raise SystemExit(1)

    return config


def load_config_from_string(yaml_str: str) -> SoupConfig:
    """Parse a YAML string and return validated SoupConfig.

    Unlike load_config(), raises ValueError on errors instead of SystemExit,
    making it suitable for API/UI usage.
    """
    raw = yaml.safe_load(yaml_str)
    if raw is None:
        raise ValueError("Config is empty")
    if not isinstance(raw, dict):
        # A non-mapping document (e.g. a bare list "- a") would make
        # SoupConfig(**raw) raise TypeError, breaking this function's
        # ValueError-only contract (API/UI callers only catch ValueError).
        raise ValueError(
            f"Config must be a YAML mapping, got {type(raw).__name__}"
        )

    unknown_error = _report_unknown_keys(raw)
    if unknown_error is not None:
        raise ValueError(unknown_error)

    try:
        return SoupConfig(**raw)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        raise ValueError("; ".join(errors))
