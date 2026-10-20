"""Load and validate soup.yaml configs."""

import warnings
from pathlib import Path

import yaml
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config.deprecation import SoupConfigDeprecationWarning
from soup_cli.config.schema import SoupConfig
from soup_cli.config.staged_fields import find_staged_config_fields, format_staged_fields
from soup_cli.config.unknown_keys import find_unknown_config_keys, format_unknown_keys
from soup_cli.utils.terminal import for_terminal
from soup_cli.utils.yaml_limits import (
    MAX_CONFIG_BYTES,
    YamlLimitError,
    load_yaml_file_limited,
    load_yaml_limited,
)

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
#: The decision was warn-then-forbid. v0.74.0 shipped ``"warn"`` with a
#: deadline named by
#: :data:`~soup_cli.config.unknown_keys.UNKNOWN_KEY_REJECTION_VERSION` -- by
#: reference, not by number, because the warning stated that version in
#: exactly one place -- and the release that reached it flipped this to
#: ``"error"``. ``TestTheDeadline`` pins the switch to the declared
#: ``__version__`` in both directions, so it can be neither forgotten nor
#: flipped early.
UNKNOWN_KEY_SEVERITY = "error"

#: What to do about a declared config field that is read by nothing (#808).
#:
#: The decision was warn-then-forbid. Ships ``"warn"`` with a deadline named
#: by :data:`~soup_cli.config.staged_fields.STAGED_FIELD_REJECTION_VERSION` -- by
#: reference, not by number, because the warning stated that version in
#: exactly one place -- and the release that reaches it flips this to
#: ``"error"``.
STAGED_FIELD_SEVERITY = "warn"


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
    # The key names in ``message`` came from the config file: escape them.
    console.print(f"[yellow]Warning:[/] {for_terminal(message)}")
    console.print(
        "[dim]An unapplied key is ignored, not defaulted -- the run proceeds as "
        "if you had not written it.[/]"
    )
    return None


def _report_staged_fields(
    raw: dict, *, config: SoupConfig
) -> "str | None":
    """Return an error string when staged fields must stop the load (#808).

    Silence is the thing being fixed: under ``"warn"`` it is printed and
    ``None`` is returned; under ``"error"`` the message is handed back for the
    caller to raise in its own contract (``SystemExit`` for the CLI,
    ``ValueError`` for the API/UI).
    """
    staged = find_staged_config_fields(raw, config=config)
    if not staged:
        return None
    warning = STAGED_FIELD_SEVERITY != "error"
    message = format_staged_fields(staged, include_deadline=warning)
    if not warning:
        return message
    console.print(f"[yellow]Warning:[/] {for_terminal(message)}")
    return None


#: Longest piece of text taken from the file that a refusal quotes.
_MAX_DETAIL_CHARS = 200


def _refuse(message: str) -> None:
    """Print a one-line refusal and exit 1; the text may quote the file."""
    console.print(f"[red]{for_terminal(message)}[/]")
    raise SystemExit(1)


def _clip(detail: str) -> str:
    """``detail`` on one line, cut to ``_MAX_DETAIL_CHARS`` with an ellipsis."""
    detail = " ".join(detail.split())
    if len(detail) <= _MAX_DETAIL_CHARS:
        return detail
    return detail[:_MAX_DETAIL_CHARS] + "..."


def _read_config_file(path: Path, *, raise_unreadable: bool = False) -> object:
    """Load the YAML document at ``path``, or exit 1 naming what stopped it.

    A file over a limit, a directory, a file the OS will not let us read, bytes
    that are not UTF-8, a YAML syntax error and a value YAML cannot build (a
    date that does not exist) each get one line that names the file and the
    problem; text taken from the file is cut to ``_MAX_DETAIL_CHARS``. A
    missing file is left to the caller as ``FileNotFoundError``, and any other
    exception is passed on as itself.

    With ``raise_unreadable`` nothing is printed and nothing exits: the caller
    gets the exception itself (``YamlLimitError``, ``yaml.YAMLError``,
    ``UnicodeDecodeError``, ``OSError``) to report in its own words.
    """
    what = f"Config file {path}"
    if raise_unreadable:
        return load_yaml_file_limited(path, what, max_bytes=MAX_CONFIG_BYTES)
    if path.is_dir():
        _refuse(f"{what} is a directory, not a file")
    try:
        return load_yaml_file_limited(path, what, max_bytes=MAX_CONFIG_BYTES)
    except YamlLimitError as exc:
        _refuse(str(exc))
    except UnicodeDecodeError as exc:
        _refuse(f"{what} is not valid UTF-8: {exc.reason} at byte {exc.start}")
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None)
        if mark is not None and problem:
            detail = f"{_clip(problem)} (line {mark.line + 1}, column {mark.column + 1})"
        else:
            detail = _clip(str(exc))
        _refuse(f"{what} is not valid YAML: {detail}")
    except FileNotFoundError:
        raise
    except OSError as exc:
        _refuse(f"{what} cannot be read: {exc.strerror or type(exc).__name__}")
    return None  # pragma: no cover - _refuse always raises


def _build_config(raw: dict) -> SoupConfig:
    """Construct ``SoupConfig``, printing each deprecated value once (#759).

    Validators report a value that will be refused in a later release as a
    :class:`SoupConfigDeprecationWarning`. Recorded here rather than left to the
    ``warnings`` filters, which print a source line, may show a message once
    per process rather than once per load, or hide it entirely. Any other
    warning raised while building is passed on unchanged.
    """
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = SoupConfig(**raw)
    finally:
        # Outside the ``catch_warnings`` block: re-raising a warning inside it
        # would record it again into the list being walked.
        printed: set = set()
        for item in caught:
            if issubclass(item.category, SoupConfigDeprecationWarning):
                message = str(item.message)
                if message not in printed:
                    printed.add(message)
                    console.print(f"[yellow]Warning:[/] {for_terminal(message)}")
            else:
                warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)
    return config


def load_config(
    path: "Path | str",
    *,
    training_overrides: dict | None = None,
    raise_unreadable: bool = False,
) -> SoupConfig:
    """Load a soup.yaml file and return validated SoupConfig.

    ``training_overrides`` are merged into the YAML ``training:`` mapping
    *before* ``SoupConfig`` is constructed, so CLI flags that map onto
    training fields participate in the same cross-validators as YAML.

    The file is read under the limits of :mod:`soup_cli.utils.yaml_limits`
    (at most :data:`MAX_CONFIG_BYTES`, bounded nesting and alias expansion);
    a file over one of them exits 1 with a message naming the file and the
    limit, before anything walks the document. So does a file that cannot
    be read as YAML at all; see :func:`_read_config_file`. A caller that
    reports such a file in its own words passes ``raise_unreadable=True``
    and receives the exception instead; a config that reads but does not
    validate still exits 1 with the loader's message either way.
    """
    raw = _read_config_file(Path(path), raise_unreadable=raise_unreadable)

    if raw is None:
        console.print("[red]Config file is empty[/]")
        raise SystemExit(1)
    if not isinstance(raw, dict):
        # A bare list ("- a") or scalar would reach SoupConfig(**raw) and die
        # with a TypeError traceback; load_config_from_string already refuses
        # this shape, and the CLI contract here is SystemExit(1).
        console.print(f"[red]Config must be a YAML mapping, got {type(raw).__name__}[/]")
        raise SystemExit(1)

    if training_overrides:
        training = raw.get("training")
        if not isinstance(training, dict):
            training = {}
            raw["training"] = training
        training.update(training_overrides)

    unknown_error = _report_unknown_keys(raw)
    if unknown_error is not None:
        console.print("[red bold]Config validation error:[/]\n")
        console.print(f"  [red]{for_terminal(unknown_error)}[/]")
        raise SystemExit(1)

    try:
        config = _build_config(raw)
    except ValidationError as e:
        console.print("[red bold]Config validation error:[/]\n")
        for err in e.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            # Both can quote config keys or values from the file.
            console.print(f"  [red]{for_terminal(loc)}:[/] {for_terminal(err['msg'])}")
        raise SystemExit(1)

    staged_error = _report_staged_fields(raw, config=config)
    if staged_error is not None:
        console.print("[red bold]Config validation error:[/]\n")
        console.print(f"  [red]{for_terminal(staged_error)}[/]")
        raise SystemExit(1)

    return config


def load_config_from_string(yaml_str: str, *, max_bytes: "int | None" = None) -> SoupConfig:
    """Parse a YAML string and return validated SoupConfig.

    Unlike load_config(), raises ValueError on errors instead of SystemExit,
    making it suitable for API/UI usage. The text is held to the same limits
    as a config file; crossing one raises
    :class:`~soup_cli.utils.yaml_limits.YamlLimitError`, a ``ValueError``.
    ``max_bytes`` replaces :data:`MAX_CONFIG_BYTES` for a caller that
    documents a size limit of its own.
    """
    limit = MAX_CONFIG_BYTES if max_bytes is None else max_bytes
    raw = load_yaml_limited(yaml_str, "Config", max_bytes=limit)
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
        config = _build_config(raw)
    except ValidationError as exc:
        errors = []
        for err in exc.errors():
            loc = " -> ".join(str(part) for part in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        raise ValueError("; ".join(errors))

    staged_error = _report_staged_fields(raw, config=config)
    if staged_error is not None:
        raise ValueError(staged_error)

    return config
