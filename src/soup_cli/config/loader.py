"""Load and validate soup.yaml configs."""

import re
import warnings
from pathlib import Path

import yaml
from pydantic import ValidationError
from rich.console import Console

from soup_cli.config.deprecation import SoupConfigDeprecationWarning, deadline_clause
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
#: The decision was warn-then-forbid. v0.76 shipped ``"warn"`` with a deadline
#: named by :data:`~soup_cli.config.staged_fields.STAGED_FIELD_REJECTION_VERSION`
#: -- by reference, not by number, because the warning stated that version in
#: exactly one place -- and the 0.77.0 release, which reached it, flipped this to
#: ``"error"`` in the same commit as ``DEPRECATED_VALUE_SEVERITY`` in
#: ``config/deprecation.py``. ``tests/test_issue808_staged_config_fields.py`` and
#: ``TestTheDeadline`` in ``tests/test_config_deprecation.py`` pin it to the
#: declared ``__version__``.
STAGED_FIELD_SEVERITY = "error"


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


def _names_a_field(err: dict, names: "tuple[str, ...]") -> bool:
    """Whether a Pydantic error entry is about one of ``names``: one is a part of
    its location, or its message names one as a whole word."""
    if any(part in names for part in err["loc"]):
        return True
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", err["msg"])
        for name in names
    )


def _staged_refusal_after_failed_build(
    raw: dict, exc: ValidationError
) -> "tuple[str, list] | None":
    """The staged-field refusal for a config that did not build, and the build
    errors that are left to report after it (#808).

    Some staged fields have a cross-validator of their own: ``grace_codebook``
    asks for ``grace_codebook_size`` and ``grace_codebook_dim``,
    ``long_context_grpo`` for ``task: grpo``, ``video_fps`` for a video format.
    Reported alone, that message is advice the next load refuses: the user adds
    what it asks for, and the staged check then refuses all of it. So when the
    build fails and the mapping sets a staged field, the staged refusal is
    reported first, followed only by the build errors that name none of the
    refused fields. The mapping is read without a built config (the
    ``config=None`` path of :func:`find_staged_config_fields`), which can only
    run here, on a load that has already failed.

    Returns ``None`` (report the build errors as they are) under ``"warn"`` or
    when no staged field is set.
    """
    if STAGED_FIELD_SEVERITY != "error":
        return None
    staged = find_staged_config_fields(raw)
    if not staged:
        return None
    names = tuple(item.name for item in staged)
    remaining = [err for err in exc.errors() if not _names_a_field(err, names)]
    return format_staged_fields(staged, include_deadline=False), remaining


#: Longest piece of text taken from the file that a refusal quotes.
_MAX_DETAIL_CHARS = 200


def _refuse(message: str) -> None:
    """Print a one-line refusal and exit 1; the text may quote the file."""
    console.print(f"[red]{for_terminal(message)}[/]")
    raise SystemExit(1)


def _print_refusal_lines(message: str) -> None:
    """Print a multi-line refusal under ``Config validation error:``, indenting
    every line, not only the first (one line per unknown key or staged field).
    The text quotes key names from the file, so each line is made terminal-safe,
    and printed with emoji codes off, so a key ``:warning:`` shows as written."""
    for line in message.splitlines():
        console.print(f"  [red]{for_terminal(line)}[/]", emoji=False)


#: What Pydantic puts in front of the text of a ``ValueError`` raised in a validator.
_PYDANTIC_VALUE_ERROR_PREFIX = "Value error, "


def _print_validation_errors(errors: list) -> None:
    """Print each entry of a ``ValidationError`` under ``Config validation error:``.

    ``errors`` is ``ValidationError.errors()``, or the part of it that is left
    after a staged-field refusal (:func:`_staged_refusal_after_failed_build`).

    A field-level entry reads ``<loc>: <message>``. A model-level one (a
    ``model_validator`` refusal) has an empty ``loc`` and is the message alone,
    not ``: <message>``. Pydantic's ``Value error, `` prefix is dropped from both:
    the text after it is Soup's own and already names the fix. Only the terminal
    is changed; ``load_config_from_string`` keeps Pydantic's text in its
    ``ValueError``, which API callers read. Both parts can quote config keys or
    values from the file, so each goes through :func:`for_terminal`.

    A deprecated-value refusal (#759) at field level is also the message alone:
    its text is the v0.76 warning's, which had no location and so starts with
    the field's dotted path, and ``<loc>: `` in front would name the field twice
    (``data -> remove_unused_columns: data.remove_unused_columns: ...``).
    """
    for err in errors:
        loc = " -> ".join(str(part) for part in err["loc"])
        msg = err["msg"].removeprefix(_PYDANTIC_VALUE_ERROR_PREFIX)
        dotted = ".".join(str(part) for part in err["loc"])
        if loc and msg.startswith(f"{dotted}: ") and deadline_clause() in msg:
            loc = ""
        if loc:
            console.print(f"  [red]{for_terminal(loc)}:[/] {for_terminal(msg)}", emoji=False)
        else:
            console.print(f"  {for_terminal(msg)}", emoji=False)


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

    Under ``DEPRECATED_VALUE_SEVERITY = "warn"`` validators report a value that a
    later release refuses as a :class:`SoupConfigDeprecationWarning`; under
    ``"error"`` they raise instead, and nothing is printed. A warning is
    recorded here rather than left to the ``warnings`` filters, which print a
    source line, may show a message once per process rather than once per load,
    or hide it entirely. Any other warning raised while building is passed on
    unchanged.
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

    return validate_config_mapping(raw)


def validate_config_mapping(raw: dict) -> SoupConfig:
    """Validate a config mapping already read from a file; the checks and
    messages of :func:`load_config`, which ends here.

    For a command that reads the file itself (``soup plan`` / ``soup apply``)
    and must refuse what ``soup train`` refuses, with the same lines: an
    unknown key, a staged field, a refused value or any other schema error
    prints ``Config validation error:`` and its lines and exits 1.
    """
    unknown_error = _report_unknown_keys(raw)
    if unknown_error is not None:
        console.print("[red bold]Config validation error:[/]\n")
        _print_refusal_lines(unknown_error)
        raise SystemExit(1)

    try:
        config = _build_config(raw)
    except ValidationError as e:
        console.print("[red bold]Config validation error:[/]\n")
        staged_first = _staged_refusal_after_failed_build(raw, e)
        if staged_first is None:
            _print_validation_errors(e.errors())
        else:
            staged_error, remaining = staged_first
            _print_refusal_lines(staged_error)
            _print_validation_errors(remaining)
        raise SystemExit(1)

    staged_error = _report_staged_fields(raw, config=config)
    if staged_error is not None:
        console.print("[red bold]Config validation error:[/]\n")
        _print_refusal_lines(staged_error)
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
        staged_first = _staged_refusal_after_failed_build(raw, exc)
        entries = exc.errors() if staged_first is None else staged_first[1]
        errors = []
        for err in entries:
            loc = " -> ".join(str(part) for part in err["loc"])
            errors.append(f"{loc}: {err['msg']}")
        if staged_first is None:
            raise ValueError("; ".join(errors))
        # The staged lines first, one per field as on a load that builds; the
        # build errors that are left follow on one more line.
        lines = [staged_first[0]]
        if errors:
            lines.append("; ".join(errors))
        raise ValueError("\n".join(lines))

    staged_error = _report_staged_fields(raw, config=config)
    if staged_error is not None:
        raise ValueError(staged_error)

    return config
