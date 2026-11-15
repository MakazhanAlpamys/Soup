"""Pack registry entries into ``.can`` artifacts (v0.26.0 Part E)."""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console

from soup_cli.cans.schema import CAN_FORMAT_VERSION, Manifest
from soup_cli.registry.store import RegistryStore
from soup_cli.utils.paths import is_under_cwd
from soup_cli.utils.redaction import RedactionReport, redact_config, redact_text
from soup_cli.utils.terminal import for_terminal

_MAX_CAN_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
# How many changed key paths are listed on the console; the count covers all of them.
_MAX_LISTED_PATHS = 20


def _add_text(tar: tarfile.TarFile, name: str, content: str) -> None:
    data = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = int(datetime.now().timestamp())
    tar.addfile(info, io.BytesIO(data))


def _redact_for_pack(
    config: object, description: Optional[str],
) -> tuple[object, Optional[str], RedactionReport]:
    """The config and description with credentials replaced, and what changed.

    ``report.paths`` holds key paths only (``data.hf_token``), never a value.
    """
    redacted, report = redact_config(config)
    if description:
        before = report.count
        description = redact_text(description, report)
        if report.count != before:
            report.add_path("description")
    return redacted, description, report


def _redact_field(value: str, label: str, report: RedactionReport) -> str:
    """One text field written next to the config (a name, a tag, an author, ...)."""
    if not isinstance(value, str) or not value:
        return value
    before = report.count
    new = redact_text(value, report)
    if report.count != before:
        report.add_path(label)
    return new


def _redact_name(name: str, label: str, report: RedactionReport) -> str:
    """A can name that is itself a credential cannot be stored: refuse, without the value."""
    before = report.count
    new = redact_text(name, report)
    if report.count != before:
        raise ValueError(
            f"the can {label} looks like a credential; rename it before packing "
            "(the value is not shown)"
        )
    return new


def _announce_redactions(report: RedactionReport, console: Optional[Console]) -> None:
    """Say which keys were changed before the can was written (paths, never values)."""
    if not report.count:
        return
    shown = [for_terminal(redact_text(path)) for path in report.paths[:_MAX_LISTED_PATHS]]
    more = len(report.paths) - len(shown)
    listing = ", ".join(shown) + (f" (+{more} more)" if more > 0 else "")
    target = console or Console()
    target.print(
        f"[yellow]Redacted {report.count} credential value(s) before packing, "
        f"keys changed:[/] {listing}",
        soft_wrap=True, highlight=False,
    )
    target.print(
        "[dim]The redacted values are not in the can: supply them again before it is run.[/]",
        soft_wrap=True, highlight=False,
    )


def pack_entry(
    *, entry_id: str, out_path: str, author: str = "unknown",
    description: Optional[str] = None,
    attestations: Optional[list[dict]] = None,
    console: Optional[Console] = None,
) -> Path:
    """Pack a registry entry as a ``.can`` tarball.

    Args:
        entry_id: registry entry id (full id or prefix).
        out_path: destination ``.can`` file. Must stay under the cwd captured
            at call time.
        author: author handle baked into the manifest.
        description: optional free-form description.
        attestations: optional list of embedded in-toto Statements (v0.71.3
            #182). Each is shape-validated by the manifest schema.
        console: where the changed key paths are listed when the config or
            description carried credentials (they are replaced before packing).

    Returns:
        The absolute path to the written file.
    """
    out = Path(out_path)
    if not is_under_cwd(out):
        raise ValueError(f"out path '{out_path}' is outside cwd - refusing")

    with RegistryStore() as store:
        resolved = store.resolve(entry_id)
        if resolved is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        entry = store.get(resolved)
        if entry is None:
            raise ValueError(f"registry entry not found: {entry_id}")
        try:
            config = json.loads(entry.get("config_json") or "{}")
        except (TypeError, ValueError):
            config = {}

    config, safe_description, report = _redact_for_pack(
        config, description or entry.get("notes"),
    )
    # the fields written next to the config are redacted too (recipe.md prints three of them)
    safe_name = _redact_name(entry["name"], "name", report)
    safe_author = _redact_field(author, "author", report)
    safe_tags = [
        _redact_field(tag, f"tags[{index}]", report)
        for index, tag in enumerate(entry.get("tags", []))
    ]
    base_model = _redact_field(entry["base_model"], "base_model", report)
    task = _redact_field(entry["task"], "task", report)
    _announce_redactions(report, console)

    manifest = Manifest(
        can_format_version=CAN_FORMAT_VERSION,
        name=safe_name,
        author=safe_author,
        created_at=datetime.now().isoformat(timespec="seconds"),
        base_hash=entry.get("entry_hash", ""),
        description=safe_description,
        tags=safe_tags,
        attestations=list(attestations or []),
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, mode="w:gz") as tar:
        _add_text(tar, "manifest.yaml", yaml.safe_dump(
            manifest.model_dump(), sort_keys=False,
        ))
        _add_text(tar, "config.yaml", yaml.safe_dump(config, sort_keys=False))
        _add_text(tar, "data_ref.yaml", yaml.safe_dump(
            {"kind": "local", "note": "user must supply data locally"},
            sort_keys=False,
        ))
        _add_text(tar, "recipe.md",
                  f"# {manifest.name}\n\nBase: {base_model}\n"
                  f"Task: {task}\n\n"
                  f"{manifest.description or ''}\n")

    # Enforce size cap — reject anything larger than 100 MB
    if out.stat().st_size > _MAX_CAN_SIZE_BYTES:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"can exceeds max size ({_MAX_CAN_SIZE_BYTES // 1024 // 1024} MB) - "
            "trim metadata or exclude large fields"
        )
    return out


_FORBIDDEN_MOD_KEYS = frozenset({
    "__class__", "__init__", "__dict__", "__globals__", "__builtins__",
    "__import__", "__bases__", "__mro__",
})


def _apply_modification(config: dict, mod: str) -> None:
    """Apply a single ``dotted.path=value`` modification in-place."""
    if "=" not in mod:
        raise ValueError(
            f"modification '{mod}' must be 'dotted.path=value'"
        )
    path, raw_value = mod.split("=", 1)
    if "\x00" in path:
        raise ValueError(f"modification key contains null byte: {path!r}")
    keys = path.split(".")
    if any(key in _FORBIDDEN_MOD_KEYS or key.startswith("__") for key in keys):
        raise ValueError(
            f"modification key '{path}' references a dunder / forbidden name"
        )
    # Try parsing as JSON (number, bool, null, string) for proper typing.
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = raw_value
    target = config
    for key in keys[:-1]:
        target = target.setdefault(key, {})
        if not isinstance(target, dict):
            raise ValueError(
                f"cannot modify '{path}': '{key}' is not a dict"
            )
    target[keys[-1]] = value


def fork_can(
    *, source: str, out_path: str, modifications: list[str],
    author: str = "unknown",
    console: Optional[Console] = None,
) -> Path:
    """Apply ``modifications`` to a can's config and re-pack.

    Credentials in the config (including a value set by a modification) and in the
    description are replaced before packing; the changed key paths are listed on
    ``console``.
    """
    from soup_cli.cans.unpack import inspect_can, read_config

    src = Path(source)
    if not is_under_cwd(src):
        raise ValueError(f"source '{source}' is outside cwd - refusing")
    if not src.exists():
        raise FileNotFoundError(f"source can not found: {source}")

    out = Path(out_path)
    if not is_under_cwd(out):
        raise ValueError(f"out path '{out_path}' is outside cwd - refusing")

    manifest = inspect_can(source)
    config = read_config(source)
    for mod in modifications:
        _apply_modification(config, mod)
    config, safe_description, report = _redact_for_pack(config, manifest.description)
    safe_name = _redact_name(manifest.name, "name", report)
    safe_author = _redact_field(author, "author", report)
    safe_tags = [
        _redact_field(tag, f"tags[{index}]", report) for index, tag in enumerate(manifest.tags)
    ]
    _announce_redactions(report, console)

    with tarfile.open(out, mode="w:gz") as tar:
        _add_text(tar, "manifest.yaml", yaml.safe_dump({
            **manifest.model_dump(),
            "description": safe_description,
            "author": safe_author,
            "tags": safe_tags,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "name": f"{safe_name}-fork",
        }, sort_keys=False))
        _add_text(tar, "config.yaml", yaml.safe_dump(config, sort_keys=False))
        _add_text(tar, "data_ref.yaml", yaml.safe_dump(
            {"kind": "local", "note": "forked"},
            sort_keys=False,
        ))

    if out.stat().st_size > _MAX_CAN_SIZE_BYTES:
        out.unlink(missing_ok=True)
        raise ValueError(
            f"forked can exceeds max size "
            f"({_MAX_CAN_SIZE_BYTES // 1024 // 1024} MB)"
        )
    return out
