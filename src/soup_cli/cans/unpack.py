"""Inspect, extract, and read ``.can`` artifacts (v0.26.0 Part E)."""

from __future__ import annotations

import os
import tarfile
import unicodedata
from pathlib import Path
from typing import Any

from soup_cli.cans.schema import Manifest
from soup_cli.utils.paths import is_under_cwd, quote_path
from soup_cli.utils.yaml_limits import MAX_CONFIG_BYTES, load_yaml_limited

#: Largest ``manifest.yaml`` read into memory. A manifest holds a handful of
#: short fields plus at most 64 attestations of <= 1 MiB each, so a real one is
#: far below this.
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
#: Most members ``extract_can`` will write. ``soup can pack`` writes four.
MAX_EXTRACT_MEMBERS = 10_000
#: Largest total declared size of the regular files ``extract_can`` writes.
#: ``soup can pack`` / ``fork`` write only small text members and refuse a can
#: over 100 MB compressed, but a hand-assembled can may carry a GGUF for a
#: ``deploy_targets`` entry (``cans/run.py:_deploy_target``). GGUF weights are
#: already quantised and barely compress, and an 8B model at Q4_K_M is ~4.9 GB,
#: so 8 GiB covers that case with margin while bounding what a small gzip
#: stream can expand to on disk (deflate reaches ~1000:1 on repetitive input).
MAX_EXTRACT_BYTES = 8 * 1024 * 1024 * 1024


def _name_key(name: str) -> str:
    """The entry a member name is written to, on any platform, as one comparable string.

    Two members whose keys are equal are the same file once extracted somewhere: the
    separator forms (``/``, ``\\``, ``//``), a ``./`` prefix, ``.`` and ``..`` components
    and a trailing separator all resolve away; case and Unicode composition are folded
    (case-insensitive file systems, macOS); and a trailing dot or space and an NTFS
    stream suffix (``:name``) are dropped, as Windows drops them. Raises ``ValueError``
    for a name that leaves the can. An empty key is the can's root.
    """
    parts: list[str] = []
    for raw in name.replace("\\", "/").split("/"):
        if raw in ("", "."):
            continue
        if raw == "..":
            if not parts:
                raise ValueError(f"member name {quote_path(name)} leaves the can; refusing")
            parts.pop()
            continue
        part = unicodedata.normalize("NFC", raw).split(":", 1)[0].rstrip(". ")
        parts.append((part or raw).casefold())
    return "/".join(parts)


def _check_members(tar: tarfile.TarFile) -> None:
    """Refuse a can whose members are not one file or directory per name.

    ``inspect_can`` and ``read_config`` read a member by its exact name while
    ``extract_can`` writes all of them, so an archive holding two members that resolve to
    the same entry (see :func:`_name_key`) would show one file to the checks and hand
    another to the code that runs. Every reader of a can calls this first. Also refused:
    a link, a device or any other member that is not a regular file or a directory (``soup
    can pack`` never writes one), a member below a file member, and more members than
    :data:`MAX_EXTRACT_MEMBERS`. Iterates lazily, so the count stops the header scan early.
    """
    first_name: dict[str, str] = {}
    files: set[str] = set()
    for count, member in enumerate(tar, start=1):
        if count > MAX_EXTRACT_MEMBERS:
            raise ValueError(
                f"can has too many members (> {MAX_EXTRACT_MEMBERS}); refusing"
            )
        if member.issym() or member.islnk():
            raise ValueError(
                f"member {quote_path(member.name)} is a link; links are not allowed in a "
                ".can file; refusing"
            )
        if not (member.isfile() or member.isdir()):
            raise ValueError(
                f"member {quote_path(member.name)} is neither a regular file nor a "
                "directory; refusing"
            )
        key = _name_key(member.name)
        if not key:
            continue
        if key in first_name:
            raise ValueError(
                "two members of this can resolve to the same name: "
                f"{quote_path(first_name[key])} and {quote_path(member.name)}; refusing"
            )
        first_name[key] = member.name
        if member.isfile():
            files.add(key)
    for key, name in first_name.items():
        parts = key.split("/")
        for depth in range(1, len(parts)):
            parent = "/".join(parts[:depth])
            if parent in files:
                raise ValueError(
                    f"member {quote_path(name)} lies under the file member "
                    f"{quote_path(first_name[parent])}; refusing"
                )


def _read_text_member(
    tar: tarfile.TarFile, name: str, max_bytes: int,
) -> str:
    """Return the text of a regular UTF-8 member, size-capped."""
    try:
        member = tar.getmember(name)
    except KeyError as exc:
        raise ValueError(f"can has no member '{name}'") from exc
    if not member.isfile():
        raise ValueError(f"member '{name}' in can is not a regular file")
    if member.size > max_bytes:
        raise ValueError(
            f"member '{name}' in can is too large "
            f"({member.size} > {max_bytes} bytes)"
        )
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"cannot read member '{name}' from can")
    # The header size is not trusted alone: the read itself is bounded.
    raw = extracted.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(
            f"member '{name}' in can is too large (> {max_bytes} bytes)"
        )
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"member '{name}' in can is not valid UTF-8: {exc}") from exc


def inspect_can(path: str) -> Manifest:
    """Load and validate the manifest from a ``.can`` file.

    Refuses paths outside the current working directory so an ``inspect``
    invocation cannot be coerced into reading arbitrary tarballs.
    """
    can_path = Path(path)
    if not is_under_cwd(can_path):
        raise ValueError(f"can path '{path}' is outside cwd - refusing")
    if not can_path.exists():
        raise FileNotFoundError(f"can not found: {path}")
    with tarfile.open(can_path, mode="r:gz") as tar:
        _check_members(tar)
        manifest_text = _read_text_member(tar, "manifest.yaml", MAX_MANIFEST_BYTES)
    data = load_yaml_limited(
        manifest_text, "manifest.yaml", max_bytes=MAX_MANIFEST_BYTES,
    ) or {}
    return Manifest(**data)


def read_config(path: str) -> dict[str, Any]:
    """Return the config dict stored in the can."""
    can_path = Path(path)
    if not is_under_cwd(can_path):
        raise ValueError(f"can path '{path}' is outside cwd - refusing")
    if not can_path.exists():
        raise FileNotFoundError(f"can not found: {path}")
    with tarfile.open(can_path, mode="r:gz") as tar:
        _check_members(tar)
        cfg_text = _read_text_member(tar, "config.yaml", MAX_CONFIG_BYTES)
    data = load_yaml_limited(cfg_text, "config.yaml", max_bytes=MAX_CONFIG_BYTES) or {}
    if not isinstance(data, dict):
        raise ValueError("config.yaml must deserialise to a mapping")
    return data


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract ``tar`` into ``dest`` without escaping it.

    Uses tarfile's ``filter="data"`` on Python 3.12+. Security-related
    errors from that filter (``FilterError`` / subclasses of ``TarError``)
    are re-raised so malicious archives cannot slip through a fallback.
    On older Python where the filter kwarg is not supported, falls back
    to a manual commonpath + symlink check.
    """
    dest_real = os.path.realpath(str(dest))

    if hasattr(tarfile, "data_filter"):
        try:
            tar.extractall(dest, filter="data")
            return
        except (TypeError, AttributeError):
            # filter="data" not supported on this tarfile build — fall through
            # to manual check. Security-relevant TarError subclasses propagate.
            pass

    for member in tar.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(
                f"symlinks / hardlinks are not allowed in .can files: {member.name}"
            )
        target_path = os.path.realpath(os.path.join(dest_real, member.name))
        try:
            common = os.path.commonpath([dest_real, target_path])
        except ValueError as exc:
            raise ValueError(
                f"tar entry '{member.name}' escapes destination"
            ) from exc
        if common != dest_real:
            raise ValueError(
                f"tar entry '{member.name}' escapes destination"
            )
        tar.extract(member, dest)


def extract_can(path: str, dest_dir: str) -> Path:
    """Extract the can into ``dest_dir`` safely."""
    can_path = Path(path)
    if not can_path.exists():
        raise FileNotFoundError(f"can not found: {path}")
    dest = Path(dest_dir)
    with tarfile.open(can_path, mode="r:gz") as tar:
        _check_members(tar)
        _check_extract_bounds(tar)
        dest.mkdir(parents=True, exist_ok=True)
        _safe_extract(tar, dest)
    return dest


def _check_extract_bounds(tar: tarfile.TarFile) -> None:
    """Refuse a can whose member count or total file size is over the limits.

    Runs before anything is written, so a refused can leaves no partial
    output. Iterates lazily so the count stops the header scan early.
    """
    count = 0
    total = 0
    for member in tar:
        count += 1
        if count > MAX_EXTRACT_MEMBERS:
            raise ValueError(
                f"can has too many members (> {MAX_EXTRACT_MEMBERS}); "
                "refusing to extract"
            )
        if member.isfile():
            total += member.size
            if total > MAX_EXTRACT_BYTES:
                raise ValueError(
                    f"can expands to more than {MAX_EXTRACT_BYTES} bytes; "
                    "refusing to extract"
                )
