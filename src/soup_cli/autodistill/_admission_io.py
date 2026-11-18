"""Bounded, read-only access to AutoDistill artifact files."""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from pathlib import Path, PureWindowsPath
from typing import TypeVar

from pydantic import BaseModel

from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    _require_safe_relative_path,
    canonical_json_bytes,
)
from soup_cli.utils.paths import open_no_follow, refuse_linked_dirs

T = TypeVar("T", bound=BaseModel)


def normalize_capture_root(root: str | os.PathLike[str]) -> Path:
    try:
        if not os.fspath(root):
            raise ValueError("capture root is empty")
        path = Path(os.path.abspath(root))
        refuse_linked_dirs(path, stop_at=path.parent)
        resolved = Path(os.path.realpath(path))
        if not resolved.is_dir():
            raise NotADirectoryError(path)
    except (OSError, ValueError) as exc:
        raise ArtifactCorruptionError(
            "capture root must be an existing, unlinked directory"
        ) from exc
    return resolved


def read_bounded_artifact(root: Path, relative_path: str, *, max_bytes: int) -> bytes:
    """Read one regular artifact once, refusing linked path components."""
    try:
        parts = _require_safe_relative_path(relative_path).split("/")
        if os.name == "nt" and any(
            ":" in part or part.endswith((".", " ")) or PureWindowsPath(part).is_reserved()
            for part in parts
        ):
            raise ValueError("artifact path contains a Windows alias")
        candidate = root.joinpath(*parts)
        refuse_linked_dirs(candidate.parent, stop_at=root.parent)
        root_identity = os.path.normcase(os.fspath(root))
        candidate_identity = os.path.normcase(os.path.realpath(candidate))
        if os.path.commonpath((root_identity, candidate_identity)) != root_identity:
            raise ValueError("artifact path escapes the capture root")
        with ExitStack() as handles:
            flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NONBLOCK", 0)
            if os.name == "posix":
                directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                parent_fd = os.open(root, directory_flags)
                handles.callback(os.close, parent_fd)
                for part in parts[:-1]:
                    parent_fd = os.open(part, directory_flags, dir_fd=parent_fd)
                    handles.callback(os.close, parent_fd)
                fd = os.open(parts[-1], flags | os.O_NOFOLLOW, dir_fd=parent_fd)
            else:
                fd = open_no_follow(candidate, flags)
            handles.callback(os.close, fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("artifact is not a regular file")
            if info.st_size > max_bytes:
                raise ValueError(f"artifact exceeds the {max_bytes}-byte limit")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError(f"artifact exceeds the {max_bytes}-byte limit")
            if os.name == "nt":
                refuse_linked_dirs(candidate.parent, stop_at=root.parent)
    except (OSError, ValueError) as exc:
        raise ArtifactCorruptionError(f"{relative_path!r}: {exc}") from exc
    return data


def load_canonical_artifact(
    root: Path,
    relative_path: str,
    model_type: type[T],
    *,
    max_bytes: int,
) -> tuple[T, bytes]:
    data = read_bounded_artifact(root, relative_path, max_bytes=max_bytes)
    try:
        model = model_type.model_validate_json(data)
        if data != canonical_json_bytes(model) + b"\n":
            raise ValueError("artifact is not canonical JSON")
    except (ValueError, TypeError) as exc:
        raise ArtifactCorruptionError(f"{relative_path!r}: {exc}") from exc
    return model, data
