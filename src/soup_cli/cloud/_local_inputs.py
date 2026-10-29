"""Validation and manifest creation for local cloud-training inputs."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig

_MAX_LOCAL_TRAIN_INPUT_BYTES = 256 * 1024 * 1024

@dataclass(frozen=True)
class LocalTrainInput:
    """A validated local file and its destination in the remote run directory."""

    local_path: str
    remote_path: str
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int

def _reject_symlink_components(path: Path, field: str) -> None:
    current = Path(path.anchor)
    for index, part in enumerate(path.parts[1:]):
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError as exc:
            if index == len(path.parts[1:]) - 1:
                raise ValueError(
                    f"{field} file does not exist: {path}"
                ) from exc
            raise ValueError(
                f"{field}: cannot access parent directory {current}: {exc}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"{field}: cannot access {current}: {exc}"
            ) from exc

        is_reparse_point = bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        )
        if stat.S_ISLNK(info.st_mode) or is_reparse_point:
            raise ValueError(f"{field}: symlinks are not allowed: {current}")

def prepare_local_train_inputs(
    cfg: SoupConfig,
    *,
    cwd: str | None = None,
) -> tuple[LocalTrainInput, ...]:
    """Validate local data.train files and rewrite them to remote-relative paths.

    Hub IDs and supported remote URIs are preserved. Local files must be regular
    files beneath the current working directory, with no symlinked components.
    """
    from soup_cli.config.schema import _classify_data_train_entry

    root = Path(cwd or os.getcwd()).resolve()
    train = cfg.data.train
    entries = [train] if isinstance(train, str) else list(train or [])
    manifest: list[LocalTrainInput] = []
    rewritten: list[str] = []
    total_bytes = 0

    for index, entry in enumerate(entries):
        if not isinstance(entry, str):
            raise ValueError(f"data.train[{index}] must be a string")

        if _classify_data_train_entry(entry) != "local":
            rewritten.append(entry)
            continue

        supplied_path = Path(entry)
        candidate = supplied_path if supplied_path.is_absolute() else root / supplied_path
        lexical_path = Path(os.path.abspath(candidate))

        _reject_symlink_components(lexical_path, f"data.train[{index}]")

        real_path = Path(os.path.realpath(lexical_path))
        try:
            if os.path.commonpath((str(root), str(real_path))) != str(root):
                raise ValueError(
                    f"data.train[{index}] must resolve inside the working directory"
                )
        except ValueError as exc:
            if "must resolve inside" in str(exc):
                raise
            raise ValueError(
                f"data.train[{index}] must resolve inside the working directory"
            ) from exc

        if not real_path.is_file():
            raise ValueError(
                f"data.train[{index}] file does not exist or is not a regular file: "
                f"{entry!r}"
            )

        file_stat = real_path.stat()
        size = file_stat.st_size
        if size > _MAX_LOCAL_TRAIN_INPUT_BYTES:
            raise ValueError(
                f"data.train[{index}] is {size} bytes; "
                f"maximum allowed is {_MAX_LOCAL_TRAIN_INPUT_BYTES} bytes"
            )

        total_bytes += size
        if total_bytes > _MAX_LOCAL_TRAIN_INPUT_BYTES:
            raise ValueError(
                f"data.train files total {total_bytes} bytes; "
                f"maximum allowed is {_MAX_LOCAL_TRAIN_INPUT_BYTES} bytes"
            )

        suffix = real_path.suffix.lower()
        remote_path = f"inputs/train-{index}{suffix}"
        manifest.append(
            LocalTrainInput(
                local_path=str(real_path),
                remote_path=remote_path,
                size_bytes=size,
                mtime_ns=file_stat.st_mtime_ns,
                device=file_stat.st_dev,
                inode=file_stat.st_ino,
            )
        )
        rewritten.append(remote_path)

    if isinstance(train, str):
        if rewritten:
            cfg.data.train = rewritten[0]
    elif train is not None:
        cfg.data.train = rewritten

    return tuple(manifest)
