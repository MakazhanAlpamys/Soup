"""Validation and manifest creation for local cloud-training inputs."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from soup_cli.utils.paths import (
    enforce_under_cwd_and_no_symlink,
    is_under,
    refuse_linked_dirs,
)

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

        field = f"data.train[{index}]"
        supplied_path = Path(entry)
        candidate = (
            supplied_path if supplied_path.is_absolute() else root / supplied_path
        )
        lexical_path = Path(os.path.abspath(candidate))

        try:
            common = os.path.commonpath((str(root), str(lexical_path)))
        except ValueError:
            common = ""

        if os.path.normcase(common) != os.path.normcase(str(root)):
            raise ValueError(f"{field} must resolve inside the working directory")

        relative_parts = Path(os.path.relpath(lexical_path, root)).parts
        if any(part.startswith(".") for part in relative_parts):
            raise ValueError(
                f"{field}: dotfiles and dot-directories are not allowed"
            )


        try:
            refuse_linked_dirs(lexical_path.parent, stop_at=root)

        except OSError as exc:
            raise ValueError(
                f"{field}: symlinks are not allowed: {exc}"
            ) from exc

        if os.path.lexists(lexical_path):
            try:
                info = os.lstat(lexical_path)
            except OSError as exc:
                raise ValueError(
                    f"{field}: cannot access file: {exc}"
                ) from exc

            is_reparse_point = bool(
                getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            )
            if stat.S_ISLNK(info.st_mode) or is_reparse_point:
                raise ValueError(f"{field}: symlinks are not allowed")

        if root == Path.cwd().resolve():
            try:
                enforce_under_cwd_and_no_symlink(str(lexical_path), field)
            except (OSError, ValueError) as exc:
                message = str(exc)
                if "must stay under cwd" in message:
                    raise ValueError(
                        f"{field} must resolve inside the working directory"
                    ) from exc
                raise ValueError(f"{field}: {message}") from exc

        real_path = Path(os.path.realpath(lexical_path))

        if not is_under(real_path, root):
            raise ValueError(
                f"{field} must resolve inside the working directory"
            )

        if not real_path.is_file():
            raise ValueError(
                f"{field} file does not exist or is not a regular file: "
                f"{entry!r}"
            )

        if not os.access(real_path, os.R_OK):
            raise ValueError(f"{field} file is not readable: {entry!r}")

        file_stat = real_path.stat()
        size = file_stat.st_size

        if size > _MAX_LOCAL_TRAIN_INPUT_BYTES:
            raise ValueError(
                f"{field} is {size / (1024 * 1024):.2f} MiB ({size} bytes); "
                f"maximum allowed is "
                f"{_MAX_LOCAL_TRAIN_INPUT_BYTES / (1024 * 1024):.2f} MiB "
                f"({_MAX_LOCAL_TRAIN_INPUT_BYTES} bytes)"
            )

        total_bytes += size
        if total_bytes > _MAX_LOCAL_TRAIN_INPUT_BYTES:
            raise ValueError(
                f"data.train files total "
                f"{total_bytes / (1024 * 1024):.2f} MiB ({total_bytes} bytes) "
                f"after adding {field}; maximum allowed is "
                f"{_MAX_LOCAL_TRAIN_INPUT_BYTES / (1024 * 1024):.2f} MiB "
                f"({_MAX_LOCAL_TRAIN_INPUT_BYTES} bytes)"
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
