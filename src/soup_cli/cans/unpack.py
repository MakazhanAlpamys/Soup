"""Inspect, extract, and read ``.can`` artifacts (v0.26.0 Part E)."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path
from typing import Any

import yaml

from soup_cli.cans.schema import Manifest
from soup_cli.utils.paths import is_under_cwd


def _read_text_member(tar: tarfile.TarFile, name: str) -> str:
    member = tar.getmember(name)
    extracted = tar.extractfile(member)
    if extracted is None:
        raise ValueError(f"cannot read member '{name}' from can")
    return extracted.read().decode("utf-8")


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
        manifest_text = _read_text_member(tar, "manifest.yaml")
    data = yaml.safe_load(manifest_text) or {}
    return Manifest(**data)


def read_config(path: str) -> dict[str, Any]:
    """Return the config dict stored in the can."""
    can_path = Path(path)
    if not is_under_cwd(can_path):
        raise ValueError(f"can path '{path}' is outside cwd - refusing")
    if not can_path.exists():
        raise FileNotFoundError(f"can not found: {path}")
    with tarfile.open(can_path, mode="r:gz") as tar:
        cfg_text = _read_text_member(tar, "config.yaml")
    data = yaml.safe_load(cfg_text) or {}
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
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(can_path, mode="r:gz") as tar:
        _safe_extract(tar, dest)
    return dest
