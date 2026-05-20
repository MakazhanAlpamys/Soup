"""`soup env` — hermetic env lockfile + ABI-mismatch detection (v0.64.0 Part C).

The "CUDA hell" problem: a fine-tune that worked on Friday breaks on
Monday because PyPI silently upgraded ``transformers`` past the trainer's
compat band, or because the box rebuilt with a different CUDA. v0.34
``soup doctor`` surfaces some of this; v0.64 makes it lockable.

``snapshot_env`` reads Python + CUDA + key package versions from the
current interpreter via ``importlib.metadata`` (no network, no shell-out
to pip beyond what stdlib already does). ``write_lock`` persists the
snapshot as ``soup-env.lock``. ``check_abi_compat`` compares two locks
and produces an ``AbiCheck`` report listing ABI-sensitive drifts.

Live full uv/nix-backed install + recreate lands in v0.64.1; v0.64.0
ships the schema, snapshotter, comparator, and CLI surface so an
operator can capture an environment + detect drift today.

Public surface:
- ``EnvEntry`` frozen dataclass (name / version / source).
- ``EnvLock`` frozen dataclass (soup version / python / platform / cuda
  / entries tuple / timestamp).
- ``AbiCheck`` frozen dataclass (ok / drift_count / changes tuple).
- ``TRACKED_PACKAGES`` tuple of names whose drift is ABI-relevant.
- ``snapshot_env()`` -> EnvLock.
- ``write_lock(lock, path)`` / ``read_lock(path)`` -> atomic JSON.
- ``check_abi_compat(a, b)`` -> AbiCheck.
- ``DEFAULT_LOCK_FILE = "soup-env.lock"``.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform as _platform
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

from soup_cli.utils.paths import atomic_write_text, is_under_cwd

DEFAULT_LOCK_FILE = "soup-env.lock"
_MAX_NAME_LEN = 256
_MAX_VERSION_LEN = 128
_MAX_PLATFORM_LEN = 256
_MAX_PY_VERSION_LEN = 64
_MAX_CUDA_VERSION_LEN = 64
_MAX_ENTRIES = 4096

# Source allowlist — defends against schema drift on read.
_VALID_SOURCES = frozenset({"pip", "conda", "system", "wheel", "unknown"})

# ABI-sensitive packages — drift here is most likely to break training.
TRACKED_PACKAGES: Tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
    "bitsandbytes",
    "huggingface-hub",
    "flash-attn",
    "xformers",
    "deepspeed",
    "unsloth",
    "vllm",
    "sentencepiece",
    "tokenizers",
)


def _check_non_empty_str(value: object, fld: str, *, max_len: int) -> None:
    if isinstance(value, bool):
        raise TypeError(f"{fld} must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"{fld} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{fld} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{fld} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{fld} too long (> {max_len} chars)")


@dataclass(frozen=True)
class EnvEntry:
    """One package -> version row in the lockfile."""

    name: str
    version: str
    source: str

    def __post_init__(self) -> None:
        _check_non_empty_str(self.name, "name", max_len=_MAX_NAME_LEN)
        _check_non_empty_str(self.version, "version", max_len=_MAX_VERSION_LEN)
        _check_non_empty_str(self.source, "source", max_len=32)
        if self.source not in _VALID_SOURCES:
            allowed = ", ".join(sorted(_VALID_SOURCES))
            raise ValueError(
                f"source must be one of {{{allowed}}}, got {self.source!r}"
            )


@dataclass(frozen=True)
class EnvLock:
    """Captured snapshot of the running environment."""

    soup_version: str
    python_version: str
    platform: str
    cuda_version: Optional[str]
    entries: Tuple[EnvEntry, ...]
    created_at: str

    def __post_init__(self) -> None:
        _check_non_empty_str(self.soup_version, "soup_version", max_len=64)
        _check_non_empty_str(
            self.python_version, "python_version", max_len=_MAX_PY_VERSION_LEN
        )
        _check_non_empty_str(self.platform, "platform", max_len=_MAX_PLATFORM_LEN)
        if self.cuda_version is not None:
            _check_non_empty_str(
                self.cuda_version, "cuda_version", max_len=_MAX_CUDA_VERSION_LEN
            )
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be a tuple of EnvEntry")
        if len(self.entries) > _MAX_ENTRIES:
            raise ValueError(
                f"too many entries ({len(self.entries)} > {_MAX_ENTRIES})"
            )
        for entry in self.entries:
            if not isinstance(entry, EnvEntry):
                raise TypeError("every entry must be EnvEntry")
        _check_non_empty_str(self.created_at, "created_at", max_len=64)


@dataclass(frozen=True)
class AbiCheck:
    """Outcome of an ABI-compat comparison between two ``EnvLock``s."""

    ok: bool
    drift_count: int
    changes: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise TypeError("ok must be bool")
        if isinstance(self.drift_count, bool) or not isinstance(self.drift_count, int):
            raise TypeError("drift_count must be int")
        if self.drift_count < 0:
            raise ValueError("drift_count must be >= 0")
        if not isinstance(self.changes, tuple):
            raise TypeError("changes must be a tuple of str")
        for entry in self.changes:
            if not isinstance(entry, str):
                raise TypeError("changes entries must be str")


def _detect_cuda_version() -> Optional[str]:
    """Best-effort CUDA version probe via env / nvidia-smi / torch.

    Returns ``None`` if no CUDA found. Lazy-imports torch only if it's
    already in ``sys.modules`` so we never pay the import on a CPU box.

    Path-parse handles POSIX (``/usr/local/cuda-12.1``) AND Windows
    (``C:\\Program Files\\...\\CUDA\\v12.1``) — splits on every common
    separator and strips a leading ``v`` when present.
    """
    cuda_env = os.environ.get("CUDA_VERSION") or os.environ.get("CUDA_HOME")
    if cuda_env:
        # Split on both POSIX and Windows separators so e.g.
        # `C:\Program Files\...\CUDA\v12.1` produces `v12.1` as the last
        # token instead of one giant path.
        tokens = cuda_env.replace("\\", "/").split("/")
        for tok in reversed(tokens):
            cleaned = tok.lstrip("v")  # strip Windows `v` prefix
            if cleaned and cleaned[0].isdigit():
                return cleaned
            if "-" in tok:
                tail = tok.rsplit("-", 1)[-1].lstrip("v")
                if tail and tail[0].isdigit():
                    return tail
    # If torch is already imported, ask it.
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        ver = getattr(getattr(torch_mod, "version", None), "cuda", None)
        if isinstance(ver, str) and ver:
            return ver
    return None


def _detect_package_version(name: str) -> Optional[str]:
    """Read installed-package version via ``importlib.metadata``.

    Returns ``None`` for either "not installed" (``PackageNotFoundError``)
    or "metadata parse failed" (``OSError`` / ``ValueError``). All other
    exceptions propagate to surface real bugs (matches v0.33.0 #47 /
    v0.40.3 #33 narrow-except policy).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover — py < 3.8
        return None
    try:
        return version(name)
    except PackageNotFoundError:
        return None
    except (OSError, ValueError):  # pragma: no cover — corrupt metadata
        return None


def snapshot_env() -> EnvLock:
    """Capture the current Python/CUDA/package versions as an ``EnvLock``."""
    import soup_cli

    py_v = ".".join(str(x) for x in sys.version_info[:3])
    plat = f"{_platform.system().lower()}-{_platform.machine().lower()}"
    cuda_v = _detect_cuda_version()

    entries: list[EnvEntry] = []
    for name in TRACKED_PACKAGES:
        ver = _detect_package_version(name)
        if ver is None:
            continue
        try:
            entries.append(EnvEntry(name=name, version=ver, source="pip"))
        except (TypeError, ValueError):
            # Skip malformed version strings.
            continue

    return EnvLock(
        soup_version=soup_cli.__version__,
        python_version=py_v,
        platform=plat,
        cuda_version=cuda_v,
        entries=tuple(entries),
        created_at=_dt.datetime.now(_dt.timezone.utc).isoformat(),
    )


def _entry_to_dict(e: EnvEntry) -> dict:
    return {"name": e.name, "version": e.version, "source": e.source}


def _entry_from_dict(d: dict) -> EnvEntry:
    if not isinstance(d, dict):
        raise ValueError("entry must be a dict")
    return EnvEntry(
        name=str(d.get("name", "")),
        version=str(d.get("version", "")),
        source=str(d.get("source", "unknown")),
    )


def write_lock(lock: EnvLock, path: str) -> None:
    """Atomically write an ``EnvLock`` to JSON under cwd containment."""
    if not isinstance(lock, EnvLock):
        raise TypeError(f"lock must be EnvLock, got {type(lock).__name__}")
    payload = {
        "schema_version": "1",
        "soup_version": lock.soup_version,
        "python_version": lock.python_version,
        "platform": lock.platform,
        "cuda_version": lock.cuda_version,
        "entries": [_entry_to_dict(e) for e in lock.entries],
        "created_at": lock.created_at,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    atomic_write_text(text, path, prefix=".envlock.", field="env lock")


def read_lock(path: str) -> EnvLock:
    """Read a previously-written ``EnvLock`` JSON.

    Containment + symlink rejection BEFORE existence probe so a crafted
    path cannot distinguish "outside cwd" from "missing" (matches v0.55.0
    / v0.62.0 ordering policy).
    """
    import stat as _stat

    if not isinstance(path, str):
        raise TypeError(f"path must be str, got {type(path).__name__}")
    if "\x00" in path:
        raise ValueError("lock path must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"lock {path!r} is outside cwd")
    if os.path.lexists(path):
        st = os.lstat(path)
        if _stat.S_ISLNK(st.st_mode):
            raise ValueError("lock path must not be a symlink")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError("lock root must be a dict")
    entries_raw = payload.get("entries", [])
    if not isinstance(entries_raw, list):
        raise ValueError("entries must be a list")
    entries = tuple(_entry_from_dict(e) for e in entries_raw)
    return EnvLock(
        soup_version=str(payload.get("soup_version", "")),
        python_version=str(payload.get("python_version", "")),
        platform=str(payload.get("platform", "unknown")),
        cuda_version=payload.get("cuda_version"),
        entries=entries,
        created_at=str(payload.get("created_at", "")),
    )


def check_abi_compat(a: EnvLock, b: EnvLock) -> AbiCheck:
    """Compare two EnvLocks; flag ABI-sensitive drifts.

    Drift sources (in order of impact):
    1. Python minor version change.
    2. Platform string change.
    3. CUDA version change.
    4. Any TRACKED_PACKAGES version change.

    Returns ``AbiCheck(ok=True)`` iff *zero* drift sources fire.
    """
    if not isinstance(a, EnvLock):
        raise TypeError(f"a must be EnvLock, got {type(a).__name__}")
    if not isinstance(b, EnvLock):
        raise TypeError(f"b must be EnvLock, got {type(b).__name__}")

    changes: list[str] = []
    # Python minor: 3.10.x vs 3.11.x is ABI-different
    a_py = ".".join(a.python_version.split(".")[:2])
    b_py = ".".join(b.python_version.split(".")[:2])
    if a_py != b_py:
        changes.append(f"python: {a.python_version} -> {b.python_version}")

    if a.platform != b.platform:
        changes.append(f"platform: {a.platform} -> {b.platform}")

    if (a.cuda_version or None) != (b.cuda_version or None):
        changes.append(f"cuda: {a.cuda_version} -> {b.cuda_version}")

    by_a = {e.name.lower(): e.version for e in a.entries}
    by_b = {e.name.lower(): e.version for e in b.entries}
    for name in sorted(set(by_a) | set(by_b)):
        va = by_a.get(name)
        vb = by_b.get(name)
        if va != vb:
            changes.append(f"{name}: {va} -> {vb}")

    return AbiCheck(
        ok=not changes,
        drift_count=len(changes),
        changes=tuple(changes),
    )


__all__ = [
    "AbiCheck",
    "DEFAULT_LOCK_FILE",
    "EnvEntry",
    "EnvLock",
    "TRACKED_PACKAGES",
    "check_abi_compat",
    "read_lock",
    "snapshot_env",
    "write_lock",
]
