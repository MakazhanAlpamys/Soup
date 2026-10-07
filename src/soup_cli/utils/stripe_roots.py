"""``SOUP_LAYER_STREAM_STRIPE_DIRS`` — extra NVMe roots the layer-shard cache stripes over.

The primary cache root (``SOUP_LAYER_STREAM_CACHE_DIR`` or ``~/.soup/layer-stream``) keeps its
meaning exactly, lenient fallback included. This variable is deliberately the opposite: setting
it is an explicit act, so every entry is validated and a bad one is refused by name. A silent
fallback here would train at single-drive speed with nothing said, which is the failure the
feature exists to remove.

Why it exists, measured: ``benchmarks/probe-rtx5070-two-drive-read.md`` — two NVMe drives read
7.65-9.15 GB/s together where one reads ~4.

No top-level torch: imported on the setup path only.
"""

import hashlib
import os
import stat
from dataclasses import dataclass
from typing import Callable, Iterator, List, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.config_bounds import (
    DEFAULT_STREAM_READ_AHEAD,
    MAX_STREAM_READ_AHEAD,
    MIN_STREAM_READ_AHEAD,
)
from soup_cli.utils.paths import is_under

STRIPE_DIRS_ENV = "SOUP_LAYER_STREAM_STRIPE_DIRS"

#: With N roots the reader wants N + 1 staging slots (one read in flight per drive plus the
#: slot the consumer holds), and ``stream_read_ahead`` stops at MAX_STREAM_READ_AHEAD, so the
#: primary root plus this many extra roots is the most the depth can serve.
MAX_STRIPE_DIRS = MAX_STREAM_READ_AHEAD - 2

#: Windows device-namespace spellings (``\\?\C:\...``, ``\\.\C:\...``), refused by name.
_DEVICE_NAMESPACE_PREFIXES = ("\\\\?\\", "\\\\.\\")


class StripeRootError(ValueError):
    """A stripe root that breaks a rule; the message names the entry and the rule."""


def effective_read_ahead(configured: int, n_roots: int) -> int:
    """The async reader's depth for ``configured`` over ``n_roots`` drives.

    Every drive needs a read in flight beside the slot the consumer holds, so the DEFAULT depth
    becomes ``n_roots + 1``, capped at MAX_STREAM_READ_AHEAD. "Default" means the configured
    value equals DEFAULT_STREAM_READ_AHEAD — a default is not a decision (``config/schema.py``),
    and a config round-tripped through ``model_dump`` looks explicitly set everywhere. Any
    other value is the operator's and is kept.
    """
    configured = int(configured)
    if n_roots <= 1 or configured != DEFAULT_STREAM_READ_AHEAD:
        return configured
    return max(configured, min(n_roots + 1, MAX_STREAM_READ_AHEAD))


@dataclass(frozen=True)
class ReadAheadDecision:
    """The depth the reader uses, what the operator configured, and over how many drives.

    Carried from the setup to every message that advises lowering the depth: after the N + 1
    bump, "lower it" is a loop — the operator writes 2, which is the default, and it is raised
    again. :meth:`lowering_advice` names a value that the rule does NOT raise back up.
    """

    depth: int
    configured: int
    n_roots: int = 1

    @property
    def raised(self) -> bool:
        """Soup deepened the configured value because the cache spans several drives."""
        return self.depth > self.configured

    def _next_lower(self) -> Optional[int]:
        """The largest configurable value whose effective depth is below this one."""
        for candidate in range(self.depth - 1, MIN_STREAM_READ_AHEAD - 1, -1):
            if effective_read_ahead(candidate, self.n_roots) < self.depth:
                return candidate
        return None

    def lowering_advice(self) -> str:
        """A clause telling the operator how to lower the depth, or ``""`` at the floor.

        One drive keeps the historical wording. With several, the clause names a value that
        actually lowers the depth, says when the depth was raised (and that unsetting the
        variable undoes that), and says when the default value would be re-raised.
        """
        lower = self._next_lower()
        if lower is None:
            return ""
        if self.n_roots <= 1:
            return f"Lower training.stream_read_ahead (currently {self.depth})"
        notes = []
        if self.raised:
            notes.append(
                f"raised from {self.configured} because the layer cache spans "
                f"{self.n_roots} drives"
            )
        if lower < DEFAULT_STREAM_READ_AHEAD < self.depth:
            bumped = effective_read_ahead(DEFAULT_STREAM_READ_AHEAD, self.n_roots)
            notes.append(
                f"with {self.n_roots} drives, {DEFAULT_STREAM_READ_AHEAD} counts as the "
                f"default and becomes {bumped}"
            )
        said = f"Set training.stream_read_ahead to {lower} (currently {self.depth}"
        said += "".join(f"; {note}" for note in notes) + ")"
        if self.raised:
            said += f", unset {STRIPE_DIRS_ENV} or name fewer folders in it"
        return said


def parse_stripe_dirs(raw: Optional[str]) -> Tuple[str, ...]:
    """Split the variable on ``os.pathsep``; blank entries are dropped, not refused."""
    if raw is None:
        return ()
    return tuple(part.strip() for part in raw.split(os.pathsep) if part.strip())


def _nearest_existing(path: str) -> str:
    anchor = os.path.realpath(os.path.expanduser(path))
    while not os.path.exists(anchor):
        parent = os.path.dirname(anchor)
        if parent == anchor:
            raise StripeRootError(f"cannot find an existing folder above {path!r}")
        anchor = parent
    return anchor


def volume_of(path: str) -> int:
    """The volume ``path`` lives on: ``st_dev``, the volume serial number on Windows."""
    return int(os.stat(_nearest_existing(path)).st_dev)


def check_stripe_root(
    entry: str,
    *,
    primary_root: str,
    accepted: Sequence[str] = (),
) -> Tuple[Optional[str], Optional[str]]:
    """Syntax, existence, symlink and overlap rules; returns ``(resolved, reason)``.

    If valid, returns ``(resolved_realpath, None)``.
    If invalid, returns ``(None, reason)`` where ``reason`` does NOT contain the
    environment variable or entry prefix.
    """
    if any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in entry):
        return None, "contains control characters"
    if os.name == "nt" and entry.replace("/", "\\").startswith(_DEVICE_NAMESPACE_PREFIXES):
        return None, (
            "a device-namespace path is not accepted; name the folder by its drive "
            "letter or UNC share"
        )
    expanded = os.path.expanduser(entry)
    if not os.path.isabs(expanded):
        return None, "must be an absolute path"
    if os.name == "nt" and not os.path.splitdrive(expanded)[0]:
        return None, "must name a drive letter or a UNC share"
    if os.path.islink(expanded):
        return None, "must not be a symlink"
    if not os.path.isdir(expanded):
        return None, (
            "must be an existing directory. Soup creates only its per-model folder "
            "inside a stripe root, never the root itself, so a drive that is not mounted "
            "cannot be mistaken for an empty one. Reconnect the drive, or remove this entry "
            f"(unset {STRIPE_DIRS_ENV} to re-shard to one root)."
        )
    resolved = os.path.realpath(expanded)
    primary = os.path.realpath(os.path.expanduser(primary_root))
    if is_under(resolved, primary) or is_under(primary, resolved):
        return None, f"overlaps the primary layer-stream cache root {primary}"
    for other in accepted:
        if is_under(resolved, other) or is_under(other, resolved):
            return None, f"overlaps another stripe root, {other}"
    return resolved, None


def validate_stripe_root(entry: str, *, primary_root: str, accepted: Sequence[str] = ()) -> str:
    """Syntax, existence, symlink and overlap rules; returns the entry's realpath.

    The distinct-volume and NVMe rules live in :func:`resolve_stripe_roots`, which the setup
    path runs once. The sharder calls this one to bound its own writes without paying the
    ~9 s disk-kind probe a second time.
    """
    resolved, reason = check_stripe_root(entry, primary_root=primary_root, accepted=accepted)
    if reason is not None:
        raise StripeRootError(f"{STRIPE_DIRS_ENV} entry {entry!r}: {reason}")
    assert resolved is not None
    return resolved


def iter_early_stripe_roots(
    primary_root: str,
    entries: Sequence[str],
) -> Iterator[Tuple[str, Optional[str], Optional[str]]]:
    """Validate stripe root entries early, yielding ``(entry, resolved, reason)``.

    ``resolved`` is the resolved canonical path when valid, or ``None``.
    ``reason`` is the refusal message without environment variable prefix when invalid,
    or ``None``.
    """
    accepted: List[str] = []
    primary_canonical = os.path.realpath(os.path.expanduser(primary_root))
    owners = {volume_of(primary_canonical): f"the primary cache root {primary_canonical}"}

    for idx, entry in enumerate(entries):
        if len(entries) > MAX_STRIPE_DIRS and idx >= MAX_STRIPE_DIRS:
            yield (
                entry,
                None,
                (
                    f"names {len(entries)} folders; at most {MAX_STRIPE_DIRS} "
                    f"(training.stream_read_ahead stops at {MAX_STREAM_READ_AHEAD}, "
                    f"and every drive needs a staging slot of its own)"
                ),
            )
            continue

        resolved, reason = check_stripe_root(
            entry, primary_root=primary_canonical, accepted=accepted
        )
        if reason is not None:
            yield (entry, None, reason)
            continue

        assert resolved is not None
        volume = volume_of(resolved)
        if volume in owners:
            yield (
                entry,
                None,
                (
                    f"on the same volume as {owners[volume]}. "
                    f"Two roots on one drive read no faster than one and cost a staging slot."
                ),
            )
            continue

        owners[volume] = f"stripe root {resolved}"
        accepted.append(resolved)
        yield (entry, resolved, None)


def validate_early_stripe_roots(
    primary_root: str,
    *,
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    """Syntax, existence, count cap, symlink, overlap and same-volume rules.

    Runs early in the CLI (``soup train``, ``--dry-run``) before expensive dataset
    loading or run creation. Omits the ~9 s NVMe disk-kind probe, which stays in
    :func:`resolve_stripe_roots` on the setup path.
    """
    env = os.environ if environ is None else environ
    entries = parse_stripe_dirs(env.get(STRIPE_DIRS_ENV))
    if not entries:
        return ()
    if len(entries) > MAX_STRIPE_DIRS:
        raise StripeRootError(
            f"{STRIPE_DIRS_ENV} names {len(entries)} folders; at most {MAX_STRIPE_DIRS} "
            f"(training.stream_read_ahead stops at {MAX_STREAM_READ_AHEAD}, and every drive "
            f"needs a staging slot of its own)"
        )
    accepted: List[str] = []
    for entry, resolved, reason in iter_early_stripe_roots(primary_root, entries):
        if reason is not None:
            raise StripeRootError(f"{STRIPE_DIRS_ENV} entry {entry!r}: {reason}")
        assert resolved is not None
        accepted.append(resolved)
    return tuple(accepted)


def resolve_stripe_roots(
    primary_root: str,
    *,
    disk_kind: Callable[[str], str],
    environ: Optional[Mapping[str, str]] = None,
) -> Tuple[str, ...]:
    """The validated stripe roots, or ``()`` when the variable is unset or blank.

    ``disk_kind`` maps a path to ``nvme`` / ``ssd`` / ``hdd`` / ``unknown``; the setup path
    passes ``resolve_disk_kind(...).kind`` so ``training.stream_disk_kind`` overrides the probe
    here exactly as it does for the primary cache.
    """
    env = os.environ if environ is None else environ
    entries = parse_stripe_dirs(env.get(STRIPE_DIRS_ENV))
    if not entries:
        return ()
    accepted = validate_early_stripe_roots(primary_root, environ=env)
    for entry, resolved in zip(entries, accepted):
        kind = disk_kind(resolved)
        if kind != "nvme":
            raise StripeRootError(
                f"{STRIPE_DIRS_ENV} entry {entry!r}: the disk tier streams from NVMe only, and "
                f"this volume classifies as {kind!r}. If the probe is wrong, "
                f"training.stream_disk_kind overrides it."
            )
    return accepted


def layer_roots_for(n_layers: int, n_roots: int) -> Tuple[int, ...]:
    """Decoder layer ``i`` lives on root ``i mod n_roots``; ``()`` means one root."""
    if n_roots < 1:
        raise ValueError(f"n_roots must be at least 1; got {n_roots}")
    if n_roots == 1:
        return ()
    return tuple(idx % n_roots for idx in range(n_layers))


# -- the per-model folder inside a stripe root ------------------------------------------------
def primary_cache_identity(shard_dir: str) -> str:
    """The primary cache a stripe folder belongs to: its realpath, case-folded on Windows."""
    return os.path.normcase(os.path.realpath(os.path.abspath(os.path.expanduser(shard_dir))))


def stripe_folder_name(shard_dir: str) -> str:
    """The folder a stripe root holds for the primary cache at ``shard_dir``.

    The model slug plus 12 hex of a hash of the primary cache's realpath. Two primary caches of
    one base (per user, or per ``SOUP_LAYER_STREAM_CACHE_DIR``, or with different dtypes) that
    share one stripe root therefore never share a folder, so neither can read the other's layer
    files as its own or overwrite them on a re-shard. The slug keeps the folder recognisable.
    """
    slug = os.path.basename(os.path.normpath(shard_dir))
    digest = hashlib.sha256(os.fsencode(primary_cache_identity(shard_dir))).hexdigest()[:12]
    return f"{slug}-{digest}"


def _is_link(info: os.stat_result) -> bool:
    """A symlink, or on Windows any reparse point: a junction does not report S_ISLNK (and
    ``os.path.islink`` misses it before 3.12), but it redirects exactly as a symlink does."""
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return os.name == "nt" and bool(getattr(info, "st_file_attributes", 0) & reparse)


def stripe_folder_problem(folder: str, root: str) -> Optional[str]:
    """Why ``folder`` must not be read, written or deleted in, or ``None``.

    ``root`` is the VALIDATED stripe root (a realpath) and ``folder`` its per-model child. A
    missing folder is not a problem here — the caller creates it or reports it. A link or
    junction at the folder is, and so is a folder whose realpath is not ``root``'s child: that
    catches a root swapped for a link after it was validated, which lstat on the folder cannot.
    """
    try:
        info = os.lstat(folder)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"the stripe folder {folder} cannot be inspected ({exc})"
    if _is_link(info):
        return (
            f"the stripe folder {folder} is a link or junction. Soup reads and writes only a "
            f"real folder it made inside a {STRIPE_DIRS_ENV} entry, so it will not follow one; "
            f"remove the link (Soup never deletes inside a stripe root) or name another folder"
        )
    if not stat.S_ISDIR(info.st_mode):
        return f"the stripe folder {folder} exists and is not a folder"
    expected = os.path.join(root, os.path.basename(folder))
    actual = os.path.realpath(folder)
    if os.path.normcase(actual) != os.path.normcase(expected):
        return (
            f"the stripe folder {folder} resolves to {actual}, not to a folder inside its "
            f"stripe root {root}; a link somewhere on that path changed after it was checked"
        )
    return None


def secure_stripe_folder(folder: str, root: str) -> None:
    """Make ``folder`` (inside the validated ``root``) Soup's own before anything touches it.

    Refuses a link or junction; then creates the folder owner-only, or verifies that an
    existing one belongs to this account (and nobody else can write it) and restricts it
    again. Files there are trusted between runs as the owner-only primary cache's are, so a
    folder that cannot be made private refuses the run by name rather than being used.
    """
    from soup_cli.utils import owner_only_dir

    problem = stripe_folder_problem(folder, root)
    if problem is not None:
        raise StripeRootError(f"{STRIPE_DIRS_ENV}: {problem}.")
    try:
        owner_only_dir.ensure_owner_only_dir(folder)
    except OSError as exc:
        raise StripeRootError(
            f"{STRIPE_DIRS_ENV}: the stripe folder {folder} cannot be made private to the "
            f"account running Soup: {exc}. Its layer files are trusted between runs, as the "
            f"owner-only primary cache's are, so Soup refuses rather than use a folder another "
            f"account could change. Remove it (Soup never deletes inside a stripe root) or "
            f"point {STRIPE_DIRS_ENV} at a folder this account owns."
        ) from exc
    problem = stripe_folder_problem(folder, root)
    if problem is not None:
        raise StripeRootError(f"{STRIPE_DIRS_ENV}: {problem}.")
