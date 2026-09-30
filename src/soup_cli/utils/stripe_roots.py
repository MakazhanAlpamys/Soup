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

import os
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

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


def validate_stripe_root(entry: str, *, primary_root: str, accepted: Sequence[str] = ()) -> str:
    """Syntax, existence, symlink and overlap rules; returns the entry's realpath.

    The distinct-volume and NVMe rules live in :func:`resolve_stripe_roots`, which the setup
    path runs once. The sharder calls this one to bound its own writes without paying the
    ~9 s disk-kind probe a second time.
    """
    where = f"{STRIPE_DIRS_ENV} entry {entry!r}"
    if any(ord(ch) < 0x20 for ch in entry):
        raise StripeRootError(f"{where}: contains control characters")
    expanded = os.path.expanduser(entry)
    if not os.path.isabs(expanded):
        raise StripeRootError(f"{where}: must be an absolute path")
    if os.path.islink(expanded):
        raise StripeRootError(f"{where}: must not be a symlink")
    if not os.path.isdir(expanded):
        raise StripeRootError(
            f"{where}: must be an existing directory. Soup creates only its per-model folder "
            f"inside a stripe root, never the root itself, so a drive that is not mounted "
            f"cannot be mistaken for an empty one."
        )
    resolved = os.path.realpath(expanded)
    primary = os.path.realpath(os.path.expanduser(primary_root))
    if is_under(resolved, primary) or is_under(primary, resolved):
        raise StripeRootError(f"{where}: overlaps the primary layer-stream cache root {primary}")
    for other in accepted:
        if is_under(resolved, other) or is_under(other, resolved):
            raise StripeRootError(f"{where}: overlaps another stripe root, {other}")
    return resolved


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
    if len(entries) > MAX_STRIPE_DIRS:
        raise StripeRootError(
            f"{STRIPE_DIRS_ENV} names {len(entries)} folders; at most {MAX_STRIPE_DIRS} "
            f"(training.stream_read_ahead stops at {MAX_STREAM_READ_AHEAD}, and every drive "
            f"needs a staging slot of its own)"
        )
    accepted: List[str] = []
    owners = {volume_of(primary_root): f"the primary cache root {primary_root}"} if entries else {}
    for entry in entries:
        resolved = validate_stripe_root(entry, primary_root=primary_root, accepted=accepted)
        volume = volume_of(resolved)
        if volume in owners:
            raise StripeRootError(
                f"{STRIPE_DIRS_ENV} entry {entry!r}: on the same volume as {owners[volume]}. "
                f"Two roots on one drive read no faster than one and cost a staging slot."
            )
        kind = disk_kind(resolved)
        if kind != "nvme":
            raise StripeRootError(
                f"{STRIPE_DIRS_ENV} entry {entry!r}: the disk tier streams from NVMe only, and "
                f"this volume classifies as {kind!r}. If the probe is wrong, "
                f"training.stream_disk_kind overrides it."
            )
        owners[volume] = f"stripe root {resolved}"
        accepted.append(resolved)
    return tuple(accepted)


def layer_roots_for(n_layers: int, n_roots: int) -> Tuple[int, ...]:
    """Decoder layer ``i`` lives on root ``i mod n_roots``; ``()`` means one root."""
    if n_roots < 1:
        raise ValueError(f"n_roots must be at least 1; got {n_roots}")
    if n_roots == 1:
        return ()
    return tuple(idx % n_roots for idx in range(n_layers))
