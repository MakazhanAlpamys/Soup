"""Canary discovery — `soup eval discover` (v0.55.0 Part B).

Splits a training dataset into three behaviour-bearing groups:

* ``held_out`` — rows representative of the dominant clusters; tests
  whether learned behaviour generalises.
* ``adjacent_skills`` — rows that look superficially similar but cover
  different lexical themes; tests for catastrophic forgetting.
* ``memorization_probes`` — partial-prompt versions of training rows that
  trip if the adapter regurgitates training prefixes verbatim.

Pure functions — no torch / no GPU. The base model is accepted as a
string for the signature compatibility with v0.56.0 ``soup diagnose``;
the helper does not load it.

Public surface
--------------
- Frozen dataclass: ``CanarySet``.
- Pure function: ``discover_canaries``, ``write_canary_set``,
  ``load_canary_set``.
"""

from __future__ import annotations

import json
import os
import random
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from soup_cli.utils._eval_text import row_text as _row_text
from soup_cli.utils._eval_text import tokenize as _tokenize
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, is_under_cwd

_MAX_ROWS = 1_000_000
_MAX_FILE_BYTES = 16 * 1024 * 1024
_MAX_CANARIES_PER_GROUP = 1024
_MIN_PER_CLUSTER = 1

# Subsample cap inside the clustering hot path. Defends against the
# O(num_clusters × N) farthest-first centroid scan blowing up on huge
# datasets (code-review HIGH fix) — every dataset over this size is
# truncated to a deterministic prefix for clustering only; the canary
# *outputs* still cover the visible prefix.
_CLUSTER_SUBSAMPLE = 10_000


@dataclass(frozen=True)
class CanarySet:
    """Three groups of canary prompts derived from training data.

    The fields are tuples (not lists) so the dataclass is genuinely
    immutable post-construction — mutation would otherwise silently
    bypass the dedup logic.
    """

    held_out: tuple[str, ...]
    adjacent_skills: tuple[str, ...]
    memorization_probes: tuple[str, ...]
    cluster_count: int
    base: str | None = None
    dimensions: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_int(value: object, *, field_name: str, lo: int, hi: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, got bool")
    if not isinstance(value, int):
        raise TypeError(
            f"{field_name} must be int, got {type(value).__name__}"
        )
    if value < lo or value > hi:
        raise ValueError(f"{field_name} must be in [{lo}, {hi}]")
    return value


def _validate_base(base: object) -> str | None:
    if base is None:
        return None
    if isinstance(base, bool):
        raise TypeError("base must be str or None, got bool")
    if not isinstance(base, str):
        raise TypeError(f"base must be str or None, got {type(base).__name__}")
    if "\x00" in base:
        raise ValueError("base must not contain NUL bytes")
    if len(base) > 512:
        raise ValueError("base exceeds 512 characters")
    return base


# ---------------------------------------------------------------------------
# Clustering — tiny k-means-flavoured token-set partitioning
# ---------------------------------------------------------------------------

def _row_signature(row: Mapping[str, object]) -> frozenset:
    """Compact lexical signature for clustering."""
    return frozenset(_tokenize(_row_text(row)))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _cluster_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    k: int,
    seed: int,
) -> list[list[int]]:
    """Greedy farthest-first clustering — deterministic given ``seed``.

    Picks ``k`` rows whose signatures maximise pairwise Jaccard distance,
    then assigns every row to its nearest centroid. Returns a list of
    index buckets in order of seed centroid pick.

    For datasets larger than ``_CLUSTER_SUBSAMPLE`` rows, only the
    deterministic prefix is clustered (the v0.55.0 DoS cap).
    """
    if not rows:
        return []
    if k <= 0:
        return [list(range(len(rows)))]
    n = min(len(rows), _CLUSTER_SUBSAMPLE)
    sigs = [_row_signature(rows[i]) for i in range(n)]
    rng = random.Random(seed)
    first = rng.randrange(n)
    centroid_idx: list[int] = [first]
    while len(centroid_idx) < min(k, n):
        best_i = -1
        best_min_dist = -1.0
        for i in range(n):
            if i in centroid_idx:
                continue
            min_dist = min(
                1.0 - _jaccard(sigs[i], sigs[c]) for c in centroid_idx
            )
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_i = i
        if best_i < 0:
            break
        centroid_idx.append(best_i)
    buckets: list[list[int]] = [[] for _ in centroid_idx]
    for i, sig in enumerate(sigs):
        # Tie-break by earliest centroid (lower index) for determinism.
        best_c = 0
        best_sim = -1.0
        for cj, c in enumerate(centroid_idx):
            sim = _jaccard(sig, sigs[c])
            if sim > best_sim:
                best_sim = sim
                best_c = cj
        buckets[best_c].append(i)
    return buckets


# ---------------------------------------------------------------------------
# Canary derivation
# ---------------------------------------------------------------------------

def _prompt_text(row: Mapping[str, object]) -> str:
    """Best-effort prompt extraction (input side)."""
    if not isinstance(row, Mapping):
        return ""
    messages = row.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        for msg in messages:
            if not isinstance(msg, Mapping):
                continue
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    return content
    for key in ("prompt", "input", "question", "instruction"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    # Fallback to the row's output text — better to have something than
    # to produce an empty canary group.
    return _row_text(row)


def _memorization_probe(prompt: str) -> str:
    """Truncate prompt to first ~25% — the rest tests for regurgitation."""
    if not prompt:
        return ""
    words = prompt.split()
    if len(words) <= 4:
        return prompt
    cut = max(4, len(words) // 4)
    return " ".join(words[:cut])


def discover_canaries(
    rows: Sequence[Mapping[str, object]],
    *,
    base: str | None = None,
    num_clusters: int = 5,
    per_cluster: int = 3,
    seed: int = 0,
    dimensions: Sequence[str] | None = None,
) -> CanarySet:
    """Build a :class:`CanarySet` from training rows.

    Algorithm:
      1. Cluster rows by token-set Jaccard distance.
      2. Pick top-N rows from each cluster as ``held_out`` (in-distribution).
      3. Pick top-N rows from the *smallest* clusters as
         ``adjacent_skills`` (low-frequency themes → forgetting probes).
      4. Truncate every held-out prompt to first 25% for
         ``memorization_probes``.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError("rows must be a sequence of mapping rows")
    if len(rows) > _MAX_ROWS:
        raise ValueError(f"rows exceed cap of {_MAX_ROWS}")
    base = _validate_base(base)
    num_clusters = _require_int(
        num_clusters, field_name="num_clusters", lo=1, hi=64,
    )
    per_cluster = _require_int(
        per_cluster, field_name="per_cluster", lo=1, hi=64,
    )
    seed = _require_int(seed, field_name="seed", lo=0, hi=2**31 - 1)
    if dimensions is not None:
        if isinstance(dimensions, (str, bytes)) or not isinstance(
            dimensions, Sequence
        ):
            raise TypeError("dimensions must be a sequence of strings")
        for d in dimensions:
            if not isinstance(d, str) or "\x00" in d:
                raise ValueError("dimension names must be NUL-free strings")

    buckets = _cluster_rows(rows, k=num_clusters, seed=seed)
    held_out: list[str] = []
    adjacent: list[str] = []
    probes: list[str] = []
    seen_held: set = set()
    seen_adjacent: set = set()
    seen_probes: set = set()

    # Held-out: take per_cluster rows from each non-empty bucket in order.
    for bucket in buckets:
        for idx in bucket[:per_cluster]:
            text = _prompt_text(rows[idx])
            if text and text not in seen_held:
                held_out.append(text)
                seen_held.add(text)
                probe = _memorization_probe(text)
                if probe and probe not in seen_probes:
                    probes.append(probe)
                    seen_probes.add(probe)
            if len(held_out) >= _MAX_CANARIES_PER_GROUP:
                break
        if len(held_out) >= _MAX_CANARIES_PER_GROUP:
            break

    # Adjacent skills: pull from the smallest buckets (rarest behaviours).
    small_buckets = sorted(buckets, key=len)[:max(1, len(buckets) // 2)]
    for bucket in small_buckets:
        for idx in bucket[-per_cluster:]:  # tail of small bucket = rarer
            text = _prompt_text(rows[idx])
            if text and text not in seen_adjacent and text not in seen_held:
                adjacent.append(text)
                seen_adjacent.add(text)
            if len(adjacent) >= _MAX_CANARIES_PER_GROUP:
                break
        if len(adjacent) >= _MAX_CANARIES_PER_GROUP:
            break

    return CanarySet(
        held_out=tuple(held_out),
        adjacent_skills=tuple(adjacent),
        memorization_probes=tuple(probes),
        cluster_count=len([b for b in buckets if b]),
        base=base,
        dimensions=tuple(dimensions) if dimensions else tuple(),
    )


def canary_set_to_dict(canary: CanarySet) -> dict[str, object]:
    if not isinstance(canary, CanarySet):
        raise TypeError("canary must be a CanarySet")
    return {
        "held_out": list(canary.held_out),
        "adjacent_skills": list(canary.adjacent_skills),
        "memorization_probes": list(canary.memorization_probes),
        "cluster_count": canary.cluster_count,
        "base": canary.base,
        "dimensions": list(canary.dimensions),
    }


def write_canary_set(canary: CanarySet, output_path: str) -> str:
    """Atomic write of a canary set with cwd containment + symlink reject."""
    enforce_under_cwd_and_no_symlink(output_path, "output_path")
    payload = json.dumps(
        canary_set_to_dict(canary), ensure_ascii=False, indent=2
    )
    if len(payload.encode("utf-8")) > _MAX_FILE_BYTES:
        raise ValueError("rendered canary set exceeds 16 MiB cap")
    parent = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".soup-canaries.", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(tmp, output_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return output_path


def load_canary_set(path: str) -> CanarySet:
    if not isinstance(path, str):
        raise TypeError("path must be str")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain NUL")
    if not is_under_cwd(path):
        raise ValueError("path must stay under cwd")
    # Unconditional lstat — TOCTOU defence parity with eval_design.py.
    try:
        st = os.lstat(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"canary set file not found: {os.path.basename(path)}"
        ) from exc
    except OSError as exc:
        raise ValueError(
            f"path unreadable: {type(exc).__name__}"
        ) from exc
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("path must not be a symlink (TOCTOU defence)")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(f"file exceeds {_MAX_FILE_BYTES} byte cap")
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, Mapping):
        raise ValueError("canary JSON root must be an object")

    def _string_list(key: str) -> tuple[str, ...]:
        raw = data.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"{key} must be a list")
        out: list[str] = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(f"{key} entries must be strings")
            if "\x00" in item:
                raise ValueError(f"{key} entry contains NUL")
            out.append(item)
        return tuple(out)

    cluster_count = data.get("cluster_count", 0)
    if isinstance(cluster_count, bool) or not isinstance(cluster_count, int):
        raise ValueError("cluster_count must be int")
    base = data.get("base")
    if base is not None and not isinstance(base, str):
        raise ValueError("base must be string or null")
    return CanarySet(
        held_out=_string_list("held_out"),
        adjacent_skills=_string_list("adjacent_skills"),
        memorization_probes=_string_list("memorization_probes"),
        cluster_count=cluster_count,
        base=base,
        dimensions=_string_list("dimensions"),
    )
