"""Airgap bundle assembler (v0.60.0 Part F).

Builds a single tarball containing model + datasets + (optional) wheels +
(optional) CUDA kernels + signed manifest with SHA-256 fingerprints for
every member. Sized for one-way physical-media transfer through a data
diode — operators on isolated networks can verify-then-unpack offline.

The bundle is reproducible: same inputs → same SHA-256 list inside the
manifest (tar headers themselves are non-deterministic — timestamps — so
the tarball bytes are not bit-stable, but the bundled-files list is).

Public surface:
- ``AirgapBundlePlan`` frozen dataclass — caller-supplied inputs.
- ``BundleFileEntry`` / ``BundleManifest`` frozen dataclasses.
- ``build_airgap_bundle(plan)`` -> ``BundleManifest``.
- ``inspect_airgap_bundle(path)`` -> ``BundleManifest`` (reads back).
- Default cap: 100 GiB; configurable.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import soup_cli
from soup_cli.utils.paths import (
    enforce_under_cwd_and_no_symlink,
    is_under_cwd,
)

_DEFAULT_BUNDLE_CAP_BYTES = 100 * 1024 * 1024 * 1024  # 100 GiB
_MANIFEST_FILENAME = "manifest.json"
_REPRO_RECEIPT_FILENAME = "repro-receipt.json"  # v0.71.3 #188
_MAX_FILES = 100_000
_HASH_CHUNK = 1024 * 1024
_MAX_MANIFEST_BYTES = 64 * 1024 * 1024  # 64 MiB cap on inspect-side manifest read


@dataclass(frozen=True)
class AirgapBundlePlan:
    """Caller-supplied bundle inputs."""

    output: str
    model_dir: str
    dataset_dirs: Tuple[str, ...]
    wheel_dirs: Tuple[str, ...]
    kernel_dirs: Tuple[str, ...]
    bundle_size_cap_bytes: int
    # v0.71.3 #188 — optional reproducibility receipt to embed.
    repro_receipt: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.output, "output"),
            (self.model_dir, "model_dir"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty str")
            if "\x00" in value:
                raise ValueError(f"{name} must not contain null bytes")
        for sequence, name in (
            (self.dataset_dirs, "dataset_dirs"),
            (self.wheel_dirs, "wheel_dirs"),
            (self.kernel_dirs, "kernel_dirs"),
        ):
            if not isinstance(sequence, tuple):
                raise ValueError(f"{name} must be tuple")
            for entry in sequence:
                if not isinstance(entry, str) or not entry:
                    raise ValueError(f"{name}[*] must be non-empty str")
                if "\x00" in entry:
                    raise ValueError(f"{name}[*] must not contain null bytes")
        if isinstance(self.bundle_size_cap_bytes, bool):
            raise ValueError("bundle_size_cap_bytes must be int, not bool")
        if not isinstance(self.bundle_size_cap_bytes, int):
            raise ValueError("bundle_size_cap_bytes must be int")
        if self.bundle_size_cap_bytes <= 0:
            raise ValueError("bundle_size_cap_bytes must be > 0")
        if self.repro_receipt is not None and not isinstance(
            self.repro_receipt, Mapping
        ):
            raise ValueError("repro_receipt must be a mapping or None")


@dataclass(frozen=True)
class BundleFileEntry:
    """One file in the bundle manifest."""

    name: str    # path-inside-tar (no leading slash)
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleManifest:
    """Signed manifest inside the airgap bundle."""

    soup_version: str
    created_at: str
    model_dir: str
    datasets: Tuple[str, ...]
    wheels: Tuple[str, ...]
    kernels: Tuple[str, ...]
    files: Tuple[BundleFileEntry, ...]
    total_bytes: int
    # v0.71.3 #188 — embedded reproducibility receipt (None when absent).
    repro_receipt: Optional[dict] = None


def _hash_file(path: str) -> tuple[int, str]:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError(f"{os.path.basename(path)!r}: must not be a symlink")
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f"{os.path.basename(path)!r}: must be a regular file")
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(_HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _enumerate_dir(root: str, *, label: str) -> list[tuple[str, str]]:
    """Walk ``root`` and return ``[(abs_path, rel_in_tar)]`` tuples.

    ``label`` becomes the top-level dir name inside the tarball
    (e.g. ``model/``, ``datasets/<i>/``).
    """
    out: list[tuple[str, str]] = []
    for dirpath, _, filenames in os.walk(root, followlinks=False):
        if not is_under_cwd(dirpath):
            continue
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            # Skip symlinks at the file level.
            try:
                st = os.lstat(full)
            except OSError:
                continue
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                continue
            rel = os.path.relpath(full, root)
            # Use forward slashes inside tar entries (POSIX-style).
            rel_tar = label + "/" + rel.replace(os.sep, "/")
            out.append((full, rel_tar))
    return out


def build_airgap_bundle(plan: AirgapBundlePlan) -> BundleManifest:
    """Assemble the bundle. Returns the in-memory manifest.

    Raises ``ValueError`` on:
    - output / model_dir / dataset_dir outside cwd (containment)
    - aggregate bytes > ``bundle_size_cap_bytes``
    - any symlink encountered (refused at the boundary)
    """
    if not isinstance(plan, AirgapBundlePlan):
        raise TypeError("plan must be AirgapBundlePlan")

    enforce_under_cwd_and_no_symlink(plan.output, "output")
    enforce_under_cwd_and_no_symlink(plan.model_dir, "model_dir")
    for entry in plan.dataset_dirs:
        enforce_under_cwd_and_no_symlink(entry, "dataset_dir")
    for entry in plan.wheel_dirs:
        enforce_under_cwd_and_no_symlink(entry, "wheel_dir")
    for entry in plan.kernel_dirs:
        enforce_under_cwd_and_no_symlink(entry, "kernel_dir")

    members: list[tuple[str, str]] = []
    members.extend(_enumerate_dir(plan.model_dir, label="model"))
    # Label by sorted basename rather than caller-supplied index so the
    # manifest is reorder-stable: `[a, b]` and `[b, a]` produce the same
    # tarball layout, fulfilling the "same inputs → same SHA-256 list"
    # contract (code-review HIGH fix).
    def _label_for(dirs: tuple[str, ...], prefix: str) -> list[tuple[str, str]]:
        sorted_dirs = sorted(dirs, key=lambda d: os.path.basename(os.path.normpath(d)))
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for d in sorted_dirs:
            base = os.path.basename(os.path.normpath(d)) or "_"
            label = f"{prefix}/{base}"
            # Disambiguate on duplicate basename collisions.
            i = 0
            while label in seen:
                i += 1
                label = f"{prefix}/{base}__{i}"
            seen.add(label)
            out.extend(_enumerate_dir(d, label=label))
        return out

    members.extend(_label_for(plan.dataset_dirs, "datasets"))
    members.extend(_label_for(plan.wheel_dirs, "wheels"))
    members.extend(_label_for(plan.kernel_dirs, "kernels"))

    if len(members) > _MAX_FILES:
        raise ValueError(f"bundle has > {_MAX_FILES} files")

    # v0.71.3 #188 — serialise the optional reproducibility receipt up front so
    # it counts toward the size cap and lands as a top-level manifest member.
    receipt_dict: Optional[dict] = None
    receipt_bytes: Optional[bytes] = None
    if plan.repro_receipt is not None:
        receipt_dict = dict(plan.repro_receipt)
        receipt_bytes = json.dumps(
            receipt_dict, indent=2, sort_keys=True
        ).encode("utf-8")

    # Hash + pre-size check (refuse early if cap exceeded).
    entries: list[BundleFileEntry] = []
    total = 0
    if receipt_bytes is not None:
        total += len(receipt_bytes)
        if total > plan.bundle_size_cap_bytes:
            raise ValueError(
                f"bundle aggregate size {total} bytes exceeds cap "
                f"{plan.bundle_size_cap_bytes} bytes"
            )
        entries.append(
            BundleFileEntry(
                name=_REPRO_RECEIPT_FILENAME,
                size=len(receipt_bytes),
                sha256=hashlib.sha256(receipt_bytes).hexdigest(),
            )
        )
    for full, rel_tar in members:
        size, digest = _hash_file(full)
        total += size
        if total > plan.bundle_size_cap_bytes:
            raise ValueError(
                f"bundle aggregate size {total} bytes exceeds cap "
                f"{plan.bundle_size_cap_bytes} bytes"
            )
        entries.append(BundleFileEntry(name=rel_tar, size=size, sha256=digest))

    manifest = BundleManifest(
        soup_version=soup_cli.__version__,
        created_at=datetime.now(tz=timezone.utc).isoformat(),
        model_dir=os.path.basename(os.path.normpath(plan.model_dir)),
        datasets=tuple(os.path.basename(os.path.normpath(d)) for d in plan.dataset_dirs),
        wheels=tuple(os.path.basename(os.path.normpath(w)) for w in plan.wheel_dirs),
        kernels=tuple(os.path.basename(os.path.normpath(k)) for k in plan.kernel_dirs),
        files=tuple(entries),
        total_bytes=total,
        repro_receipt=receipt_dict,
    )

    # Write tar via a temp file in the same dir, then atomic-rename.
    parent = os.path.dirname(os.path.realpath(plan.output)) or "."
    os.makedirs(parent, exist_ok=True)
    # TOCTOU defence: reject if parent dir became a symlink between the
    # containment check and the makedirs (security-review HIGH fix).
    parent_st = os.lstat(parent)
    if stat.S_ISLNK(parent_st.st_mode):
        raise ValueError(f"output parent dir {parent!r} is a symlink")
    # Reject pre-placed symlink at the final output path.
    if os.path.lexists(plan.output):
        out_st = os.lstat(plan.output)
        if stat.S_ISLNK(out_st.st_mode):
            raise ValueError("output path is a symlink (TOCTOU defence)")
    fd, tmp_path = tempfile.mkstemp(prefix=".airgap.", suffix=".tar.tmp", dir=parent)
    os.close(fd)
    try:
        with tarfile.open(tmp_path, "w") as tar:
            # manifest.json first so streaming readers see it without
            # walking the entire archive.
            manifest_bytes = _manifest_to_bytes(manifest)
            manifest_info = tarfile.TarInfo(name=_MANIFEST_FILENAME)
            manifest_info.size = len(manifest_bytes)
            manifest_info.mtime = 0
            import io as _io
            tar.addfile(manifest_info, _io.BytesIO(manifest_bytes))
            # v0.71.3 #188 — write the repro receipt as a top-level member.
            if receipt_bytes is not None:
                receipt_info = tarfile.TarInfo(name=_REPRO_RECEIPT_FILENAME)
                receipt_info.size = len(receipt_bytes)
                receipt_info.mtime = 0
                tar.addfile(receipt_info, _io.BytesIO(receipt_bytes))
            for full, rel_tar in members:
                info = tar.gettarinfo(name=full, arcname=rel_tar)
                # Force regular-file type; suppress owner / group / mtime
                # variance so the bundle hashes stay portable.
                info.uname = ""
                info.gname = ""
                info.uid = 0
                info.gid = 0
                info.mtime = 0
                with open(full, "rb") as fh:
                    tar.addfile(info, fh)
        os.replace(tmp_path, plan.output)
    finally:
        # Use lexists/lstat so a hostile replacement of tmp_path with a
        # symlink between failure and cleanup cannot trick us into
        # unlink-ing the symlink's target (code-review MEDIUM fix).
        if os.path.lexists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return manifest


def _manifest_to_bytes(manifest: BundleManifest) -> bytes:
    payload = {
        "soup_version": manifest.soup_version,
        "created_at": manifest.created_at,
        "model_dir": manifest.model_dir,
        "datasets": list(manifest.datasets),
        "wheels": list(manifest.wheels),
        "kernels": list(manifest.kernels),
        "total_bytes": manifest.total_bytes,
        "repro_receipt": manifest.repro_receipt,
        "files": [
            {"name": e.name, "size": e.size, "sha256": e.sha256}
            for e in manifest.files
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _manifest_from_payload(payload: dict) -> BundleManifest:
    files_raw = payload.get("files", [])
    if not isinstance(files_raw, list):
        raise ValueError("manifest 'files' must be a list")
    files = tuple(
        BundleFileEntry(
            name=str(entry["name"]),
            size=int(entry["size"]),
            sha256=str(entry["sha256"]),
        )
        for entry in files_raw
    )
    receipt = payload.get("repro_receipt")
    if receipt is not None and not isinstance(receipt, dict):
        receipt = None
    return BundleManifest(
        soup_version=str(payload.get("soup_version", "")),
        created_at=str(payload.get("created_at", "")),
        model_dir=str(payload.get("model_dir", "")),
        datasets=tuple(payload.get("datasets", [])),
        wheels=tuple(payload.get("wheels", [])),
        kernels=tuple(payload.get("kernels", [])),
        files=files,
        total_bytes=int(payload.get("total_bytes", 0)),
        repro_receipt=receipt,
    )


def inspect_airgap_bundle(bundle_path: str) -> BundleManifest:
    """Read back the manifest from a bundle tarball.

    Containment-checked; symlink at the bundle path is rejected via the
    shared helper. Raises ``FileNotFoundError`` on missing file,
    ``ValueError`` on out-of-cwd / null bytes / missing manifest.json.
    """
    enforce_under_cwd_and_no_symlink(bundle_path, "bundle")
    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"{bundle_path}: not a file")

    # Force PAX format and disable seek-back so the open call cannot be
    # tricked by a crafted header (defence-in-depth — we only read
    # manifest.json via getmember + extractfile, which does NOT honour
    # ``extraction_filter``; the filter assignment below is for the
    # benefit of any FUTURE caller adding ``tar.extractall``).
    with tarfile.open(bundle_path, "r") as tar:
        if hasattr(tarfile, "data_filter"):
            # Project policy: data_filter is the v3.12+ extract-time
            # symlink/tar-bomb defence. It is a no-op on our manifest
            # read path but locks the behaviour for future maintainers
            # (security-review HIGH fix).
            tar.extraction_filter = tarfile.data_filter
        try:
            member: tarfile.TarInfo = tar.getmember(_MANIFEST_FILENAME)
        except KeyError as exc:
            raise ValueError(
                f"{bundle_path}: missing {_MANIFEST_FILENAME}"
            ) from exc
        # Cap manifest size — reject crafted bundles with multi-GiB manifest.
        if member.size > _MAX_MANIFEST_BYTES:
            raise ValueError(
                f"{bundle_path}: manifest.json exceeds {_MAX_MANIFEST_BYTES} bytes"
            )
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"{bundle_path}: manifest.json is not a regular file")
        payload = json.loads(extracted.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{bundle_path}: manifest must be a JSON object")
    return _manifest_from_payload(payload)
