"""Transactional publication of immutable AutoDistill capture shards."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    CaptureToken,
    PayloadDigest,
    ShardManifest,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize_jsonl_bytes,
    decide_resume,
    ensure_shard_transition,
    verify_payload_bytes,
)

PublicationState = Literal["staging", "complete", "verified", "available"]
_STATES: tuple[PublicationState, ...] = ("staging", "complete", "verified", "available")
_PAYLOAD_NAME = "capture.jsonl"


def _contained_path(root: Path, *parts: str) -> Path:
    root_real = os.path.realpath(root)
    candidate = os.path.abspath(os.path.join(root_real, *parts))
    candidate_real = os.path.realpath(candidate)
    try:
        contained = os.path.commonpath((root_real, candidate_real)) == root_real
    except ValueError as exc:
        raise ValueError("artifact path is outside the publication root") from exc
    if not contained:
        raise ValueError("artifact path is outside the publication root")
    return Path(candidate)


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ArtifactCorruptionError(f"refusing to replace symlink {path.name!r}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _sync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _manifest_name(state: PublicationState) -> str:
    return f"manifest.{state}.json"


class CaptureShardPublisher:
    """Advance one capture shard through a write-last manifest transaction."""

    def __init__(
        self,
        *,
        root: str | os.PathLike[str],
        plan: AutoDistillPlan,
        shard_id: str,
        transaction_id: str,
    ) -> None:
        self.root = Path(os.path.realpath(root))
        self.plan = plan
        self.shard_id = shard_id
        self.transaction_id = transaction_id
        self.transaction_dir = _contained_path(self.root, ".transactions", transaction_id)
        self.available_dir = _contained_path(self.root, "shards", shard_id)
        self.quarantine_root = _contained_path(self.root, "quarantine")

    def publish(
        self,
        captures: Sequence[CaptureToken],
        *,
        stop_after: PublicationState | None = None,
    ) -> ShardManifest:
        """Resume or publish exactly one immutable shard.

        ``stop_after`` is an explicit interruption hook for tests and controllers;
        every returned state is already committed by an atomic manifest.
        """

        if stop_after is not None and stop_after not in _STATES:
            raise ValueError("unknown stop_after state")
        payload = self._capture_payload(captures)
        directory, manifest = self._inspect_existing()
        if manifest is not None:
            if not self._integrity_matches(directory, manifest, payload):
                self._quarantine(directory, manifest)
                raise ArtifactCorruptionError("existing transaction is corrupt or mismatched")
            decision = decide_resume(
                state=manifest.state,
                payloads_valid=True,
                fingerprints_match=True,
            )
            if decision == "reuse":
                return manifest
        else:
            manifest = self._stage(payload, len(captures))
            directory = self.transaction_dir
        if stop_after == manifest.state:
            return manifest

        if manifest.state == "staging":
            manifest = self._commit_state(directory, manifest, "complete")
            if stop_after == "complete":
                return manifest
        if manifest.state == "complete":
            self._verify_semantics(directory, manifest)
            manifest = self._commit_state(directory, manifest, "verified")
            if stop_after == "verified":
                return manifest
        if manifest.state == "verified":
            manifest = self._make_available(directory, manifest)
        return manifest

    def _capture_payload(self, captures: Sequence[CaptureToken]) -> bytes:
        rows = tuple(captures)
        if not rows:
            raise ValueError("a capture shard must contain at least one token row")
        closed_examples: set[str] = set()
        current_example: str | None = None
        expected_position = 0
        for row in rows:
            if not isinstance(row, CaptureToken):
                raise TypeError("captures must contain CaptureToken instances")
            if row.trajectory_kind != "teacher_expert":
                raise ValueError("Milestone B1 publishes teacher_expert rows only")
            if row.vocab_size != self.plan.capture.vocab_size:
                raise ValueError("capture vocabulary does not match plan")
            if row.temperature != self.plan.probability_policy.temperature:
                raise ValueError("capture temperature does not match plan")
            if len(row.top_k_token_ids) != self.plan.probability_policy.top_k:
                raise ValueError("capture top-k cardinality does not match plan")
            if len(row.context_token_ids) > self.plan.capture.max_sequence_length:
                raise ValueError("capture context exceeds plan max_sequence_length")
            if row.example_id != current_example:
                if row.example_id in closed_examples:
                    raise ValueError("capture examples must form contiguous groups")
                if current_example is not None:
                    closed_examples.add(current_example)
                current_example = row.example_id
                expected_position = 0
            if row.position != expected_position:
                raise ValueError("capture positions must be contiguous and start at zero")
            expected_position += 1
        return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)

    def _stage(self, payload: bytes, row_count: int) -> ShardManifest:
        if self.transaction_dir.exists() or self.available_dir.exists():
            raise ArtifactCorruptionError("transaction target appeared while staging")
        self.root.mkdir(parents=True, exist_ok=True)
        plan_path = _contained_path(self.root, "plan.json")
        plan_bytes = canonical_json_bytes(self.plan) + b"\n"
        if plan_path.exists():
            if plan_path.is_symlink() or plan_path.read_bytes() != plan_bytes:
                raise ArtifactCorruptionError("publication root is bound to a different plan")
        else:
            _atomic_write(plan_path, plan_bytes)
        self.transaction_dir.mkdir(parents=True)
        payload_path = _contained_path(self.transaction_dir, _PAYLOAD_NAME)
        _atomic_write(payload_path, payload)
        payload_digest = PayloadDigest(
            path=_PAYLOAD_NAME,
            bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
            rows=row_count,
            tokens=row_count,
        )
        manifest = ShardManifest(
            schema="soup.autodistill.shard-manifest.v1",
            shard_id=self.shard_id,
            transaction_id=self.transaction_id,
            state="staging",
            plan_sha256=canonical_sha256(self.plan),
            previous_manifest_sha256=None,
            row_count=row_count,
            token_count=row_count,
            payloads=(payload_digest,),
        )
        self._write_manifest(self.transaction_dir, manifest)
        return manifest

    def _commit_state(
        self,
        directory: Path,
        previous: ShardManifest,
        state: Literal["complete", "verified"],
    ) -> ShardManifest:
        ensure_shard_transition(previous.state, state)
        manifest = previous.model_copy(
            update={
                "state": state,
                "previous_manifest_sha256": canonical_sha256(previous),
            }
        )
        manifest = ShardManifest.model_validate(manifest.model_dump(by_alias=True))
        self._write_manifest(directory, manifest)
        return manifest

    def _make_available(self, directory: Path, previous: ShardManifest) -> ShardManifest:
        ensure_shard_transition(previous.state, "available")
        self.available_dir.parent.mkdir(parents=True, exist_ok=True)
        if directory == self.transaction_dir:
            if self.available_dir.exists():
                raise ArtifactCorruptionError("available shard path already exists")
            os.replace(directory, self.available_dir)
            _sync_directory(self.available_dir.parent)
            directory = self.available_dir
        elif directory != self.available_dir:
            raise ArtifactCorruptionError("transaction is outside known publication paths")
        manifest = previous.model_copy(
            update={
                "state": "available",
                "previous_manifest_sha256": canonical_sha256(previous),
            }
        )
        manifest = ShardManifest.model_validate(manifest.model_dump(by_alias=True))
        self._write_manifest(directory, manifest)
        return manifest

    def _inspect_existing(self) -> tuple[Path, ShardManifest | None]:
        directories = [path for path in (self.available_dir, self.transaction_dir) if path.exists()]
        if len(directories) > 1:
            raise ArtifactCorruptionError("duplicate transaction and available shard directories")
        if not directories:
            return self.transaction_dir, None
        directory = directories[0]
        if directory.is_symlink():
            raise ArtifactCorruptionError("transaction directory must not be a symlink")
        found: list[tuple[PublicationState, ShardManifest]] = []
        for state in _STATES:
            path = _contained_path(directory, _manifest_name(state))
            if path.exists():
                found.append((state, self._read_manifest(path)))
        if not found:
            raise ArtifactCorruptionError("transaction has no committed manifest")
        self._verify_manifest_chain(found)
        return directory, found[-1][1]

    def _verify_manifest_chain(
        self,
        found: Sequence[tuple[PublicationState, ShardManifest]],
    ) -> None:
        expected_states = list(_STATES[: len(found)])
        if [state for state, _ in found] != expected_states:
            raise ArtifactCorruptionError("transaction manifest states are not contiguous")
        previous: ShardManifest | None = None
        for state, manifest in found:
            if manifest.state != state:
                raise ArtifactCorruptionError("manifest filename and state disagree")
            if manifest.shard_id != self.shard_id or manifest.transaction_id != self.transaction_id:
                raise ArtifactCorruptionError("manifest identity does not match publisher")
            expected_previous = None if previous is None else canonical_sha256(previous)
            if manifest.previous_manifest_sha256 != expected_previous:
                raise ArtifactCorruptionError("manifest hash chain is broken")
            if previous is not None and manifest.payloads != previous.payloads:
                raise ArtifactCorruptionError("manifest transition changed payload commitments")
            previous = manifest

    def _integrity_matches(
        self,
        directory: Path,
        manifest: ShardManifest,
        expected_payload: bytes,
    ) -> bool:
        try:
            if manifest.plan_sha256 != canonical_sha256(self.plan):
                return False
            plan_path = _contained_path(self.root, "plan.json")
            if plan_path.is_symlink():
                return False
            if plan_path.read_bytes() != canonical_json_bytes(self.plan) + b"\n":
                return False
            payload_path = _contained_path(directory, _PAYLOAD_NAME)
            if payload_path.is_symlink():
                return False
            payload = payload_path.read_bytes()
            verify_payload_bytes(manifest, {_PAYLOAD_NAME: payload})
            if payload != expected_payload:
                return False
            self._verify_semantics(directory, manifest)
        except (OSError, TypeError, ValueError):
            return False
        return True

    def _verify_semantics(self, directory: Path, manifest: ShardManifest) -> None:
        payload = _contained_path(directory, _PAYLOAD_NAME).read_bytes()
        if canonicalize_jsonl_bytes(payload) != payload:
            raise ArtifactCorruptionError("capture payload is not canonical JSONL")
        rows = tuple(
            CaptureToken.model_validate(json.loads(line))
            for line in payload.splitlines()
        )
        rebuilt = self._capture_payload(rows)
        if rebuilt != payload:
            raise ArtifactCorruptionError("capture payload semantic verification failed")
        if len(rows) != manifest.row_count or len(rows) != manifest.token_count:
            raise ArtifactCorruptionError("capture logical counts do not match manifest")
        verify_payload_bytes(manifest, {_PAYLOAD_NAME: payload})

    def _write_manifest(self, directory: Path, manifest: ShardManifest) -> None:
        path = _contained_path(directory, _manifest_name(manifest.state))
        if path.exists():
            if path.read_bytes() != canonical_json_bytes(manifest) + b"\n":
                raise ArtifactCorruptionError("refusing to overwrite a committed manifest")
            return
        _atomic_write(path, canonical_json_bytes(manifest) + b"\n")

    def _read_manifest(self, path: Path) -> ShardManifest:
        if path.is_symlink():
            raise ArtifactCorruptionError("manifest must not be a symlink")
        data = path.read_bytes()
        try:
            manifest = ShardManifest.model_validate_json(data)
        except ValueError as exc:
            raise ArtifactCorruptionError(f"invalid manifest {path.name!r}") from exc
        if data != canonical_json_bytes(manifest) + b"\n":
            raise ArtifactCorruptionError("manifest bytes are not canonical")
        return manifest

    def _quarantine(self, directory: Path, previous: ShardManifest) -> None:
        self.quarantine_root.mkdir(parents=True, exist_ok=True)
        target = _contained_path(
            self.quarantine_root,
            f"{self.shard_id}.{self.transaction_id}",
        )
        if target.exists():
            raise ArtifactCorruptionError("quarantine target already exists")
        os.replace(directory, target)
        _sync_directory(self.quarantine_root)
        quarantined = previous.model_copy(
            update={
                "state": "quarantined",
                "previous_manifest_sha256": canonical_sha256(previous),
            }
        )
        quarantined = ShardManifest.model_validate(quarantined.model_dump(by_alias=True))
        self._write_manifest(target, quarantined)
