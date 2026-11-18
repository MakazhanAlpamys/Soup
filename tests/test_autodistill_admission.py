"""Read-only admission of published teacher-expert shards (#580)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from soup_cli.autodistill import admission
from soup_cli.autodistill.capture import (
    TeacherExpertExample,
    capture_teacher_expert_trajectory,
)
from soup_cli.autodistill.contract import (
    AutoDistillPlan,
    CaptureToken,
    ShardManifest,
    canonical_json_bytes,
    canonical_sha256,
)
from soup_cli.autodistill.publisher import CaptureShardPublisher

_STATES = ("staging", "complete", "verified", "available")
_PLAN_FILE = Path(__file__).parent / "fixtures" / "autodistill" / "v1" / "plan.json"


@dataclass(frozen=True)
class _Shard:
    root: Path
    plan: AutoDistillPlan
    publisher: CaptureShardPublisher
    rows: tuple[CaptureToken, ...]

    def manifest_path(self, state: str = "available") -> Path:
        return self.publisher.available_dir / f"manifest.{state}.json"

    @property
    def payload(self) -> Path:
        return self.publisher.available_dir / "capture.jsonl"

    def admit(self, expected_artifact_sha256: str, *, expected_plan: AutoDistillPlan | None = None):
        return admission.admit_teacher_expert_shard(
            capture_root=self.root,
            expected_plan=self.plan if expected_plan is None else expected_plan,
            shard_id=self.publisher.shard_id,
            expected_artifact_sha256=expected_artifact_sha256,
        )


@pytest.fixture
def shard(tmp_path: Path) -> _Shard:
    plan = AutoDistillPlan.model_validate_json(_PLAN_FILE.read_bytes())
    logits = [0.0] * plan.capture.vocab_size
    logits[10] = 4.0
    logits[11] = 3.0
    rows = tuple(
        row
        for example_id in ("example-1", "example-2")
        for row in capture_teacher_expert_trajectory(
            example=TeacherExpertExample(
                example_id=example_id,
                prompt_token_ids=(1, 2),
                target_token_ids=(10, 12),
            ),
            teacher_logits_by_position=(logits, logits),
            vocab_size=plan.capture.vocab_size,
            probability_policy=plan.probability_policy,
            max_sequence_length=plan.capture.max_sequence_length,
            truncation=plan.capture.truncation,
        )
    )
    root = tmp_path / "capture-cache"
    publisher = CaptureShardPublisher(
        root=root,
        plan=plan,
        shard_id="shard-0001",
        transaction_id="transaction-0001",
    )
    return _Shard(root=root, plan=plan, publisher=publisher, rows=rows)


@pytest.fixture
def published(shard: _Shard) -> ShardManifest:
    return shard.publisher.publish(shard.rows)


def _files(root: Path) -> dict[Path, tuple[int, bytes]]:
    return {
        path.relative_to(root): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def _rewrite_chain(shard: _Shard, edit) -> str:
    """Recommit prior hashes so a semantic mutation is not rejected as a bad link."""
    previous = None
    for state in _STATES:
        path = shard.manifest_path(state)
        manifest = json.loads(path.read_bytes())
        manifest["previous_manifest_sha256"] = previous
        edit(state, manifest)
        path.write_bytes(canonical_json_bytes(manifest) + b"\n")
        previous = canonical_sha256(manifest)
    assert previous is not None
    return previous


def _recommit_payload(shard: _Shard, payload: bytes, row_count: int) -> str:
    shard.payload.write_bytes(payload)

    def edit(_state, manifest):
        manifest.update(row_count=row_count, token_count=row_count)
        manifest["payloads"][0].update(
            bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest(),
            rows=row_count, tokens=row_count,
        )

    return _rewrite_chain(shard, edit)


def test_admission_returns_verified_rows_after_one_payload_read(shard, published, monkeypatch):
    shard.payload.with_name("uncommitted.jsonl").write_bytes(b"sidecar data\n")
    before = _files(shard.root)
    directories = {path.relative_to(shard.root) for path in shard.root.rglob("*") if path.is_dir()}
    real_read = admission.read_bounded_artifact
    payload_reads = []

    def read_artifact(root, relative_path, *, max_bytes):
        assert not relative_path.endswith("/uncommitted.jsonl")
        data = real_read(root, relative_path, max_bytes=max_bytes)
        if relative_path.endswith("/capture.jsonl"):
            payload_reads.append(data)
        return data

    monkeypatch.setattr(admission, "read_bounded_artifact", read_artifact)

    result = shard.admit(canonical_sha256(published))

    assert result.manifest == published
    assert result.rows == shard.rows
    [checked_payload] = payload_reads
    assert checked_payload == before[shard.payload.relative_to(shard.root)][1]
    assert _files(shard.root) == before
    assert {
        path.relative_to(shard.root) for path in shard.root.rglob("*") if path.is_dir()
    } == directories

    # The consumer's rows are already materialized from the checked bytes.
    shard.payload.write_bytes(b"different bytes\n")
    assert result.rows == shard.rows


def test_admission_uses_the_payload_bytes_it_checked_when_the_path_changes(
    shard, published, monkeypatch,
):
    verify = admission.verify_payload_bytes

    def verify_then_replace(manifest, payloads):
        verify(manifest, payloads)
        shard.payload.write_bytes(b"changed after verification\n")

    monkeypatch.setattr(admission, "verify_payload_bytes", verify_then_replace)

    assert shard.admit(canonical_sha256(published)).rows == shard.rows
    assert shard.payload.read_bytes() == b"changed after verification\n"


@pytest.mark.parametrize("state", ["staging", "complete", "verified", "quarantined"])
def test_admission_refuses_any_unpublished_state(shard, state):
    if state == "quarantined":
        shard.publisher.publish(shard.rows, stop_after="staging")
        (shard.publisher.transaction_dir / "capture.jsonl").write_bytes(b"damaged")
        with pytest.raises(ValueError, match="corrupt or mismatched"):
            shard.publisher.publish(shard.rows)
        directory = next(shard.publisher.quarantine_root.iterdir())
        manifest = ShardManifest.model_validate_json(
            (directory / "manifest.quarantined.json").read_bytes()
        )
    else:
        manifest = shard.publisher.publish(shard.rows, stop_after=state)
        directory = shard.publisher.transaction_dir

    # A directory's presence at the final path is not proof of publication.
    shard.publisher.available_dir.parent.mkdir()
    directory.rename(shard.publisher.available_dir)
    before = _files(shard.root)

    with pytest.raises(ValueError, match=r"(?i)available|quarantin"):
        shard.admit(canonical_sha256(manifest))

    assert _files(shard.root) == before


def test_admission_refuses_a_missing_capture_without_creating_it(shard):
    with pytest.raises(ValueError, match=r"(?i)capture root|missing|not found|available"):
        shard.admit("f" * 64)

    assert not shard.root.exists()


def test_an_available_shard_with_a_quarantine_marker_is_refused(shard, published):
    marker = published.model_copy(
        update={"state": "quarantined", "previous_manifest_sha256": canonical_sha256(published)}
    )
    shard.manifest_path("quarantined").write_bytes(canonical_json_bytes(marker) + b"\n")
    before = _files(shard.root)

    with pytest.raises(ValueError, match="publication state is not available"):
        shard.admit(canonical_sha256(published))

    assert _files(shard.root) == before


def test_admission_refuses_a_different_available_artifact(shard, published):
    actual = canonical_sha256(published)
    foreign = ("a" if actual[0] != "a" else "b") + actual[1:]

    with pytest.raises(ValueError, match="available manifest SHA-256 does not match"):
        shard.admit(foreign)


@pytest.mark.parametrize(
    ("fingerprint", "field", "changed"),
    [
        ("teacher", "revision", "d" * 40),
        ("student", "revision", "d" * 40),
        ("tokenizer", "chat_template_sha256", "d" * 64),
        ("dataset", "normalized_sha256", "d" * 64),
    ],
)
def test_admission_refuses_a_foreign_plan_fingerprint(
    shard, published, fingerprint, field, changed,
):
    data = shard.plan.model_dump(mode="json", by_alias=True)
    data[fingerprint][field] = changed
    foreign = AutoDistillPlan.model_validate(data)

    with pytest.raises(ValueError, match=rf"plan\.json {fingerprint} fingerprint does not match"):
        shard.admit(canonical_sha256(published), expected_plan=foreign)


def test_admission_refuses_a_missing_historical_manifest(shard, published):
    shard.manifest_path("complete").unlink()

    with pytest.raises(ValueError, match=r"manifest\.complete\.json"):
        shard.admit(canonical_sha256(published))


def test_admission_refuses_a_corrupt_on_disk_plan(shard, published):
    plan_path = shard.root / "plan.json"
    plan_path.write_bytes(b"not a plan\n")
    before = _files(shard.root)

    with pytest.raises(ValueError, match=r"(?i)plan"):
        shard.admit(canonical_sha256(published))

    assert _files(shard.root) == before


def test_admission_refuses_a_broken_manifest_chain(shard, published):
    def edit(state, manifest):
        if state == "complete":
            manifest["previous_manifest_sha256"] = "f" * 64

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match=r"manifest\.complete\.json manifest hash chain is broken"):
        shard.admit(artifact)


def test_admission_refuses_a_historical_manifest_for_another_transaction(shard, published):
    def edit(state, manifest):
        if state == "verified":
            manifest["transaction_id"] = "other-transaction"

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match=r"manifest\.verified\.json shard or transaction identity"):
        shard.admit(artifact)


def test_admission_refuses_a_historical_manifest_for_another_plan(shard, published):
    def edit(state, manifest):
        if state == "verified":
            manifest["plan_sha256"] = "f" * 64

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match=r"manifest\.verified\.json plan SHA-256 does not match"):
        shard.admit(artifact)


def test_admission_refuses_a_valid_link_that_changes_payload_commitments(shard, published):
    def edit(state, manifest):
        if state == "staging":
            manifest["payloads"][0]["sha256"] = "f" * 64

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match=r"manifest\.staging\.json payload commitments changed"):
        shard.admit(artifact)


def test_admission_refuses_payload_byte_count_drift(shard, published):
    shard.payload.write_bytes(shard.payload.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=r"(?i)byte count|size"):
        shard.admit(canonical_sha256(published))


def test_admission_refuses_a_hash_change_even_if_the_row_still_parses(shard, published):
    original = shard.payload.read_bytes()
    changed = original.replace(b'"example-2"', b'"example-X"')
    assert len(changed) == len(original) and changed != original
    assert all(CaptureToken.model_validate_json(line) for line in changed.splitlines())
    shard.payload.write_bytes(changed)
    before = _files(shard.root)

    with pytest.raises(ValueError, match=r"(?i)sha256|sha-256|digest"):
        shard.admit(canonical_sha256(published))

    assert _files(shard.root) == before


def test_admission_refuses_an_unexpected_committed_payload_member(shard, published):
    second_payload = shard.payload.with_name("other.jsonl")
    second_payload.write_bytes(shard.payload.read_bytes())

    def edit(_state, manifest):
        manifest["payloads"].append(dict(manifest["payloads"][0], path=second_payload.name))
        manifest["row_count"] *= 2
        manifest["token_count"] *= 2

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match=r"payload membership must be exactly capture\.jsonl"):
        shard.admit(artifact)


def test_admission_refuses_a_missing_payload_member(shard, published):
    shard.payload.unlink()

    with pytest.raises(ValueError, match=r"(?i)membership|capture\.jsonl|payload"):
        shard.admit(canonical_sha256(published))


def test_admission_refuses_a_noncanonical_row_even_with_correct_hash(shard, published):
    payload = shard.payload.read_bytes()
    spaced = payload.replace(b'{"', b'{ "', 1)
    assert spaced != payload
    assert CaptureToken.model_validate_json(spaced.splitlines()[0]) == shard.rows[0]
    artifact = _recommit_payload(shard, spaced, len(shard.rows))

    with pytest.raises(ValueError, match=r"capture\.jsonl row 1: row is not canonical JSONL"):
        shard.admit(artifact)


def test_admission_refuses_duplicate_positions_even_with_correct_payload_hash(shard, published):
    rows = list(shard.rows)
    rows[1] = rows[1].model_copy(update={"position": 0})
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    artifact = _recommit_payload(shard, payload, len(rows))

    with pytest.raises(ValueError, match="capture positions must be contiguous and start at zero"):
        shard.admit(artifact)


def test_admission_refuses_an_example_reappearing_after_its_group(shard, published):
    rows = [shard.rows[0], *shard.rows[2:], shard.rows[1].model_copy(update={"position": 0})]
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    artifact = _recommit_payload(shard, payload, len(rows))

    with pytest.raises(ValueError, match="capture examples must form contiguous groups"):
        shard.admit(artifact)


def test_admission_refuses_a_committed_row_count_mismatch(shard, published):
    artifact = _recommit_payload(shard, shard.payload.read_bytes(), len(shard.rows) + 1)

    with pytest.raises(ValueError, match=r"capture\.jsonl row or token count does not match"):
        shard.admit(artifact)


def test_admission_refuses_a_valid_student_rollout_row(shard, published):
    rows = list(shard.rows)
    data = rows[0].model_dump(mode="json", by_alias=True)
    data.update(
        trajectory_kind="student_rollout",
        student_sampled_token_id=rows[0].target_token_id,
        target_token_id=None,
    )
    rows[0] = CaptureToken.model_validate(data)
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    artifact = _recommit_payload(shard, payload, len(rows))

    with pytest.raises(ValueError, match="teacher_expert rows only"):
        shard.admit(artifact)


def test_admission_refuses_a_manifest_payload_path_escape(shard, published):
    def edit(_state, manifest):
        manifest["payloads"][0]["path"] = "../capture.jsonl"

    artifact = _rewrite_chain(shard, edit)

    with pytest.raises(ValueError, match="artifact path must be normalized"):
        shard.admit(artifact)


@pytest.mark.requires_symlink
def test_admission_refuses_even_a_symlink_to_the_exact_payload_bytes(shard, published):
    target = shard.root / "linked-capture.jsonl"
    shard.payload.rename(target)
    original = target.read_bytes()
    shard.payload.symlink_to(target)

    with pytest.raises(ValueError, match=r"(?i)symlink|symbolic link"):
        shard.admit(canonical_sha256(published))

    assert shard.payload.is_symlink()
    assert target.read_bytes() == original


@pytest.mark.parametrize(
    ("limit", "size_of", "match"),
    [
        (
            "_MAX_MANIFEST_BYTES",
            lambda shard: (shard.root / "plan.json").stat().st_size,
            r"plan\.json.*exceeds the .*byte limit",
        ),
        (
            "_MAX_PAYLOAD_BYTES",
            lambda shard: shard.payload.stat().st_size,
            r"capture\.jsonl.*exceeds the .*byte limit",
        ),
        ("_MAX_CAPTURE_ROWS", lambda shard: len(shard.rows), r"row count is outside the .*limit"),
    ],
)
def test_admission_accepts_exact_limit_and_refuses_one_over(
    shard, published, monkeypatch, limit, size_of, match,
):
    artifact = canonical_sha256(published)
    size = size_of(shard)
    monkeypatch.setattr(admission, limit, size)
    assert shard.admit(artifact).rows == shard.rows

    monkeypatch.setattr(admission, limit, size - 1)

    with pytest.raises(ValueError, match=match):
        shard.admit(artifact)


def test_actual_payload_growth_one_byte_past_limit_is_refused(shard, published, monkeypatch):
    payload = shard.payload.read_bytes()
    monkeypatch.setattr(admission, "_MAX_PAYLOAD_BYTES", len(payload))
    # The authenticated manifest still commits a size exactly at the limit.
    shard.payload.write_bytes(payload + b"\n")

    with pytest.raises(ValueError, match=r"capture\.jsonl.*artifact exceeds the .*byte limit"):
        shard.admit(canonical_sha256(published))


def test_available_manifest_one_byte_past_limit_is_refused(shard, published, monkeypatch):
    manifest_path = shard.manifest_path()
    manifest = manifest_path.read_bytes()
    monkeypatch.setattr(admission, "_MAX_MANIFEST_BYTES", len(manifest))
    manifest_path.write_bytes(manifest + b"\n")

    with pytest.raises(ValueError, match=r"manifest\.available\.json.*exceeds the .*byte limit"):
        shard.admit(canonical_sha256(published))
