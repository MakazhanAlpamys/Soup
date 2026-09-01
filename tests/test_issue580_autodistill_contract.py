"""Milestone A contract tests for issue #580 (AutoDistill).

These tests are intentionally model-free.  They exercise only versioned artifacts,
deterministic arithmetic, integrity checks, and state-machine semantics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from soup_cli.autodistill.contract import (
    ArtifactCorruptionError,
    AutoDistillPlan,
    CaptureToken,
    ConsumptionEvent,
    ShardManifest,
    ThroughputProfile,
    build_plan_estimate,
    canonical_json_bytes,
    canonical_sha256,
    canonicalize_jsonl_bytes,
    coarse_tail_forward_kl,
    decide_resume,
    ensure_consumption_transition,
    ensure_example_transition,
    ensure_shard_transition,
    validate_consumption_ledger,
    verify_payload_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures" / "autodistill" / "v1"


def _load_json(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _canonical_capture_payload(name: str = "capture_token.json") -> bytes:
    token = CaptureToken.model_validate(_load_json(name))
    return canonical_json_bytes(token) + b"\n"


def test_plan_fixture_is_versioned_frozen_and_same_tokenizer_only():
    plan = AutoDistillPlan.model_validate(_load_json("plan.json"))

    assert plan.schema_id == "soup.autodistill.plan.v1"
    assert plan.capture_boundary == "same_tokenizer"
    assert plan.probability_policy.name == "topk_union_forced_tail.v1"
    assert plan.probability_policy.renormalize_selected is False
    assert plan.probability_policy.forced_token_sources == (
        "target",
        "student_sample",
    )
    assert len(canonical_sha256(plan)) == 64
    assert b'"schema":"soup.autodistill.plan.v1"' in canonical_json_bytes(plan)
    with pytest.raises(ValidationError):
        plan.teacher.model_id = "moving-target"


def test_plan_rejects_implicit_compression_policy():
    payload = _load_json("plan.json")
    del payload["probability_policy"]["top_k"]

    with pytest.raises(ValidationError):
        AutoDistillPlan.model_validate(payload)


def test_plan_rejects_tampered_size_estimate():
    payload = _load_json("plan.json")
    payload["estimate"]["sparse_payload_bytes_upper_bound"] += 1

    with pytest.raises(ValidationError, match="estimate does not match"):
        AutoDistillPlan.model_validate(payload)


def test_dataset_normalization_is_deterministic_across_formatting_and_newlines():
    compact = b'{"a":1,"b":"caf\xc3\xa9"}\n'
    formatted = b'\xef\xbb\xbf{ "b": "caf\xc3\xa9", "a": 1 }\r\n'

    assert canonicalize_jsonl_bytes(compact) == compact
    assert canonicalize_jsonl_bytes(formatted) == compact


def test_dataset_normalization_rejects_duplicate_keys():
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        canonicalize_jsonl_bytes(b'{"prompt":"a","prompt":"b"}\n')


def test_plan_only_size_estimate_is_exact_and_model_free(monkeypatch):
    imported_heavy: list[str] = []
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"mlx", "torch", "transformers"}:
            imported_heavy.append(name)
            raise AssertionError(f"plan-only imported heavy runtime {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    estimate = build_plan_estimate(
        token_count=1_000,
        vocab_size=1_024,
        top_k=8,
        max_forced_tokens_per_position=2,
        token_id_bytes=4,
        log_probability_bytes=2,
        tail_mass_bytes=4,
        entropy_bytes=4,
    )

    assert estimate.dense_payload_bytes == 2_048_000
    assert estimate.sparse_payload_bytes_upper_bound == 68_000
    assert estimate.runtime.status == "unknown"
    assert estimate.runtime.seconds_min is None
    assert estimate.runtime.seconds_max is None
    assert imported_heavy == []


def test_plan_only_runtime_range_uses_cached_profile_only():
    profile = ThroughputProfile(
        profile_sha256="9" * 64,
        teacher_fingerprint_sha256="a" * 64,
        hardware_fingerprint_sha256="b" * 64,
        backend="transformers",
        backend_version="5.16.1",
        dtype="bfloat16",
        quantization="none",
        sequence_length_min=1,
        sequence_length_max=2_048,
        tokens_per_second_min=40.0,
        tokens_per_second_max=50.0,
    )
    estimate = build_plan_estimate(
        token_count=1_000,
        vocab_size=1_024,
        top_k=8,
        max_forced_tokens_per_position=2,
        token_id_bytes=4,
        log_probability_bytes=2,
        tail_mass_bytes=4,
        entropy_bytes=4,
        throughput_profile=profile,
    )

    assert estimate.runtime.status == "profiled"
    assert estimate.runtime.seconds_min == 20.0
    assert estimate.runtime.seconds_max == 25.0
    assert estimate.runtime.profile_sha256 == "9" * 64


def test_capture_fixture_preserves_selected_and_tail_probability_mass():
    token = CaptureToken.model_validate(_load_json("capture_token.json"))

    selected_mass = math.fsum(math.exp(value) for value in token.teacher_log_probabilities)
    assert selected_mass == pytest.approx(0.9)
    assert selected_mass + token.tail_mass == pytest.approx(1.0)
    assert token.selected_token_ids == (0, 2)
    assert token.top_k_token_ids == (0,)
    assert token.forced_token_ids == (2,)
    assert token.context_token_ids == (1,)
    assert token.target_token_id == 2


def test_student_rollout_fixture_requires_sampled_token_and_forces_it():
    payload = _load_json("student_rollout_capture_token.json")
    token = CaptureToken.model_validate(payload)

    assert token.trajectory_kind == "student_rollout"
    assert token.student_sampled_token_id == 2
    assert token.student_sampled_token_id in token.forced_token_ids

    missing_sample = payload.copy()
    missing_sample["student_sampled_token_id"] = None
    with pytest.raises(ValidationError, match="student_rollout rows require"):
        CaptureToken.model_validate(missing_sample)

    unforced_sample = payload.copy()
    unforced_sample["student_sampled_token_id"] = 1
    with pytest.raises(ValidationError, match="student_sampled_token_id must be forced"):
        CaptureToken.model_validate(unforced_sample)


def test_capture_rejects_renormalized_top_k_that_discards_tail():
    payload = _load_json("capture_token.json")
    payload["teacher_log_probabilities"] = [math.log(0.75), math.log(0.25)]
    payload["tail_mass"] = 0.0

    with pytest.raises(ValidationError, match="tail mass must be positive"):
        CaptureToken.model_validate(payload)


def test_capture_rejects_probability_mass_other_than_one():
    payload = _load_json("capture_token.json")
    payload["tail_mass"] = 0.2

    with pytest.raises(ValidationError, match="selected probability mass plus tail_mass"):
        CaptureToken.model_validate(payload)


def test_capture_rejects_selected_ids_outside_top_k_union_forced():
    payload = _load_json("capture_token.json")
    payload["selected_token_ids"] = [0]
    payload["teacher_log_probabilities"] = [math.log(0.9)]

    with pytest.raises(ValidationError, match="selected_token_ids must equal"):
        CaptureToken.model_validate(payload)


def test_coarse_tail_kl_matches_collapsed_dense_distribution():
    teacher = (0.7, 0.2, 0.1)
    student = (0.6, 0.25, 0.15)
    dense = math.fsum(
        teacher_prob * math.log(teacher_prob / student_prob)
        for teacher_prob, student_prob in zip(teacher, student)
    )
    collapsed = coarse_tail_forward_kl(
        teacher_log_probabilities=(math.log(teacher[0]), math.log(teacher[1])),
        student_log_probabilities=(math.log(student[0]), math.log(student[1])),
        teacher_tail_mass=teacher[2],
        student_tail_mass=student[2],
    )

    assert collapsed == pytest.approx(dense, abs=1e-15)


def test_k_equals_vocab_matches_dense_forward_kl_exactly():
    teacher = (0.5, 0.3, 0.2)
    student = (0.4, 0.4, 0.2)
    dense = math.fsum(
        teacher_prob * math.log(teacher_prob / student_prob)
        for teacher_prob, student_prob in zip(teacher, student)
    )
    offline = coarse_tail_forward_kl(
        teacher_log_probabilities=tuple(math.log(value) for value in teacher),
        student_log_probabilities=tuple(math.log(value) for value in student),
        teacher_tail_mass=0.0,
        student_tail_mass=0.0,
    )

    assert offline == pytest.approx(dense, abs=1e-15)


def test_shard_fixture_detects_parseable_payload_corruption():
    manifest = ShardManifest.model_validate(_load_json("shard_manifest.json"))
    plan = AutoDistillPlan.model_validate(_load_json("plan.json"))
    payload = _canonical_capture_payload()

    assert manifest.plan_sha256 == canonical_sha256(plan)
    verify_payload_bytes(manifest, {"capture.jsonl": payload})
    payload_rows = [json.loads(line) for line in payload.splitlines()]
    assert payload_rows == [_load_json("capture_token.json")]
    CaptureToken.model_validate(payload_rows[0])
    changed = payload.replace(b'"ex-0001"', b'"ex-9999"')
    with pytest.raises(ArtifactCorruptionError, match="sha256 mismatch"):
        verify_payload_bytes(manifest, {"capture.jsonl": changed})


def test_canonical_capture_payload_is_independent_of_checkout_newlines():
    fixture_bytes = (FIXTURES / "capture_token.json").read_bytes()
    crlf_fixture_bytes = fixture_bytes.replace(b"\n", b"\r\n")
    token = CaptureToken.model_validate(json.loads(crlf_fixture_bytes))

    rebuilt = canonical_json_bytes(token) + b"\n"

    assert rebuilt == _canonical_capture_payload()
    assert b"\r\n" not in rebuilt


def test_plan_rejects_parent_path_traversal():
    payload = _load_json("plan.json")
    payload["teacher"]["weights"][0]["path"] = "../model.safetensors"

    with pytest.raises(ValidationError, match="may not contain '..'"):
        AutoDistillPlan.model_validate(payload)


def test_state_machines_fail_closed():
    assert ensure_example_transition("proposed", "probed") == "probed"
    assert ensure_example_transition("verified", "admitted") == "admitted"
    assert ensure_shard_transition("staging", "complete") == "complete"
    assert ensure_shard_transition("verified", "available") == "available"

    with pytest.raises(ValueError, match="invalid example transition"):
        ensure_example_transition("proposed", "admitted")
    with pytest.raises(ValueError, match="invalid shard transition"):
        ensure_shard_transition("complete", "available")


def test_consumption_is_exactly_once_for_student_rollouts():
    assert (
        ensure_consumption_transition(
            view="student_rollout",
            current="available",
            target="reserved",
        )
        == "reserved"
    )
    assert (
        ensure_consumption_transition(
            view="student_rollout",
            current="reserved",
            target="committed",
            checkpoint_sha256="8" * 64,
        )
        == "committed"
    )
    with pytest.raises(ValueError, match="student_rollout replay is forbidden"):
        ensure_consumption_transition(
            view="student_rollout",
            current="committed",
            target="reserved",
            replay_of="7" * 64,
        )


def test_consumption_ledger_fixture_commits_exactly_once():
    events = tuple(
        ConsumptionEvent.model_validate(payload)
        for payload in _load_json("consumption_ledger.json")
    )

    assert validate_consumption_ledger(events) == "committed"
    assert events[1].checkpoint_sha256 == "05" * 32


def test_consumption_ledger_rejects_noncontiguous_sequence():
    payloads = _load_json("consumption_ledger.json")
    payloads[1]["sequence"] = 2
    events = tuple(ConsumptionEvent.model_validate(payload) for payload in payloads)

    with pytest.raises(ValueError, match="sequence must be contiguous"):
        validate_consumption_ledger(events)


def test_expert_replay_must_chain_to_prior_committed_event():
    payloads = _load_json("consumption_ledger.json")
    for payload in payloads:
        payload["view"] = "teacher_expert"
    committed_events = tuple(
        ConsumptionEvent.model_validate(payload) for payload in payloads
    )
    replay_payload = {
        "schema": "soup.autodistill.consumption-event.v1",
        "event_id": "consume-0001-replay",
        "sequence": 2,
        "artifact_sha256": "04" * 32,
        "view": "teacher_expert",
        "from": "committed",
        "to": "reserved",
        "run_id": "student-run-0001",
        "reservation_id": "reservation-0002",
        "checkpoint_sha256": None,
        "replay_of": "07" * 32,
    }
    bad_replay = ConsumptionEvent.model_validate(replay_payload)

    with pytest.raises(ValueError, match="replay_of must identify"):
        validate_consumption_ledger((*committed_events, bad_replay))

    replay_payload["replay_of"] = canonical_sha256(committed_events[-1])
    replay = ConsumptionEvent.model_validate(replay_payload)
    assert validate_consumption_ledger((*committed_events, replay)) == "reserved"


def test_expert_replay_requires_an_explicit_prior_commit():
    assert (
        ensure_consumption_transition(
            view="teacher_expert",
            current="committed",
            target="reserved",
            replay_of="7" * 64,
        )
        == "reserved"
    )
    with pytest.raises(ValueError, match="replay_of"):
        ensure_consumption_transition(
            view="teacher_expert",
            current="committed",
            target="reserved",
        )


@pytest.mark.parametrize(
    ("state", "payloads_valid", "fingerprints_match", "expected"),
    [
        ("staging", True, True, "resume_staging"),
        ("complete", True, True, "verify_then_publish"),
        ("verified", True, True, "publish"),
        ("available", True, True, "reuse"),
        ("quarantined", True, True, "refuse"),
        ("complete", False, True, "quarantine"),
        ("available", True, False, "quarantine"),
    ],
)
def test_resume_decisions_are_explicit(
    state, payloads_valid, fingerprints_match, expected
):
    assert (
        decide_resume(
            state=state,
            payloads_valid=payloads_valid,
            fingerprints_match=fingerprints_match,
        )
        == expected
    )
