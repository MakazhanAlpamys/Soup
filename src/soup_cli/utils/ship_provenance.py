"""Versioned, opt-in drift binding for trusted ship evidence (not attestation)."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_data_identity(value: str | None) -> None:
    """Caller-supplied identity is an assertion, never inferred training evidence."""
    if value is not None and (not isinstance(value, str) or not _SHA256.fullmatch(value)):
        raise ValueError("--data-sha256 must be a lowercase 64-hex SHA-256")


def policy_hash(policy: dict[str, Any]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_binding(
    *, adapter: str | None, task_eval_sha256: str | None,
    source_data_sha256: str | None, data_sha256: str | None,
    suites: list[str], policy: dict[str, Any],
) -> dict[str, Any]:
    """Bind known inputs; unknown inputs stay explicit for ordinary emit flows."""
    from soup_cli.eval.gate import current_baseline_stamp
    from soup_cli.utils.adapter_sign import compute_adapter_manifest

    validate_data_identity(data_sha256)
    # Suite order is not part of score identity; verdict deltas sort names too.
    # Canonicalize both declarations before hashing, without changing callers.
    suites = sorted(suites)
    policy = {**policy, "general_suite": sorted(policy["general_suite"])}
    root = None
    if adapter is not None:
        try:
            manifest = compute_adapter_manifest(adapter)
            if manifest.files:
                root = manifest.merkle_root
        except (OSError, ValueError, TypeError):
            # Existing emit flows may refer to unavailable/non-local artifacts.
            # Opt-in verification refuses unknowns; default emission stays compatible.
            pass
    stamp = current_baseline_stamp()
    return {
        "schema_version": 1,
        "adapter_merkle_root": root,
        "task_eval_sha256": task_eval_sha256,
        "source_data_at_verdict_sha256": source_data_sha256,
        "training_data": {"sha256": data_sha256, "source": "caller"}
        if data_sha256 is not None else None,
        "suites": {"names": suites, **stamp},
        "policy": policy,
        "policy_sha256": policy_hash(policy),
    }


def check_binding(payload: dict[str, Any], expected: dict[str, Any]) -> None:
    """Refuse unknown/drifted identities without echoing untrusted field values."""
    provenance = payload.get("provenance")
    got = provenance.get("binding") if isinstance(provenance, dict) else None
    if not isinstance(got, dict) or type(got.get("schema_version")) is not int:
        raise ValueError("provenance binding is missing or malformed; re-emit trusted scores")
    try:
        recorded_policy = got.get("policy")
        valid_policy = (isinstance(recorded_policy, dict)
                        and policy_hash(recorded_policy) == got.get("policy_sha256"))
    except (ValueError, TypeError, RecursionError):
        valid_policy = False
    if not valid_policy:
        raise ValueError("provenance policy hash is missing or inconsistent")
    for field in ("adapter_merkle_root", "task_eval_sha256"):
        value = expected.get(field)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"provenance comparison needs a known {field}")
    if expected["source_data_at_verdict_sha256"] is None and expected["training_data"] is None:
        raise ValueError("provenance needs a known local training source or explicit --data-sha256")
    if (expected["policy"]["task_mode"] in {"judge_score", "pairwise"}
            and expected["policy"]["judge_model_sha256"] is None):
        raise ValueError("provenance for judge modes needs a declared --judge-model")
    if got != expected:
        raise ValueError(
            "stale provenance: artifact, task, training source, suites or policy differs"
        )
    # The stored descriptor must describe the scores being replayed, not merely
    # compare equal to an independent block the caller attached to other scores.
    task = payload.get("task")
    benchmarks = payload.get("benchmarks")
    if (not isinstance(task, dict) or task.get("mode") != expected["policy"]["task_mode"]
            or not isinstance(benchmarks, dict)
            or set(benchmarks) != set(expected["suites"]["names"])):
        raise ValueError("provenance does not describe the evidence task mode or benchmark set")


def judge_identity(judge: str | None) -> str | None:
    """Record endpoint identity without leaking URL paths, credentials or query values."""
    return hashlib.sha256(judge.encode("utf-8")).hexdigest() if judge is not None else None
