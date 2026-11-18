"""A training run that failed is not repeated on the same inputs (review 5, F3).

The watch daemon polls every few seconds. A train stage that failed (the fit gate refused, the
card was busy, a missing dependency) was started again on the next poll, and the next, on a
dataset rebuilt byte for byte: four attempts in 35 s, each with its own ``iter-*`` directory.
Now a failure is remembered in ``loop.yaml`` (``failed_train``) against a fingerprint of what the
run was made of:

- the dataset the harvest built (its bytes), the training method, the base model and the
  recipe the loop trains with (for sft the effective ``sft_recipe``);
- the same fingerprint again after a failure waits: ``RETRY_DELAYS_SEC`` after the first
  failure, longer after the second, and after ``ATTEMPT_LIMIT`` failures the loop HOLDS: no new
  attempt until the inputs change (new feedback makes a new dataset; ``soup loop set base_model``
  or a new recipe makes a new fingerprint) or the operator clears it with ``soup loop resume``;
- a different fingerprint trains at once, and a run that trained clears the record.

A waiting or held cycle is a skipped iteration whose notes say so (``train_backoff: ...`` or
``train_held: ...``), and ``soup loop status`` shows the ``train_retry`` row. The deploy retry
(``DEPLOY_RETRY_LIMIT``) is separate and unchanged.

A stage of the cycle that RAISES (the cost, harvest, train or gate stage; a locked thumbs
database, a directory that went away) uses a separate loop-wide ``failed_stage`` record and the
same waits. Its stable fingerprint is shared across stages, so alternating failures still reach
the hold; its reason starts with ``stage error:``. The notes then say ``stage_backoff: ...`` or
``stage_held: ...``.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

#: Seconds to wait before the next attempt on the same inputs, after the first and the second
#: failure. A third failure holds.
RETRY_DELAYS_SEC = (300.0, 1800.0)
#: Failures on the same inputs after which the loop holds.
ATTEMPT_LIMIT = len(RETRY_DELAYS_SEC) + 1

_REASON_CHARS = 300
#: The stages whose exception is remembered (a deploy has its own retry).
STAGE_ERROR_STAGES = ("cost", "harvest", "train", "gate")
#: A ``failed_train`` reason that starts this way is a stage that raised, not a failed training.
STAGE_ERROR_PREFIX = "stage error: "
_MAX_DATASET_BYTES = 256 * 1024 * 1024
_READ_CHUNK = 1024 * 1024


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _parse(text: object) -> Optional[datetime]:
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def train_inputs_fingerprint(state: Any, data_path: str, method: str) -> Optional[str]:
    """A SHA-256 over the dataset bytes, the method, the base and the recipe; None if unreadable.

    ``state`` is the ``LoopState`` (its ``base_model`` / ``served_model`` and, for sft, its
    effective recipe). The dataset path is not part of it, only what the file holds.
    """
    digest = hashlib.sha256()
    try:
        if os.path.getsize(data_path) > _MAX_DATASET_BYTES:
            return None
        with open(data_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    recipe = state.effective_sft_recipe() if method == "sft" else None
    identity = {
        "method": method,
        "base": state.base_model or state.served_model,
        "recipe": recipe,
    }
    digest.update(json.dumps(identity, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def stage_error_fingerprint(stage: str) -> str:
    """Stable loop-wide fingerprint so errors in different stages share one retry limit."""
    del stage  # Keep the argument for callers and persisted-state migration.
    return hashlib.sha256(b"loop stage error").hexdigest()


def stage_error_reason(stage: str, error_type: str, detail: str = "") -> str:
    """The failure record reason of a stage that raised: its name and exception type.

    ``detail`` (the exception's message) follows when there is one; the record keeps one short
    line (``after_failure`` collapses the whitespace and cuts it).
    """
    text = f"{STAGE_ERROR_PREFIX}the {stage} stage raised {error_type}"
    return f"{text}: {detail}" if detail else text


def is_stage_error(failed: Mapping[str, Any]) -> bool:
    return str(failed.get("reason") or "").startswith(STAGE_ERROR_PREFIX)


def stage_error_hold(failed: Optional[Mapping[str, Any]], now: datetime) -> Optional[str]:
    """Why the cycle must not call its stages now (the skip note), or None to go ahead.

    Only a record of a stage that raised is judged here; the hold of a failed training is the
    train stage's own (it knows the dataset).
    """
    if not isinstance(failed, Mapping) or not is_stage_error(failed):
        return None
    # A failed training run has its own fingerprinted retry/hold in ``failed_train``. The
    # loop-wide failure streak still counts that cycle, but must not shadow the training retry:
    # doing so prevents the same dataset from being retried after its own wait expires.
    reason = str(failed.get("reason") or "")
    if reason.startswith("training failed:") or reason.startswith(
        f"{STAGE_ERROR_PREFIX}the train stage raised TrainingFailed:"
    ):
        return None
    # The record is loop-wide in current state. Using its own fingerprint also accepts a legacy
    # v0.78.0 per-stage fingerprint while that record is migrated on the next write.
    fingerprint = failed.get("inputs")
    return hold_reason(failed, fingerprint if isinstance(fingerprint, str) else None, now)


def after_failure(
    previous: Optional[Mapping[str, Any]],
    fingerprint: str,
    now: datetime,
    reason: str,
) -> "dict[str, Any]":
    """The ``failed_train`` record after one more failure on ``fingerprint``.

    The count continues from ``previous`` when it is for the same fingerprint, else starts at 1.
    ``retry_after`` is None once the loop holds.
    """
    same = isinstance(previous, Mapping) and previous.get("inputs") == fingerprint
    attempts = int(previous["attempts"]) + 1 if same else 1
    held = attempts >= ATTEMPT_LIMIT
    wait = timedelta(seconds=RETRY_DELAYS_SEC[min(attempts, len(RETRY_DELAYS_SEC)) - 1])
    return {
        "inputs": fingerprint,
        "attempts": attempts,
        "last_at": _iso(now),
        "retry_after": None if held else _iso(now + wait),
        "held": held,
        "reason": " ".join(str(reason).split())[:_REASON_CHARS] or "train failed",
    }


def hold_reason(
    failed: Optional[Mapping[str, Any]], fingerprint: Optional[str], now: datetime
) -> Optional[str]:
    """Why training on ``fingerprint`` must wait now (the skip note), or None to go ahead."""
    if not isinstance(failed, Mapping) or fingerprint is None:
        return None
    if failed.get("inputs") != fingerprint:
        return None
    attempts = failed.get("attempts")
    times = "1 time" if attempts == 1 else f"{attempts} times"
    if is_stage_error(failed):
        detail = str(failed.get("reason"))[len(STAGE_ERROR_PREFIX):]
        seen = f"{detail} ({times}, last {failed.get('last_at')})"
        if failed.get("held"):
            return f"stage_held: {seen}; not retrying until soup loop resume clears it"
        retry_after = _parse(failed.get("retry_after"))
        if retry_after is not None and now < retry_after:
            return f"stage_backoff: {seen}; next attempt after {failed.get('retry_after')}"
        return None
    seen = (
        f"the same inputs failed {times} "
        f"(last {failed.get('last_at')}: {failed.get('reason')})"
    )
    if failed.get("held"):
        return (
            f"train_held: {seen}; not retrying until the data, the recipe or the base model "
            "changes, or soup loop resume clears it"
        )
    retry_after = _parse(failed.get("retry_after"))
    if retry_after is not None and now < retry_after:
        return f"train_backoff: {seen}; next attempt after {failed.get('retry_after')}"
    return None


def describe(failed: Optional[Mapping[str, Any]], now: Optional[datetime] = None) -> Optional[str]:
    """The ``train_retry`` row of ``soup loop status``, or None when nothing is remembered."""
    if not isinstance(failed, Mapping):
        return None
    attempts = failed.get("attempts")
    plural = "" if attempts == 1 else "s"
    if is_stage_error(failed):
        detail = str(failed.get("reason"))[len(STAGE_ERROR_PREFIX):]
        base = f"{detail} ({attempts} time{plural} in a row)"
    else:
        base = f"failed {attempts} time{plural} on the same inputs ({failed.get('reason')})"
    if failed.get("held"):
        until = (
            "soup loop resume clears it" if is_stage_error(failed)
            else "the data, the recipe or the base model changes, or soup loop resume clears it"
        )
        return f"held: {base}; no new attempt until {until}"
    moment = now or datetime.now(timezone.utc)
    retry_after = _parse(failed.get("retry_after"))
    if retry_after is not None and moment < retry_after:
        return f"waiting: {base}; next attempt after {failed.get('retry_after')}"
    return f"due: {base}; the next cycle tries again"
