"""A training run that failed is not repeated on the same inputs (review 5, F3).

The watch daemon polls every few seconds. A train stage that failed (the fit gate refused, the
card was busy, a missing dependency) was started again on the next poll, and the next, on a
dataset rebuilt byte for byte: four attempts in 35 s, each with its own ``iter-*`` directory.
Now a failure is remembered in ``loop.yaml`` (``failed_train``) against a fingerprint of what the
run was made of:

- the dataset the harvest built (its bytes; a file over ``_MAX_DATASET_BYTES`` is
  fingerprinted from its size and its head and tail, and an unreadable one from its
  path, so either failure is held rather than retried on every poll — #3043), the
  training method, the base model and the recipe the loop trains with (for sft the
  effective ``sft_recipe``);
- the same fingerprint again after a failure waits: ``RETRY_DELAYS_SEC`` after the first
  failure, longer after the second, and after ``ATTEMPT_LIMIT`` failures the loop HOLDS: no new
  attempt until the inputs change (new feedback makes a new dataset; ``soup loop set base_model``
  or a new recipe makes a new fingerprint) or the operator clears it with ``soup loop resume``;
- a different fingerprint trains at once, and a run that trained clears the record.

A waiting or held cycle is a skipped iteration whose notes say so (``train_backoff: ...`` or
``train_held: ...``), and ``soup loop status`` shows the ``train_retry`` row. The deploy retry
(``DEPLOY_RETRY_LIMIT``) is separate and unchanged.

A stage of the cycle that RAISES (the cost, harvest, train or gate stage; a locked thumbs
database, a directory that went away) uses the same record and the same waits: its fingerprint is
the stage's name (``stage_error_fingerprint``) and its reason starts with ``stage error:``. The
notes then say ``stage_backoff: ...`` or ``stage_held: ...``.
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
    """A SHA-256 over the dataset bytes, the method, the base and the recipe.

    ``state`` is the ``LoopState`` (its ``base_model`` / ``served_model`` and, for sft, its
    effective recipe). The dataset path is not part of it, only what the file holds, so a
    byte-identical copy fingerprints the same wherever it sits.

    A dataset over ``_MAX_DATASET_BYTES`` is not read whole: its exact size and a head
    and tail hash are used instead. A dataset that cannot be read gets a fingerprint
    keyed on its path, so its failures are held like any others instead of being
    retried on every poll forever; once it becomes readable the content fingerprint
    replaces that key and training resumes. ``None`` is reserved for a caller that has
    no dataset path at all.

    Limit: over the cap, a change that keeps the size and both ends of the file is
    not seen, so such a hold ends only with ``soup loop resume``; the full hash below
    the cap has no such blind spot.
    """
    digest = hashlib.sha256()
    try:
        size = os.path.getsize(data_path)
        if size > _MAX_DATASET_BYTES:
            with open(data_path, "rb") as handle:
                head = handle.read(_READ_CHUNK)
                handle.seek(max(0, size - _READ_CHUNK))
                tail = handle.read(_READ_CHUNK)
            digest.update(f"large:{size}:".encode())
            digest.update(head)
            digest.update(tail)
        else:
            with open(data_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
                    digest.update(chunk)
    except OSError:
        digest.update(f"unreadable:{os.path.abspath(os.path.normpath(data_path))}:".encode())
    recipe = state.effective_sft_recipe() if method == "sft" else None
    identity = {
        "method": method,
        "base": state.base_model or state.served_model,
        "recipe": recipe,
    }
    digest.update(json.dumps(identity, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def stage_error_fingerprint(stage: str) -> str:
    """The ``failed_train`` fingerprint of a stage that raised: a SHA-256 over its name."""
    return hashlib.sha256(f"loop stage error:{stage}".encode("utf-8")).hexdigest()


def stage_error_reason(stage: str, error_type: str, detail: str = "") -> str:
    """The ``failed_train`` reason of a stage that raised: its name and the exception's type.

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
    for stage in STAGE_ERROR_STAGES:
        note = hold_reason(failed, stage_error_fingerprint(stage), now)
        if note is not None:
            return note
    return None


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
