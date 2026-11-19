"""A promotion reaches three places; ``loop.yaml`` decides which version is production.

(Review 5b, N1.) A promotion changes the server (its unkeyed traffic goes to the adapter),
``loop.yaml`` (``production_name`` / ``production_version`` ...) and the version log
(``versions.jsonl`` and the iteration's closing line in ``iterations.jsonl``). The daemon used to
write ``loop.yaml`` only when the cycle ended, and the same cycle went on into the next harvest,
train and ship (about an hour): a kill in that hour left the log at version 2, the server on the
adapter and ``loop.yaml`` on version 1 with the canary still running, and nothing said so.

The order is now: the server switch first (it may fail, and then nothing is recorded as promoted),
then ``loop.yaml`` (atomic replace under the state lock, at the moment of the promotion and not at
the end of the cycle), then the log lines. ``loop.yaml`` decides; the logs come after it and can be
recovered from it:

- ``loop.yaml`` ahead of the log (killed between the two writes): the version line and the
  iteration's closing line are missing. ``finish`` writes them (both are idempotent).
- the log ahead of ``loop.yaml`` (the order of an older daemon, or a hand edit): the log has the
  promote line of version N and ``loop.yaml`` still holds that adapter as the running canary.
  ``finish`` applies the promotion to ``loop.yaml`` (the change ``_promote`` makes), under the
  lock, if nothing moved meanwhile.
- anything else that does not add up is NOT touched: it is reported, by name, for a person.

``detect`` is read-only (``soup loop status`` shows its sentence in the ``promotion`` row);
``finish`` runs when ``soup loop watch`` starts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Optional

from soup_cli.utils.loop_holdout import promotion_cursor_changes
from soup_cli.utils.loop_log import (
    LoopLog,
    progress_line,
    promotion_changes,
    resolved_line,
    utc_now_iso,
    version_line,
)
from soup_cli.utils.loop_state import LoopState, read_state, update_state

#: The most of a log read to compare it with ``loop.yaml``; a bigger file is not read.
_MAX_LOG_BYTES = 64 * 1024 * 1024
_MAX_LINE_BYTES = 1024 * 1024
_RECOVERED = (
    "recovered: loop.yaml held this promotion and the log did not "
    "(the daemon stopped between the two writes)"
)


@dataclass(frozen=True)
class Mismatch:
    """What differs between ``loop.yaml`` and the version log.

    ``kind`` is ``yaml_behind`` (the log has the promotion, ``loop.yaml`` has not),
    ``log_behind`` (``loop.yaml`` has it, the log has not) or ``unexplained`` (the numbers differ
    in a way this tool does not finish by itself). ``text`` is one sentence for ``soup loop
    status`` and the daemon's log.
    """

    kind: str
    version: int
    text: str
    adapter: Optional[str] = None
    iteration_id: Optional[str] = None


def promoted_state(state: LoopState) -> LoopState:
    """``state`` with its running canary promoted to production (the change ``_promote`` makes)."""
    return replace(
        state,
        previous_production_name=state.production_name,
        previous_production_adapter=state.production_adapter,
        production_name=state.canary_active,
        production_adapter=state.canary_adapter,
        canary_active=None,
        canary_traffic_pct=None,
        canary_rollout_id=None,
        canary_adapter=None,
        canary_started_ts=None,
        canary_override=None,
        **promotion_changes(state),
        **promotion_cursor_changes(state),
    )


def _read_lines(path: str) -> "list[dict]":
    """The JSON objects of a log, in order; a torn or huge line, or a huge file, reads as none."""
    lines: "list[dict]" = []
    try:
        if os.path.getsize(path) > _MAX_LOG_BYTES:
            return []
        with open(path, "rb") as handle:
            for raw in handle:
                if len(raw) > _MAX_LINE_BYTES:
                    continue
                try:
                    record = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    lines.append(record)
    except (OSError, UnicodeDecodeError):
        return []
    return lines


def _int(value: object) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def detect(state: LoopState, log: Optional[LoopLog] = None) -> Optional[Mismatch]:
    """The mismatch between ``loop.yaml`` and ``versions.jsonl``, or ``None``. Never writes."""
    log = LoopLog() if log is None else log
    if not state.version_log_started:
        return None  # a loop without a version log has nothing to compare
    lines = _read_lines(log.path("version"))
    versions = [number for number in (_int(line.get("to_version")) for line in lines) if number]
    if not versions:
        return None
    top = max(versions)
    latest = state.latest_version
    if top == latest:
        return None
    if top > latest:
        line = next(
            (item for item in reversed(lines)
             if _int(item.get("to_version")) == top and item.get("cause") == "promote"),
            None,
        )
        if line is not None and top == latest + 1 and state.canary_active == line.get("served_as"):
            name = line.get("served_as")
            return Mismatch(
                "yaml_behind", top,
                f"versions.jsonl records version {top} ({name}) as production, but loop.yaml "
                f"still names version {state.production_version} and {name} as the running "
                "canary: an unfinished promotion; the next soup loop watch start finishes it",
                adapter=str(name), iteration_id=line.get("iteration_id"),
            )
    elif (
        latest == top + 1
        and state.production_version == latest
        and state.production_name is not None
    ):
        return Mismatch(
            "log_behind", latest,
            f"loop.yaml names version {latest} ({state.production_name}) as production, but "
            f"versions.jsonl ends at version {top}: the next soup loop watch start writes the "
            "missing line",
            adapter=state.production_name, iteration_id=state.production_iteration_id,
        )
    return Mismatch(
        "unexplained", top,
        f"versions.jsonl ends at version {top} but loop.yaml's latest version is {latest}: not "
        "a half-done promotion this tool can finish; check .soup-loops/versions.jsonl against "
        "loop.yaml",
    )


def _latest_iteration_line(log: LoopLog, iteration_id: Optional[str]) -> Optional[dict]:
    if not iteration_id:
        return None
    found = None
    for line in _read_lines(log.path("iteration")):
        if line.get("iteration_id") == iteration_id:
            found = line  # the latest line of an iteration wins
    return found


def _finish_yaml(state_path: Optional[str], found: Mismatch) -> "list[str]":
    adapter, number = found.adapter, found.version

    def apply(fresh: LoopState) -> LoopState:
        # Only if nothing moved meanwhile: the canary is still that adapter, one version behind.
        if fresh.canary_active == adapter and fresh.latest_version == number - 1:
            return promoted_state(fresh)
        return fresh

    try:
        done = update_state(apply, state_path)
    except (OSError, ValueError):
        return [f"{found.text} (loop.yaml could not be written now)"]
    if done.latest_version == number:
        return [
            f"finished an unfinished promotion: loop.yaml now names version {number} "
            f"({adapter}) as production, as versions.jsonl already did"
        ]
    return [f"{found.text} (loop.yaml changed meanwhile: not touched)"]


def _finish_log(state: LoopState, log: LoopLog, found: Mismatch) -> "list[str]":
    now = utc_now_iso()
    wrote = log.version(version_line(
        from_version=state.previous_production_version, to_version=found.version,
        cause="promote", actor="loop", iteration_id=found.iteration_id,
        served_as=found.adapter, reason=_RECOVERED, ts=now,
    ))
    snapshot = _latest_iteration_line(log, found.iteration_id)
    if snapshot is not None and snapshot.get("outcome") == "canary":
        log.iteration(resolved_line(
            snapshot, outcome="promoted", reason=_RECOVERED, finished_at=now,
            to_version=found.version,
        ))
    log.progress(progress_line(
        found.iteration_id, "promote", "end", ts=now,
        message=f"recovered the log line of version {found.version} ({found.adapter})",
    ))
    if not wrote:
        return [f"{found.text} (the log could not be written now)"]
    return [
        f"finished an unfinished promotion: wrote the missing line of version {found.version} "
        f"({found.adapter}) to versions.jsonl"
    ]


def finish(state_path: Optional[str], log: LoopLog) -> "list[str]":
    """Finish a half-done promotion this tool can finish; the sentences say what was done.

    ``[]`` when ``loop.yaml`` and the log agree. An ``unexplained`` mismatch is reported only.
    """
    try:
        state = read_state(state_path)
    except (OSError, ValueError):
        return []
    found = detect(state, log)
    if found is None:
        return []
    if found.kind == "yaml_behind":
        return _finish_yaml(state_path, found)
    if found.kind == "log_behind":
        return _finish_log(state, log, found)
    return [found.text]
