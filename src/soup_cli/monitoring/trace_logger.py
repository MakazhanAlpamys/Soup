"""Append-only JSONL request log for ``soup serve --trace-log``.

``soup serve`` writes one conversation record per answer in the version 1 format of
``soup_cli/schemas/loop/conversation.schema.json`` (see ``docs/loop-formats.md``)::

    {"schema_version": 1, "id": <response id>, "conversation_id": ..., "ts": ...,
     "model": ..., "version": null, "adapter": ..., "canary_bucket": ...,
     "messages": [...], "response": ..., "finish_reason": ..., "latency_ms": ...,
     "tokens": {"prompt": ..., "completion": ...}, "source": "soup-serve",
     "prompt": <last user message>}

``prompt`` is kept so readers of the version 0 line keep working.
:meth:`TraceLogWriter.record` still writes the version 0 line
``{"prompt", "response", "latency_ms", "tokens": <int>, "ts": <epoch>}``.

Hard rotation cap of ``cap_mb`` (default 100 MB). When the active file
exceeds the cap, it is renamed ``<path>.1`` (one backup retained) and a
new active file is started. The writer is thread-safe — FastAPI runs
sync handlers in a threadpool, and the request log is shared.

Path containment via shared :func:`soup_cli.utils.paths.is_under_cwd`
prevents ``--trace-log /etc/cron.d/x`` writes (mirrors v0.30.0 / v0.32.0
policy).

Added in v0.40.3 (#33 (b)).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from soup_cli.utils.paths import is_under_cwd

logger = logging.getLogger(__name__)

#: ``schema_version`` of the conversation record :meth:`TraceLogWriter.record_conversation` writes.
CONVERSATION_SCHEMA_VERSION = 1

#: ``source`` of every record ``soup serve`` writes.
SERVE_SOURCE = "soup-serve"

#: The message roles ``conversation.schema.json`` allows.
CONVERSATION_ROLES: frozenset[str] = frozenset(
    {"system", "developer", "user", "assistant", "tool"}
)

_MAX_MODEL_CHARS = 512
_MAX_ADAPTER_CHARS = 256
_MAX_FINISH_REASON_CHARS = 64
_MAX_CONVERSATION_ID_CHARS = 512

# Mirror v0.34.0 crash.py policy: redact common secret shapes BEFORE writing
# request/response strings to disk. Operators frequently embed tokens in
# system prompts or chat-template stubs; logging them verbatim is a leak.
_SECRET_RE = re.compile(
    # `Bearer` token body excludes `.` so an end-of-sentence period is not
    # silently consumed (e.g. `Use Bearer abc12345xyz.` keeps the period).
    r"(?:hf_[A-Za-z0-9]{8,}"
    r"|sk-[A-Za-z0-9_-]{16,}"
    r"|Bearer\s+[A-Za-z0-9_\-]{8,}"
    # GitHub tokens: classic (`ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_`) and fine-grained.
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
    # AWS access key ids (long-term `AKIA`, temporary `ASIA`).
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"
    # Slack tokens (`xoxb-`, `xoxp-`, `xoxa-`, `xoxr-`, `xoxs-`).
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    # JSON Web Tokens: three base64url parts, the first two starting `eyJ` ({"...).
    r"|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"
    # PEM private keys: the block, or from the header on when the footer is missing. The
    # body is capped so a header without a footer cannot make the scan quadratic.
    r"|-----BEGIN [A-Z ]{0,24}PRIVATE KEY-----[A-Za-z0-9+/=\s]{0,16384}"
    r"(?:-----END [A-Z ]{0,24}PRIVATE KEY-----)?"
    # `api_key=...`, `api-key: ...`, `apikey=...`, `secret=...`, `password: ...`, `token=...`:
    # the whole assignment goes, name included. A value of at most five characters stays.
    r"|(?i:\b(?:api[_-]?key|secret|password|passwd|token)\b[\"']?\s*[=:]\s*)"
    r"[\"']?[^\s\"',;]{6,})"
)


def _redact_secrets(text: str) -> str:
    """Replace common token shapes with ``<redacted>``."""
    if not isinstance(text, str):
        return text
    return _SECRET_RE.sub("<redacted>", text)


def _redact_value(value: Any) -> Any:
    """Recursively walk and redact any nested string secrets."""
    if isinstance(value, str):
        return _redact_secrets(value)
    if isinstance(value, dict):
        return {k: _redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_value(v) for v in value]
    return value


# Public alias so other subsystems (e.g. the v0.71.26 reward-hack mitigation
# log) reuse the ONE redaction implementation instead of duplicating the regex.
redact_value = _redact_value

DEFAULT_CAP_MB: int = 100

_MIN_CAP_MB: int = 1
_MAX_CAP_MB: int = 10_000


#: Permission bits of a log ``soup serve`` creates: the owner alone reads and writes it. The
#: logs hold whole prompts, answers and corrections; on POSIX another local user must not
#: read them. Windows ignores the mode (access follows the folder's ACL), so there it is no
#: protection beyond what the directory already gives.
LOG_FILE_MODE = 0o600


def open_private_append(path: "Path | str") -> Any:
    """Open ``path`` for binary appending, creating it ``0o600`` when it is new.

    The mode applies to a file this call creates; a log that already exists keeps the
    permissions it has (a ``chmod`` of someone else's file is not this function's call).
    """
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
    fd = os.open(os.fspath(path), flags, LOG_FILE_MODE)
    try:
        return os.fdopen(fd, "ab")
    except Exception:
        os.close(fd)
        raise


def resolve_log_path(path: str, *, flag: str) -> Path:
    """Validate a log path given to ``flag`` and return it resolved.

    The path must be a non-empty string without null bytes and stay under the
    current directory. Its parent directory is created. Shared by the trace and
    feedback writers so both apply one containment rule.
    """
    if not isinstance(path, str) or not path or "\x00" in path:
        raise ValueError("path must be a non-empty string with no null bytes")
    if not is_under_cwd(path):
        raise ValueError(f"{flag} path must stay under cwd: {path}")
    resolved = Path(os.path.realpath(path))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def utc_timestamp() -> str:
    """The current time as ISO 8601 UTC with a ``+00:00`` suffix, as the loop formats want."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _count_or_none(value: Any) -> Optional[int]:
    """A token count as the engine reported it, or ``None`` when it did not."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _latency_or_none(value: Any) -> Optional[float]:
    """Latency in ms, or ``None`` when it is not a finite non-negative number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 3)


def _text_or_none(value: Any, *, max_chars: int) -> Optional[str]:
    """``value`` when it is a non-empty string within ``max_chars``, else ``None``."""
    if isinstance(value, str) and 0 < len(value) <= max_chars:
        return value
    return None


def _last_user_message(messages: list[dict[str, Any]]) -> str:
    return next(
        (str(msg.get("content", "")) for msg in reversed(messages) if msg.get("role") == "user"),
        "",
    )


def _conversation_problem(
    model: Any, adapter: Any, messages: list[dict[str, Any]]
) -> Optional[str]:
    """Why a conversation record would not be schema-valid, or ``None`` when it would be."""
    if _text_or_none(model, max_chars=_MAX_MODEL_CHARS) is None:
        return "the model name is empty or longer than 512 characters"
    if adapter is not None and _text_or_none(adapter, max_chars=_MAX_ADAPTER_CHARS) is None:
        return "the adapter name is empty or longer than 256 characters"
    if not messages:
        return "the request has no messages"
    for msg in messages:
        role = msg.get("role")
        if not isinstance(role, str) or role not in CONVERSATION_ROLES:
            return (
                f"message role {str(role)[:40]!r} is not one of "
                f"{', '.join(sorted(CONVERSATION_ROLES))}"
            )
    return None


class TraceLogWriter:
    """Thread-safe append-only JSONL writer with size-based rotation."""

    def __init__(self, path: str, *, cap_mb: int = DEFAULT_CAP_MB) -> None:
        if isinstance(cap_mb, bool) or not isinstance(cap_mb, int):
            raise TypeError("cap_mb must be int")
        if not (_MIN_CAP_MB <= cap_mb <= _MAX_CAP_MB):
            raise ValueError(
                f"cap_mb must be in [{_MIN_CAP_MB}, {_MAX_CAP_MB}], got {cap_mb}"
            )
        self._path = resolve_log_path(path, flag="--trace-log")
        self._cap_bytes = cap_mb * 1024 * 1024
        # Single-process lock: protects rotate + write within ONE serve
        # process. Multi-worker (e.g. `--workers 4`) deployments writing to
        # the same path would race on rotation; the OS-level rename is
        # best-effort and one missed rotation cycle is acceptable for a
        # passive log. Documented limitation.
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def cap_bytes(self) -> int:
        return self._cap_bytes

    def record(
        self,
        *,
        prompt: str,
        response: str,
        latency_ms: float,
        tokens: int,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Append one version 0 entry. Never raises on serialisation issues."""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "prompt": _redact_secrets(str(prompt)),
            "response": _redact_secrets(str(response)),
            "latency_ms": float(latency_ms),
            "tokens": int(tokens),
        }
        if extra:
            for key, value in extra.items():
                if key in entry:
                    continue
                # Defence in depth — redact secrets in caller-supplied
                # extras too (e.g. `extra={"system_prompt": ...}`).
                entry[str(key)] = _redact_value(value)
        self._append(entry)

    def record_conversation(
        self,
        *,
        response_id: str,
        model: str,
        messages: list[dict[str, Any]],
        response: str,
        conversation_id: Optional[str] = None,
        version: Optional[int] = None,
        adapter: Optional[str] = None,
        canary_bucket: Optional[str] = None,
        finish_reason: Optional[str] = None,
        latency_ms: Optional[float] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ) -> None:
        """Append one version 1 conversation record (``conversation.schema.json``).

        ``messages`` is the conversation the model saw, without the answer; its
        contents and ``response`` are redacted before they reach disk. A value that
        is unknown is written as ``null`` (never ``0``): token counts that are not
        non-negative ints, a latency that is not finite, a ``finish_reason`` that is
        not a short string, a ``conversation_id`` outside 1-512 characters.

        Only schema-valid lines are written. A request the schema cannot describe
        (a message role outside ``system``/``developer``/``user``/``assistant``/
        ``tool``, an empty ``messages`` list, an empty model name) is answered
        normally but not logged; the reason goes to the server log. Never raises.
        """
        problem = _conversation_problem(model, adapter, messages)
        if problem is not None:
            logger.warning("conversation record %s not written: %s", response_id, problem)
            return
        clean_messages = [
            {
                "role": msg["role"],
                "content": _redact_secrets(str(msg.get("content", ""))),
            }
            for msg in messages
        ]
        entry: dict[str, Any] = {
            "schema_version": CONVERSATION_SCHEMA_VERSION,
            "id": response_id,
            "conversation_id": _text_or_none(
                _redact_secrets(conversation_id), max_chars=_MAX_CONVERSATION_ID_CHARS
            ),
            "ts": utc_timestamp(),
            "model": model,
            "version": _count_or_none(version) or None,
            "adapter": adapter,
            "canary_bucket": canary_bucket if canary_bucket in ("stable", "canary") else None,
            "messages": clean_messages,
            "response": _redact_secrets(str(response)),
            "finish_reason": _text_or_none(finish_reason, max_chars=_MAX_FINISH_REASON_CHARS),
            "latency_ms": _latency_or_none(latency_ms),
            "tokens": {
                "prompt": _count_or_none(prompt_tokens),
                "completion": _count_or_none(completion_tokens),
            },
            "source": SERVE_SOURCE,
            # Version 0 readers (soup data from-traces, soup loop) read this key.
            "prompt": _last_user_message(clean_messages),
        }
        self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        """Serialise ``entry`` and append it, rotating first. Never raises."""
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return  # drop unserialisable entries silently — passive log
        try:
            line_bytes = (line + "\n").encode("utf-8")
        except UnicodeEncodeError:
            # A lone surrogate cannot be written as UTF-8: \u-escape it instead.
            try:
                line = json.dumps(entry, ensure_ascii=True, default=str)
            except (TypeError, ValueError):
                return
            line_bytes = (line + "\n").encode("ascii")
        with self._lock:
            self._maybe_rotate(extra=len(line_bytes))
            try:
                with open_private_append(self._path) as fh:
                    fh.write(line_bytes)
            except OSError:
                # Disk full / permissions — never crash the request handler.
                return

    def _maybe_rotate(self, *, extra: int) -> None:
        try:
            current = self._path.stat().st_size
        except OSError:
            return
        if current + extra <= self._cap_bytes:
            return
        backup = self._path.with_suffix(self._path.suffix + ".1")
        try:
            # Mirror v0.33.0 #22 TOCTOU policy: refuse to overwrite a symlink
            # at the backup path. An attacker who pre-places <log>.1 as a
            # symlink to /etc/cron.d/x would otherwise have the active log
            # renamed onto that target.
            try:
                backup_stat = os.lstat(backup)
                if stat.S_ISLNK(backup_stat.st_mode):
                    return  # refuse to overwrite a symlink
                backup.unlink()
            except FileNotFoundError:
                pass
            self._path.rename(backup)
        except OSError:
            # rename failure (Windows file-lock / antivirus / cross-volume)
            # is non-fatal: rotation is best-effort.
            return
