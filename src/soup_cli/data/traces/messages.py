"""Shared, text-only rendering for trace and ingest message payloads."""

from __future__ import annotations

import json
from typing import Any


def _message_role(message: dict[str, Any]) -> str | None:
    """Prefer explicit roles, accepting LangChain's serialized message types."""
    role = message.get("role")
    if isinstance(role, str):
        return role
    message_type = message.get("type")
    if isinstance(message_type, str):
        return {"human": "user", "ai": "assistant", "system": "system"}.get(message_type)
    return None


def content_text(value: Any) -> str:
    """Join text parts with newlines, omitting images, audio and other non-text parts."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    return "\n".join(
        part["text"] for part in value
        if isinstance(part, dict)
        and part.get("type", "text") in ("text", "input_text", "output_text")
        and isinstance(part.get("text"), str)
    )


def conversation_prompt(messages: Any) -> str:
    """Render context through the last user turn, never a trailing response.

    Without an earlier user or assistant turn, keep the last user's plain text,
    including system-plus-user requests. Otherwise use compact role/content JSON
    through that user turn. System-only requests and last user turns without
    text are omitted, not relabelled.
    """
    if not isinstance(messages, list):
        return ""
    turns = []
    for message in messages:
        if isinstance(message, dict):
            role = _message_role(message)
            if role is not None:
                turns.append({"role": role, "content": content_text(message.get("content"))})
    last_user = next((idx for idx in range(len(turns) - 1, -1, -1)
                      if turns[idx]["role"] == "user"), None)
    if last_user is None or not turns[last_user]["content"].strip():
        return ""
    if not any(turn["role"] in ("user", "assistant") for turn in turns[:last_user]):
        return turns[last_user]["content"]
    context = [turn for turn in turns[:last_user + 1] if turn["content"].strip()]
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def message_list_text(messages: list[Any], *, prompt: bool = True) -> str:
    """Distinguish message lists from content-part lists and extract text."""
    if any(isinstance(message, dict) and ("role" in message or _message_role(message) is not None)
           for message in messages):
        if prompt:
            return conversation_prompt(messages)
        # Output envelopes may include several turns; choose the final assistant.
        for message in reversed(messages):
            if isinstance(message, dict) and _message_role(message) == "assistant":
                return content_text(message.get("content"))
        return ""
    return content_text(messages)
