"""Quality diagnostics for assistant turns after dataset format conversion."""

from __future__ import annotations

from soup_cli.utils.data_clean import has_tool_calls, sanitize_text

_NON_TEXT_PART_TYPES = (
    "image", "image_url", "input_image", "audio", "audio_url", "input_audio", "video", "video_url",
)


def _visible_text(value: object) -> bool:
    return isinstance(value, str) and bool(sanitize_text(value)[0].strip())


def _has_answer_content(content: object) -> bool:
    if not isinstance(content, list):
        return _visible_text(content)
    return any(
        _visible_text(part)
        or (isinstance(part, dict) and (
            _visible_text(part.get("text")) or part.get("type") in _NON_TEXT_PART_TYPES
        ))
        for part in content
    )


def count_empty_assistant_turns(row: dict) -> int:
    """Count assistant turns with neither visible content nor a tool call."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and not has_tool_calls(message)
        and not _has_answer_content(message.get("content"))
        for message in messages
    )
