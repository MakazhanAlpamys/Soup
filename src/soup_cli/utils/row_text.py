"""Shared multi-format text extraction for expectations and quality scans."""

from __future__ import annotations

from typing import Any, List, Mapping

_ASSISTANT_ROLES = frozenset({"assistant", "gpt", "chatgpt", "bot", "model"})


def _extract_content_parts(content: Any) -> List[str]:
    """Extract strings from string or multimodal content parts list."""
    import json

    parts: List[str] = []
    if isinstance(content, str):
        if content:
            parts.append(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                if part:
                    parts.append(part)
            elif isinstance(part, Mapping):
                val = part.get("text")
                if isinstance(val, str) and val:
                    parts.append(val)
                args = part.get("arguments") or part.get("input")
                if isinstance(args, str) and args:
                    parts.append(args)
                elif isinstance(args, (dict, list)):
                    parts.append(json.dumps(args))
    return parts


def _extract_tool_calls_text(tool_calls: Any) -> List[str]:
    """Extract function call arguments and names from tool_calls list."""
    import json

    parts: List[str] = []
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if isinstance(call, Mapping):
                func = call.get("function")
                if isinstance(func, Mapping):
                    args = func.get("arguments")
                    if isinstance(args, str) and args:
                        parts.append(args)
                    elif isinstance(args, (dict, list)):
                        parts.append(json.dumps(args))
                args = call.get("arguments")
                if isinstance(args, str) and args:
                    parts.append(args)
                elif isinstance(args, (dict, list)):
                    parts.append(json.dumps(args))
    return parts


def _extract_messages_text(messages: Any, *, assistant_only: bool = False) -> List[str]:
    """Extract text from a messages list (chatml, audio, tool-calling, multimodal)."""
    parts: List[str] = []
    if not isinstance(messages, list):
        return parts
    for msg in messages:
        if not isinstance(msg, Mapping):
            continue
        role = msg.get("role")
        if assistant_only and role not in _ASSISTANT_ROLES:
            continue
        content = msg.get("content")
        parts.extend(_extract_content_parts(content))
        if "tool_calls" in msg:
            parts.extend(_extract_tool_calls_text(msg.get("tool_calls")))
    return parts


def _extract_conversations_text(
    conversations: Any, *, assistant_only: bool = False
) -> List[str]:
    """Extract text from a conversations list (sharegpt, llava, sharegpt4v)."""
    parts: List[str] = []
    if not isinstance(conversations, list):
        return parts
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        speaker = turn.get("from")
        if assistant_only and speaker not in _ASSISTANT_ROLES:
            continue
        val = turn.get("value")
        parts.extend(_extract_content_parts(val))
    return parts


def _extract_field_text_or_messages(
    val: Any, *, assistant_only: bool = False
) -> List[str]:
    """Extract text from a field that may be str or list of messages/strings."""
    if isinstance(val, str):
        return [val] if val else []
    if isinstance(val, list):
        msg_parts = _extract_messages_text(val, assistant_only=assistant_only)
        if msg_parts:
            return msg_parts
        str_parts: List[str] = []
        for item in val:
            if isinstance(item, str) and item:
                str_parts.append(item)
            elif isinstance(item, Mapping):
                if assistant_only:
                    role = item.get("role") or item.get("from")
                    if role is not None and role not in _ASSISTANT_ROLES:
                        continue
                for k in ("content", "text", "value"):
                    v = item.get(k)
                    if isinstance(v, str) and v:
                        str_parts.append(v)
        return str_parts
    return []


def extract_row_parts(row: Mapping[str, Any]) -> List[str]:
    """Extract all text parts across all format fields (PII / length scans)."""
    parts: List[str] = []

    for key in (
        "instruction",
        "input",
        "output",
        "system",
        "response",
        "anchor",
        "positive",
        "negative",
        "text",
        "content",
        "query",
        "golden_doc",
        "answer",
    ):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)
        elif isinstance(val, list) and key in ("anchor", "positive", "negative"):
            for s in val:
                if isinstance(s, str) and s:
                    parts.append(s)

    for key in ("prompt", "chosen", "rejected", "completion"):
        val = row.get(key)
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=False))

    completions = row.get("completions")
    if isinstance(completions, list):
        for c in completions:
            if isinstance(c, str) and c:
                parts.append(c)

    distractors = row.get("distractor_docs")
    if isinstance(distractors, list):
        for d in distractors:
            if isinstance(d, str) and d:
                parts.append(d)

    segments = row.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, Mapping):
                t = seg.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)

    if "conversations" in row:
        parts.extend(_extract_conversations_text(row.get("conversations"), assistant_only=False))

    if "messages" in row:
        parts.extend(_extract_messages_text(row.get("messages"), assistant_only=False))

    if "tool_calls" in row:
        parts.extend(_extract_tool_calls_text(row.get("tool_calls")))

    tools = row.get("tools")
    if isinstance(tools, list):
        for tool in tools:
            if isinstance(tool, Mapping):
                func = tool.get("function")
                if isinstance(func, Mapping):
                    desc = func.get("description")
                    if isinstance(desc, str) and desc:
                        parts.append(desc)

    return parts


def extract_row_text(row: Mapping[str, Any]) -> str:
    """Extract all text across all format fields as a single string."""
    return "\n".join(extract_row_parts(row))


def extract_assistant_parts(row: Mapping[str, Any]) -> List[str]:
    """Extract assistant-side text parts (for refusal scans)."""
    parts: List[str] = []

    for key in ("output", "response", "answer"):
        val = row.get(key)
        if isinstance(val, str) and val:
            parts.append(val)

    if "conversations" in row:
        parts.extend(_extract_conversations_text(row.get("conversations"), assistant_only=True))

    if "messages" in row:
        parts.extend(_extract_messages_text(row.get("messages"), assistant_only=True))

    if "tool_calls" in row:
        parts.extend(_extract_tool_calls_text(row.get("tool_calls")))

    if "chosen" in row:
        val = row.get("chosen")
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=True))

    if "completion" in row:
        val = row.get("completion")
        if val is not None:
            parts.extend(_extract_field_text_or_messages(val, assistant_only=True))

    completions = row.get("completions")
    if isinstance(completions, list):
        for c in completions:
            if isinstance(c, str) and c:
                parts.append(c)

    segments = row.get("segments")
    if isinstance(segments, list):
        for seg in segments:
            if isinstance(seg, Mapping) and seg.get("label", True):
                t = seg.get("text")
                if isinstance(t, str) and t:
                    parts.append(t)

    if not parts and ("text" in row or "content" in row):
        is_chat_or_pref = any(
            k in row
            for k in (
                "conversations",
                "messages",
                "instruction",
                "output",
                "prompt",
                "chosen",
                "completion",
                "query",
                "answer",
                "segments",
                "completions",
            )
        )
        if not is_chat_or_pref:
            val = row.get("text") or row.get("content")
            if isinstance(val, str) and val:
                parts.append(val)

    return parts


def extract_assistant_text(row: Mapping[str, Any]) -> str:
    """Extract assistant-side text as a single string."""
    return "\n".join(extract_assistant_parts(row))
