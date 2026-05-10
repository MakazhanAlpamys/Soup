"""v0.45.0 Part B — Anthropic Messages API converter (schema-only).

Pure-Python converter between OpenAI ``/v1/chat/completions`` payloads and
Anthropic ``/v1/messages`` payloads. The wire-up of the Anthropic-shaped
endpoint inside ``soup serve`` is deferred to v0.45.1 (matches the
project's stub-then-live policy).

Surface kept narrow on purpose:

- ``to_anthropic(openai_payload)`` -> dict in Anthropic shape
- ``from_anthropic(anthropic_payload)`` -> dict in OpenAI shape
- ``validate_anthropic_payload(p)`` -> raises on schema violations
"""

from __future__ import annotations

from typing import Any, Dict, List

# Caps mirror the v0.30.0 inference-server caps (max_tokens) and v0.40.3
# trace-log message-size policy.
_MAX_MESSAGES = 1024
_MAX_CONTENT_LEN = 1_048_576  # 1 MiB per message
_MAX_TOKENS_CAP = 16384  # matches /v1/chat/completions

_VALID_ROLES_OPENAI = frozenset({"system", "user", "assistant", "tool"})
_VALID_ROLES_ANTHROPIC = frozenset({"user", "assistant"})


def _check_str(value: Any, name: str, *, max_len: int = _MAX_CONTENT_LEN) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{name} exceeds {max_len} chars")
    return value


def _check_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        raise TypeError("messages must be a list")
    if not messages:
        raise ValueError("messages must not be empty")
    if len(messages) > _MAX_MESSAGES:
        raise ValueError(f"messages exceeds {_MAX_MESSAGES} entries")
    out: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be a dict")
        out.append(message)
    return out


def to_anthropic(openai_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI chat-completions payload to Anthropic Messages shape.

    - The (single) ``system`` message becomes the top-level ``system`` field.
    - ``user`` / ``assistant`` messages are passed through.
    - ``tool`` messages are surfaced as ``tool_result`` content blocks under
      a ``user`` role (Anthropic convention).
    - ``max_tokens`` is required by Anthropic and defaults to ``1024`` if
      missing on the OpenAI side; capped at ``_MAX_TOKENS_CAP``.
    """
    if not isinstance(openai_payload, dict):
        raise TypeError("openai_payload must be a dict")
    messages = _check_messages(openai_payload.get("messages"))
    model = _check_str(openai_payload.get("model", ""), "model", max_len=256)

    system_text: List[str] = []
    out_messages: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if not isinstance(role, str) or role not in _VALID_ROLES_OPENAI:
            raise ValueError(
                f"messages[{index}].role must be one of {_VALID_ROLES_OPENAI}"
            )
        content = message.get("content", "")
        if isinstance(content, str):
            _check_str(content, f"messages[{index}].content")
        elif isinstance(content, list):
            # Pass-through structured content (e.g. multi-modal). Caller is
            # responsible for shape validity beyond NUL-byte checks.
            for inner in content:
                if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                    _check_str(inner["text"], f"messages[{index}].content[].text")
        else:
            raise TypeError(
                f"messages[{index}].content must be str or list"
            )

        if role == "system":
            if isinstance(content, str):
                system_text.append(content)
            else:
                raise TypeError("system message content must be a string")
            continue
        if role == "tool":
            tool_call_id = _check_str(
                message.get("tool_call_id", ""),
                f"messages[{index}].tool_call_id",
                max_len=256,
            )
            # Structured (list) content from OpenAI is concatenated into a
            # single text block instead of being silently dropped — Anthropic
            # ``tool_result`` accepts either str or a list of text/image
            # blocks, but we keep the conversion lossless for the common
            # text-only path. Non-text inner items are stringified via repr.
            if isinstance(content, str):
                tool_content: Any = content
            elif isinstance(content, list):
                parts: List[str] = []
                for inner in content:
                    if isinstance(inner, dict) and isinstance(
                        inner.get("text"), str
                    ):
                        parts.append(inner["text"])
                    elif isinstance(inner, str):
                        parts.append(inner)
                tool_content = "\n".join(parts)
            else:
                raise TypeError(
                    f"messages[{index}].content must be str or list"
                )
            out_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": tool_content,
                        }
                    ],
                }
            )
            continue
        out_messages.append({"role": role, "content": content})

    raw_max = openai_payload.get("max_tokens", 1024)
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise TypeError("max_tokens must be an int")
    if raw_max < 1:
        raise ValueError("max_tokens must be >= 1")
    max_tokens = min(raw_max, _MAX_TOKENS_CAP)

    out: Dict[str, Any] = {
        "model": model,
        "messages": out_messages,
        "max_tokens": max_tokens,
    }
    if system_text:
        out["system"] = "\n\n".join(system_text)
    if "temperature" in openai_payload:
        temperature = openai_payload["temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be int or float")
        if not 0.0 <= float(temperature) <= 2.0:
            raise ValueError("temperature must be in [0.0, 2.0]")
        out["temperature"] = float(temperature)
    return out


def from_anthropic(anthropic_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic Messages payload back to OpenAI chat shape."""
    validate_anthropic_payload(anthropic_payload)
    out_messages: List[Dict[str, Any]] = []
    system_field = anthropic_payload.get("system")
    if isinstance(system_field, str) and system_field:
        out_messages.append({"role": "system", "content": system_field})
    for message in anthropic_payload["messages"]:
        # validate_anthropic_payload above guarantees the role key is
        # present and on the allowlist; ``.get`` is defence-in-depth so a
        # post-validation iteration cannot raise ``KeyError``.
        out_messages.append(
            {
                "role": message.get("role"),
                "content": message.get("content", ""),
            }
        )
    return {
        "model": anthropic_payload["model"],
        "messages": out_messages,
        "max_tokens": anthropic_payload["max_tokens"],
    }


def validate_anthropic_payload(payload: Dict[str, Any]) -> None:
    """Raise if the Anthropic payload is malformed."""
    if not isinstance(payload, dict):
        raise TypeError("payload must be a dict")
    _check_str(payload.get("model", ""), "model", max_len=256)
    messages = _check_messages(payload.get("messages"))
    for index, message in enumerate(messages):
        role = message.get("role")
        if role not in _VALID_ROLES_ANTHROPIC:
            raise ValueError(
                f"messages[{index}].role must be one of {_VALID_ROLES_ANTHROPIC}"
            )
    raw_max = payload.get("max_tokens")
    if isinstance(raw_max, bool) or not isinstance(raw_max, int):
        raise TypeError("max_tokens must be an int")
    if raw_max < 1 or raw_max > _MAX_TOKENS_CAP:
        raise ValueError(f"max_tokens must be in [1, {_MAX_TOKENS_CAP}]")
    if "system" in payload:
        if not isinstance(payload["system"], str):
            raise TypeError("system must be a string")
        _check_str(payload["system"], "system")


__all__ = [
    "to_anthropic",
    "from_anthropic",
    "validate_anthropic_payload",
]
