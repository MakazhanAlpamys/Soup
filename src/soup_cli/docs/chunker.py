"""Heading-aware Markdown chunker for documentation search."""

from __future__ import annotations

import re
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def split_markdown_by_headings(text: str, doc_name: str) -> list[dict[str, Any]]:
    """Split markdown text on ATX headings into searchable document chunks.

    Each record contains:
      - doc_name: Identifier of the source document
      - section: Title of the current heading section
      - heading_prefix: Breadcrumb hierarchy (e.g. 'Training > SFT > LoRA')
      - text: Section body text
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []

    current_section: str | None = None
    heading_stack: list[tuple[int, str]] = []  # (level, text)
    body_lines: list[str] = []
    in_fence: str | None = None

    def _flush() -> None:
        nonlocal current_section, body_lines
        body = "\n".join(body_lines).strip()
        if body or current_section:
            prefix = " > ".join(title for _, title in heading_stack) if heading_stack else doc_name
            chunks.append(
                {
                    "doc_name": doc_name,
                    "section": current_section or "Overview",
                    "heading_prefix": prefix,
                    "text": body,
                }
            )
        body_lines = []

    for line in lines:
        fence_match = _FENCE_RE.match(line)
        if fence_match:
            fence = fence_match.group(1)
            if in_fence is None:
                in_fence = fence[0] * len(fence)
            elif fence.startswith(in_fence[0]) and len(fence) >= len(in_fence):
                in_fence = None
            body_lines.append(line)
            continue

        if in_fence is None:
            m = _HEADING_RE.match(line)
            if m:
                _flush()
                level = len(m.group(1))
                heading_title = m.group(2).strip()

                # Unwind heading stack to parent level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, heading_title))
                current_section = heading_title
                continue

        body_lines.append(line)

    _flush()
    return chunks
