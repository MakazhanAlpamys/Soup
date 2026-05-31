"""Shared utilities for data generation providers."""

import json


def parse_json_array(content: str) -> list[dict]:
    """Parse a JSON array from LLM output, handling markdown code blocks.

    Handles:
    - Clean JSON arrays
    - Markdown code fences (```json ... ```)
    - JSON arrays with surrounding text
    - Line-by-line JSON objects (NDJSON fallback)
    """
    content = content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Try to find JSON array in content
    start = content.find("[")
    end = content.rfind("]")
    if start != -1 and end != -1 and end > start:
        content = content[start:end + 1]

    try:
        result = json.loads(content)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # Try line-by-line JSON objects
    results = []
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                continue
    return results
