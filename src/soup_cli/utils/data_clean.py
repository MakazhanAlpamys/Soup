"""Pure dataset cleaning and sanity repair engine (Feature 3).

Provides format-aware dataset sanitization:
1. Control character & invisible space stripping (C0 controls, zero-width spaces, CRLF)
2. Markdown code block balancing (auto-closing unclosed triple backticks)
3. Boilerplate AI disclaimer stripping ("As an AI...", "Certainly!...", "I hope this helps!")
4. JSON / tool-calling argument repair (trailing commas, unescaped quotes, markdown fences)
5. Empty and degenerate turn pruning
6. Target leakage and echo turn detection
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# C0 controls (strip under 0x20 except \t, \n, \r) + DEL (0x7F) + Unicode zero-width/BOM chars
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F\u200B\u200C\u200D\uFEFF]")

# Common LLM preamble/boilerplate prefixes in assistant turns
_BOILERPLATE_PREFIXES = [
    re.compile(
        r"^(?:sure(?: thing)?|certainly|of course|happy to help|absolutely)[!,.]?\s*"
        r"(?:here(?:'s| is)[^:\n]*:)?\s*",
        re.IGNORECASE,
    ),
    re.compile(
        r"^as an? (?:ai|artificial intelligence|large language model|assistant)[^,.\n]*[,.]?\s*"
        r"(?:i (?:can|will|would be happy to|am ready to) [^,.\n]*[,.]?\s*)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^i am an? (?:ai|virtual assistant|automated system)[^,.\n]*[,.]?\s*",
        re.IGNORECASE,
    ),
]

# Common LLM canned sign-offs / suffixes in assistant turns
_BOILERPLATE_SUFFIXES = [
    re.compile(
        r"\s*(?:i )?hope (?:this|that) helps[!,.]?(?:\s*(?:please )?let me know if you "
        r"(?:need|have) (?:anything|any(?: other)? questions)[^.\n]*[!,.]?)?$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\s*(?:please )?let me know if you (?:have|need) (?:any )?"
        r"(?:further|other|more)? (?:questions|assistance|help)[!,.]?$",
        re.IGNORECASE,
    ),
]

# Markdown code block extractor for tool-call arguments
_MARKDOWN_JSON_RE = re.compile(r"^```(?:json)?\s*\n?([\s\S]*?)\n?```$", re.IGNORECASE)

# Regex to fix trailing commas in JSON object / array before closing brace/bracket
_TRAILING_COMMA_RE = re.compile(r",\s*([\]}])")


@dataclass
class CleanReport:
    """Statistics report summarizing actions taken across a dataset."""

    total_scanned: int = 0
    total_clean: int = 0
    total_modified: int = 0
    total_dropped: int = 0
    rule_counts: Dict[str, int] = field(default_factory=dict)

    def record_rule(self, rule_name: str, count: int = 1) -> None:
        """Increment count for a specific rule."""
        self.rule_counts[rule_name] = self.rule_counts.get(rule_name, 0) + count


def sanitize_text(text: str) -> Tuple[str, bool]:
    """Strip control characters, zero-width spaces, and normalize CRLF/CR to LF."""
    if not isinstance(text, str):
        return str(text), False
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL_CHARS_RE.sub("", normalized)
    return cleaned, cleaned != text


def repair_code_fences(text: str) -> Tuple[str, bool]:
    """If backticks count (```) is odd, append a closing code fence."""
    if not isinstance(text, str):
        return text, False
    fences = re.findall(r"```", text)
    if len(fences) % 2 != 0:
        cleaned = text.rstrip() + "\n```"
        return cleaned, True
    return text, False


def strip_boilerplate(text: str) -> Tuple[str, bool]:
    """Strip canned AI preamble and sign-offs from an assistant response."""
    if not isinstance(text, str):
        return text, False
    current = text.strip()
    original = current

    # Strip prefix boilerplate (loop to catch chained preambles like "Certainly! As an AI...")
    changed = True
    passes = 0
    while changed and passes < 5:
        changed = False
        passes += 1
        for pattern in _BOILERPLATE_PREFIXES:
            match = pattern.match(current)
            if match:
                current = current[match.end():].lstrip()
                changed = True
                break

    # Strip suffix boilerplate
    for pattern in _BOILERPLATE_SUFFIXES:
        match = pattern.search(current)
        if match:
            current = current[: match.start()].rstrip()
            break

    return current, current != original


def repair_json_string(text: str) -> Tuple[str, bool]:
    """Repair common JSON corruptions in tool-call arguments.

    1. Extracts JSON if wrapped in markdown ```json ... ``` fences
    2. Strips trailing commas (e.g. `{"a": 1,}`)
    3. Validates parseability with json.loads()
    """
    if not isinstance(text, str):
        return text, False

    current = text.strip()
    modified = False

    # Check if wrapped in markdown ```json
    markdown_match = _MARKDOWN_JSON_RE.match(current)
    if markdown_match:
        current = markdown_match.group(1).strip()
        modified = True

    # Check if already valid JSON
    try:
        json.loads(current)
        return current, modified
    except (json.JSONDecodeError, ValueError):
        pass

    # Try removing trailing commas
    candidate = _TRAILING_COMMA_RE.sub(r"\1", current)
    try:
        json.loads(candidate)
        return candidate, True
    except (json.JSONDecodeError, ValueError):
        pass

    return text, False


def is_echo_turn(prompt: str, response: str) -> bool:
    """Check if the assistant response is merely echoing the user prompt verbatim."""
    clean_p = prompt.strip().lower()
    clean_r = response.strip().lower()
    if not clean_p or not clean_r:
        return False
    return clean_p == clean_r


def clean_row(
    row: Dict[str, Any],
    fmt: str,
    *,
    min_tokens: int = 1,
    prune_echo: bool = False,
    strip_ai_boilerplate: bool = False,
    repair_code: bool = False,
    repair_json: bool = False,
    drop_invalid_json: bool = False,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Clean a single row according to its format.

    Returns:
        A tuple of (cleaned_row_or_None, list_of_rules_applied).
        If the row should be dropped, returns (None, [drop_reason]).
    """
    applied_rules: List[str] = []

    if fmt == "tool-calling" or "tool_calls" in row:
        # Tool calling clean (messages + tool_calls arguments)
        cleaned_row = dict(row)
        tool_calls = row.get("tool_calls", [])
        if isinstance(tool_calls, list):
            cleaned_calls: List[Dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                args = call.get("arguments", "")
                if isinstance(args, str) and repair_json:
                    san_args, _ = sanitize_text(args)
                    repaired_args, was_repaired = repair_json_string(san_args)
                    if was_repaired:
                        applied_rules.append("Malformed JSON in Tools")
                        call = dict(call)
                        call["arguments"] = repaired_args
                    elif drop_invalid_json:
                        try:
                            json.loads(san_args)
                        except (json.JSONDecodeError, ValueError):
                            return None, ["Invalid JSON in Tool Calls"]
                cleaned_calls.append(call)
            cleaned_row["tool_calls"] = cleaned_calls

        # Also sanitize messages if present
        if "messages" in row and isinstance(row["messages"], list):
            cleaned_messages: List[Dict[str, Any]] = []
            for msg in row["messages"]:
                if not isinstance(msg, dict):
                    continue
                role = str(msg.get("role", "")).strip()
                content = msg.get("content")
                if content is not None:
                    san_c, was_san = sanitize_text(str(content))
                    if was_san:
                        applied_rules.append("Invisible & Control Chars")
                else:
                    san_c = None
                cleaned_messages.append({"role": role, "content": san_c})
            cleaned_row["messages"] = cleaned_messages

        return cleaned_row, list(dict.fromkeys(applied_rules))

    elif fmt == "chatml" or ("messages" in row and isinstance(row.get("messages"), list)):
        messages = row.get("messages", [])
        if not messages or not isinstance(messages, list):
            return None, ["Empty / malformed messages"]

        cleaned_messages = []
        user_prompt = ""

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip()
            content = message.get("content")

            if content is None:
                content = ""
            elif not isinstance(content, str):
                content = str(content)

            # Sanitize control characters
            sanitized_content, was_sanitized = sanitize_text(content)
            if was_sanitized:
                applied_rules.append("Invisible & Control Chars")

            if role == "user":
                user_prompt = sanitized_content
                cleaned_messages.append({"role": role, "content": sanitized_content})
            elif role == "assistant":
                # Check for echo turn (opt-in)
                if prune_echo and user_prompt and is_echo_turn(user_prompt, sanitized_content):
                    return None, ["Target Leakage / Echo"]

                # Boilerplate stripping
                if strip_ai_boilerplate:
                    sanitized_content, was_boilerplate = strip_boilerplate(sanitized_content)
                    if was_boilerplate:
                        applied_rules.append("Boilerplate Disclaimers")

                # Code fence balancing
                if repair_code:
                    sanitized_content, was_code_repaired = repair_code_fences(sanitized_content)
                    if was_code_repaired:
                        applied_rules.append("Markdown Code Fence Repair")

                # Empty or short check
                if len(sanitized_content.strip()) < min_tokens:
                    return None, ["Empty / Whitespace Turns"]

                cleaned_messages.append({"role": role, "content": sanitized_content})
            elif role in ("system", "tool", "function"):
                cleaned_messages.append({"role": role, "content": sanitized_content})

        if not any(m.get("role") == "assistant" for m in cleaned_messages):
            return None, ["Empty / Whitespace Turns"]

        cleaned_row = dict(row)
        cleaned_row["messages"] = cleaned_messages
        return cleaned_row, list(dict.fromkeys(applied_rules))

    elif fmt == "alpaca" or ("instruction" in row and "output" in row):
        instruction = str(row.get("instruction", ""))
        input_text = str(row.get("input", ""))
        output_text = str(row.get("output", ""))

        san_inst, inst_san = sanitize_text(instruction)
        san_in, in_san = sanitize_text(input_text)
        san_out, out_san = sanitize_text(output_text)

        if inst_san or in_san or out_san:
            applied_rules.append("Invisible & Control Chars")

        prompt_combined = (san_inst + "\n" + san_in).strip()
        if prune_echo and is_echo_turn(prompt_combined, san_out):
            return None, ["Target Leakage / Echo"]

        if strip_ai_boilerplate:
            san_out, was_boilerplate = strip_boilerplate(san_out)
            if was_boilerplate:
                applied_rules.append("Boilerplate Disclaimers")

        if repair_code:
            san_out, was_code_repaired = repair_code_fences(san_out)
            if was_code_repaired:
                applied_rules.append("Markdown Code Fence Repair")

        if len(san_out.strip()) < min_tokens:
            return None, ["Empty / Whitespace Turns"]

        cleaned_row = dict(row)
        cleaned_row["instruction"] = san_inst
        cleaned_row["input"] = san_in
        cleaned_row["output"] = san_out
        return cleaned_row, list(dict.fromkeys(applied_rules))

    elif fmt == "sharegpt" or (
        "conversations" in row and isinstance(row.get("conversations"), list)
    ):
        conversations = row.get("conversations", [])
        if not conversations or not isinstance(conversations, list):
            return None, ["Empty / malformed conversations"]

        cleaned_convs: List[Dict[str, Any]] = []
        last_human = ""

        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            from_role = str(turn.get("from", "")).strip()
            value = str(turn.get("value", ""))

            san_val, was_sanitized = sanitize_text(value)
            if was_sanitized:
                applied_rules.append("Invisible & Control Chars")

            if from_role in ("human", "user"):
                last_human = san_val
                cleaned_convs.append({"from": from_role, "value": san_val})
            elif from_role in ("gpt", "assistant", "chatgpt"):
                if prune_echo and last_human and is_echo_turn(last_human, san_val):
                    return None, ["Target Leakage / Echo"]

                if strip_ai_boilerplate:
                    san_val, was_boilerplate = strip_boilerplate(san_val)
                    if was_boilerplate:
                        applied_rules.append("Boilerplate Disclaimers")

                if repair_code:
                    san_val, was_code_repaired = repair_code_fences(san_val)
                    if was_code_repaired:
                        applied_rules.append("Markdown Code Fence Repair")

                if len(san_val.strip()) < min_tokens:
                    return None, ["Empty / Whitespace Turns"]

                cleaned_convs.append({"from": from_role, "value": san_val})
            else:
                cleaned_convs.append({"from": from_role, "value": san_val})

        if not any(t.get("from") in ("gpt", "assistant", "chatgpt") for t in cleaned_convs):
            return None, ["Empty / Whitespace Turns"]

        cleaned_row = dict(row)
        cleaned_row["conversations"] = cleaned_convs
        return cleaned_row, list(dict.fromkeys(applied_rules))

    elif fmt in ("dpo", "kto"):
        prompt = str(row.get("prompt", ""))
        chosen = str(row.get("chosen", "") or row.get("completion", ""))
        rejected = str(row.get("rejected", ""))

        san_p, p_san = sanitize_text(prompt)
        san_c, c_san = sanitize_text(chosen)
        san_r, r_san = sanitize_text(rejected)

        if p_san or c_san or r_san:
            applied_rules.append("Invisible & Control Chars")

        if repair_code:
            san_c, c_rep = repair_code_fences(san_c)
            san_r, r_rep = repair_code_fences(san_r)
            if c_rep or r_rep:
                applied_rules.append("Markdown Code Fence Repair")

        if len(san_c.strip()) < min_tokens:
            return None, ["Empty / Whitespace Turns"]

        cleaned_row = dict(row)
        cleaned_row["prompt"] = san_p
        if "chosen" in row:
            cleaned_row["chosen"] = san_c
        if "completion" in row:
            cleaned_row["completion"] = san_c
        if "rejected" in row:
            cleaned_row["rejected"] = san_r

        return cleaned_row, list(dict.fromkeys(applied_rules))

    # Plaintext or unknown format fallback: clean control characters
    cleaned_row = {}
    for key, val in row.items():
        if isinstance(val, str):
            san_val, was_san = sanitize_text(val)
            if was_san:
                applied_rules.append("Invisible & Control Chars")
            cleaned_row[key] = san_val
        else:
            cleaned_row[key] = val

    return cleaned_row, list(dict.fromkeys(applied_rules))


def clean_dataset(
    data: List[Dict[str, Any]],
    fmt: str,
    *,
    min_tokens: int = 1,
    prune_echo: bool = False,
    strip_ai_boilerplate: bool = False,
    repair_code: bool = False,
    repair_json: bool = False,
    drop_invalid_json: bool = False,
) -> Tuple[List[Dict[str, Any]], CleanReport]:
    """Clean a full dataset in memory and return (cleaned_rows, report)."""
    report = CleanReport(total_scanned=len(data))
    cleaned_rows: List[Dict[str, Any]] = []

    for row in data:
        cleaned, rules = clean_row(
            row,
            fmt,
            min_tokens=min_tokens,
            prune_echo=prune_echo,
            strip_ai_boilerplate=strip_ai_boilerplate,
            repair_code=repair_code,
            repair_json=repair_json,
            drop_invalid_json=drop_invalid_json,
        )
        if cleaned is None:
            report.total_dropped += 1
            for rule in rules:
                report.record_rule(rule)
        else:
            if rules:
                report.total_modified += 1
                for rule in rules:
                    report.record_rule(rule)
            else:
                report.total_clean += 1
            cleaned_rows.append(cleaned)

    return cleaned_rows, report

