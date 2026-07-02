"""Dataset format detection and conversion.

Supported formats:
- alpaca: {"instruction": ..., "input": ..., "output": ...}
- sharegpt: {"conversations": [{"from": "human", "value": ...}, ...]}
- chatml: {"messages": [{"role": "user", "content": ...}, ...]}
- dpo: {"prompt": ..., "chosen": ..., "rejected": ...}
- kto: {"prompt": ..., "completion": ..., "label": true/false}
- llava: {"image": ..., "conversations": [{"from": "human", "value": ...}, ...]}
- sharegpt4v: {"image": ..., "conversations": [{"from": "human", "value": ...}, ...]}
- plaintext: {"text": "..."} — raw text for continued pre-training
- embedding: {"anchor": ..., "positive": ..., "negative": ...} — sentence embedding pairs/triplets
- audio: {"audio": ..., "messages": [...]} — audio + conversation for speech models
- tool-calling: {"messages": [...], "tools": [...], "tool_calls": [...]} — function-calling training
"""

import json
from typing import Optional

from rich.console import Console

console = Console()

# Required keys per format
FORMAT_SIGNATURES = {
    "alpaca": {"instruction", "output"},
    "sharegpt": {"conversations"},
    "chatml": {"messages"},
    "dpo": {"prompt", "chosen", "rejected"},
    "kto": {"prompt", "completion", "label"},
    "llava": {"image", "conversations"},
    "sharegpt4v": {"image", "conversations"},
    "embedding": {"anchor", "positive"},
    "audio": {"audio", "messages"},
    "plaintext": {"text"},
    "tool-calling": {"messages", "tools", "tool_calls"},
}


def detect_format(data: list[dict]) -> str:
    """Auto-detect dataset format from first few rows."""
    if not data:
        raise ValueError("Empty dataset - cannot detect format")

    sample = data[0]
    keys = set(sample.keys())

    # Check more specific formats first (llava/sharegpt4v before sharegpt).
    # tool-calling (messages+tools+tool_calls) is checked BEFORE audio
    # (audio+messages): a row carrying both would otherwise match audio first
    # and silently drop its tools/tool_calls. tool-calling before chatml
    # (signature is a superset of chatml). plaintext ("text" key only) last.
    check_order = [
        "alpaca", "llava", "sharegpt4v", "kto", "dpo", "embedding",
        "tool-calling", "audio", "sharegpt", "chatml", "plaintext",
    ]
    for fmt in check_order:
        required_keys = FORMAT_SIGNATURES[fmt]
        if required_keys.issubset(keys):
            return fmt

    raise ValueError(
        f"Cannot detect format. Keys found: {keys}. "
        f"Expected one of: alpaca (instruction, output), "
        f"sharegpt (conversations), chatml (messages), "
        f"dpo (prompt, chosen, rejected), "
        f"kto (prompt, completion, label), "
        f"llava/sharegpt4v (image, conversations), "
        f"embedding (anchor, positive), "
        f"audio (audio, messages), "
        f"tool-calling (messages, tools, tool_calls), "
        f"plaintext (text)"
    )


def format_to_messages(row: dict, fmt: str) -> Optional[dict]:
    """Convert any format to normalized structure for training.

    Returns:
    - SFT formats: {"messages": [{"role": ..., "content": ...}, ...]}
    - Vision formats: {"messages": [...], "image": "path"}
    - DPO format: {"prompt": ..., "chosen": ..., "rejected": ...}
    - KTO format: {"prompt": ..., "completion": ..., "label": bool}
    """
    valid_formats = (
        "chatml", "alpaca", "sharegpt", "dpo", "kto", "llava", "sharegpt4v",
        "plaintext", "embedding", "audio", "tool-calling",
        # v0.42.0 Part A
        "prm", "pre_tokenized", "input_output", "video", "multimodal",
        # v0.62.0 Part A — RAFT (Retrieval-Augmented Fine-Tuning).
        "raft",
    )
    if fmt not in valid_formats:
        raise ValueError(f"Unknown format: {fmt}")
    try:
        if fmt == "chatml":
            return _convert_chatml(row)
        elif fmt == "alpaca":
            return _convert_alpaca(row)
        elif fmt == "sharegpt":
            return _convert_sharegpt(row)
        elif fmt == "dpo":
            return _convert_dpo(row)
        elif fmt == "kto":
            return _convert_kto(row)
        elif fmt == "plaintext":
            return _convert_plaintext(row)
        elif fmt == "embedding":
            return _convert_embedding(row)
        elif fmt == "audio":
            return _convert_audio(row)
        elif fmt == "tool-calling":
            return _convert_tool_calling(row)
        elif fmt == "prm":
            return _convert_prm(row)
        elif fmt == "pre_tokenized":
            return _convert_pre_tokenized(row)
        elif fmt == "input_output":
            return _convert_input_output(row)
        elif fmt == "video":
            return _convert_video(row)
        elif fmt == "multimodal":
            return _convert_multimodal(row)
        elif fmt == "raft":
            return _convert_raft(row)
        else:
            return _convert_vision(row)
    except (KeyError, TypeError, IndexError, ValueError):
        return None


def _require_str_content(value: object, field: str) -> str:
    """Reject non-string message content (e.g. a JSON ``null``).

    The older converters passed row values through verbatim, so a JSON
    ``null`` became literal ``None`` message content. Raising here routes the
    row to ``format_to_messages``'s drop path instead of silently corrupting
    the dataset with a ``None``-content turn.
    """
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    return value


def _convert_alpaca(row: dict) -> dict:
    instruction = _require_str_content(row["instruction"], "alpaca.instruction")
    input_text = row.get("input") or ""  # missing / null -> ""
    output = _require_str_content(row["output"], "alpaca.output")

    user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction

    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]

    if row.get("system"):
        messages.insert(0, {"role": "system", "content": row["system"]})

    return {"messages": messages}


def _convert_sharegpt(row: dict) -> dict:
    conversations = row["conversations"]
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    messages = []
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        messages.append(
            {"role": role, "content": _require_str_content(turn["value"], "sharegpt.value")}
        )

    return {"messages": messages}


def _convert_chatml(row: dict) -> dict:
    # Already in the right format
    return {"messages": row["messages"]}


def _convert_dpo(row: dict) -> dict:
    """Convert DPO preference row to {prompt, chosen, rejected} for trl.DPOTrainer.

    Note: chosen/rejected may legitimately be message LISTS (conversational
    DPO), so this converter only rejects an explicit null rather than requiring
    a plain string (unlike the alpaca / sharegpt / vision text converters).
    """
    for field in ("prompt", "chosen", "rejected"):
        if row.get(field) is None:
            raise TypeError(f"dpo.{field} must not be null")
    return {
        "prompt": row["prompt"],
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }


def _convert_kto(row: dict) -> dict:
    """Convert KTO row to {prompt, completion, label} for trl.KTOTrainer."""
    raw_label = row["label"]
    if isinstance(raw_label, str):
        low = raw_label.strip().lower()
        if low in ("true", "1", "yes"):
            label = True
        elif low in ("false", "0", "no"):
            label = False
        else:
            raise ValueError(
                f"KTO label must be true/false, got string: {raw_label!r}"
            )
    elif isinstance(raw_label, bool):
        label = raw_label
    elif isinstance(raw_label, (int, float)):
        # Both the ±1 convention (+1 desirable / -1 undesirable) and the 0/1
        # convention map "positive == desirable". `bool(-1)` is True, which
        # would silently INVERT a -1 "bad" label — flip it to False here.
        label = raw_label > 0
    else:
        label = bool(raw_label)
    return {
        "prompt": row["prompt"],
        "completion": row["completion"],
        "label": label,
    }


def _convert_plaintext(row: dict) -> dict:
    """Convert plaintext row to {text} for continued pre-training.

    Input: {"text": "raw document text..."}
    Output: {"text": "raw document text..."}
    """
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Plaintext row must have a non-empty 'text' field")
    return {"text": text}


def _convert_embedding(row: dict) -> dict:
    """Convert embedding row to {anchor, positive, negative?} for embedding training.

    Input: {"anchor": "query text", "positive": "similar text", "negative": "dissimilar text"}
    Output: {"anchor": ..., "positive": ..., "negative": ...} (negative is optional)
    """
    anchor = row["anchor"]
    positive = row["positive"]
    if not isinstance(anchor, str) or not anchor.strip():
        raise ValueError("Embedding row must have a non-empty 'anchor' field")
    if not isinstance(positive, str) or not positive.strip():
        raise ValueError("Embedding row must have a non-empty 'positive' field")
    result = {"anchor": anchor, "positive": positive}
    negative = row.get("negative")
    if isinstance(negative, str) and negative.strip():
        result["negative"] = negative
    return result


def _convert_audio(row: dict) -> dict:
    """Convert audio format to unified messages + audio path.

    Input: {"audio": "path.wav", "messages": [{"role": "user", "content": ...}, ...]}
    Output: {"messages": [...], "audio": "path.wav"}
    """
    audio = row["audio"]
    if not isinstance(audio, str) or not audio.strip():
        raise ValueError("Audio row must have a non-empty 'audio' field")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) < 1:
        raise ValueError("Audio row must have a 'messages' list with at least one message")
    return {"messages": messages, "audio": audio}


def _convert_vision(row: dict) -> dict:
    """Convert LLaVA / ShareGPT4V vision format to unified messages + image.

    Input: {"image": "path.jpg", "conversations": [{"from": "human", "value": ...}, ...]}
    Output: {"messages": [...], "image": "path.jpg"}
    """
    conversations = row["conversations"]
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}

    messages = []
    for turn in conversations:
        role = role_map.get(turn["from"], turn["from"])
        messages.append(
            {"role": role, "content": _require_str_content(turn["value"], "vision.value")}
        )

    result = {"messages": messages, "image": row["image"]}
    # Preserve optional id field
    if "id" in row:
        result["id"] = row["id"]
    return result


def _convert_tool_calling(row: dict) -> dict:
    """Normalize tool-calling row to unified messages format.

    Input:
        {
            "messages": [{"role": "user", "content": ...}],
            "tools": [{"type": "function", "function": {...}}, ...],
            "tool_calls": [{"function": {"name": ..., "arguments": "json-string"}}],
        }

    Output (unified format — tool schema embedded in system message,
    tool_calls attached to final assistant turn):
        {
            "messages": [
                {"role": "system", "content": "<tool schema description>"},
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "", "tool_calls": [...]},
            ]
        }

    Security: every tool_call's 'arguments' must be JSON-parseable. Tool schemas
    must be a list of dicts. Invalid rows raise ValueError and are mapped to None
    by the outer handler.
    """
    tools = row["tools"]
    tool_calls = row["tool_calls"]

    if not isinstance(tools, list):
        raise ValueError("tool-calling 'tools' must be a list")
    if not isinstance(tool_calls, list):
        raise ValueError("tool-calling 'tool_calls' must be a list")

    for tool in tools:
        if not isinstance(tool, dict):
            raise ValueError("tool-calling tool entries must be dicts")

    normalized_tool_calls = []
    for call in tool_calls:
        if not isinstance(call, dict):
            raise ValueError("tool_calls entries must be dicts")
        func = call.get("function")
        if not isinstance(func, dict):
            raise ValueError("tool_calls entry missing 'function' dict")
        name = func.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_calls 'function.name' must be a non-empty string")
        args = func.get("arguments", "{}")
        if isinstance(args, str):
            try:
                json.loads(args)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"tool_calls 'arguments' must be JSON-parseable: {exc}"
                ) from exc
            args_str = args
        elif isinstance(args, dict):
            args_str = json.dumps(args)
        else:
            raise ValueError("tool_calls 'arguments' must be str or dict")
        normalized_tool_calls.append({
            "function": {"name": name, "arguments": args_str},
        })

    original_messages = row["messages"]
    if not isinstance(original_messages, list) or not original_messages:
        raise ValueError("tool-calling 'messages' must be a non-empty list")

    tool_schema_descriptions = []
    for tool in tools:
        function_def = tool.get("function", {})
        tool_name = function_def.get("name", "unknown")
        description = function_def.get("description", "")
        params = function_def.get("parameters", {})
        tool_schema_descriptions.append(
            f"- {tool_name}: {description}\n  parameters: {json.dumps(params)}"
        )

    system_content = (
        "You have access to the following tools. When a tool call is needed, "
        "respond with a function call in JSON.\n\n"
        + "\n".join(tool_schema_descriptions)
    )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    for msg in original_messages:
        if not isinstance(msg, dict) or "role" not in msg:
            raise ValueError("tool-calling messages must be dicts with 'role'")
        if msg["role"] == "system":
            # Merge user system message into our synthesized system content
            messages[0]["content"] = msg.get("content", "") + "\n\n" + messages[0]["content"]
            continue
        messages.append({"role": msg["role"], "content": msg.get("content", "")})

    if normalized_tool_calls:
        messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": normalized_tool_calls,
        })

    return {"messages": messages}


def is_vision_format(fmt: str) -> bool:
    """Check if a format is a vision/multimodal format."""
    return fmt in ("llava", "sharegpt4v")


def is_audio_format(fmt: str) -> bool:
    """Check if a format is an audio/speech format."""
    return fmt == "audio"


# --- Reverse conversion: messages → target format ---

CONVERTIBLE_FORMATS = ("alpaca", "sharegpt", "chatml")


def messages_to_format(row: dict, target_fmt: str) -> Optional[dict]:
    """Convert unified messages format back to a specific format.

    Input: {"messages": [{"role": ..., "content": ...}, ...]}
    Output: dict in target format (alpaca, sharegpt, chatml)
    """
    try:
        if target_fmt == "chatml":
            return row  # already in chatml/messages format
        elif target_fmt == "alpaca":
            return _to_alpaca(row["messages"])
        elif target_fmt == "sharegpt":
            return _to_sharegpt(row["messages"])
        else:
            raise ValueError(f"Cannot convert to format: {target_fmt}")
    except (KeyError, TypeError, IndexError):
        return None


def _to_alpaca(messages: list[dict]) -> dict:
    """Convert messages to alpaca format."""
    result: dict = {"instruction": "", "input": "", "output": ""}

    for msg in messages:
        if msg["role"] == "system":
            result["system"] = msg["content"]
        elif msg["role"] == "user":
            result["instruction"] = msg["content"]
        elif msg["role"] == "assistant":
            result["output"] = msg["content"]

    return result


def _to_sharegpt(messages: list[dict]) -> dict:
    """Convert messages to sharegpt format."""
    role_map = {"user": "human", "assistant": "gpt", "system": "system"}
    conversations = []
    for msg in messages:
        conversations.append({
            "from": role_map.get(msg["role"], msg["role"]),
            "value": msg["content"],
        })
    return {"conversations": conversations}


# --- v0.42.0 Part A: New format converters ---------------------------------

_MAX_PRM_STEPS = 10_000

# v0.62.0 Part A — RAFT (Retrieval-Augmented Fine-Tuning) caps.
_MAX_RAFT_DISTRACTORS = 64
_MAX_RAFT_FIELD_LEN = 65_536  # 64 KiB per document — generous for legal/RAG corpora.


def _convert_prm(row: dict) -> dict:
    """PRM (Process Reward Model) stepwise-supervised format.

    Schema: {"prompt": str, "completions": [str, ...], "labels": [bool, ...]}
    Each completion is a reasoning step; each label is True if that step is
    correct. Live PPO/RL wiring lands in v0.50 — v0.42.0 stores the row as-is
    after schema validation so downstream consumers can opt in.
    """
    prompt = row["prompt"]
    completions = row["completions"]
    labels = row["labels"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("PRM 'prompt' must be a non-empty string")
    if not isinstance(completions, list) or not isinstance(labels, list):
        raise ValueError("PRM completions/labels must be lists")
    if len(completions) != len(labels):
        raise ValueError("PRM completions and labels must be same length")
    if not completions:
        raise ValueError("PRM row must have at least one completion")
    if len(completions) > _MAX_PRM_STEPS:
        raise ValueError(f"PRM row exceeds {_MAX_PRM_STEPS} steps")
    for index, comp in enumerate(completions):
        if not isinstance(comp, str):
            raise ValueError(f"PRM completions[{index}] must be a string")
    for index, lab in enumerate(labels):
        if not isinstance(lab, bool):
            raise ValueError(f"PRM labels[{index}] must be a bool")
    return {"prompt": prompt, "completions": completions, "labels": labels}


def _convert_pre_tokenized(row: dict) -> dict:
    """Already-tokenized rows — pass through input_ids / labels / attention_mask."""
    if "input_ids" not in row:
        raise ValueError("pre_tokenized row must have 'input_ids'")
    out: dict = {"input_ids": row["input_ids"]}
    if "labels" in row:
        out["labels"] = row["labels"]
    if "attention_mask" in row:
        out["attention_mask"] = row["attention_mask"]
    return out


def _convert_input_output(row: dict) -> dict:
    """Template-free segments+labels format (axolotl `input_output`).

    Schema: {"segments": [{"text": str, "label": bool}, ...]}
    Each segment is rendered verbatim — no chat template applied — and only
    segments with label=True contribute to the loss.
    """
    segments = row["segments"]
    if not isinstance(segments, list) or not segments:
        raise ValueError("input_output row must have non-empty 'segments' list")
    cleaned: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError("input_output segment must be a dict")
        if "text" not in seg or "label" not in seg:
            raise ValueError("input_output segment must have 'text' and 'label'")
        if not isinstance(seg["text"], str):
            raise ValueError("input_output segment.text must be a string")
        if not isinstance(seg["label"], bool):
            raise ValueError("input_output segment.label must be a bool")
        cleaned.append({"text": seg["text"], "label": seg["label"]})
    return {"segments": cleaned}


def _convert_video(row: dict) -> dict:
    """Video format. Schema: {"video": "path/url", "messages": [...]}."""
    if "video" not in row:
        raise ValueError("video row must have 'video' key")
    video = row["video"]
    if not isinstance(video, str) or not video:
        raise ValueError("video row 'video' must be a non-empty string")
    if "\x00" in video:
        raise ValueError("video row 'video' must not contain null bytes")
    if len(video) > 2048:
        raise ValueError("video row 'video' must be <= 2048 chars")
    messages = row.get("messages") or []
    return {"video": video, "messages": messages}


def _convert_multimodal(row: dict) -> dict:
    """Axolotl multimodal content-parts schema.

    Schema: {"messages": [{"role": ..., "content": [{"type": "text"|"image"
    |"audio"|"video", ...}, ...]}, ...]}
    Each message's content is a list of typed parts. Validates the part
    types but stores them verbatim.
    """
    messages = row["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("multimodal row must have non-empty 'messages' list")
    valid_types = {"text", "image", "audio", "video"}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            continue  # back-compat with plain strings
        if not isinstance(content, list):
            raise ValueError(
                "multimodal message.content must be a list of parts or a string"
            )
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("multimodal content part must be a dict")
            ptype = part.get("type")
            if ptype not in valid_types:
                raise ValueError(
                    f"multimodal content part.type must be in {sorted(valid_types)}"
                )
    return {"messages": messages}


# --- v0.62.0 Part A: RAFT (Retrieval-Augmented Fine-Tuning) ----------------


def _check_raft_string(name: str, value: object) -> str:
    """Shared validator for RAFT string fields (query / golden_doc / answer).

    Returns the canonical value. Rejects non-string, empty, null-byte, and
    oversize values (mirrors v0.42.0 `_convert_video` policy). The cap is
    generous (64 KiB) because legal/RAG corpora frequently embed full
    paragraphs verbatim in the golden_doc field.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"RAFT '{name}' must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"RAFT '{name}' must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"RAFT '{name}' must not contain null bytes")
    if len(value) > _MAX_RAFT_FIELD_LEN:
        raise ValueError(
            f"RAFT '{name}' must be <= {_MAX_RAFT_FIELD_LEN} chars"
        )
    return value


def _convert_raft(row: dict) -> dict:
    """RAFT (Retrieval-Augmented Fine-Tuning) format — Stanford 2024.

    Schema: ``{"query": str, "golden_doc": str, "distractor_docs": [str, ...],
    "answer": str}``. The trainer composes the prompt by concatenating the
    query with the golden doc + N distractor docs in randomised order; the
    model learns to attend to the relevant doc while ignoring distractors.

    Distractor list MAY be empty (effectively reduces to closed-book QA on
    the golden doc). Live RAFT training loop ships in v0.62.1; v0.62.0
    locks the schema + recipe surface.
    """
    if "query" not in row:
        raise ValueError("RAFT row must have 'query'")
    if "golden_doc" not in row:
        raise ValueError("RAFT row must have 'golden_doc'")
    if "answer" not in row:
        raise ValueError("RAFT row must have 'answer'")

    query = _check_raft_string("query", row["query"])
    golden_doc = _check_raft_string("golden_doc", row["golden_doc"])
    answer = _check_raft_string("answer", row["answer"])

    raw_distractors = row.get("distractor_docs", [])
    if not isinstance(raw_distractors, list):
        raise ValueError("RAFT 'distractor_docs' must be a list")
    if len(raw_distractors) > _MAX_RAFT_DISTRACTORS:
        raise ValueError(
            f"RAFT 'distractor_docs' must have <= {_MAX_RAFT_DISTRACTORS} entries "
            f"(got {len(raw_distractors)})"
        )
    cleaned_distractors: list[str] = []
    for index, doc in enumerate(raw_distractors):
        cleaned_distractors.append(
            _check_raft_string(f"distractor_docs[{index}]", doc)
        )

    return {
        "query": query,
        "golden_doc": golden_doc,
        "distractor_docs": cleaned_distractors,
        "answer": answer,
    }
