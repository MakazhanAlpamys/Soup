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
"""

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
    "plaintext": {"text"},
}


def detect_format(data: list[dict]) -> str:
    """Auto-detect dataset format from first few rows."""
    if not data:
        raise ValueError("Empty dataset - cannot detect format")

    sample = data[0]
    keys = set(sample.keys())

    # Check more specific formats first (llava/sharegpt4v before sharegpt)
    # plaintext ("text" key only) checked last to avoid false matches
    check_order = [
        "alpaca", "llava", "sharegpt4v", "kto", "dpo", "embedding",
        "sharegpt", "chatml", "plaintext",
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
        "plaintext", "embedding",
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
        else:
            return _convert_vision(row)
    except (KeyError, TypeError, IndexError, ValueError):
        return None


def _convert_alpaca(row: dict) -> dict:
    instruction = row["instruction"]
    input_text = row.get("input", "")
    output = row["output"]

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
        messages.append({"role": role, "content": turn["value"]})

    return {"messages": messages}


def _convert_chatml(row: dict) -> dict:
    # Already in the right format
    return {"messages": row["messages"]}


def _convert_dpo(row: dict) -> dict:
    """Convert DPO preference row to {prompt, chosen, rejected} for trl.DPOTrainer."""
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
        messages.append({"role": role, "content": turn["value"]})

    result = {"messages": messages, "image": row["image"]}
    # Preserve optional id field
    if "id" in row:
        result["id"] = row["id"]
    return result


def is_vision_format(fmt: str) -> bool:
    """Check if a format is a vision/multimodal format."""
    return fmt in ("llava", "sharegpt4v")


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
