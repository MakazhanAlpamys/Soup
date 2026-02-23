"""Dataset format detection and conversion.

Supported formats:
- alpaca: {"instruction": ..., "input": ..., "output": ...}
- sharegpt: {"conversations": [{"from": "human", "value": ...}, ...]}
- chatml: {"messages": [{"role": "user", "content": ...}, ...]}
- dpo: {"prompt": ..., "chosen": ..., "rejected": ...}
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
}


def detect_format(data: list[dict]) -> str:
    """Auto-detect dataset format from first few rows."""
    if not data:
        raise ValueError("Empty dataset — cannot detect format")

    sample = data[0]
    keys = set(sample.keys())

    for fmt, required_keys in FORMAT_SIGNATURES.items():
        if required_keys.issubset(keys):
            return fmt

    raise ValueError(
        f"Cannot detect format. Keys found: {keys}. "
        f"Expected one of: alpaca (instruction, output), "
        f"sharegpt (conversations), chatml (messages), "
        f"dpo (prompt, chosen, rejected)"
    )


def format_to_messages(row: dict, fmt: str) -> Optional[dict]:
    """Convert any format to unified messages format for training.

    Returns: {"messages": [{"role": ..., "content": ...}, ...]}
    """
    try:
        if fmt == "chatml":
            return _convert_chatml(row)
        elif fmt == "alpaca":
            return _convert_alpaca(row)
        elif fmt == "sharegpt":
            return _convert_sharegpt(row)
        elif fmt == "dpo":
            return _convert_dpo(row)
        else:
            raise ValueError(f"Unknown format: {fmt}")
    except (KeyError, TypeError, IndexError):
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
