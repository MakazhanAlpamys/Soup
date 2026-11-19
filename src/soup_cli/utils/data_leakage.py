"""Read-only exact input-text overlap between training and evaluation rows."""

from collections.abc import Mapping, Sequence
from typing import Any

from soup_cli.utils.row_text import extract_row_text

_ASSISTANT = frozenset({"assistant", "gpt", "chatgpt", "bot", "model"})


def _without_response(messages: object) -> list:
    if not isinstance(messages, list) or any(not isinstance(x, Mapping) for x in messages):
        raise ValueError("conversation must be a list of message objects")
    if messages and (messages[-1].get("role") or messages[-1].get("from")) in _ASSISTANT:
        return messages[:-1]
    return messages


def _input_text(row: Mapping[str, Any]) -> str:
    """Reuse the shared text extractor on input fields, excluding the final answer."""
    if not isinstance(row, Mapping):
        raise TypeError("record must be a JSON object")
    if "prompt" in row:
        fields = {"prompt": row["prompt"]}
        if "system" in row:
            fields["system"] = row["system"]
    elif "instruction" in row:
        fields = {key: row[key] for key in ("instruction", "input", "system") if key in row}
    elif "messages" in row or "conversations" in row:
        key = "messages" if "messages" in row else "conversations"
        fields = {key: _without_response(row[key])}
    elif "chosen" in row and "rejected" in row:
        chosen = _without_response(row["chosen"])
        rejected = _without_response(row["rejected"])
        if chosen != rejected:
            raise ValueError("chosen/rejected conversations do not have the same prompt")
        fields = {"messages": chosen}
    else:
        fields = {key: row[key] for key in ("query", "text", "content") if key in row}
    text = extract_row_text(fields).strip()
    if not text:
        raise ValueError("record has no usable input text")
    return text


def check_leakage(
    train: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    *,
    samples: int = 5,
) -> dict[str, Any]:
    """Count eval records matching any train input, with bounded example pairs.

    Record numbers are one-based and exclude blank JSONL lines. Each eval record
    counts once even if several train records match. Only outer whitespace is
    stripped; case, internal whitespace and previous conversation turns remain.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or not 0 <= samples <= 20:
        raise ValueError("samples must be an integer in [0, 20]")
    if not train or not evaluation:
        raise ValueError("train and eval must both contain records")
    index: dict[str, tuple[int, int]] = {}
    for number, row in enumerate(train, 1):
        try:
            text = _input_text(row)
        except (TypeError, ValueError) as error:
            raise ValueError(f"train record {number}: {error}") from error
        first, count = index.get(text, (number, 0))
        index[text] = (first, count + 1)
    matched = 0
    examples = []
    for number, row in enumerate(evaluation, 1):
        try:
            text = _input_text(row)
        except (TypeError, ValueError) as error:
            raise ValueError(f"eval record {number}: {error}") from error
        if text in index:
            matched += 1
            if len(examples) < samples:
                first, count = index[text]
                examples.append(
                    {
                        "train_row": first,
                        "eval_row": number,
                        "train_matches": count,
                        "prompt_preview": text[:160],
                    }
                )
    return {
        "schema_version": 1,
        "matching": "exact extracted input text; outer whitespace stripped",
        "row_numbers": "one-based records, excluding blank lines",
        "train_rows": len(train),
        "eval_rows": len(evaluation),
        "overlap_eval_rows": matched,
        "overlap_percent": 100.0 * matched / len(evaluation),
        "samples": examples,
    }
