"""Shared constants and helpers of the ds_compare harness.

Nothing here imports ``soup_cli``: ``hf_baseline.py`` and ``make_rows.py`` use this
module and must stay independent of the tool they are compared against.

Contents
--------
* the fixed experimental constants (model revision, seed, LoRA, the three row shapes),
* ``render_alpaca_row``: the chat-template rendering and loss mask Soup applies to an
  alpaca row, reimplemented in a few lines so the Hugging Face script feeds the same
  token stream (see the docstring for the exact correspondence),
* small JSON helpers shared by the runners.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Fixed experimental constants (protocol section 1)
# ---------------------------------------------------------------------------

MODEL_ID = "NousResearch/Meta-Llama-3.1-8B-Instruct"
# Full commit id of the pinned revision (short form d10aef7999a2).
MODEL_REVISION = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"

SEED = 42
LR = 2e-4
WEIGHT_DECAY = 0.0
MAX_GRAD_NORM = 1.0
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# The LoRA target modules Soup resolves for a Llama model.
#   soup.yaml ``training.lora.target_modules`` defaults to "auto"
#   (src/soup_cli/config/schema.py:92-95). For Llama, ``resolve_lora_target_modules``
#   returns None for an architecture PEFT already maps
#   (src/soup_cli/utils/peft_wiring.py:381-439), so PEFT's own default list is used:
#   TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING["llama"] = ["q_proj", "v_proj"]
#   (Soup mirrors the same table at src/soup_cli/utils/model_shape.py:422-444).
# ``orchestrate.py`` compares this list with the ``adapter_config.json`` each streaming
# run writes, and records the comparison in the run JSON.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

ROWS_PER_SHAPE = 64
#: Shape table. ``target_tokens`` is the exact tokenized length of EVERY row of the shape
#: after the chat template (bos, headers, the trailing header of this template, eot).
SHAPES: dict[str, dict[str, Any]] = {
    "A": {
        "rows_file": "ds_compare_rows_a.jsonl",
        "max_length": 256,
        "epochs": 4,
        "steps": 256,
        "target_tokens": 32,
        "repeats": 3,
    },
    "B": {
        "rows_file": "ds_compare_rows_b.jsonl",
        "max_length": 512,
        "epochs": 2,
        "steps": 128,
        "target_tokens": 512,
        "repeats": 2,
    },
    "C": {
        "rows_file": "ds_compare_rows_c.jsonl",
        "max_length": 2048,
        "epochs": 1,
        "steps": 64,
        "target_tokens": 2048,
        "repeats": 2,
    },
}

RUN_SCHEMA = "ds_compare.run.v1"
IGNORE_INDEX = -100


def rows_path(shape: str) -> Path:
    return HERE / SHAPES[shape]["rows_file"]


def load_rows(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Chat-template rendering: what Soup does to an alpaca row
# ---------------------------------------------------------------------------


def alpaca_messages(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """alpaca row -> chat messages, as ``_convert_alpaca`` does it.

    Soup: src/soup_cli/data/formats.py:355-371. ``instruction`` and a non-empty
    ``input`` are joined with a newline and stripped; an optional ``system`` field
    becomes a leading system message.
    """
    instruction = row["instruction"]
    output = row["output"]
    if not isinstance(instruction, str) or not isinstance(output, str):
        raise ValueError("alpaca rows need string instruction and output")
    input_text = row.get("input") or ""
    user_content = f"{instruction}\n{input_text}".strip() if input_text else instruction
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    if row.get("system"):
        messages.insert(0, {"role": "system", "content": row["system"]})
    return messages


def _token_ids(out: Any) -> list[int]:
    """ids from a list or from a mapping (BatchEncoding) with ``input_ids``."""
    values = out.get("input_ids") if hasattr(out, "get") else out
    if values is None:
        raise ValueError("tokenizer output has no input_ids")
    return [int(value) for value in values]


def _template_has_generation_marker(tokenizer: Any) -> bool:
    template = getattr(tokenizer, "chat_template", None) or ""
    return isinstance(template, str) and "generation" in template and "{%" in template


def render_alpaca_row(tokenizer: Any, row: Mapping[str, Any], max_length: int) -> dict[str, list]:
    """Tokenize one alpaca row exactly the way ``soup train`` does (default config).

    Soup path: ``build_format_row`` with ``data.train_on_responses_only: true`` (the
    default, src/soup_cli/config/schema.py:419) -> ``build_assistant_only_labels``
    (src/soup_cli/data/loss_mask.py:273-361):

    * the row goes through ``tokenizer.apply_chat_template(tokenize=True,
      add_generation_prompt=False, add_special_tokens=False)``, so the template is
      the only source of special tokens;
    * labels are -100 everywhere except the assistant turn. A template with a
      ``{% generation %}`` marker gets its own assistant mask; any other template (the
      Llama-3.1 one is such a template) uses the incremental fallback: render the
      conversation up to the previous turn and up to this turn, and train on the
      difference;
    * ids and labels are cut to ``max_length`` and the attention mask is all ones.

    Known differences from Soup: (1) ``data.mask_history``, ``train_on_eot`` and
    ``reasoning_effort`` are off in the benchmark config and are not implemented here;
    (2) when a template declares a generation marker but the mask comes back all zero,
    Soup falls back to the incremental render and this function trusts the marker; the
    Llama-3.1 template has no marker, so the benchmark never reaches that case.
    ``selftest.py`` compares both implementations on every row of the three files when
    the tokenizer is present.
    """
    messages = alpaca_messages(row)

    full_ids: list[int]
    labels: list[int]
    if _template_has_generation_marker(tokenizer):
        out = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        full_ids = _token_ids(out)
        mask = [int(flag) for flag in out.get("assistant_masks", [])]
        if len(mask) != len(full_ids) or not any(mask):
            raise ValueError("template reported no assistant tokens for an alpaca row")
        labels = [tok if flag else IGNORE_INDEX for tok, flag in zip(full_ids, mask)]
    else:
        full_ids = _token_ids(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                add_special_tokens=False,
            )
        )
        labels = [IGNORE_INDEX] * len(full_ids)
        prev_len = 0
        cumulative: list[dict[str, str]] = []
        for message in messages:
            cumulative.append(message)
            if message["role"] != "assistant" and not any(
                item["role"] == "user" for item in cumulative
            ):
                continue
            rendered = _token_ids(
                tokenizer.apply_chat_template(
                    cumulative,
                    tokenize=True,
                    add_generation_prompt=False,
                    add_special_tokens=False,
                )
            )
            new_len = len(rendered)
            if message["role"] == "assistant":
                end = min(new_len, len(full_ids))
                labels[prev_len:end] = full_ids[prev_len:end]
            prev_len = new_len

    input_ids = full_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != IGNORE_INDEX for label in labels[1:]):
        raise ValueError(f"no causal-loss target left after truncation at max_length={max_length}")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> None:
    """Write ``payload`` so that a reader never sees a half-written file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        raise ValueError("median of an empty list")
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def spread(values: list[float]) -> float:
    """(max - min) / median, the protocol's spread."""
    mid = median(values)
    if mid == 0:
        return float("inf")
    return (max(values) - min(values)) / mid
