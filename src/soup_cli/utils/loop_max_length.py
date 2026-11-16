"""``data.max_length`` of the loop's train recipe, from the data the loop just built.

``soup train`` prices a run before it starts (``utils/hardware_fit.py``): activations and the
logits scale with ``data.max_length`` x batch, and the estimate assumes every row of a
micro-batch FILLS that length. The loop's generated recipe left ``max_length`` at the schema
default (2048), so a 0.6B model trained on 26 rows of about 25 tokens was priced at 21.7 GB
and refused on an 8 GB card, where the same run peaks at 1.2 GB. Every iteration of the loop
failed there and no loop flag got under it (review 5, F1).

The loop knows its rows, so it writes the length they need:

- ``longest`` = the longest row of the dataset it just built, in tokens of the BASE model's
  tokenizer (the chat template applied for sft, the prompt and the answer for dpo / kto);
- ``max_length`` = ``longest`` plus ``HEADROOM_TOKENS``, rounded up to a multiple of
  ``ROUND_TO``, never below ``MIN_MAX_LENGTH`` (the smallest length the estimate prices, and
  the schema's own floor) and never above the configured ceiling (the schema default): a row
  longer than the ceiling is cut by the trainer, as it was before this module;
- the tokenizer is NOT loaded in the watch daemon: transformers imports torch, and a process
  that lives for days must not carry it (nor a CUDA context). A short-lived child does it
  (``CHILD_SCRIPT``, as ``loop_quantization``'s probe does), prints one JSON line and exits;
- if the length cannot be measured (no tokenizer, no rows, a child that died) the schema
  default stays, and the note says why. The hardware-fit gate is never bypassed and
  ``--allow-oom-attempt`` is never passed: a smaller ``max_length`` is the honest way under
  the gate, an opt-out is the operator's to choose.

The note (``MaxLength.note``) goes into the iteration record's notes, so the value that was
used and the reason are on the record.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

#: The smallest ``data.max_length`` the memory estimate prices (``hardware_fit._MIN_SEQ_LEN``)
#: and the schema's lower bound.
MIN_MAX_LENGTH = 64
#: ``max_length`` is a multiple of this.
ROUND_TO = 64
#: Tokens kept free above the longest row: what a trainer adds that a plain render does not
#: (an end-of-sequence token, a generation prompt, role tags of a template).
HEADROOM_TOKENS = 16

#: Seconds the measuring child may take (importing transformers, loading the tokenizer).
CHILD_TIMEOUT_S = 180
#: A dataset the loop built is far smaller; a bigger file is not read by the child.
MAX_DATASET_BYTES = 256 * 1024 * 1024
#: How many rows the child measures at most.
MAX_ROWS = 200_000

_PREFIX = "max_length"

#: What the child runs. It drops the working directory from ``sys.path`` first, so a
#: ``transformers.py`` next to the loop cannot be imported in its place.
CHILD_SCRIPT = (
    "import sys\n"
    "sys.path[:] = [entry for entry in sys.path if entry not in ('', '.')]\n"
    "from soup_cli.utils.loop_max_length import child_main\n"
    "child_main(sys.argv[1:])\n"
)


class MeasureFailedError(Exception):
    """The longest row could not be measured; ``str()`` says why (one short line)."""


@dataclass(frozen=True)
class MaxLength:
    """The decision: ``value`` for ``data.max_length`` (None = leave the default) and its note."""

    value: Optional[int]
    note: str


def configured_ceiling() -> int:
    """The configured ``data.max_length``: the schema default, the most the loop writes."""
    from soup_cli.config.schema import DataConfig

    return int(DataConfig.model_fields["max_length"].default)


def round_up(tokens: int, *, ceiling: int) -> int:
    """``tokens`` + ``HEADROOM_TOKENS`` rounded up to a multiple of ``ROUND_TO``, clamped.

    At least ``MIN_MAX_LENGTH``, at most ``ceiling``.
    """
    wanted = tokens + HEADROOM_TOKENS
    rounded = -(-wanted // ROUND_TO) * ROUND_TO
    return max(MIN_MAX_LENGTH, min(rounded, ceiling))


# ---------------------------------------------------------------------------
# The measurement (runs in the child, or in a test with a stand-in tokenizer)
# ---------------------------------------------------------------------------


def _text(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _chat_texts(row: Mapping[str, Any], tokenizer: Any) -> "list[str]":
    """The one text of an sft row: its messages through the chat template.

    The tokenizer's own template when it has one; else ``role: content`` lines (a base without
    a template is trained with the plain join by the trainer too, near enough for a length).
    """
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return []
    clean = [
        {"role": str(item.get("role", "user")), "content": _text(item.get("content")) or ""}
        for item in messages
        if isinstance(item, Mapping)
    ]
    if not clean:
        return []
    render = getattr(tokenizer, "apply_chat_template", None)
    if render is not None:
        try:
            rendered = render(clean, tokenize=False)
        except Exception:  # noqa: BLE001 - no template, or one that refuses these roles
            rendered = None
        if isinstance(rendered, str) and rendered:
            return [rendered]
    return ["\n".join(f"{item['role']}: {item['content']}" for item in clean)]


def row_texts(row: object, method: str, tokenizer: Any = None) -> "list[str]":
    """The texts whose longest token count is the row's length; empty for a row to ignore.

    sft: the chat-templated messages. dpo: prompt + chosen and prompt + rejected. kto: prompt +
    completion. A row that is not a mapping, or lacks the method's fields, is ignored.
    """
    if not isinstance(row, Mapping):
        return []
    if method == "sft":
        return _chat_texts(row, tokenizer)
    prompt = _text(row.get("prompt")) or ""
    if method == "dpo":
        answers = [_text(row.get("chosen")), _text(row.get("rejected"))]
    elif method == "kto":
        answers = [_text(row.get("completion"))]
    else:
        return []
    return [f"{prompt}\n{answer}" for answer in answers if answer]


def _token_count(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"]
    return len(ids)


def longest_row_tokens(rows: Iterable[object], tokenizer: Any, method: str) -> "tuple[int, int]":
    """``(longest, rows measured)`` of ``rows`` under ``tokenizer``; ``(0, 0)`` when none.

    ``tokenizer`` is called as ``tokenizer(text, add_special_tokens=False)["input_ids"]``;
    sft rows also use its ``apply_chat_template(messages, tokenize=False)`` when it has one.
    """
    longest = measured = 0
    for row in rows:
        texts = row_texts(row, method, tokenizer)
        if not texts:
            continue
        measured += 1
        longest = max(longest, *(_token_count(tokenizer, text) for text in texts))
    return longest, measured


def _read_rows(path: str) -> "list[object]":
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise MeasureFailedError(f"the dataset cannot be read ({type(exc).__name__})") from exc
    if size > MAX_DATASET_BYTES:
        raise MeasureFailedError("the dataset is larger than the loop measures")
    rows: "list[object]" = []
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue  # a torn line: the trainer's loader reports it, not this probe
            if len(rows) >= MAX_ROWS:
                break
    return rows


def _load_tokenizer(base: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base)


def child_main(
    argv: "list[str]",
    *,
    load_tokenizer: Callable[[str], Any] = _load_tokenizer,
    out: Any = None,
) -> None:
    """The measuring child: ``argv`` = ``[base, dataset path, method]``; one JSON line on ``out``.

    ``{"longest": N, "rows": M}``, or ``{"error": "<short reason>"}``. Always exits normally: the
    parent reads the line, and treats a missing one as a failure.
    """
    out = sys.stdout if out is None else out

    def say(payload: "dict[str, object]") -> None:
        out.write(json.dumps(payload) + "\n")
        out.flush()

    try:
        base, path, method = argv
        rows = _read_rows(path)
        tokenizer = load_tokenizer(base)
        longest, measured = longest_row_tokens(rows, tokenizer, method)
    except MeasureFailedError as exc:
        say({"error": str(exc)})
        return
    except Exception as exc:  # noqa: BLE001 - the parent only needs the kind of failure
        say({"error": f"{type(exc).__name__} while loading the tokenizer or tokenising"})
        return
    if measured == 0:
        say({"error": "the dataset has no row to measure"})
        return
    say({"longest": longest, "rows": measured})


# ---------------------------------------------------------------------------
# The parent: ask the child, decide, say why
# ---------------------------------------------------------------------------


def _measure_in_child(base: str, data_path: str, method: str) -> "tuple[int, int]":
    """``(longest, rows)`` from a short-lived child; ``MeasureFailedError`` when it cannot tell."""
    import subprocess  # noqa: S404 - argv list, no shell

    from soup_cli.utils.loop_spawn import child_popen_kwargs, describe_exit

    argv = [sys.executable, "-c", CHILD_SCRIPT, base, data_path, method]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, capture_output=True, text=True, timeout=CHILD_TIMEOUT_S,
            stdin=subprocess.DEVNULL, check=False, **child_popen_kwargs(),
        )
    except subprocess.TimeoutExpired as exc:
        raise MeasureFailedError(f"the tokenizer child took over {CHILD_TIMEOUT_S} s") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        kind = type(exc).__name__
        raise MeasureFailedError(f"the tokenizer child could not start ({kind})") from exc
    if result.returncode != 0:
        ended = describe_exit(result.returncode)
        raise MeasureFailedError(f"the tokenizer child ended with {ended}")
    lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
    try:
        answer = json.loads(lines[-1]) if lines else None
    except (ValueError, RecursionError):
        answer = None
    if not isinstance(answer, dict):
        raise MeasureFailedError("the tokenizer child printed no answer")
    reason = answer.get("error")
    if isinstance(reason, str) and reason:
        raise MeasureFailedError(reason[:200])
    longest, rows = answer.get("longest"), answer.get("rows")
    for value in (longest, rows):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MeasureFailedError("the tokenizer child printed a bad answer")
    return longest, rows


def decide_max_length(
    base: str,
    data_path: str,
    method: str,
    *,
    measure: Optional[Callable[[str, str, str], "tuple[int, int]"]] = None,
    ceiling: Optional[int] = None,
) -> MaxLength:
    """The ``data.max_length`` to write for ``data_path`` (rows of ``method``), and why.

    ``measure(base, data_path, method) -> (longest, rows)`` replaces the child (tests); it
    raises ``MeasureFailedError`` when it cannot tell.
    """
    cap = configured_ceiling() if ceiling is None else ceiling
    measure = _measure_in_child if measure is None else measure
    try:
        longest, rows = measure(base, data_path, method)
    except MeasureFailedError as exc:
        return MaxLength(
            None,
            f"{_PREFIX}: {cap} kept, the longest row could not be measured ({exc})",
        )
    value = round_up(longest, ceiling=cap)
    if longest + HEADROOM_TOKENS > cap:
        why = f"longest of {rows} rows is {longest} tokens, above the ceiling {cap}"
    else:
        why = (
            f"longest of {rows} rows is {longest} tokens, +{HEADROOM_TOKENS}, rounded up to a "
            f"multiple of {ROUND_TO}, ceiling {cap}"
        )
    return MaxLength(value, f"{_PREFIX}: {value} ({why})")
