"""The longest row a padded text-SFT run trains on, in tokens (the pre-flight's second look).

The VRAM estimate prices rows that FILL ``data.max_length`` (``utils/vram_estimate.py``), which
is what its calibration runs did. A run on short rows needs a fraction of that: TRL's collator
pads a micro-batch to the longest row IN it, so ``batch_size`` rows of at most N tokens never
build more than ``batch_size x N`` positions (``utils/profiler._not_padded_to_the_longest_row``
says which runs pad that way). ``soup train`` therefore asks, once, for a run its pre-flight
would refuse at ``data.max_length``: how long is the longest row the trainer will see?

The answer is measured on the data the trainer will really see, with the code the trainer
runs: ``data.sft_format.build_format_row`` (the chat template, ``data.chat_template``, the
reasoning-effort prefix, ``data.prompt_strategy`` and the truncation at ``data.max_length``)
over the loader's rows, with the run's own tokenizer. No model is loaded. Every row of both
splits is read: a sample is not a maximum, so past ``MAX_ROWS`` rows, or past
``TIME_BUDGET_SECONDS`` of reading, nothing is measured and the run is priced at
``data.max_length``, as before. So is every run this function does not apply to.

Lazy imports: ``transformers`` is imported only when a tokenizer is loaded.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

#: Rows read at most, train and validation together. The trainer's own tokenisation runs at
#: about 650 rows a second (the chat template is rendered up to four times per row), so this
#: is about half a minute for short rows, and it is only paid by a run the pre-flight would
#: otherwise refuse.
MAX_ROWS = 20_000
#: Seconds of reading allowed before the measurement is dropped (rows longer than the
#: ones this was sized on, a slow tokenizer).
TIME_BUDGET_SECONDS = 90.0
#: How many rows are read between looks at the clock.
_CLOCK_EVERY = 64


@dataclass(frozen=True)
class LongestRow:
    """The longest row of the data, as the trainer will tokenise it."""

    #: Tokens in the longest row after the template and the truncation at ``data.max_length``.
    tokens: int
    #: Rows read (train and validation).
    rows: int


@dataclass(frozen=True)
class RealisedLengthUnavailable:
    """Why the longest row was not measured; the run is then priced at ``data.max_length``."""

    reason: str


def unpriceable_reason(cfg: Any, *, remote_code: bool = False) -> Optional[str]:
    """Why ``cfg``'s run is not priced at the longest row of its data; ``None`` when it is.

    A run that pads each micro-batch to its longest row (``_not_padded_to_the_longest_row``:
    text SFT on the transformers backend, with no packing, multipack, streamed data or layer
    streaming) whose rows reach the trainer through the format conversion this module reads
    them through. Not these: RAFT and pre-tokenised data (rows that are not chat messages), a
    model downloaded from another hub after the pre-flight, and a model that ships its own
    code (``remote_code``: the pre-flight does not run it to read a tokenizer).
    """
    from soup_cli.utils.profiler import _not_padded_to_the_longest_row

    reason = _not_padded_to_the_longest_row(cfg)
    if reason is not None:
        return reason
    data, training = getattr(cfg, "data", None), getattr(cfg, "training", None)
    fmt = getattr(data, "format", None)
    if fmt in ("raft", "pre_tokenized"):
        return f"data.format {fmt} does not hand the trainer chat messages to tokenise"
    hub = getattr(training, "hub", "hf") or "hf"
    if hub != "hf":
        return f"training.hub is {hub}: the model is fetched from there after the pre-flight"
    if remote_code:
        return (
            "the model ships its own code, which the pre-flight does not run to read its "
            "tokenizer"
        )
    return None


def _load_tokenizer(cfg: Any, *, trust_remote_code: bool) -> Any:
    """The run's tokenizer, as ``SFTTrainerWrapper._setup_transformers`` loads it."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base, trust_remote_code=bool(trust_remote_code))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def longest_row_tokens(
    cfg: Any,
    dataset: Any,
    *,
    trust_remote_code: bool = False,
    remote_code: bool = False,
    tokenizer_loader: Optional[Callable[..., Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Union[LongestRow, RealisedLengthUnavailable]:
    """The longest row of ``dataset`` (``load_dataset``'s ``{"train": rows, "val": rows}``).

    ``trust_remote_code`` is the run's own ``--trust-remote-code`` flag and nothing more;
    ``remote_code`` is whether the model's config names custom code (the answer is then
    unavailable). Never raises for a cause of the data or the tokenizer: it returns
    :class:`RealisedLengthUnavailable` naming the cause, so the caller prices ``data.max_length``.
    """
    reason = unpriceable_reason(cfg, remote_code=remote_code)
    if reason is not None:
        return RealisedLengthUnavailable(reason)
    splits = [rows for rows in (dataset.get("train"), dataset.get("val")) if rows]
    total = sum(len(rows) for rows in splits)
    if total == 0:
        return RealisedLengthUnavailable("the data has no rows")
    if total > MAX_ROWS:
        return RealisedLengthUnavailable(
            f"the data has {total:,} rows, more than the {MAX_ROWS:,} the pre-flight reads: a "
            "sample is not a maximum"
        )
    deadline = clock() + TIME_BUDGET_SECONDS
    try:
        from soup_cli.data.sft_format import build_format_row

        tokenizer = (tokenizer_loader or _load_tokenizer)(cfg, trust_remote_code=trust_remote_code)
        format_row = build_format_row(
            tokenizer=tokenizer, data_cfg=cfg.data, console=None, training_cfg=cfg.training
        )
        longest = 0
        for index, row in enumerate(itertools.chain.from_iterable(splits)):
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                return RealisedLengthUnavailable(
                    f"row {index + 1} has no list of chat messages to tokenise"
                )
            longest = max(longest, len(format_row(row)["input_ids"]))
            if index % _CLOCK_EVERY == 0 and clock() > deadline:
                return RealisedLengthUnavailable(
                    f"reading the rows took more than {TIME_BUDGET_SECONDS:.0f} seconds "
                    f"(stopped at row {index + 1:,} of {total:,})"
                )
    except Exception as exc:  # noqa: BLE001 - the pre-flight must not fail on a cause it can name
        return RealisedLengthUnavailable(
            f"the rows could not be tokenised ({type(exc).__name__}); the trainer reports why"
        )
    return LongestRow(tokens=longest, rows=total)


__all__ = [
    "MAX_ROWS",
    "TIME_BUDGET_SECONDS",
    "LongestRow",
    "RealisedLengthUnavailable",
    "longest_row_tokens",
    "unpriceable_reason",
]
