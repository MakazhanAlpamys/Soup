"""Pure helpers behind ``soup rewind`` — spike detection and row ranking.

No heavy imports at module scope: ``soup rewind`` is a light command, and the
only thing here that needs the training stack (rebuilding the dataset for row
previews) imports it inside the function.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from statistics import median
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soup_cli.monitoring.rewind_log import RewindRun

# A ratio spike needs a baseline that is actually a baseline. Two points is a
# line through any two losses; three is the smallest window whose median is not
# simply one of the two values it sits between.
MIN_BASELINE = 3


@dataclass(frozen=True)
class Spike:
    step: int
    loss: float
    baseline: float
    ratio: float
    severity: str


@dataclass(frozen=True)
class RankedRow:
    row: int
    loss: float
    tokens: int
    share: float


def find_spikes(run: RewindRun, *, window: int = 20, factor: float = 2.0) -> list[Spike]:
    """Steps whose loss went non-finite, or past ``factor`` x the recent median.

    Walks ``run.steps()`` in order and carries a rolling history of the last
    ``window`` MEASURED, finite step losses.

    A step whose ``step_loss`` is ``None`` is unmeasured — no batches, or no
    supervised tokens in the ones it has. It is neither reported as a spike nor
    allowed into the baseline: counting it as ``0.0`` would drag the median
    down and manufacture spikes out of the ordinary steps after it.

    A non-finite loss fires regardless of how much history there is (there is
    no recovering from a NaN), with ``ratio`` at infinity; a ratio spike needs
    at least :data:`MIN_BASELINE` finite values behind it. The comparison is
    strict: a step exactly at ``factor`` x baseline is not a spike.

    Returned worst-first: every critical before every warning, then by ratio
    descending.
    """
    history: list[float] = []
    spikes: list[Spike] = []

    for step in run.steps():
        loss = run.step_loss(step)
        if loss is None:
            continue

        if not math.isfinite(loss):
            recent = history[-window:]
            spikes.append(
                Spike(
                    step=step,
                    loss=loss,
                    baseline=median(recent) if recent else math.nan,
                    ratio=math.inf,
                    severity="critical",
                )
            )
            # Deliberately NOT appended to history: a NaN median would poison
            # every later comparison.
            continue

        recent = history[-window:]
        if len(recent) >= MIN_BASELINE:
            baseline = median(recent)
            if loss > factor * baseline:
                spikes.append(
                    Spike(
                        step=step,
                        loss=loss,
                        baseline=baseline,
                        ratio=(loss / baseline) if baseline else math.inf,
                        severity="warning",
                    )
                )
        history.append(loss)

    spikes.sort(key=lambda s: (0 if s.severity == "critical" else 1, -s.ratio))
    return spikes


def rank_rows(run: RewindRun, step: int) -> list[RankedRow]:
    """Every row of every micro-batch of ``step``, by share of the step's loss.

    ``share`` is ``loss * tokens`` over the step's total, because the row that
    moved the optimizer is the one that carried the loss, not the one with the
    largest per-token mean. A NaN loss contributes 0 to the total (so the
    shares of the finite rows still sum to 1) but is reported as NaN rather
    than smoothed away. A step with no supervised tokens gives every row 0.0.
    """
    entries: list[tuple[int, float, int]] = []
    for batch in run.batches_for(step):
        for row, loss, tokens in zip(batch.rows, batch.row_loss, batch.row_tokens):
            entries.append((int(row), float(loss), int(tokens)))

    weights = [0.0 if math.isnan(loss) else loss * tokens for _, loss, tokens in entries]
    total = sum(weights)

    ranked = [
        RankedRow(
            row=row,
            loss=loss,
            tokens=tokens,
            share=(weight / total) if total else 0.0,
        )
        for (row, loss, tokens), weight in zip(entries, weights)
    ]
    ranked.sort(key=lambda r: (-r.share, r.row))
    return ranked


def _preview_text(row: Any, width: int) -> str:
    """One dataset row as a single collapsed line of at most ``width`` chars."""
    text: Any = None
    if isinstance(row, Mapping):
        messages = row.get("messages")
        if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
            if messages and isinstance(messages[-1], Mapping):
                text = messages[-1].get("content")
        if text is None:
            text = row.get("text") or row.get("output")
    if text is None:
        text = json.dumps(row, ensure_ascii=False, default=str)

    collapsed = " ".join(str(text).split())
    if len(collapsed) > width:
        return collapsed[:width] + "…"
    return collapsed


def load_row_previews(
    config: dict,
    row_ids: Sequence[int],
    *,
    expected_fingerprint: str,
    width: int = 80,
) -> dict[int, str] | None:
    """Rebuild this run's training rows and preview the ones in ``row_ids``.

    Returns ``None`` — and previews nothing — when the rebuilt list does not
    fingerprint to ``expected_fingerprint``. The row indices in the log are
    positions in the list the trainer saw; against a dataset that has since
    been edited they point at whatever now sits at that position, which is a
    confident answer about the wrong row.

    Heavy: this is the one function here that loads the data stack, so the
    imports stay inside it.
    """
    from soup_cli.config.schema import SoupConfig
    from soup_cli.data.loader import load_dataset
    from soup_cli.monitoring.rewind_log import dataset_fingerprint

    cfg = SoupConfig.model_validate(config)
    # Exactly what `soup train` calls for an SFT run: preserve_source_columns
    # is GRPO-only, and the rewind log is written for SFT.
    rows = load_dataset(cfg.data)["train"]

    if dataset_fingerprint(rows) != expected_fingerprint:
        return None

    previews: dict[int, str] = {}
    for raw_id in row_ids:
        row_id = int(raw_id)
        if 0 <= row_id < len(rows):
            previews[row_id] = _preview_text(rows[row_id], width)
        else:
            previews[row_id] = "<row not in dataset>"
    return previews
