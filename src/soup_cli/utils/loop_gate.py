"""The loop's side of ``soup ship --gate loop-v2``: the two files it is given, the answer it reads.

``soup ship --gate loop-v2 --corrections FILE --held-out FILE`` (docs/evaluation.md) decides
whether a candidate learned the corrections people made (check 1) and did not lose the edits kept
out of training (check 2). This module builds the two JSONL files from the loop's edits and
reads the evidence ``soup ship --output`` writes back, in the shapes of the loop iteration format
(``verdict.corrections`` / ``verdict.held_out``, docs/loop-formats.md). It does not call ``soup
ship``: ``utils/loop_stages.py`` does, as a subprocess, and no ship code is copied here.

- ``--corrections``: ``{"edit_id", "expected", "original", "paraphrases"}`` per trained edit that
  has paraphrases (``loop_paraphrase``): the corrected text, the answer it corrected, and the
  rewordings of the question both versions are asked;
- ``--held-out``: ``{"edit_id", "prompt", "expected", "original"}`` per edit kept out of training
  in this iteration: both versions are asked the original question.

``soup ship`` refuses a file with a row it cannot read, and then there is no verdict at all, so
a row that would be refused (too long, blank, too many) is left out here and counted.

Standard library only: ``loop_iteration`` validates its evidence fields with this module.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

#: The ``--gate`` value, and the ``verdict.gate`` of the iteration format.
GATE = "loop-v2"
#: What the iteration says when ``soup ship`` found nothing to check and the canary decides.
NO_CORRECTIONS_NOTE = "No corrections to check - the canary's ratings decide"

#: Limits ``soup ship`` applies to the files (``utils/ship_loop_gate.py``).
SHIP_MAX_ROWS = 10_000
SHIP_MAX_TEXT_CHARS = 20_000
SHIP_MAX_ID_CHARS = 256
SHIP_MAX_PARAPHRASES = 20

_MAX_REASON_CHARS = 4096


# ---------------------------------------------------------------------------
# The two files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateFiles:
    """The text of the two files and what was left out of them."""

    corrections: str
    held_out: str
    n_corrections: int
    n_held_out: int
    #: Trained edits that have no paraphrases: left out of the corrections file.
    without_paraphrases: int
    #: Rows (either file) that ``soup ship`` would refuse, so were not written.
    over_limit: int


def _text_ok(value: object, *, allow_empty: bool = False, limit: int = SHIP_MAX_TEXT_CHARS) -> bool:
    if not isinstance(value, str) or len(value) > limit:
        return False
    return allow_empty or bool(value.strip())


def _paraphrases(value: object) -> "list[str]":
    """The usable paraphrases of an edit: non-blank strings within the limit, at most 20."""
    if not isinstance(value, (list, tuple)):
        return []
    usable = [item for item in value if _text_ok(item)]
    return usable[:SHIP_MAX_PARAPHRASES]


def _dump(rows: "list[dict]") -> str:
    # ensure_ascii: a lone surrogate in a logged answer cannot break the UTF-8 write
    return "".join(json.dumps(row) + "\n" for row in rows)


def build_gate_files(
    trained: Sequence[Any],
    kept: Sequence[Any],
    paraphrases: Mapping[str, Sequence[str]],
) -> GateFiles:
    """The ``--corrections`` and ``--held-out`` files for one iteration.

    ``trained`` are the edits trained in this iteration (graduated and new), ``kept`` the edits
    kept out of it, both ``loop_feedback.EditedAnswer``; ``paraphrases`` maps an edit's id to its
    paraphrases. A trained edit without paraphrases is left out of the corrections file and
    counted. Identical held-out corrections (the same prompt and text) count once.
    """
    from soup_cli.utils.loop_holdout import holdout_key, last_user_text

    over = missing = 0
    corrections: "list[dict]" = []
    for edit in trained:
        items = _paraphrases(paraphrases.get(edit.id))
        if not items:
            missing += 1
            continue
        row = {
            "edit_id": edit.id, "expected": edit.edited, "original": edit.original,
            "paraphrases": items,
        }
        if (
            not _text_ok(edit.id, limit=SHIP_MAX_ID_CHARS)
            or not _text_ok(edit.edited)
            or not _text_ok(edit.original, allow_empty=True)
            or len(corrections) >= SHIP_MAX_ROWS
        ):
            over += 1
            continue
        corrections.append(row)
    held_out: "list[dict]" = []
    seen: "set[tuple[str, str]]" = set()
    for edit in kept:
        prompt = last_user_text(edit.messages)
        if (
            prompt is None
            or not _text_ok(prompt)
            or not _text_ok(edit.id, limit=SHIP_MAX_ID_CHARS)
            or not _text_ok(edit.edited)
            or not _text_ok(edit.original, allow_empty=True)
            or len(held_out) >= SHIP_MAX_ROWS
        ):
            over += 1
            continue
        identical = (holdout_key(edit.messages) or "", edit.edited.strip())
        if identical in seen:
            continue
        seen.add(identical)
        held_out.append({
            "edit_id": edit.id, "prompt": prompt, "expected": edit.edited,
            "original": edit.original,
        })
    return GateFiles(
        corrections=_dump(corrections), held_out=_dump(held_out),
        n_corrections=len(corrections), n_held_out=len(held_out),
        without_paraphrases=missing, over_limit=over,
    )


# ---------------------------------------------------------------------------
# What soup ship answered
# ---------------------------------------------------------------------------

_CORRECTIONS_OUTCOMES = ("pass", "fail", "insufficient", "not_applicable")
_HELD_OUT_OUTCOMES = ("pass", "fail", "weak", "not_measured")
_SCORING_METHODS = ("added_recall", "reference_judge")


def _is_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _is_fraction(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _fraction_or_null(value: object) -> bool:
    return value is None or _is_fraction(value)


def _fields(value: object, name: str, required: tuple, optional: tuple = ()) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{name} lacks {', '.join(missing)}")
    unknown = [key for key in value if key not in required and key not in optional]
    if unknown:
        raise ValueError(f"{name} has the unknown field {unknown[0]!r}")
    return value


def _check_scoring(value: object, name: str) -> None:
    scoring = _fields(value, f"{name}.scoring", ("method", "threshold", "template"))
    if scoring["method"] not in _SCORING_METHODS:
        raise ValueError(f"{name}.scoring.method must be one of {', '.join(_SCORING_METHODS)}")
    if not _fraction_or_null(scoring["threshold"]):
        raise ValueError(f"{name}.scoring.threshold must be a number in [0, 1] or null")
    template = scoring["template"]
    if template is not None:
        _fields(template, f"{name}.scoring.template", ("id", "version"))
        if not (_text_ok(template["id"]) and _text_ok(template["version"])):
            raise ValueError(f"{name}.scoring.template must hold an id and a version")


def _check_counts(value: dict, name: str, keys: tuple) -> None:
    for key in keys:
        if not _is_count(value[key]):
            raise ValueError(f"{name}.{key} must be a non-negative integer")


def check_corrections(value: object) -> dict:
    """``value`` as ``verdict.corrections`` of the iteration format, or ``ValueError``.

    Returns a copy. All eight fields are required and no other is allowed.
    """
    name = "verdict.corrections"
    data = _fields(
        value, name,
        ("fixed", "broken", "ties", "n_edits", "n_paraphrases", "p_value", "scoring", "outcome"),
    )
    _check_counts(data, name, ("fixed", "broken", "ties", "n_edits", "n_paraphrases"))
    if not _fraction_or_null(data["p_value"]):
        raise ValueError(f"{name}.p_value must be a number in [0, 1] or null")
    _check_scoring(data["scoring"], name)
    if data["outcome"] not in _CORRECTIONS_OUTCOMES:
        raise ValueError(f"{name}.outcome must be one of {', '.join(_CORRECTIONS_OUTCOMES)}")
    return json.loads(json.dumps(data))


def check_style(value: object) -> dict:
    """``value`` as ``verdict.held_out.style`` (information only), or ``ValueError``."""
    name = "verdict.held_out.style"
    data = _fields(value, name, ("preferred_new", "preferred_old", "n", "p_value"))
    _check_counts(data, name, ("preferred_new", "preferred_old", "n"))
    if not _fraction_or_null(data["p_value"]):
        raise ValueError(f"{name}.p_value must be a number in [0, 1] or null")
    return json.loads(json.dumps(data))


def check_held_out(value: object) -> dict:
    """``value`` as ``verdict.held_out`` of the iteration format, or ``ValueError``.

    Returns a copy. The seven fields are required; ``style`` is optional.
    """
    name = "verdict.held_out"
    data = _fields(
        value, name, ("better", "worse", "ties", "n", "p_value", "scoring", "outcome"), ("style",)
    )
    _check_counts(data, name, ("better", "worse", "ties", "n"))
    if not _fraction_or_null(data["p_value"]):
        raise ValueError(f"{name}.p_value must be a number in [0, 1] or null")
    _check_scoring(data["scoring"], name)
    if data["outcome"] not in _HELD_OUT_OUTCOMES:
        raise ValueError(f"{name}.outcome must be one of {', '.join(_HELD_OUT_OUTCOMES)}")
    if "style" in data:
        check_style(data["style"])
    return json.loads(json.dumps(data))


def clean_corrections(value: object) -> Optional[dict]:
    """``check_corrections`` that gives ``None`` instead of raising."""
    try:
        return check_corrections(value)
    except (ValueError, TypeError, RecursionError):
        return None


def clean_held_out(value: object) -> Optional[dict]:
    """``check_held_out`` that gives ``None`` instead of raising.

    A bad ``style`` alone is dropped: the check beside it stands.
    """
    try:
        return check_held_out(value)
    except (ValueError, TypeError, RecursionError):
        pass
    if isinstance(value, dict) and "style" in value:
        try:
            return check_held_out({key: item for key, item in value.items() if key != "style"})
        except (ValueError, TypeError, RecursionError):
            return None
    return None


def clean_reason(value: object) -> Optional[str]:
    """A reason from ``soup ship`` as one printable line of at most 4096 characters, or ``None``."""
    if not isinstance(value, str):
        return None
    text = "".join(char if char.isprintable() else " " for char in value)
    text = " ".join(text.split())[:_MAX_REASON_CHARS]
    return text or None


@dataclass(frozen=True)
class GateResult:
    """What ``soup ship --gate loop-v2`` wrote besides its decision."""

    corrections: Optional[dict]
    held_out: Optional[dict]
    #: DON'T SHIP because there was nothing to check: the canary's ratings decide the rollout.
    canary_decides: bool
    #: Why it did not ship (``None`` on SHIP), as one line.
    reason: Optional[str]


def read_gate_result(data: object, decision: object) -> Optional[GateResult]:
    """The gate's evidence from a ``soup ship --output`` object, or ``None`` if it ran no gate.

    ``None`` unless the object says ``gate: "loop-v2"``. An evidence object that does not follow
    the iteration format is dropped on its own (the decision stands); ``canary_decides`` counts
    only as a real ``true`` next to a DON'T SHIP.
    """
    if not isinstance(data, Mapping) or data.get("gate") != GATE:
        return None
    return GateResult(
        corrections=clean_corrections(data.get("corrections")),
        held_out=clean_held_out(data.get("held_out")),
        canary_decides=data.get("canary_decides") is True and decision == "DON'T SHIP",
        reason=clean_reason(data.get("reason")),
    )
