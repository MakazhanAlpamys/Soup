"""``soup data doctor`` — chat-template compatibility report + loss-mask X-ray
(v0.71.27).

Kills the top *silent* fine-tune failures before a single training step:
missing chat_template, a template that can't render the data, no
``{% generation %}`` markers (weaker assistant-only masking), a missing
EOS/EOT token after the trained span (the #1 "model never stops
generating" bug), duplicated BOS tokens, unsupported system role
(Mistral-style templates), unknown message roles, and truncation risk.

Design mirrors ``utils/diagnose/report.py``'s OK / MINOR / MAJOR taxonomy
(worst-verdict-wins aggregation) without literally reusing its
``FailureScore`` class — that dataclass validates ``mode`` against a
CLOSED tuple of diagnose's own failure modes, so a parallel
``DoctorCheck`` / ``DoctorReport`` pair lives here instead.

``--show-mask`` renders sample rows through the REAL collator paths
(``data.loss_mask.build_assistant_only_labels`` / per-message-train-field /
``utils.raft`` span-mask) so the trained/masked colouring in the terminal
is exactly what the trainer would produce — not a re-implementation.

No top-level torch / transformers import — tokenizer loading is lazy so
`soup data doctor --help` stays fast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from soup_cli.data.loss_mask import IGNORE_INDEX

VERDICTS: Tuple[str, ...] = ("OK", "MINOR", "MAJOR")

CHECKS: Tuple[str, ...] = (
    "chat_template",
    "template_render",
    "generation_markers",
    "eos_in_labels",
    "bos_duplication",
    "system_role",
    "unknown_roles",
    "truncation_risk",
)

_KNOWN_ROLES = frozenset({"system", "user", "assistant", "tool", "function"})
_DEFAULT_SAMPLE_SIZE = 200
_MAX_SAMPLE_SIZE = 20_000
_MAX_PREVIEW_ROWS = 50
# `--show-mask N` scans raw rows in order until it finds N renderable ones —
# on a large dataset where every row fails to convert/render (wrong
# --format, or a template that rejects every row), an unbounded scan turns
# a "preview a few rows" op into a full-dataset pass. Bound it the same way
# `run_doctor` bounds its own sampling.
_MAX_MASK_SCAN_ROWS = 2_000
_MASK_STRATEGIES = frozenset({"assistant_only", "per_message_train", "raft", "legacy_text"})


# ---------------------------------------------------------------------------
# Dataclasses + taxonomy (mirrors utils/diagnose/report.py's pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoctorCheck:
    """One check's verdict + human-readable message/evidence."""

    name: str
    verdict: str
    message: str
    evidence: str = ""

    def __post_init__(self) -> None:
        if self.name not in CHECKS:
            raise ValueError(f"unknown doctor check {self.name!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        for attr in ("message", "evidence"):
            value = getattr(self, attr)
            if not isinstance(value, str):
                raise TypeError(f"{attr} must be str")
            if "\x00" in value:
                raise ValueError(f"{attr} must not contain null bytes")
            if len(value) > 2048:
                raise ValueError(f"{attr} too long (max 2048 chars)")


def overall_verdict(checks: Sequence[DoctorCheck]) -> str:
    """Worst-case verdict across ``checks``; empty -> OK."""
    rank = {"OK": 0, "MINOR": 1, "MAJOR": 2}
    worst = "OK"
    for check in checks:
        if not isinstance(check, DoctorCheck):
            raise TypeError("checks must contain DoctorCheck instances")
        if rank[check.verdict] > rank[worst]:
            worst = check.verdict
    return worst


@dataclass(frozen=True)
class DoctorReport:
    """Aggregated report returned by :func:`run_doctor`."""

    checks: Tuple[DoctorCheck, ...]
    overall: str
    rows_scanned: int
    total_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.checks, tuple):
            object.__setattr__(self, "checks", tuple(self.checks))
        for check in self.checks:
            if not isinstance(check, DoctorCheck):
                raise TypeError("checks must contain DoctorCheck instances")
        if self.overall not in VERDICTS:
            raise ValueError(f"overall must be one of {VERDICTS}")
        for attr in ("rows_scanned", "total_rows"):
            value = getattr(self, attr)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{attr} must be a non-negative int")
        if self.rows_scanned > self.total_rows:
            raise ValueError("rows_scanned cannot exceed total_rows")

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "rows_scanned": self.rows_scanned,
            "total_rows": self.total_rows,
            "checks": [
                {
                    "name": c.name,
                    "verdict": c.verdict,
                    "message": c.message,
                    "evidence": c.evidence,
                }
                for c in self.checks
            ],
        }


def compose_doctor_report(
    checks: Sequence[DoctorCheck], *, rows_scanned: int, total_rows: int
) -> DoctorReport:
    return DoctorReport(
        checks=tuple(checks),
        overall=overall_verdict(checks),
        rows_scanned=rows_scanned,
        total_rows=total_rows,
    )


# ---------------------------------------------------------------------------
# Sampling + tokenizer resolution
# ---------------------------------------------------------------------------


def sample_indices(total: int, n: int) -> List[int]:
    """Evenly-spaced sample of up to ``n`` indices across ``[0, total)``.

    Even spacing (rather than ``data[:n]``) catches issues confined to the
    tail of a large dataset instead of only ever inspecting the head.
    """
    if total <= 0 or n <= 0:
        return []
    n = min(n, total)
    if n == total:
        return list(range(total))
    step = total / n
    return sorted({int(i * step) for i in range(n)})


def resolve_tokenizer(tokenizer: Any, *, trust_remote_code: bool = False) -> Any:
    """Return a tokenizer object from a name (lazy AutoTokenizer) or object.

    Mirrors ``utils/prune_prompt.py::_resolve_tokenizer`` — a duck-typed
    pre-built tokenizer is returned as-is (the test/injection seam); a
    string is treated as an HF model id / local path and lazily loaded so
    importing this module never pulls in transformers.
    """
    if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
        return tokenizer
    if not isinstance(tokenizer, str):
        raise TypeError(
            "tokenizer must be a model id / path string or a tokenizer object, "
            f"got {type(tokenizer).__name__}"
        )
    if not tokenizer:
        raise ValueError("tokenizer name must be non-empty")
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except ImportError as exc:
        raise ValueError(
            "soup data doctor needs transformers to load a tokenizer — "
            "install with: pip install \"soup-cli[train]\""
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=trust_remote_code)
    except Exception as exc:  # noqa: BLE001 — surface a friendly message
        raise ValueError(
            f"could not load tokenizer {tokenizer!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _eos_token_ids(tokenizer: Any) -> set:
    """Normalise ``tokenizer.eos_token_id`` (int / list[int] / None) to a set."""
    candidate = getattr(tokenizer, "eos_token_id", None)
    ids: set = set()
    if isinstance(candidate, bool):
        return ids
    if isinstance(candidate, int):
        ids.add(candidate)
    elif isinstance(candidate, (list, tuple)):
        for entry in candidate:
            if isinstance(entry, int) and not isinstance(entry, bool):
                ids.add(entry)
    return ids


def _trained_spans(labels: Sequence[int]) -> List[Tuple[int, int]]:
    """Return every ``[start, end]`` (inclusive) contiguous trained run in
    ``labels`` — one entry per assistant turn, in order.

    Real chat templates commonly emit a formatting token (e.g. a trailing
    ``\\n``) immediately after a turn-closing EOS/EOT tag, and for the LAST
    assistant turn specifically that trailing token has no later message to
    fold into a masked prefix, so it stays INSIDE the trained span too — the
    literal last trained position is not reliably the EOS token itself.
    Callers must search each whole span, not just its final position.
    """
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, label in enumerate(labels):
        if label != IGNORE_INDEX:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))
    return spans


def _build_row_labels(
    tokenizer: Any,
    messages: Sequence[Mapping],
    *,
    max_length: int,
    train_on_responses_only: bool,
    train_on_messages_with_train_field: bool,
    include_eot: bool,
) -> dict:
    """Build ``{input_ids, labels}`` via the SAME masking-strategy dispatch
    ``data.sft_format.build_format_row`` uses at train time (per-message
    train field / assistant-only / legacy full-sequence), so every check in
    this module and ``--show-mask`` agree on what the trainer would
    actually train on for a given ``soup.yaml``. Raises on a row the
    template can't render — callers decide whether to skip or propagate.
    """
    from soup_cli.data.loss_mask import build_assistant_only_labels, build_per_message_train_labels

    if train_on_messages_with_train_field:
        return build_per_message_train_labels(messages, tokenizer, max_length=max_length)
    if train_on_responses_only:
        return build_assistant_only_labels(
            messages, tokenizer, max_length=max_length, include_eot=include_eot
        )
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", [])
    # Truncate to max_length like the other two strategies (both delegate
    # to data.loss_mask._truncate) — otherwise legacy_text is the only path
    # where --show-mask can render more tokens than the trainer actually
    # would, and the max_length-truncation check above would never trigger
    # for this strategy either.
    ids = list(ids)[:max_length]
    return {"input_ids": ids, "labels": ids}


def _mask_strategy_name(
    train_on_responses_only: bool, train_on_messages_with_train_field: bool
) -> str:
    if train_on_messages_with_train_field:
        return "per_message_train"
    if train_on_responses_only:
        return "assistant_only"
    return "legacy_text"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_chat_template(tokenizer: Any) -> DoctorCheck:
    """MAJOR when the tokenizer has no ``chat_template`` at all — every
    other check below depends on one and degrades to a no-op skip."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return DoctorCheck(
            name="chat_template",
            verdict="MAJOR",
            message="tokenizer has no chat_template — the mask/EOS/role checks below cannot run",
            evidence=(
                "pass --model with a tokenizer that ships tokenizer_config.json's "
                "chat_template, or set data.chat_template in soup.yaml"
            ),
        )
    return DoctorCheck(
        name="chat_template", verdict="OK", message="tokenizer has a chat_template",
        evidence=f"{len(template)} chars",
    )


def check_template_render(tokenizer: Any, rows: Sequence[Mapping]) -> DoctorCheck:
    """MAJOR when >=10% of sampled rows fail to render through the chat
    template (e.g. an unsupported role); MINOR for any lower non-zero rate."""
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(
            name="template_render", verdict="OK", message="skipped (no chat_template)"
        )
    if not rows:
        return DoctorCheck(name="template_render", verdict="OK", message="no rows to render")
    failures: List[Tuple[int, str]] = []
    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not messages:
            continue
        try:
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception as exc:  # noqa: BLE001 — collecting failures, not crashing
            failures.append((idx, type(exc).__name__))
    frac = len(failures) / len(rows)
    if not failures:
        verdict = "OK"
    elif frac >= 0.10:
        verdict = "MAJOR"
    else:
        verdict = "MINOR"
    sample = ", ".join(f"row {i} ({name})" for i, name in failures[:3])
    return DoctorCheck(
        name="template_render",
        verdict=verdict,
        message=f"{len(failures)}/{len(rows)} rows failed to render through the chat template",
        evidence=sample,
    )


def check_generation_markers(tokenizer: Any) -> DoctorCheck:
    """MINOR when the template lacks ``{% generation %}`` markers — assistant-
    only masking then falls back to the looser incremental-delta heuristic
    instead of HF's exact ``return_assistant_tokens_mask`` path."""
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(
            name="generation_markers", verdict="OK", message="skipped (no chat_template)"
        )
    probe = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    try:
        out = tokenizer.apply_chat_template(
            probe, tokenize=True, add_generation_prompt=False,
            return_assistant_tokens_mask=True, return_dict=True,
        )
        has_markers = isinstance(out, dict) and bool(out.get("assistant_masks")) and any(
            out["assistant_masks"]
        )
    except Exception:  # noqa: BLE001 — any failure means no usable markers
        has_markers = False
    if has_markers:
        return DoctorCheck(
            name="generation_markers", verdict="OK",
            message="template supports return_assistant_tokens_mask (exact assistant-only masking)",
        )
    return DoctorCheck(
        name="generation_markers",
        verdict="MINOR",
        message=(
            "template lacks {% generation %} markers — assistant-only masking falls back "
            "to a heuristic that may include role-prefix tokens in the loss"
        ),
    )


def check_eos_in_labels(
    tokenizer: Any,
    rows: Sequence[Mapping],
    *,
    max_length: int,
    include_eot: bool = False,
    train_on_responses_only: bool = True,
    train_on_messages_with_train_field: bool = False,
) -> DoctorCheck:
    """The #1 'model never stops generating' bug. MAJOR when >=50% of rows
    have at least one assistant turn whose trained span never trains an
    EOS/EOT token; MINOR for any lower non-zero rate. Checks every turn's
    span, not just the last, since an early un-closed turn also teaches the
    model to run turns together. Uses the SAME masking strategy as
    ``--show-mask`` (default: answer-only) so the two never disagree."""
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(
            name="eos_in_labels", verdict="OK", message="skipped (no chat_template)"
        )
    eos_ids = _eos_token_ids(tokenizer)
    if not eos_ids:
        return DoctorCheck(
            name="eos_in_labels", verdict="MINOR",
            message="tokenizer has no eos_token_id — cannot verify the model is taught to stop",
        )

    missing = 0
    checked = 0
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        try:
            built = _build_row_labels(
                tokenizer,
                messages,
                max_length=max_length,
                train_on_responses_only=train_on_responses_only,
                train_on_messages_with_train_field=train_on_messages_with_train_field,
                include_eot=include_eot,
            )
        except Exception:  # noqa: BLE001 — a row the template can't render (e.g. an
            # unsupported system role — jinja2.TemplateError, not a ValueError/
            # TypeError) is reported by template_render/system_role; skip it here.
            continue
        input_ids = built["input_ids"]
        # `build_*_labels` truncates to the FIRST max_length tokens
        # (data.loss_mask._truncate), which can cut off the trailing
        # EOS/EOT that would otherwise close the last assistant turn — that
        # would misdiagnose a max_length mismatch as a template bug.
        # truncation_risk already reports this; exclude the row here rather
        # than double-count it under the wrong check.
        if len(input_ids) >= max_length:
            continue
        spans = _trained_spans(built["labels"])
        if not spans:
            continue
        checked += 1
        # Every assistant turn (not just the last) must close on an EOS/EOT
        # token — a template that drops it on an earlier turn only would
        # still teach the model to run turns together mid-generation.
        if any(eos_ids.isdisjoint(input_ids[start : end + 1]) for start, end in spans):
            missing += 1
    if checked == 0:
        return DoctorCheck(
            name="eos_in_labels", verdict="OK", message="no assistant turns to check"
        )
    frac = missing / checked
    if frac == 0:
        verdict = "OK"
    elif frac >= 0.5:
        verdict = "MAJOR"
    else:
        verdict = "MINOR"
    return DoctorCheck(
        name="eos_in_labels",
        verdict=verdict,
        message=(
            f"{missing}/{checked} rows ({frac:.1%}) never train an EOS/EOT token — "
            "the model will not learn to stop generating"
        ),
        evidence=f"eos_token_id={sorted(eos_ids)}",
    )


def check_bos_duplication(
    tokenizer: Any,
    rows: Sequence[Mapping],
    *,
    max_length: int,
    train_on_responses_only: bool = True,
    train_on_messages_with_train_field: bool = False,
) -> DoctorCheck:
    """MAJOR when >=50% of rows start with two consecutive BOS tokens (the
    template AND the tokenizer both prepend one — a classic double-BOS
    footgun); MINOR for any lower non-zero rate.

    Unlike ``check_eos_in_labels``, this check needs no max_length-truncation
    exclusion: ``data.loss_mask._truncate`` keeps the FIRST max_length
    tokens (tail-truncates), so BOS at position 0 is never affected.
    """
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(
            name="bos_duplication", verdict="OK", message="skipped (no chat_template)"
        )
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos_id, bool) or not isinstance(bos_id, int):
        return DoctorCheck(
            name="bos_duplication", verdict="OK", message="tokenizer has no bos_token_id"
        )

    dup_rows = 0
    checked = 0
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        try:
            built = _build_row_labels(
                tokenizer,
                messages,
                max_length=max_length,
                train_on_responses_only=train_on_responses_only,
                train_on_messages_with_train_field=train_on_messages_with_train_field,
                include_eot=False,
            )
        except Exception:  # noqa: BLE001 — a row the template can't render is
            # reported by template_render/system_role; skip it here.
            continue
        ids = built["input_ids"]
        checked += 1
        if len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id:
            dup_rows += 1
    if checked == 0:
        return DoctorCheck(name="bos_duplication", verdict="OK", message="no rows to check")
    frac = dup_rows / checked
    verdict = "MAJOR" if frac >= 0.5 else "MINOR" if dup_rows > 0 else "OK"
    return DoctorCheck(
        name="bos_duplication",
        verdict=verdict,
        message=f"{dup_rows}/{checked} rows ({frac:.1%}) start with a duplicated BOS token",
        evidence=f"bos_token_id={bos_id}",
    )


def check_system_role(tokenizer: Any, rows: Sequence[Mapping]) -> DoctorCheck:
    """MAJOR when any row uses a system message but the template rejects one
    (Mistral-style templates commonly do) — training would crash on those
    rows. OK (not just skipped) when no row uses a system message at all."""
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(name="system_role", verdict="OK", message="skipped (no chat_template)")
    system_rows = sum(
        1
        for row in rows
        if any(
            isinstance(m, Mapping) and m.get("role") == "system"
            for m in (row.get("messages") or [])
        )
    )
    if system_rows == 0:
        return DoctorCheck(name="system_role", verdict="OK", message="no rows use a system message")
    probe = [{"role": "system", "content": "x"}, {"role": "user", "content": "y"}]
    try:
        tokenizer.apply_chat_template(probe, tokenize=False, add_generation_prompt=False)
        supported = True
    except Exception:  # noqa: BLE001 — template raised => role unsupported
        supported = False
    if supported:
        return DoctorCheck(
            name="system_role", verdict="OK",
            message=f"template supports the system role ({system_rows} rows use one)",
        )
    return DoctorCheck(
        name="system_role",
        verdict="MAJOR",
        message=(
            f"{system_rows} rows use a system message but this tokenizer's chat "
            "template does not support one — training will crash on those rows"
        ),
        evidence="Mistral-style templates commonly reject a leading system turn",
    )


def check_unknown_roles(rows: Sequence[Mapping]) -> DoctorCheck:
    """MAJOR when >=10% of rows contain a message role outside the known
    allowlist (typo, or an un-mapped source-format role like "human");
    MINOR for any lower non-zero rate. Runs regardless of chat_template
    presence — role typos are a data problem, not a template one."""
    if not rows:
        return DoctorCheck(name="unknown_roles", verdict="OK", message="no rows to check")
    bad_rows = 0
    seen_roles: set = set()
    for row in rows:
        row_bad = False
        for m in row.get("messages") or []:
            if not isinstance(m, Mapping):
                continue
            role = m.get("role")
            if not isinstance(role, str) or role not in _KNOWN_ROLES:
                row_bad = True
                seen_roles.add(str(role))
        if row_bad:
            bad_rows += 1
    frac = bad_rows / len(rows)
    verdict = "MAJOR" if frac >= 0.10 else "MINOR" if bad_rows > 0 else "OK"
    sample = ", ".join(sorted(seen_roles)[:5])
    return DoctorCheck(
        name="unknown_roles",
        verdict=verdict,
        message=(
            f"{bad_rows}/{len(rows)} rows ({frac:.1%}) contain a message role "
            f"outside {sorted(_KNOWN_ROLES)}"
        ),
        evidence=f"unexpected roles seen: {sample}" if sample else "",
    )


def check_truncation_risk(
    tokenizer: Any, rows: Sequence[Mapping], *, max_length: int
) -> DoctorCheck:
    """OK when the p95 rendered-token length is within ``max_length``.
    Otherwise MAJOR when >=20% of rows would be truncated, else MINOR — a
    truncated row typically loses the trailing (often the answer) content."""
    if not getattr(tokenizer, "chat_template", None):
        return DoctorCheck(
            name="truncation_risk", verdict="OK", message="skipped (no chat_template)"
        )

    from soup_cli.utils.tail_latency import percentile

    lengths: List[float] = []
    for row in rows:
        messages = row.get("messages")
        if not messages:
            continue
        try:
            ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False
            )
        except Exception:  # noqa: BLE001 — render failures are reported by template_render
            continue
        if isinstance(ids, dict):
            ids = ids.get("input_ids", [])
        lengths.append(float(len(ids)))
    if not lengths:
        return DoctorCheck(name="truncation_risk", verdict="OK", message="no rows to measure")
    p95 = percentile(lengths, 95.0) or 0.0
    p50 = percentile(lengths, 50.0) or 0.0
    over = sum(1 for length in lengths if length > max_length)
    frac_over = over / len(lengths)
    if p95 <= max_length:
        verdict = "OK"
    elif frac_over >= 0.20:
        verdict = "MAJOR"
    else:
        verdict = "MINOR"
    return DoctorCheck(
        name="truncation_risk",
        verdict=verdict,
        message=(
            f"p95 length = {p95:.0f} tokens vs max_length={max_length}; "
            f"{over}/{len(lengths)} rows ({frac_over:.1%}) would be truncated"
        ),
        evidence=f"p50={p50:.0f}, max={max(lengths):.0f}",
    )


# ---------------------------------------------------------------------------
# run_doctor — end to end
# ---------------------------------------------------------------------------


def run_doctor(
    raw_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    fmt: str,
    max_length: int = 2048,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    include_eot: bool = False,
    train_on_responses_only: bool = True,
    train_on_messages_with_train_field: bool = False,
) -> DoctorReport:
    """Run the full chat-template compat report over a sample of ``raw_rows``.

    ``train_on_responses_only``/``train_on_messages_with_train_field`` pick
    the SAME masking strategy ``--show-mask`` uses (mirrors
    ``data.sft_format.build_format_row``'s dispatch), so the compat report's
    ``eos_in_labels``/``bos_duplication`` verdicts and the mask preview
    agree on what the trainer would actually train on for a given
    ``soup.yaml`` — defaults match ``DataConfig``'s own defaults.

    Raises ``ValueError`` when nothing converts to chat messages (wrong
    ``fmt``, or ``fmt`` is a preference/RAFT shape this command doesn't
    cover — routes the caller to ``soup data lint`` / ``--show-mask``).
    """
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("sample_size must be a positive int")
    if sample_size > _MAX_SAMPLE_SIZE:
        raise ValueError(f"sample_size must be <= {_MAX_SAMPLE_SIZE}")
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive int")
    if not isinstance(include_eot, bool):
        raise TypeError("include_eot must be bool")
    if not isinstance(train_on_responses_only, bool):
        raise TypeError("train_on_responses_only must be bool")
    if not isinstance(train_on_messages_with_train_field, bool):
        raise TypeError("train_on_messages_with_train_field must be bool")

    from soup_cli.data.formats import format_to_messages

    total = len(raw_rows)
    idxs = sample_indices(total, sample_size)
    normalized: List[dict] = []
    for i in idxs:
        converted = format_to_messages(raw_rows[i], fmt)
        if converted and converted.get("messages"):
            normalized.append(converted)

    if total > 0 and not normalized:
        raise ValueError(
            f"no rows converted to chat messages for format={fmt!r} — soup data doctor "
            "targets chat/SFT data (chatml/alpaca/sharegpt/llava/audio/tool-calling/video/"
            "multimodal); preference data (dpo/kto) should use `soup data lint`, and RAFT "
            "data has no chat-template compat surface (use --show-mask instead)"
        )

    checks = [
        check_chat_template(tokenizer),
        check_template_render(tokenizer, normalized),
        check_generation_markers(tokenizer),
        check_eos_in_labels(
            tokenizer, normalized, max_length=max_length, include_eot=include_eot,
            train_on_responses_only=train_on_responses_only,
            train_on_messages_with_train_field=train_on_messages_with_train_field,
        ),
        check_bos_duplication(
            tokenizer, normalized, max_length=max_length,
            train_on_responses_only=train_on_responses_only,
            train_on_messages_with_train_field=train_on_messages_with_train_field,
        ),
        check_system_role(tokenizer, normalized),
        check_unknown_roles(normalized),
        check_truncation_risk(tokenizer, normalized, max_length=max_length),
    ]
    return compose_doctor_report(checks, rows_scanned=len(normalized), total_rows=total)


# ---------------------------------------------------------------------------
# --show-mask — per-token trained/masked X-ray through the REAL collator path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskedToken:
    """One rendered token + whether it contributes to the training loss."""

    text: str
    trained: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be str")
        if not isinstance(self.trained, bool):
            raise TypeError("trained must be bool")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("weight must be a number")
        weight = float(self.weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be a finite number >= 0")


@dataclass(frozen=True)
class MaskPreviewRow:
    """One sample row rendered as a token/trained-flag sequence."""

    row_index: int
    strategy: str
    tokens: Tuple[MaskedToken, ...]

    def __post_init__(self) -> None:
        if self.strategy not in _MASK_STRATEGIES:
            raise ValueError(f"unknown mask strategy {self.strategy!r}")
        if not isinstance(self.tokens, tuple):
            object.__setattr__(self, "tokens", tuple(self.tokens))
        if (
            isinstance(self.row_index, bool)
            or not isinstance(self.row_index, int)
            or self.row_index < 0
        ):
            raise ValueError("row_index must be a non-negative int")


def _tokens_from_ids(tokenizer: Any, ids: Sequence[int]) -> List[str]:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert):
        try:
            return [str(t) for t in convert(list(ids))]
        except Exception:  # noqa: BLE001 — fall through to per-id decode
            pass
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        out = []
        for tid in ids:
            try:
                out.append(decode([tid]))
            except Exception:  # noqa: BLE001
                out.append(f"<{tid}>")
        return out
    return [str(tid) for tid in ids]


def _build_chat_preview_row(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    fmt: str,
    max_length: int,
    train_on_responses_only: bool,
    train_on_messages_with_train_field: bool,
    include_eot: bool,
) -> Optional[Tuple[str, Tuple[MaskedToken, ...]]]:
    from soup_cli.data.formats import format_to_messages

    converted = format_to_messages(row, fmt)
    if not converted or not converted.get("messages"):
        return None
    if not getattr(tokenizer, "chat_template", None):
        return None
    messages = converted["messages"]
    strategy = _mask_strategy_name(train_on_responses_only, train_on_messages_with_train_field)
    try:
        built = _build_row_labels(
            tokenizer,
            messages,
            max_length=max_length,
            train_on_responses_only=train_on_responses_only,
            train_on_messages_with_train_field=train_on_messages_with_train_field,
            include_eot=include_eot,
        )
    except Exception:  # noqa: BLE001 — a row the template can't render (e.g. an
        # unsupported system role) is a per-row skip, not a crash — mirrors the
        # "fewer than n results is a valid outcome" contract documented above.
        return None
    token_texts = _tokens_from_ids(tokenizer, built["input_ids"])
    tokens = tuple(
        MaskedToken(text=t, trained=(lbl != IGNORE_INDEX))
        for t, lbl in zip(token_texts, built["labels"])
    )
    return strategy, tokens


def _build_raft_preview_row(
    row: Mapping[str, Any], tokenizer: Any, *, max_length: int
) -> Optional[Tuple[str, Tuple[MaskedToken, ...]]]:
    from soup_cli.data.formats import format_to_messages
    from soup_cli.utils.raft import build_raft_prompt, tokenize_raft_example

    converted = format_to_messages(row, "raft")
    if not converted:
        return None
    try:
        composed = build_raft_prompt(converted)
        built = tokenize_raft_example(tokenizer, composed, max_length=max_length)
    except Exception:  # noqa: BLE001 — same per-row skip contract as the chat path
        return None
    token_texts = _tokens_from_ids(tokenizer, built["input_ids"])
    tokens = tuple(
        MaskedToken(text=t, trained=(weight > 0), weight=float(weight))
        for t, weight in zip(token_texts, built["loss_weights"])
    )
    return "raft", tokens


def render_mask_preview(
    raw_rows: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    fmt: str,
    n: int = 3,
    max_length: int = 2048,
    train_on_responses_only: bool = True,
    train_on_messages_with_train_field: bool = False,
    include_eot: bool = False,
) -> List[MaskPreviewRow]:
    """Render up to ``n`` rows with per-token trained/masked colouring.

    Routes to the REAL collator path per format: RAFT rows go through
    ``utils.raft`` (span-mask via ``loss_weights``); everything else goes
    through ``data.loss_mask`` (answer-only / per-message-train-field /
    legacy full-sequence). Rows that fail to convert or render are skipped
    (fewer than ``n`` results is a valid outcome, not an error).
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive int")
    if n > _MAX_PREVIEW_ROWS:
        raise ValueError(f"n must be <= {_MAX_PREVIEW_ROWS}")

    previews: List[MaskPreviewRow] = []
    for row_index, row in enumerate(raw_rows[:_MAX_MASK_SCAN_ROWS]):
        if len(previews) >= n:
            break
        if fmt == "raft":
            result = _build_raft_preview_row(row, tokenizer, max_length=max_length)
        else:
            result = _build_chat_preview_row(
                row,
                tokenizer,
                fmt=fmt,
                max_length=max_length,
                train_on_responses_only=train_on_responses_only,
                train_on_messages_with_train_field=train_on_messages_with_train_field,
                include_eot=include_eot,
            )
        if result is None:
            continue
        strategy, tokens = result
        previews.append(MaskPreviewRow(row_index=row_index, strategy=strategy, tokens=tokens))
    return previews


__all__ = [
    "CHECKS",
    "VERDICTS",
    "DoctorCheck",
    "DoctorReport",
    "MaskedToken",
    "MaskPreviewRow",
    "check_bos_duplication",
    "check_chat_template",
    "check_eos_in_labels",
    "check_generation_markers",
    "check_system_role",
    "check_template_render",
    "check_truncation_risk",
    "check_unknown_roles",
    "compose_doctor_report",
    "overall_verdict",
    "render_mask_preview",
    "resolve_tokenizer",
    "run_doctor",
    "sample_indices",
]
