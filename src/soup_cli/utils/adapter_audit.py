"""Compare a finished adapter's record against the config that asked for it (#762).

Five merged fixes taught the MLX path to record what it actually did into
``adapter_config.json`` -- #683 (masking), #684 (accumulation), #685 (grad
checkpoint), #686 (optimizer and schedule), #749 (gradient clipping). Each
existed as a defect where a setting was accepted and dropped without a word,
and the record is the remedy. Nothing read it back until this module.

**Unknown is never agreement.** A setting the record cannot speak to is
reported ``unknown`` and never ``ok``. A false clean bill converts "I do not
know" into "I checked", which is precisely the substitution this command
exists to undo -- and it is how #683 and #686 survived as long as they did.
For the same reason ``unknown`` is not a failure: an older adapter predates the
keys, which is not the user's fault. It has to be visible, not fatal.

Pure data in, pure data out: no filesystem, no console, no imports beyond the
standard library, so the comparison can be tested without a CLI or a training
run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: Soup optimizer names -> the MLX class names `mlx_optim` resolves them to.
#: Kept here rather than imported so this module stays free of the trainer
#: package, which pulls heavy dependencies at import time.
_MLX_OPTIMIZER_ALIASES = {
    "adamw_torch": "AdamW",
    "adamw_hf": "AdamW",
    "adamw_torch_fused": "AdamW",
    "adamw": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
    "lion": "Lion",
    "adafactor": "Adafactor",
    "adagrad": "Adagrad",
    "adamax": "Adamax",
    "rmsprop": "RMSprop",
    "adadelta": "AdaDelta",
    "muon": "Muon",
}

OK = "ok"
DIVERGED = "diverged"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class AuditRow:
    """One setting, as asked for and as recorded."""

    setting: str
    asked: Any
    ran: Any
    status: str
    detail: str = ""


@dataclass
class AuditResult:
    rows: List[AuditRow] = field(default_factory=list)
    record_kind: str = "unknown"

    @property
    def diverged_count(self) -> int:
        return sum(1 for r in self.rows if r.status == DIVERGED)

    @property
    def unknown_count(self) -> int:
        return sum(1 for r in self.rows if r.status == UNKNOWN)

    @property
    def checked_count(self) -> int:
        """Rows the record could actually speak to -- ``ok`` or ``diverged``.

        Without it, a ``{}`` record (ten ``unknown``, nothing checked) and a
        run that genuinely agreed on everything both present as "No
        divergences" and exit 0, and a CI job cannot tell them apart (#763
        review). The exit contract is deliberately unchanged: ``unknown``
        still does not fail, it is now merely countable.
        """
        return sum(1 for r in self.rows if r.status in (OK, DIVERGED))

    @property
    def exit_code(self) -> int:
        """Non-zero only for divergence, so this composes into CI and `soup
        ship`. `unknown` deliberately does not fail: see the module docstring."""
        return 1 if self.diverged_count else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "diverged_count": self.diverged_count,
            "unknown_count": self.unknown_count,
            "checked_count": self.checked_count,
            "rows": [
                {
                    "setting": r.setting,
                    "asked": r.asked,
                    "ran": r.ran,
                    "status": r.status,
                    "detail": r.detail,
                }
                for r in self.rows
            ],
        }


def classify_record(record: Dict[str, Any]) -> str:
    """Which writer produced this ``adapter_config.json``.

    MLX writes Soup's own effective-settings record; the transformers path
    leaves PEFT's, which carries LoRA shape and nothing about the schedule.
    The distinction decides how much of the config can be audited at all, so
    the caller reports it rather than inferring a clean bill from silence.
    """
    if "peft_type" in record:
        return "peft"
    if "fine_tune_type" in record or "total_updates" in record:
        return "mlx"
    return "unknown"


def _cmp(
    setting: str,
    asked: Any,
    ran: Any,
    *,
    detail: str = "",
    normalise: "Optional[Any]" = None,
) -> AuditRow:
    """Compare one setting. ``normalise`` is applied to both sides before the
    equality test, never to what is displayed -- the user should see the value
    they wrote, not a lowered copy of it."""
    if ran is None:
        return AuditRow(setting, asked, None, UNKNOWN, detail or "not in the record")
    # `True == 1` and `False == 0` in Python, so a record carrying
    # `max_grad_norm: true` compared equal to a requested 1.0 and was handed a
    # clean bill. Only a malformed record does this, but "the record is
    # nonsense" must not read as "the run agreed".
    if isinstance(asked, bool) != isinstance(ran, bool):
        return AuditRow(
            setting, asked, ran, DIVERGED,
            f"the record holds {ran!r} ({type(ran).__name__}) where the config "
            f"asks for {asked!r} ({type(asked).__name__}); Python compares "
            "bools equal to 0/1, so this is not the agreement it looks like",
        )
    left, right = (normalise(asked), normalise(ran)) if normalise else (asked, ran)
    status = OK if left == right else DIVERGED
    return AuditRow(setting, asked, ran, status, detail if status == DIVERGED else "")


def _audit_optimizer(training: Dict[str, Any], record: Dict[str, Any]) -> AuditRow:
    asked = training.get("optimizer", "adamw_torch")
    ran = record.get("optimizer")
    if ran is None:
        return AuditRow("optimizer", asked, None, UNKNOWN, "not in the record")
    # The record holds the MLX class name; the config holds Soup's name.
    expected = _MLX_OPTIMIZER_ALIASES.get(str(asked).strip().lower(), asked)
    if expected == ran:
        return AuditRow("optimizer", asked, ran, OK)
    return AuditRow(
        "optimizer", asked, ran, DIVERGED,
        f"{asked!r} resolves to {expected!r} on this backend, and the run used {ran!r}",
    )


def _audit_warmup(training: Dict[str, Any], record: Dict[str, Any]) -> AuditRow:
    """The motivating case: a ratio that rounds away.

    `warmup_ratio: 0.03` over twelve optimizer updates is zero warmup steps.
    #686 warns at the time; if nobody was watching the terminal, the record is
    the only surviving evidence.
    """
    asked = training.get("warmup_ratio", 0.03)
    ran = record.get("warmup_updates")
    total = record.get("total_updates")
    if ran is None or total is None:
        return AuditRow("warmup_ratio", asked, ran, UNKNOWN, "not in the record")

    expected = int(float(asked) * int(total))

    # Asking for warmup and getting none is a divergence even though the
    # arithmetic is faithful. `int(0.03 * 12)` really is 0, so a purely
    # numerical comparison calls this agreement -- but the user asked for a
    # ramp and the run started at the peak learning rate, which is the
    # difference they care about and the reason #686 warns at the time.
    # Reporting "ok" here would be the false clean bill this module exists to
    # refuse, just arrived by arithmetic rather than by a missing key.
    if float(asked) > 0 and ran == 0:
        return AuditRow(
            "warmup_ratio", asked, f"{ran} of {total} updates", DIVERGED,
            f"{asked} x {total} optimizer updates rounds to 0, so training "
            "started at the peak learning rate; raise warmup_ratio or lower "
            "gradient_accumulation_steps to produce more updates",
        )
    if expected == ran:
        return AuditRow("warmup_ratio", asked, f"{ran} of {total} updates", OK)
    return AuditRow(
        "warmup_ratio", asked, f"{ran} of {total} updates", DIVERGED,
        f"expected {expected} warmup updates from {asked} x {total}",
    )


def _ran_alpha(record: Dict[str, Any]) -> Optional[float]:
    """The alpha a run used, from whichever shape its writer chose.

    PEFT stores `lora_alpha` directly. MLX stores **`scale`**, which is
    `alpha / rank` -- there is no `alpha` key at all. Reading one and hoping
    is how this returned `unknown` for every real MLX adapter: the fixture
    invented an `alpha` key, the fixture passed, and a live run on Metal
    showed `lora_parameters = {"rank": 4, "scale": 2.0, ...}`.
    """
    direct = record.get("lora_alpha")
    if direct is not None:
        return direct
    params = record.get("lora_parameters") or {}
    if "alpha" in params:
        return params["alpha"]
    scale, rank = params.get("scale"), params.get("rank")
    if scale is None or rank is None:
        return None
    return scale * rank


def _audit_masking(data: Dict[str, Any], record: Dict[str, Any]) -> AuditRow:
    """Did response-only masking actually happen -- not: was it requested.

    ``mlx_sft.py:742-744`` writes three keys, and only two of them are
    evidence. ``train_on_responses_only`` is ``responses_only`` echoed back --
    the config's own request, copied into the record. The effect is
    ``mask_prompt`` (upstream's single masked prefix, correct for
    prompt/completion rows) and ``response_token_mask`` (Soup's per-token mask,
    the only shape correct for multi-turn chat).

    ``plan_response_masking`` returns *neither* for plain-text rows: they carry
    no role boundaries, upstream raises if the flag is set, so the run warns
    once and trains on the full sequence. The record it leaves says
    ``train_on_responses_only: true`` beside ``mask_prompt: false`` and
    ``response_token_mask: false``.

    Comparing the config against the echo therefore agrees with itself on the
    one run that most needs reporting -- #683 itself, handed a clean bill by
    the command written to catch it.
    """
    asked = bool(data.get("train_on_responses_only", True))
    prefix_mask = record.get("mask_prompt")
    token_mask = record.get("response_token_mask")

    # Presence is provable from one key, absence needs both. A truthy key
    # settles it whatever the other would have said; but reading a *missing*
    # key as False would turn "this record does not say" into "the run masked
    # nothing", which is the same substitution as the false clean bill,
    # arrived at from the other direction. `mlx_sft.py` writes the pair
    # together, so a record holding one alone is foreign or truncated.
    if bool(prefix_mask) or bool(token_mask):
        ran = True
    elif prefix_mask is None or token_mask is None:
        missing = [
            name
            for name, value in (
                ("mask_prompt", prefix_mask),
                ("response_token_mask", token_mask),
            )
            if value is None
        ]
        return AuditRow(
            "data.train_on_responses_only", asked, None, UNKNOWN,
            f"the record does not carry {' or '.join(missing)}, so it cannot "
            "say whether masking took effect"
            + (" (adapter predates #683)" if len(missing) == 2 else ""),
        )
    else:
        ran = False
    if ran == asked:
        return AuditRow("data.train_on_responses_only", asked, ran, OK)
    if asked and not ran:
        return AuditRow(
            "data.train_on_responses_only", asked, ran, DIVERGED,
            "requested, but the run masked nothing -- plain-text rows carry no "
            "role boundaries, so the loss covered prompt tokens too; use chatml "
            "or prompt/completion data",
        )
    return AuditRow(
        "data.train_on_responses_only", asked, ran, DIVERGED,
        "not requested, but the run masked anyway "
        f"(mask_prompt={prefix_mask!r}, response_token_mask={token_mask!r})",
    )


def _lora_asked(training: Dict[str, Any]) -> Dict[str, Any]:
    lora = training.get("lora") or {}
    if not isinstance(lora, dict):
        return {}
    return lora


def audit_adapter(config: Dict[str, Any], record: Dict[str, Any]) -> AuditResult:
    """Compare a soup config against an adapter's own record of what ran."""
    training = config.get("training") or {}
    data = config.get("data") or {}
    kind = classify_record(record)
    rows: List[AuditRow] = []

    rows.append(_audit_optimizer(training, record))
    # mlx_optim stores `str(scheduler).strip().lower()`, so `scheduler: Cosine`
    # in a config would report DIVERGED against a record saying `cosine`.
    # _audit_optimizer already normalises; this did not.
    _asked_sched = training.get("scheduler", "cosine")
    _ran_sched = record.get("scheduler")
    rows.append(
        _cmp(
            "scheduler",
            _asked_sched,
            _ran_sched,
            normalise=lambda v: str(v).strip().lower(),
        )
    )
    rows.append(_audit_warmup(training, record))
    rows.append(
        _cmp("weight_decay", training.get("weight_decay", 0.01), record.get("weight_decay"))
    )
    rows.append(
        _cmp("max_grad_norm", training.get("max_grad_norm", 1.0), record.get("max_grad_norm"))
    )
    rows.append(
        _cmp(
            "gradient_accumulation_steps",
            training.get("gradient_accumulation_steps", 4),
            record.get("grad_accumulation_steps"),
        )
    )
    rows.append(
        _cmp(
            "gradient_checkpointing",
            bool(training.get("gradient_checkpointing", False)),
            record.get("grad_checkpoint"),
        )
    )
    rows.append(_audit_masking(data, record))

    lora = _lora_asked(training)
    if "r" in lora:
        ran_r = record.get("r")
        if ran_r is None:
            ran_r = (record.get("lora_parameters") or {}).get("rank")
        rows.append(_cmp("lora.r", lora["r"], ran_r))
    if "alpha" in lora:
        rows.append(_cmp("lora.alpha", lora["alpha"], _ran_alpha(record)))

    return AuditResult(rows=rows, record_kind=kind)


def unknown_reason(kind: str) -> Optional[str]:
    """Why a whole class of settings could not be checked.

    Said out loud by the caller, because a list of `unknown` rows with no
    explanation reads as a tool failure rather than as a limit of the record.
    """
    if kind == "peft":
        return (
            "This adapter carries PEFT's own adapter_config.json, which records "
            "LoRA shape and nothing about the optimizer, schedule or masking. "
            "Those settings cannot be audited on the transformers path."
        )
    if kind == "unknown":
        return "Unrecognised adapter_config.json; only settings it names can be checked."
    return None
