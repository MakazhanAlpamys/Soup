"""Compare a finished adapter's record against the config that asked for it (#762).

Four merged fixes taught the MLX path to record what it actually did into
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
    def exit_code(self) -> int:
        """Non-zero only for divergence, so this composes into CI and `soup
        ship`. `unknown` deliberately does not fail: see the module docstring."""
        return 1 if self.diverged_count else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_kind": self.record_kind,
            "diverged_count": self.diverged_count,
            "unknown_count": self.unknown_count,
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


def _cmp(setting: str, asked: Any, ran: Any, *, detail: str = "") -> AuditRow:
    if ran is None:
        return AuditRow(setting, asked, None, UNKNOWN, detail or "not in the record")
    status = OK if asked == ran else DIVERGED
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
    asked = training.get("warmup_ratio", 0.0)
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
    rows.append(_cmp("scheduler", training.get("scheduler", "cosine"), record.get("scheduler")))
    rows.append(_audit_warmup(training, record))
    rows.append(
        _cmp("weight_decay", training.get("weight_decay", 0.0), record.get("weight_decay"))
    )
    rows.append(
        _cmp("max_grad_norm", training.get("max_grad_norm", 1.0), record.get("max_grad_norm"))
    )
    rows.append(
        _cmp(
            "gradient_accumulation_steps",
            training.get("gradient_accumulation_steps", 1),
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
    rows.append(
        _cmp(
            "data.train_on_responses_only",
            bool(data.get("train_on_responses_only", True)),
            record.get("train_on_responses_only"),
        )
    )

    lora = _lora_asked(training)
    if "r" in lora:
        ran_r = record.get("r")
        if ran_r is None:
            ran_r = (record.get("lora_parameters") or {}).get("rank")
        rows.append(_cmp("lora.r", lora["r"], ran_r))
    if "alpha" in lora:
        ran_a = record.get("lora_alpha")
        if ran_a is None:
            ran_a = (record.get("lora_parameters") or {}).get("alpha")
        rows.append(_cmp("lora.alpha", lora["alpha"], ran_a))

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
