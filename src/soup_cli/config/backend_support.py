"""#755 — declared per-(task, backend) config support.

A field can be declared, validated, documented and still be read by nothing on
the backend you chose. That has been filed one field at a time: #683, #686,
#745, #749.

Which fields a backend honours is **declared here, not inferred**. Three
inference strategies were measured and rejected (see #755): reachability over
the import graph detects none of the known gaps, because every trainer's
transitive closure is most of the package; reads of the trainer module alone
invent gaps for fields that live in helper modules (``train_on_responses_only``
in ``data/sft_format.py``, ``loraplus_lr_ratio`` in the #738 helper); and
``--dry-run`` exits before a trainer is ever constructed, so runtime tracing
observes nothing.

So this table is maintained by review. ``tests/test_issue755_*`` bounds the
drift -- every entry must name a real field, and every dotted field name in
``trainer/mlx_sft.py``'s own warning list must appear here. It cannot originate
the truth; it can only stop it rotting, which is the failure that produced #749.

Scope today: ``task=sft`` on ``backend=mlx``. Other pairs report nothing rather
than guessing, which is the honest default for a table that has not been
reviewed for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soup_cli.config.schema import SoupConfig

#: A setting the backend reads and refuses, versus one it never reads at all.
IGNORED = "ignored"
REJECTED = "rejected"
STATUSES = frozenset({IGNORED, REJECTED})

DEFAULT_BACKEND = "transformers"


@dataclass(frozen=True)
class SupportEntry:
    """One declared gap: a field, what happens to it, and why."""

    field: str
    status: str
    reason: str
    issue: int | None = None
    #: Does the backend's trainer read this field *in order to warn about it*?
    #: The guard uses this in both directions: a ``False`` entry that the
    #: trainer starts reading means the field was wired and the entry is stale
    #: (this is what happens to ``max_grad_norm`` when #750 lands), and a
    #: ``True`` entry the trainer stops reading means the warning was deleted.
    trainer_reads: bool = False

    def describe(self) -> str:
        suffix = f" (#{self.issue})" if self.issue else ""
        return f"{self.reason}{suffix}"


_MLX_SFT: tuple[SupportEntry, ...] = (
    # Never read at all by trainer/mlx_sft.py.
    SupportEntry(
        "training.max_grad_norm",
        IGNORED,
        "MLX clips nowhere; sixteen transformers trainers clip at this value",
        issue=749,
    ),
    SupportEntry(
        "training.warmup_ratio",
        IGNORED,
        "MLX uses mlx-lm's own schedule",
        issue=686,
    ),
    SupportEntry(
        "training.weight_decay",
        IGNORED,
        "not forwarded to the MLX optimizer",
        issue=686,
    ),
    SupportEntry(
        "training.optimizer",
        IGNORED,
        "MLX always builds mlx.optimizers.Adam",
        issue=686,
    ),
    SupportEntry(
        "training.scheduler",
        IGNORED,
        "MLX uses mlx-lm's own schedule",
        issue=686,
    ),
    # Read by trainer/mlx_sft.py solely to warn about them.
    SupportEntry(
        "training.seed",
        IGNORED,
        "MLX seeds through mx.random, not this field",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.data_seed",
        IGNORED,
        "MLX seeds through mx.random, not this field",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_galore",
        IGNORED,
        "GaLore has no MLX implementation",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_ring_attention",
        IGNORED,
        "Ring Attention has no MLX path",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_flash_attn",
        IGNORED,
        "MLX has its own attention kernels",
        trainer_reads=True,
    ),
    SupportEntry(
        "data.train_on_messages_with_train_field",
        IGNORED,
        "MLX supervises every assistant turn; the per-message flag is not read",
        trainer_reads=True,
    ),
    SupportEntry(
        "data.train_on_prompt",
        IGNORED,
        "MLX masks the prompt or supervises the whole sequence; no per-field switch",
        trainer_reads=True,
    ),
)


#: ``(task, backend)`` -> the settings that pair does not honour.
REGISTRY: dict[tuple[str, str], tuple[SupportEntry, ...]] = {
    ("sft", "mlx"): _MLX_SFT,
}

#: The trainer module each reviewed pair dispatches to. The guard reads it to
#: check ``trainer_reads`` against the real source.
TRAINER_MODULES: dict[tuple[str, str], str] = {
    ("sft", "mlx"): "soup_cli/trainer/mlx_sft.py",
}


def unsupported_for(task: str, backend: str) -> tuple[SupportEntry, ...]:
    """Every declared gap for a task/backend pair, or ``()`` if unreviewed."""
    return REGISTRY.get((task, backend), ())


def check_config(cfg: "SoupConfig") -> list[SupportEntry]:
    """The declared gaps that apply to the fields this config actually sets.

    Only fields the user wrote are reported. Pydantic's ``model_fields_set``
    distinguishes those from the ones sitting at their schema default, which is
    the difference between a useful pre-flight check and a wall of 275 rows.
    """
    entries = unsupported_for(cfg.task, getattr(cfg, "backend", DEFAULT_BACKEND))
    if not entries:
        return []

    written: set[str] = set()
    for namespace in ("training", "data"):
        section = getattr(cfg, namespace, None)
        if section is None:
            continue
        for name in getattr(section, "model_fields_set", ()):
            written.add(f"{namespace}.{name}")

    return [entry for entry in entries if entry.field in written]
