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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soup_cli.config.schema import SoupConfig

#: A setting the backend reads and refuses, versus one it never reads at all.
#:
#: #755's AC3 named three statuses. Only ``ignored`` has live instances; the
#: other two are **deliberately deferred**, not forgotten:
#:
#: * ``honoured`` would mean enumerating 275 declared fields against every
#:   reviewed pair -- a table nobody can review honestly, and the opposite of
#:   this module's premise that a gap list is short enough to be checked;
#: * ``rejected`` covers settings rejected at the schema boundary for a
#:   backend (e.g. callback monitoring flags on backend=mlx, #1069).
IGNORED = "ignored"
REJECTED = "rejected"
STATUSES = frozenset({IGNORED, REJECTED})

DEFAULT_BACKEND = "transformers"
DEFAULT_MODALITY = "text"


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
    # Every entry here is read by trainer/mlx_sft.py solely to build its
    # "MLX backend ignores:" line. The five fields this table used to carry --
    # max_grad_norm, warmup_ratio, weight_decay, optimizer, scheduler -- were
    # wired by #734 and #750 and are now honoured, so they are gone rather than
    # reclassified. The guard is what noticed; see the PR body.
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
        "training.use_lorafa",
        IGNORED,
        "LoRA-FA has no MLX implementation",
        issue=725,
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
        "data.mask_history",
        IGNORED,
        "MLX supervises every assistant turn, not only the last; mask_history is "
        "applied by the transformers label builder, which MLX SFT does not use",
        trainer_reads=True,
    ),
    SupportEntry(
        "data.train_on_prompt",
        IGNORED,
        "MLX masks the prompt or supervises the whole sequence; no per-field switch",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_liger",
        IGNORED,
        "Liger fused kernels have no MLX implementation (and soup train "
        "refuses the run on non-CUDA before MLX ever sees it)",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.neftune_alpha",
        IGNORED,
        "NEFT noise is applied on the transformers training path, not MLX",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_mod",
        IGNORED,
        "Mixture-of-Depths routing is wired on the transformers path, not MLX",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.moe_lora",
        IGNORED,
        "ScatterMoE LoRA targets expert layers on the transformers path; no MLX implementation",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.quantization_aware",
        IGNORED,
        "QAT/FP8 prepare runs on the transformers path, not MLX",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.use_fsdp2_compile",
        IGNORED,
        "torch.compile on FSDP2 requires CUDA and the transformers backend "
        "(and `soup train` refuses backend=mlx in validate_fsdp2_compile_config, "
        "before resolve_trainer)",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.loss_watchdog",
        REJECTED,
        "Soup does not implement the watchdog on the MLX callback, which has no stop control",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.loss_spike_recovery",
        REJECTED,
        "spike recovery is driven by the watchdog and the watchdog cannot fire on MLX",
        trainer_reads=True,
    ),
    SupportEntry(
        "training.grad_accum_auto_tune",
        REJECTED,
        "there is no VRAM total to measure pressure against on unified memory",
        trainer_reads=True,
    ),
)


_TRANSFORMERS_VISION_SFT: tuple[SupportEntry, ...] = (
    SupportEntry(
        "data.train_on_responses_only",
        IGNORED,
        "multimodal vision collator supervises all tokens; assistant-only masking "
        "is not implemented for vision datasets",
        issue=1156,
        trainer_reads=False,
    ),
    SupportEntry(
        "data.mask_history",
        IGNORED,
        "multimodal vision collator supervises all tokens; mask_history is applied "
        "only by the transformers text label builder",
        issue=1156,
        trainer_reads=False,
    ),
)

_TRANSFORMERS_AUDIO_SFT: tuple[SupportEntry, ...] = (
    SupportEntry(
        "data.train_on_responses_only",
        IGNORED,
        "audio training supervises all tokens; assistant-only masking "
        "is not implemented for audio datasets",
        issue=1156,
        trainer_reads=False,
    ),
    SupportEntry(
        "data.mask_history",
        IGNORED,
        "audio training supervises all tokens; mask_history is applied "
        "only by the transformers text label builder",
        issue=1156,
        trainer_reads=False,
    ),
)


class _RegistryDict(dict):
    """Registry supporting 3-tuple (task, backend, modality) with 2-tuple fallback."""

    def __getitem__(self, key: tuple[str, ...]) -> tuple[Any, ...]:
        if isinstance(key, tuple) and len(key) == 2:
            val = super().get(key)
            if val is not None:
                return val
            val = super().get((key[0], key[1], DEFAULT_MODALITY))
            if val is not None:
                return val
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: tuple[str, ...], default=None):
        try:
            val = self[key]
            return default if val is None else val
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        if isinstance(key, tuple) and len(key) == 2:
            return super().__contains__(key) or super().__contains__(
                (key[0], key[1], DEFAULT_MODALITY)
            )
        return super().__contains__(key)


#: ``(task, backend, modality)`` -> the settings that combination does not honour.
REGISTRY: dict[tuple[str, ...], tuple[SupportEntry, ...]] = _RegistryDict({
    ("sft", "mlx", "text"): _MLX_SFT,
    ("sft", "transformers", "vision"): _TRANSFORMERS_VISION_SFT,
    ("sft", "transformers", "audio"): _TRANSFORMERS_AUDIO_SFT,
})

_TRANSFORMERS_SFT_MODULES = (
    "soup_cli/trainer/sft.py",
    "soup_cli/trainer/loss_summary.py",
    "soup_cli/trainer/stream_setup.py",
    "soup_cli/trainer/rewind_hf.py",
    "soup_cli/trainer/raft.py",
)

#: Every module a reviewed combination can read config through -- the trainer plus the
#: helpers it delegates to. The guard scans the **union**.
TRAINER_MODULES: dict[tuple[str, ...], tuple[str, ...]] = _RegistryDict({
    ("sft", "mlx", "text"): (
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/mlx_masking.py",
        "soup_cli/trainer/rewind_mlx.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("sft", "transformers", "vision"): _TRANSFORMERS_SFT_MODULES,
    ("sft", "transformers", "audio"): _TRANSFORMERS_SFT_MODULES,
})


def unsupported_for(
    task: str, backend: str, modality: str = DEFAULT_MODALITY
) -> tuple[SupportEntry, ...]:
    """Every declared gap for a task/backend/modality combination, or ``()`` if unreviewed."""
    return REGISTRY.get((task, backend, modality), ())


def check_config(cfg: "SoupConfig") -> list[SupportEntry]:
    """The declared gaps that apply to the fields this config actually sets.

    Only fields the user wrote are reported. Pydantic's ``model_fields_set``
    distinguishes those from the ones sitting at their schema default, which is
    the difference between a useful pre-flight check and a wall of 275 rows.
    """
    backend = getattr(cfg, "backend", DEFAULT_BACKEND)
    modality = getattr(cfg, "modality", DEFAULT_MODALITY)
    entries = unsupported_for(cfg.task, backend, modality)
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
