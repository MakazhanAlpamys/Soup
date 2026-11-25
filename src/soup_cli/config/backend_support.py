"""#755 — declared per-(task, backend, modality) config support.

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

Reviewed combinations include SFT on MLX and Unsloth (#1124), DPO, ORPO, SimPO,
KTO and GRPO on MLX, and SFT on Transformers for text, vision and audio
(#1156, #1815). :func:`is_reviewed` tells doctor whether a combination has a
reviewed table, including tables with no ignored settings. The shipped-recipe
test tracks the combinations still waiting for a review.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from soup_cli.config.schema import SoupConfig

#: A setting the backend reads and refuses, versus one it never reads at all.
#:
#: ``honoured`` is intentionally not enumerated: the table records gaps only.
#: ``rejected`` describes runtime refusals visible to the doctor; config-load
#: refusals cannot reach it (#1330). Native selective non-text masks are refused
#: before model loading rather than claimed as qualified multimodal support.
IGNORED = "ignored"
REJECTED = "rejected"
STATUSES = frozenset({IGNORED, REJECTED})

DEFAULT_BACKEND = "transformers"
DEFAULT_MODALITY = "text"


def _default_active(value: Any) -> bool:
    """A setting is active when it is switched on (truthy)."""
    return bool(value)


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
    active: Callable[[Any], bool] = field(default=_default_active, compare=False, repr=False)

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
        "training.moe_lora",
        IGNORED,
        "ScatterMoE LoRA targets expert layers on the transformers path; no MLX implementation",
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
)


# Note: The drift guard accepts these unread entries because the text path
# reads data.train_on_responses_only and data.mask_history in
# data/sft_format.py and data/loss_mask.py, outside the declared trainer modules.
_TRANSFORMERS_VISION_SFT: tuple[SupportEntry, ...] = (
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
        "data.mask_history",
        IGNORED,
        "audio training supervises all tokens; mask_history is applied "
        "only by the transformers text label builder",
        issue=1156,
        trainer_reads=False,
    ),
)


# Unsloth precision settings are refused at config load, not reported by doctor.
_UNSLOTH_SFT: tuple[SupportEntry, ...] = ()


#: Fields MLX SFT warns about that have no meaning for DPO: SFT's
#: response-masking switches (DPO supervises completions by design).
_NOT_DPO_GAPS = frozenset({
    "data.train_on_messages_with_train_field",
    "data.mask_history",
    "data.train_on_prompt",
})

# These settings are refused at config load for preference tasks, so they can
# never reach doctor. LoRA-FA and QAT currently filter no SFT rows; keeping them
# here ensures a future SFT row cannot accidentally leak into preference tables.
_MLX_PREFERENCE_REFUSED = frozenset({
    "training.use_lorafa",
    "training.quantization_aware",
    "training.use_flash_attn",
    "training.use_liger",
})

_MLX_DPO: tuple[SupportEntry, ...] = tuple(
    entry
    for entry in _MLX_SFT
    if entry.field not in _NOT_DPO_GAPS | _MLX_PREFERENCE_REFUSED
)

#: MLX ORPO runs MLX DPO's loop with its own loss, so it ignores the same fields.
_MLX_ORPO: tuple[SupportEntry, ...] = _MLX_DPO

#: MLX SimPO is MLX ORPO's tokenization and MLX DPO's loop: the same fields.
_MLX_SIMPO: tuple[SupportEntry, ...] = _MLX_ORPO

#: MLX KTO runs MLX DPO's loop and reference precompute: the same fields.
_MLX_KTO: tuple[SupportEntry, ...] = _MLX_DPO

#: MLX GRPO (#1976) prints MLX DPO's warning list, but only these of its rows can
#: load with ``task: grpo``: LoRA-FA, FlashAttention, Liger, NEFTune and QAT are
#: refused at config load for this task, and load-time refusals have no rows.
_MLX_GRPO_FIELDS = frozenset({
    "training.use_galore",
    "training.use_ring_attention",
    "training.moe_lora",
    "training.use_fsdp2_compile",
})
_MLX_GRPO: tuple[SupportEntry, ...] = tuple(
    entry for entry in _MLX_DPO if entry.field in _MLX_GRPO_FIELDS
)

_MLX_SELECTIVE_NON_TEXT = tuple(
    SupportEntry(
        name, REJECTED, "native selective message masks support text SFT only", trainer_reads=True,
    )
    for name in ("data.mask_history", "data.train_on_messages_with_train_field")
)


# Multimodal schema guards already refuse these settings before doctor can run.
_MLX_NON_TEXT_SFT = tuple(
    entry for entry in _MLX_SFT
    if entry.field not in {
        "training.use_flash_attn", "training.use_liger", "training.moe_lora",
    }
) + _MLX_SELECTIVE_NON_TEXT


def _other_task_knob(field_name: str, trainer: str, task: str) -> SupportEntry:
    """A knob another task's trainer reads, accepted on ``task: sft`` and never read."""
    return SupportEntry(
        field_name,
        IGNORED,
        f"used by {trainer} (task: {task}); SFT does not read it",
        issue=1815,
        trainer_reads=False,
    )


# #1815: these other-task knobs are accepted on Transformers text SFT.
# The review covered the SFT setup path and its helpers, shared training CLI,
# data loading, and monitoring. Globally unconsumed settings are tracked separately.
_TRANSFORMERS_TEXT_SFT: tuple[SupportEntry, ...] = (
    _other_task_knob("training.dpo_beta", "trainer/dpo.py and trainer/online_dpo.py",
                     "dpo and online_dpo"),
    _other_task_knob("training.kto_beta", "trainer/kto.py", "kto"),
    _other_task_knob("training.orpo_beta", "trainer/orpo.py", "orpo"),
    _other_task_knob("training.simpo_gamma", "trainer/simpo.py", "simpo"),
    _other_task_knob("training.cpo_alpha", "trainer/simpo.py", "simpo"),
    _other_task_knob("training.ipo_tau", "trainer/ipo.py", "ipo"),
    _other_task_knob("training.bco_beta", "trainer/bco.py", "bco"),
    _other_task_knob("training.grpo_beta", "trainer/grpo.py", "grpo"),
    _other_task_knob("training.num_generations", "trainer/grpo.py", "grpo"),
    _other_task_knob("training.reward_fn", "trainer/grpo.py and trainer/ppo.py",
                     "grpo and ppo"),
    _other_task_knob("training.verifiable_domain", "trainer/grpo.py and trainer/ppo.py",
                     "grpo and ppo"),
    _other_task_knob("training.ppo_epochs", "trainer/ppo.py", "ppo"),
    _other_task_knob("training.ppo_clip_ratio", "trainer/ppo.py", "ppo"),
    _other_task_knob("training.ppo_kl_penalty", "trainer/ppo.py", "ppo"),
    _other_task_knob("training.reward_model", "trainer/ppo.py and trainer/online_dpo.py",
                     "ppo and online_dpo"),
    _other_task_knob("training.embedding_loss", "trainer/embedding.py", "embedding"),
    _other_task_knob("training.embedding_margin", "trainer/embedding.py", "embedding"),
    _other_task_knob("training.embedding_pooling", "trainer/embedding.py", "embedding"),
    _other_task_knob("training.embedding_temperature", "trainer/embedding.py", "embedding"),
)


#: ``(task, backend, modality)`` -> the settings that combination does not honour.
#: An empty table is reviewed and has no known gaps. A missing key may use the text fallback.
REGISTRY: dict[tuple[str, str, str], tuple[SupportEntry, ...]] = {
    ("sft", "mlx", "text"): _MLX_SFT,
    ("sft", "mlx", "vision"): _MLX_NON_TEXT_SFT,
    ("sft", "mlx", "audio"): _MLX_NON_TEXT_SFT,
    ("dpo", "mlx", "text"): _MLX_DPO,
    ("orpo", "mlx", "text"): _MLX_ORPO,
    ("simpo", "mlx", "text"): _MLX_SIMPO,
    ("kto", "mlx", "text"): _MLX_KTO,
    ("grpo", "mlx", "text"): _MLX_GRPO,
    ("sft", "unsloth", "text"): _UNSLOTH_SFT,
    ("sft", "transformers", "text"): _TRANSFORMERS_TEXT_SFT,
    ("sft", "transformers", "vision"): _TRANSFORMERS_VISION_SFT,
    ("sft", "transformers", "audio"): _TRANSFORMERS_AUDIO_SFT,
}

_TRANSFORMERS_SFT_MODULES = (
    "soup_cli/trainer/sft.py",
    "soup_cli/trainer/save_args.py",
    "soup_cli/trainer/loss_summary.py",
    "soup_cli/trainer/stream_setup.py",
    "soup_cli/trainer/rewind_hf.py",
    "soup_cli/trainer/raft.py",
)

#: Every module a reviewed combination can read config through -- the trainer plus the
#: helpers it delegates to. The guard scans the **union**.
#:
#: Naming only the trainer was a real hole, found by mutation on #756: the
#: schedule fields #734 wired appear 7-12 times in ``mlx_optim.py`` and twice in
#: ``mlx_sft.py``. The guard caught that drift only because #734 happened to
#: leave ``warmup_ratio=`` and ``weight_decay=`` visible at the call site. A
#: field wired entirely inside a helper would have passed.
TRAINER_MODULES: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("sft", "mlx", "text"): (
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_full.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/mlx_masking.py",
        "soup_cli/trainer/mlx_pretokenized.py",
        "soup_cli/trainer/rewind_mlx.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("dpo", "mlx", "text"): (
        "soup_cli/trainer/mlx_dpo.py",
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/_trl_compat.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("simpo", "mlx", "text"): (
        "soup_cli/trainer/mlx_simpo.py",
        "soup_cli/trainer/mlx_orpo.py",
        "soup_cli/trainer/mlx_dpo.py",
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/_trl_compat.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("orpo", "mlx", "text"): (
        "soup_cli/trainer/mlx_orpo.py",
        "soup_cli/trainer/mlx_dpo.py",
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/_trl_compat.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("kto", "mlx", "text"): (
        "soup_cli/trainer/mlx_kto.py",
        "soup_cli/trainer/mlx_orpo.py",
        "soup_cli/trainer/mlx_dpo.py",
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/_trl_compat.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    ("grpo", "mlx", "text"): (
        "soup_cli/trainer/mlx_grpo.py",
        "soup_cli/trainer/mlx_dpo.py",
        "soup_cli/trainer/mlx_sft.py",
        "soup_cli/trainer/mlx_optim.py",
        "soup_cli/trainer/grpo.py",
        "soup_cli/trainer/rewards.py",
        "soup_cli/trainer/_trl_compat.py",
        "soup_cli/trainer/loss_summary.py",
    ),
    # utils/unsloth.py handles the loader path. trainer/sft.py::_setup_unsloth is not listed
    # here because listing sft.py would cause the drift guard to inspect transformers-only
    # precision reads; an AST test in test_issue1124 guards _setup_unsloth instead.
    ("sft", "unsloth", "text"): ("soup_cli/utils/unsloth.py",),
    ("sft", "transformers", "text"): _TRANSFORMERS_SFT_MODULES,
    ("sft", "transformers", "vision"): _TRANSFORMERS_SFT_MODULES,
    ("sft", "transformers", "audio"): _TRANSFORMERS_SFT_MODULES,
}
for _modality in ("vision", "audio"):
    TRAINER_MODULES[("sft", "mlx", _modality)] = TRAINER_MODULES[("sft", "mlx", "text")]


def _registry_key(task: str, backend: str, modality: str) -> tuple[str, str, str] | None:
    """The REGISTRY key that covers this combination, or None when it is not reviewed."""
    if (task, backend, modality) in REGISTRY:
        return (task, backend, modality)
    # An explicit modality table takes precedence over the shared text policy.
    if backend in ("mlx", "unsloth") and (task, backend, DEFAULT_MODALITY) in REGISTRY:
        return (task, backend, DEFAULT_MODALITY)
    return None


def is_reviewed(task: str, backend: str, modality: str = DEFAULT_MODALITY) -> bool:
    """Whether this combination's table was reviewed (#1815).

    ``unsupported_for`` returns ``()`` both for a reviewed combination that
    ignores nothing and for one nobody reviewed; this tells them apart.
    """
    return _registry_key(task, backend, modality) is not None


def unsupported_for(
    task: str, backend: str, modality: str = DEFAULT_MODALITY
) -> tuple[SupportEntry, ...]:
    """Every declared gap for a task/backend/modality combination, or ``()`` if unreviewed."""
    key = _registry_key(task, backend, modality)
    return REGISTRY[key] if key is not None else ()


def check_config(cfg: "SoupConfig") -> list[SupportEntry]:
    """The declared gaps that apply to the fields this config actually sets.

    Only non-default fields the user wrote and switched on are reported. Pydantic's
    ``model_fields_set`` distinguishes the written fields from the ones sitting
    at their schema default, which is the difference between a useful pre-flight
    check and a wall of 275 rows. ``entry.active`` then drops a field written in
    its off position (e.g. ``false``). A written schema default counts as
    unwritten, including a default of ``true``.
    """
    backend = getattr(cfg, "backend", DEFAULT_BACKEND)
    modality = getattr(cfg, "modality", DEFAULT_MODALITY)
    task: str = cfg.task
    if backend == "mlx":
        from soup_cli.config.schema import resolve_mlx_task

        task = resolve_mlx_task(task, cfg.training.preference_loss)
    entries = unsupported_for(task, backend, modality)
    if not entries:
        return []

    written: dict[str, Any] = {}
    for namespace in ("training", "data"):
        section = getattr(cfg, namespace, None)
        if section is None:
            continue
        for name in getattr(section, "model_fields_set", ()):
            value = getattr(section, name)
            schema_field = type(section).model_fields[name]
            if value != schema_field.get_default(call_default_factory=True):
                written[f"{namespace}.{name}"] = value

    return [
        entry for entry in entries if entry.field in written and entry.active(written[entry.field])
    ]
