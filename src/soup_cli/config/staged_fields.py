"""Detect config fields staged for features that have not landed (#808).

Warn-then-refuse across two releases following the #627 pattern. While warning
(v0.76, or ``STAGED_FIELD_SEVERITY = "warn"`` in ``config/loader.py``), each of
the staged fields prints:
    <field> is accepted but read by nothing; v<STAGED_FIELD_REJECTION_VERSION> will refuse it
Once refused (as of v0.77, the shipped setting), the load stops with one line per field:
    <field> is accepted but read by nothing; refused as of v<...>. Delete the line.

The rejection version is pinned to :data:`STAGED_FIELD_REJECTION_VERSION` ("0.77").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STAGED_FIELD_REJECTION_VERSION: str = "0.77"

__all__ = [
    "STAGED_FIELDS",
    "STAGED_FIELD_REJECTION_VERSION",
    "StagedField",
    "find_staged_config_fields",
    "format_staged_fields",
]


@dataclass(frozen=True)
class StagedField:
    """A staged config field provided in the user's config."""

    section: str
    name: str
    value: Any

    @property
    def path(self) -> str:
        return f"{self.section}.{self.name}"


#: Schema defaults for fields accepted by the schema but not consumed by any trainer
#: or data pipeline. A field is only flagged if the user provides a value that
#: DIFFERS from its schema default.
#:
#: Note: Training intelligence parent enable flags (``forgetting_detection``,
#: ``checkpoint_intelligence``, ``early_stop_on_regression``,
#: ``convergence_detection``) continue to be reported by ``soup train``'s
#: live honesty guard. The tunables themselves (Group B) are staged here (#808),
#: so non-default values are refused at load as of v0.77 (they warned in v0.76).
STAGED_FIELDS: dict[tuple[str, str], Any] = {
    ("training", "long_context_grpo"): False,
    ("training", "vision_grpo"): False,
    ("training", "quantize_ref_model"): False,
    ("training", "grace_codebook"): False,
    ("training", "grace_codebook_size"): None,
    ("training", "grace_codebook_dim"): None,
    # #761: documented, validated, and read by nothing.
    ("training", "lr_groups"): None,
    ("training", "early_stop_patience"): 2,
    ("training", "citation_recall_threshold"): None,
    # #808 Group B: training-intelligence and convergence tunables
    ("training", "forgetting_eval_steps"): 100,
    ("training", "forgetting_benchmark"): "mini_mmlu",
    ("training", "forgetting_stop"): False,
    ("training", "checkpoint_eval_steps"): 200,
    ("training", "checkpoint_eval_metric"): "composite",
    ("training", "checkpoint_eval_tasks"): None,
    ("training", "checkpoint_keep_top"): 3,
    ("training", "convergence_window"): 50,
    ("training", "convergence_rel_tol"): 0.005,
    # #808 item 2: accepted, but no save writes reference-model or rollout-buffer state.
    ("training", "rl_checkpoint_include_ref_model"): False,
    ("training", "rl_checkpoint_include_rollout_buffer"): False,
    ("data", "video_dir"): None,
    ("data", "video_fps"): None,
    ("data", "video_maxlen"): None,
    ("data", "image_min_pixels"): None,
    ("data", "image_max_pixels"): None,
    ("data", "image_resize_algorithm"): None,
    ("data", "split_thinking"): False,
    ("data", "eval_on_each_dataset"): False,
    ("data", "resize_vocab"): False,
    ("data", "extend_conversation"): False,
    ("data", "skip_prepare_dataset"): False,
}


def _coerce_staged_value(section: str, name: str, val: Any) -> Any:
    """Coerce a raw config value through the schema field's type if needed (#808)."""
    if val is None:
        return None
    from pydantic import TypeAdapter, ValidationError

    from soup_cli.config.schema import DataConfig, TrainingConfig

    model_cls = TrainingConfig if section == "training" else DataConfig
    field_info = model_cls.model_fields.get(name)
    if field_info is None:
        return val
    try:
        return TypeAdapter(field_info.annotation).validate_python(val)
    except ValidationError:
        return val


def find_staged_config_fields(
    raw: dict, config: Any | None = None
) -> list[StagedField]:
    """Find any staged config field set to a non-default value in the raw mapping.

    If a field is omitted or explicitly set to its schema default value, it is
    not reported.
    """
    if not isinstance(raw, dict):
        return []

    from soup_cli.config.schema import remap_root_level_misplaced_keys

    normalised = remap_root_level_misplaced_keys(raw)
    found: list[StagedField] = []
    for (section, name), default_val in STAGED_FIELDS.items():
        sec_dict = normalised.get(section)
        if isinstance(sec_dict, dict) and name in sec_dict:
            raw_val = sec_dict[name]
            if config is not None:
                sec_obj = getattr(config, section, None)
                val = getattr(sec_obj, name, raw_val) if sec_obj is not None else raw_val
            else:
                val = _coerce_staged_value(section, name, raw_val)
            if val != default_val:
                found.append(StagedField(section=section, name=name, value=val))
    return found


def format_staged_fields(
    staged: list[StagedField], *, include_deadline: bool = True
) -> str:
    """Format staged fields warning or refusal message.

    Per maintainer directive on #808:
    While warning, each staged field prints:
        <field> is accepted but read by nothing; v<STAGED_FIELD_REJECTION_VERSION> will refuse it
    Once refused, each line also says what to do about it:
        <field> is accepted but read by nothing; refused as of v<...>. Delete the line.
    """
    lines = []
    for item in staged:
        if include_deadline:
            lines.append(
                f"{item.path} is accepted but read by nothing; "
                f"v{STAGED_FIELD_REJECTION_VERSION} will refuse it"
            )
        else:
            lines.append(
                f"{item.path} is accepted but read by nothing; "
                f"refused as of v{STAGED_FIELD_REJECTION_VERSION}. Delete the line."
            )
    return "\n".join(lines)
