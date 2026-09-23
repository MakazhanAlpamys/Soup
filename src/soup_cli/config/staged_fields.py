"""Detect config fields staged for features that have not landed (#808).

Warn-then-refuse across two releases following the #627 pattern:
While warning, each of the staged fields prints:
    <field> is accepted but read by nothing; v<STAGED_FIELD_REJECTION_VERSION> will refuse it

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
#: Note: Training intelligence tunables already reported by ``soup train``
#: (e.g. ``forgetting_*``, ``checkpoint_*``, ``early_stop_patience``) are excluded
#: here to avoid conflicting diagnostics and adhere to their respective issue ownership (#808).
STAGED_FIELDS: dict[tuple[str, str], Any] = {
    ("training", "long_context_grpo"): False,
    ("training", "vision_grpo"): False,
    ("training", "quantize_ref_model"): False,
    ("training", "grace_codebook"): False,
    ("training", "grace_codebook_size"): None,
    ("training", "grace_codebook_dim"): None,
    ("training", "convergence_window"): 50,
    ("training", "convergence_rel_tol"): 0.005,
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


def find_staged_config_fields(raw: dict) -> list[StagedField]:
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
            val = sec_dict[name]
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
                f"refused as of v{STAGED_FIELD_REJECTION_VERSION}"
            )
    return "\n".join(lines)
