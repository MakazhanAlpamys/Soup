"""v0.61.0 Part C — Knowledge editing (ROME / MEMIT / AlphaEdit).

Surgical locate-and-edit methods for patching factual associations
WITHOUT a full fine-tuning loop. Three method backends:

* ``rome`` — Rank-One Model Editing (Meng et al., 2022). Closed-form
  rank-1 weight update at a single MLP layer.
* ``memit`` — Mass-Editing Memory in a Transformer (Meng et al., 2023).
  Distributes the update across multiple layers for higher capacity.
* ``alphaedit`` — Null-space-projected variant (2024). Better survival
  across sequential edits.

Schema-only release: validators + plan dataclasses lock the CLI surface.
Live editing kernel + Registry attach land in v0.61.1 (mirrors v0.50.0
stub-then-live pattern). The CLI's ``--plan-only`` mode is the supported
exit-0 path until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover — type-only import, no runtime cost
    from soup_cli.utils.edit_governor import EditGovernor

SUPPORTED_EDIT_METHODS: frozenset[str] = frozenset(
    {"rome", "memit", "alphaedit", "grace"}
)

_MAX_METHOD_LEN: int = 32
_MAX_SUBJECT_LEN: int = 2048
_MAX_TARGET_LEN: int = 2048
_MAX_BASE_LEN: int = 512
_MAX_LAYER_IDX: int = 256  # Reject crazy-large layer indices.


@dataclass(frozen=True)
class EditMethodSpec:
    """Metadata for a knowledge-edit method backend."""

    name: str
    description: str
    multi_edit_capable: bool
    live_wired: bool


_EDIT_METHOD_METADATA: Mapping[str, EditMethodSpec] = MappingProxyType({
    "rome": EditMethodSpec(
        name="rome",
        description=(
            "Rank-One Model Editing — closed-form rank-1 update at a "
            "single MLP layer. Best for one-shot factual patches."
        ),
        multi_edit_capable=False,
        live_wired=False,
    ),
    "memit": EditMethodSpec(
        name="memit",
        description=(
            "Mass-Editing Memory in a Transformer — distributes the "
            "update across multiple layers for higher capacity."
        ),
        multi_edit_capable=True,
        live_wired=False,
    ),
    "alphaedit": EditMethodSpec(
        name="alphaedit",
        description=(
            "Null-space-projected ROME variant — survives sequential "
            "edits better than vanilla ROME / MEMIT."
        ),
        multi_edit_capable=True,
        live_wired=False,
    ),
    "grace": EditMethodSpec(
        name="grace",
        description=(
            "GRACE codebook — discrete latent-space (key, value) store "
            "that survives thousands of sequential edits without "
            "norm-blowup. v0.62.0 Part E ships the schema; live "
            "lookup / write kernel lands in v0.62.1."
        ),
        multi_edit_capable=True,
        live_wired=False,
    ),
})


def validate_edit_method(value: object) -> str:
    """Normalise + validate an edit-method name (case-insensitive)."""
    if isinstance(value, bool):
        raise TypeError("edit_method must not be bool")
    if not isinstance(value, str):
        raise TypeError(
            f"edit_method must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("edit_method must be non-empty")
    if "\x00" in value:
        raise ValueError("edit_method must not contain null bytes")
    if len(value) > _MAX_METHOD_LEN:
        raise ValueError(
            f"edit_method must be <= {_MAX_METHOD_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_EDIT_METHODS:
        supported = ", ".join(sorted(SUPPORTED_EDIT_METHODS))
        raise ValueError(
            f"unknown edit method {value!r}; supported: {supported}"
        )
    return canonical


def _validate_text_field(value: object, name: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{name} must not be bool")
    if not isinstance(value, str):
        raise TypeError(
            f"{name} must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{name} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{name} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{name} must be <= {max_len} chars")
    return value


def parse_edit_subject_target(*, subject: str, target: str) -> tuple[str, str]:
    """Validate the (subject, target) pair for an edit.

    The subject is the prefix sentence (e.g. "Paris is the capital of
    France"); the target is the new completion (e.g. "Lyon"). Both
    fields are length-capped + null-byte-rejected. Returns the
    validated pair unchanged.
    """
    s = _validate_text_field(subject, "subject", _MAX_SUBJECT_LEN)
    t = _validate_text_field(target, "target", _MAX_TARGET_LEN)
    return s, t


def _validate_layer(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("layer must not be bool")
    if not isinstance(value, int):
        raise TypeError(
            f"layer must be int, got {type(value).__name__}"
        )
    if value < 0:
        raise ValueError(f"layer must be >= 0, got {value}")
    if value > _MAX_LAYER_IDX:
        raise ValueError(
            f"layer must be <= {_MAX_LAYER_IDX}, got {value}"
        )
    return value


@dataclass(frozen=True)
class EditRequest:
    """Operator-supplied request for a single knowledge edit."""

    base: str
    method: str
    subject: str
    target: str
    layer: Optional[int]


@dataclass(frozen=True)
class EditPlan:
    """Resolved + validated edit plan ready for ``apply_edit``.

    Differs from :class:`EditRequest` in that ``method`` is canonical
    (lowercase) and ``layer`` is always concretely set (defaulted to
    the spec's recommended layer when ``None`` was supplied).
    """

    base: str
    method: str
    subject: str
    target: str
    layer: int
    spec: EditMethodSpec

    def __post_init__(self) -> None:
        # Re-validate so callers that bypass build_edit_plan can't
        # smuggle in an inconsistent plan.
        if self.method not in SUPPORTED_EDIT_METHODS:
            raise ValueError(
                f"method must be in {sorted(SUPPORTED_EDIT_METHODS)}, "
                f"got {self.method!r}"
            )
        if not isinstance(self.layer, int) or isinstance(self.layer, bool):
            raise TypeError("layer must be int (not bool)")
        if self.layer < 0 or self.layer > _MAX_LAYER_IDX:
            raise ValueError(
                f"layer must be in [0, {_MAX_LAYER_IDX}], got {self.layer}"
            )


# Per-method default edit layer (heuristic — operator can override via
# CLI). ROME papers target the middle-to-late MLP layer; AlphaEdit
# follows the same convention. MEMIT updates a range so we treat the
# "layer" arg as the centre of the spread.
_DEFAULT_EDIT_LAYER: Mapping[str, int] = MappingProxyType({
    "rome": 5,
    "memit": 8,
    "alphaedit": 5,
    # GRACE writes to a single dedicated codebook so the "layer" arg is
    # the residual-stream layer where the lookup hook is installed.
    # v0.62.0 Part E ships the schema; default mirrors AlphaEdit.
    "grace": 5,
})


def build_edit_plan(
    *,
    base: str,
    method: str,
    subject: str,
    target: str,
    layer: Optional[int] = None,
) -> EditPlan:
    """Resolve + validate an edit request into an :class:`EditPlan`.

    Validates every operator-supplied field at construction time so
    misconfigured edits fail fast (mirrors v0.50.0 build_verdict /
    v0.55.0 ``design_evals_from_data`` policy).
    """
    canonical_base = _validate_text_field(base, "base", _MAX_BASE_LEN)
    canonical_method = validate_edit_method(method)
    canonical_subject, canonical_target = parse_edit_subject_target(
        subject=subject, target=target,
    )
    canonical_layer = _validate_layer(layer)
    if canonical_layer is None:
        canonical_layer = _DEFAULT_EDIT_LAYER[canonical_method]
    spec = _EDIT_METHOD_METADATA[canonical_method]
    return EditPlan(
        base=canonical_base,
        method=canonical_method,
        subject=canonical_subject,
        target=canonical_target,
        layer=canonical_layer,
        spec=spec,
    )


@dataclass(frozen=True)
class EditResult:
    """Outcome of a live :func:`apply_edit` call (v0.71.9 #194)."""

    method: str
    layer: int
    norm_delta: float
    layers_edited: Tuple[int, ...]
    output_dir: Optional[str]
    target_prob_before: float
    target_prob_after: float
    governed: bool


def apply_edit(
    plan: EditPlan,
    *,
    output_dir: Optional[str] = None,
    governor: "Optional[EditGovernor]" = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    grad_steps: int = 25,
    lr: float = 0.5,
    cov_corpus: Optional[Sequence[str]] = None,
) -> EditResult:
    """Apply a knowledge edit live (v0.71.9 #194 / #197 / #203; v0.71.16 #250).

    ROME / MEMIT / AlphaEdit run the rank-1 weight-edit kernel; ``grace``
    builds a discrete codebook sidecar. When a ``governor`` is supplied it is
    consulted automatically (#197): :meth:`EditGovernor.check_can_edit` runs
    BEFORE the edit (raising :class:`GovernedEditError` on refusal) and
    :meth:`EditGovernor.record_edit` runs AFTER with the measured norm delta.

    ``cov_corpus`` (v0.71.16 #250) preconditions the ROME update with the key
    covariance estimated over the supplied prompts (``C^{-1} k*``). It is only
    valid for ``method='rome'`` — covariance preconditioning is the ROME
    closed-form update — and is rejected (fail-loud) for any other method.

    Re-validates the method so callers passing a duck-typed plan still hit a
    meaningful error before any model load. Returns an :class:`EditResult`.
    """
    method_attr = getattr(plan, "method", None)
    canonical = validate_edit_method(method_attr)

    # v0.71.16 #250 — reject --cov-corpus for non-ROME methods (fail-loud, no
    # silent no-op). Runs BEFORE any model load.
    if cov_corpus is not None and canonical != "rome":
        raise ValueError(
            "--cov-corpus is only supported for method='rome' (got "
            f"method={canonical!r}); covariance preconditioning is the ROME "
            "closed-form update."
        )

    # #197: consult the governor BEFORE doing any work. A refusal must abort
    # before we burn a model load.
    if governor is not None:
        governor.check_can_edit()

    if canonical == "grace":
        from soup_cli.utils.grace_codebook import apply_grace_edit

        return apply_grace_edit(
            plan,
            output_dir=output_dir,
            governor=governor,
            device=device,
            trust_remote_code=trust_remote_code,
            grad_steps=grad_steps,
            lr=lr,
        )

    from soup_cli.utils.edit_kernels import measure_target_prob, run_edit_kernel
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    model, tokenizer, dev = load_model_and_tokenizer(
        plan.base, device=device, trust_remote_code=trust_remote_code,
    )
    prob_before = measure_target_prob(
        model, tokenizer, subject=plan.subject, target=plan.target, device=dev,
    )
    kernel_result = run_edit_kernel(
        model, tokenizer,
        method=canonical, subject=plan.subject, target=plan.target,
        layer=plan.layer, device=dev, grad_steps=grad_steps, lr=lr,
        cov_corpus=cov_corpus,
    )
    prob_after = measure_target_prob(
        model, tokenizer, subject=plan.subject, target=plan.target, device=dev,
    )

    saved_dir: Optional[str] = None
    if output_dir is not None:
        saved_dir = _save_edited_model(model, tokenizer, output_dir)

    # #197: record the completed edit (governor persists the norm delta +
    # verdict so subsequent edits can be refused).
    if governor is not None:
        governor.record_edit(method=canonical, norm_delta=kernel_result.norm_delta)

    return EditResult(
        method=canonical,
        layer=kernel_result.layer,
        norm_delta=kernel_result.norm_delta,
        layers_edited=kernel_result.layers_edited,
        output_dir=saved_dir,
        target_prob_before=prob_before,
        target_prob_after=prob_after,
        governed=governor is not None,
    )


def _save_edited_model(model: object, tokenizer: object, output_dir: str) -> str:
    """Save an edited model + tokenizer to a cwd-contained ``output_dir``."""
    import os
    import stat

    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if "\x00" in output_dir:
        raise ValueError("output_dir must not contain null bytes")
    if not is_under_cwd(output_dir):
        raise ValueError(f"output_dir must stay under cwd: {output_dir!r}")
    # TOCTOU: reject a pre-placed symlink at output_dir (a symlinked dir would
    # let save_pretrained write weights outside cwd despite is_under_cwd).
    if os.path.lexists(output_dir) and stat.S_ISLNK(os.lstat(output_dir).st_mode):
        raise ValueError("output_dir must not be a symlink")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)  # type: ignore[attr-defined]
    tokenizer.save_pretrained(output_dir)  # type: ignore[attr-defined]
    return output_dir


def get_edit_method_spec(name: str) -> EditMethodSpec:
    """Return the frozen :class:`EditMethodSpec` for ``name`` or raise."""
    canonical = validate_edit_method(name)
    return _EDIT_METHOD_METADATA[canonical]
