"""Universal Logit Distillation (ULD) — v0.70.0 Part B.

Cross-tokenizer knowledge distillation extending v0.53.2 DistillTrainer
to teacher/student pairs with **different vocabularies** (e.g. Llama →
Mistral, Llama → Qwen, Qwen → Llama). The existing v0.53.2 path
assumes column-wise logit alignment, which fails the moment student
and teacher have different vocab sizes.

Two strategies ship in v0.70.0 (schema-only; live projection module
wired in v0.70.1):

- ``wasserstein``: Boizard et al. 2024 ULD — uses 1D Wasserstein
  distance between sorted teacher / student logit distributions. No
  alignment required; works across arbitrary vocab boundaries.
  (arXiv 2402.12030)

- ``topk_align``: Top-K projection — pick top-K teacher logits, map
  via BPE-overlap heuristic to student token ids, distil only on the
  aligned subset. Requires ``top_k`` to be set.

- ``wasserstein_aligned`` (v0.71.18 #258): for *fully disjoint* tokenizers
  (different BPE merges, e.g. Llama -> GPT-2) the two token sequences are
  first aligned over their decoded character spans, the teacher logits are
  mean-pooled onto the student positions, then the sorted-W1 is applied.

Live wiring landed in v0.71.11 (#236) for wasserstein / topk_align; the
aligned strategy lands in v0.71.18 (#258).

Security:
- Closed allowlist (frozenset); arbitrary strategy rejected.
- Bool / null-byte / non-string / oversize rejection on every validator.
- Vocab-size bounds: [1, 262144] (covers multilingual SentencePiece +
  GPT-OSS 200K vocab; matches v0.42 token-cap policy).
- ``top_k`` mutually exclusive with non-topk_align strategies (silent
  no-op footgun rejection mirroring v0.52 distill / classifier policy).
- No top-level torch import — lazy import inside ``build_uld_projection``.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Optional

_MAX_STRATEGY_NAME_LEN = 32
# 262144 covers multilingual SentencePiece (e.g. NLLB) + GPT-OSS 200K.
_MAX_VOCAB_SIZE = 262144
# v0.71.18 #258 — alignment caps. Token/char alignment is O(N·M) in the
# offset-overlap case; cap both so a pathological sequence can't blow up.
_MAX_ALIGN_TOKENS = 1024
_MAX_ALIGN_CHARS = 8192

SUPPORTED_ULD_STRATEGIES: frozenset[str] = frozenset(
    {"wasserstein", "topk_align", "wasserstein_aligned"}
)


@dataclass(frozen=True)
class ULDStrategySpec:
    """Static metadata for a ULD strategy."""

    name: str
    description: str
    requires_top_k: bool


_STRATEGY_METADATA = types.MappingProxyType({
    "wasserstein": ULDStrategySpec(
        name="wasserstein",
        description=(
            "1D Wasserstein distance on sorted logit distributions. "
            "No alignment required across vocabularies."
        ),
        requires_top_k=False,
    ),
    "topk_align": ULDStrategySpec(
        name="topk_align",
        description=(
            "Top-K teacher logit alignment via BPE overlap. Requires "
            "top_k to be set."
        ),
        requires_top_k=True,
    ),
    "wasserstein_aligned": ULDStrategySpec(
        name="wasserstein_aligned",
        description=(
            "Wasserstein-1 after aligning the student/teacher token "
            "sequences over their decoded character spans (handles fully "
            "disjoint tokenizers, e.g. Llama -> GPT-2). No top_k required."
        ),
        requires_top_k=False,
    ),
})


def validate_uld_strategy(name: object) -> str:
    """Validate and normalise a ULD strategy name.

    Returns the canonical (lower-cased) name. Raises ``ValueError`` on
    any failure.
    """
    if isinstance(name, bool):
        raise ValueError("uld_strategy must be a string, got bool")
    if not isinstance(name, str):
        raise ValueError(
            f"uld_strategy must be a string, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("uld_strategy must be a non-empty string")
    if "\x00" in name:
        raise ValueError("uld_strategy must not contain null bytes")
    if len(name) > _MAX_STRATEGY_NAME_LEN:
        raise ValueError(
            f"uld_strategy exceeds {_MAX_STRATEGY_NAME_LEN} chars"
        )
    normalised = name.lower()
    if normalised not in SUPPORTED_ULD_STRATEGIES:
        raise ValueError(
            f"uld_strategy={name!r} is not supported. "
            f"Valid: {sorted(SUPPORTED_ULD_STRATEGIES)}"
        )
    return normalised


def get_strategy_spec(name: str) -> ULDStrategySpec:
    """Return the :class:`ULDStrategySpec` for ``name``."""
    normalised = validate_uld_strategy(name)
    return _STRATEGY_METADATA[normalised]


def validate_uld_projection_dim(value: object) -> int:
    """Validate a projection dimensionality (vocab size).

    Bounds: ``[1, _MAX_VOCAB_SIZE=262144]``. Bool rejected per project
    bool-as-int policy.
    """
    if isinstance(value, bool):
        raise ValueError("dim must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"dim must be int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"dim must be >= 1, got {value}")
    if value > _MAX_VOCAB_SIZE:
        raise ValueError(
            f"dim={value} exceeds {_MAX_VOCAB_SIZE} cap"
        )
    return value


def validate_uld_top_k(value: object) -> int:
    """Validate ``top_k`` for the topk_align strategy.

    Bounds: ``[1, _MAX_VOCAB_SIZE=262144]``. Bool rejected.
    """
    if isinstance(value, bool):
        raise ValueError("uld_top_k must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"uld_top_k must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(f"uld_top_k must be >= 1, got {value}")
    if value > _MAX_VOCAB_SIZE:
        raise ValueError(
            f"uld_top_k={value} exceeds {_MAX_VOCAB_SIZE} cap"
        )
    return value


@dataclass(frozen=True)
class ULDConfig:
    """Frozen ULD configuration.

    - ``strategy``: one of :data:`SUPPORTED_ULD_STRATEGIES`.
    - ``student_vocab_size`` / ``teacher_vocab_size``: positive ints,
      bounded by ``_MAX_VOCAB_SIZE``.
    - ``top_k``: required when ``strategy='topk_align'``, rejected
      otherwise (silent no-op footgun rejection).
    """

    strategy: str
    student_vocab_size: int
    teacher_vocab_size: int
    top_k: Optional[int] = None

    def __post_init__(self) -> None:
        normalised = validate_uld_strategy(self.strategy)
        if normalised != self.strategy:
            object.__setattr__(self, "strategy", normalised)
        validate_uld_projection_dim(self.student_vocab_size)
        validate_uld_projection_dim(self.teacher_vocab_size)
        spec = _STRATEGY_METADATA[normalised]
        if spec.requires_top_k:
            if self.top_k is None:
                raise ValueError(
                    f"uld_strategy='{normalised}' requires top_k to be set"
                )
            validate_uld_top_k(self.top_k)
        elif self.top_k is not None:
            raise ValueError(
                f"top_k is only valid when uld_strategy='topk_align'; "
                f"got uld_strategy='{normalised}'"
            )


def _sorted_w1(p_s_sorted, p_t_sorted):
    """1D Wasserstein-1 between two descending-sorted prob tensors.

    Pads the shorter vocab to the longer with zeros, then returns the L1
    norm of the CDF difference (the closed-form W1 for distributions on a
    discrete line). Both inputs are ``[..., V*]`` (last dim = sorted
    probabilities summing to ~1). Returns ``[...]`` (per-position W1).
    """
    import torch

    v_s = p_s_sorted.shape[-1]
    v_t = p_t_sorted.shape[-1]
    common = max(v_s, v_t)
    if v_s < common:
        p_s_sorted = torch.nn.functional.pad(p_s_sorted, (0, common - v_s))
    if v_t < common:
        p_t_sorted = torch.nn.functional.pad(p_t_sorted, (0, common - v_t))
    cdf_s = torch.cumsum(p_s_sorted, dim=-1)
    cdf_t = torch.cumsum(p_t_sorted, dim=-1)
    return (cdf_s - cdf_t).abs().sum(dim=-1)


def uld_distill_loss(
    student_logits,
    teacher_logits,
    *,
    config: ULDConfig,
    attention_mask=None,
):
    """Cross-tokenizer ULD distillation loss (v0.71.11 #236).

    Computes a vocab-mismatch-tolerant distillation loss between
    ``student_logits`` ``[B, T, Vs]`` and ``teacher_logits`` ``[B, T, Vt]``
    where ``Vs`` and ``Vt`` may differ. Two strategies:

    - ``wasserstein``: softmax both, sort descending, pad to the common
      vocab length, take the 1D Wasserstein-1 (L1 of the CDF difference)
      between the sorted distributions. No alignment required — this is
      the Boizard et al. 2024 ULD surrogate.
    - ``topk_align``: take the top-``top_k`` probabilities of each model
      (rank-aligned, renormalised) and the same sorted-W1 between them.
      Distils only on the high-probability subset.

    Differentiable w.r.t. the student logits (sort / topk both use
    gather). Returns a scalar mean over the (masked) token positions.
    """
    import torch

    if not isinstance(config, ULDConfig):
        raise TypeError(
            f"config must be ULDConfig, got {type(config).__name__}"
        )
    p_s = torch.softmax(student_logits, dim=-1)
    p_t = torch.softmax(teacher_logits, dim=-1)

    if config.strategy == "topk_align":
        k = int(config.top_k)
        k_s = min(k, p_s.shape[-1])
        k_t = min(k, p_t.shape[-1])
        s_top, _ = torch.topk(p_s, k_s, dim=-1)
        t_top, _ = torch.topk(p_t, k_t, dim=-1)
        # Renormalise the truncated top-k so each sums to ~1.
        s_top = s_top / s_top.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        t_top = t_top / t_top.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        per_pos = _sorted_w1(s_top, t_top)
    else:
        # wasserstein — full sorted distributions.
        s_sorted, _ = torch.sort(p_s, dim=-1, descending=True)
        t_sorted, _ = torch.sort(p_t, dim=-1, descending=True)
        per_pos = _sorted_w1(s_sorted, t_sorted)

    if attention_mask is not None:
        mask = attention_mask.to(per_pos.dtype)
        denom = mask.sum().clamp(min=1.0)
        return (per_pos * mask).sum() / denom
    return per_pos.mean()


class ULDProjection:
    """Callable wrapper around :func:`uld_distill_loss` (v0.71.11 #236).

    Holds the frozen :class:`ULDConfig` and exposes ``__call__`` so the
    distill trainer can swap in the cross-tokenizer ULD loss without
    re-validating per step. There is no learned projection matrix — the
    ULD surrogate operates directly on sorted logit distributions, which
    is the whole point (no alignment matrix to fit).
    """

    def __init__(self, config: ULDConfig) -> None:
        if not isinstance(config, ULDConfig):
            raise TypeError(
                f"config must be ULDConfig, got {type(config).__name__}"
            )
        self.config = config

    def __call__(self, student_logits, teacher_logits, *, attention_mask=None):
        return uld_distill_loss(
            student_logits,
            teacher_logits,
            config=self.config,
            attention_mask=attention_mask,
        )


def build_uld_projection(config) -> "ULDProjection":
    """Build the live cross-tokenizer ULD projection (v0.71.11 #236).

    Lifts the v0.70.0 ``NotImplementedError`` stub. Validates the config
    type at the public boundary (fail-fast policy), then returns a
    :class:`ULDProjection` callable that computes the ULD distillation
    loss for a (student_logits, teacher_logits) pair.
    """
    if not isinstance(config, ULDConfig):
        raise TypeError(
            f"config must be ULDConfig, got {type(config).__name__}"
        )
    return ULDProjection(config)


# ---------------------------------------------------------------------------
# v0.71.18 #258 — token-sequence alignment for fully-disjoint tokenizers.
#
# The v0.71.11 wasserstein / topk_align paths compare the student and
# teacher distributions per-position assuming the two token sequences line
# up. For genuinely different tokenizations (Llama -> GPT-2) the sequences
# have different lengths AND different token boundaries, so they must be
# aligned over their decoded text first. Both sides decode to (nearly) the
# same underlying text, so we align by character span; for decode artefacts
# we fall back to a difflib (Ratcliff-Obershelp) character match — the
# stdlib analog of the Levenshtein/DTW alignment from the ULD paper
# (Boizard et al. 2024, arXiv:2402.12030).
# ---------------------------------------------------------------------------
def _char_spans(tokens):
    """Concatenate per-token decoded strings; return (text, char_spans).

    ``char_spans[i]`` is ``(start, end)`` of token ``i`` in the joined text.
    Non-string tokens are ``str()``-coerced (defence-in-depth).
    """
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for tok in tokens:
        s = tok if isinstance(tok, str) else str(tok)
        spans.append((pos, pos + len(s)))
        parts.append(s)
        pos += len(s)
    return "".join(parts), spans


def _char_map(a: str, b: str) -> list[int]:
    """Map each char index of ``a`` to an aligned char index of ``b``.

    Uses :class:`difflib.SequenceMatcher` matching blocks. Unmatched
    positions map to ``-1``. ``autojunk=False`` so long runs aren't
    silently treated as junk.
    """
    import difflib

    amap = [-1] * len(a)
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for i, j, n in matcher.get_matching_blocks():
        for k in range(n):
            amap[i + k] = j + k
    return amap


def align_token_sequences(student_tokens, teacher_tokens):
    """Align student token positions to teacher token positions.

    Returns ``list[list[int]]`` — for each student position, the (sorted)
    list of teacher positions whose decoded character span aligns to it.

    When both token sequences decode to the *same* text (the common
    cross-tokenizer-on-identical-input case) this is an exact character
    offset-overlap. When the decoded texts differ (decode artefacts) it
    falls back to a difflib character match. Both inputs are truncated to
    ``_MAX_ALIGN_TOKENS`` so the O(N·M) overlap stays bounded.
    """
    s_tokens = list(student_tokens)[:_MAX_ALIGN_TOKENS]
    t_tokens = list(teacher_tokens)[:_MAX_ALIGN_TOKENS]
    if not s_tokens:
        return []
    if not t_tokens:
        return [[] for _ in s_tokens]

    s_text, s_spans = _char_spans(s_tokens)
    t_text, t_spans = _char_spans(t_tokens)
    # Cap the char strings so the difflib fallback stays bounded.
    s_text = s_text[:_MAX_ALIGN_CHARS]
    t_text = t_text[:_MAX_ALIGN_CHARS]

    if s_text == t_text:
        result: list[list[int]] = []
        for (s, e) in s_spans:
            aligned = [
                ti for ti, (t, f) in enumerate(t_spans) if s < f and t < e
            ]
            result.append(aligned)
        return result

    # Different decoded text → difflib character alignment.
    cmap = _char_map(s_text, t_text)
    result = []
    for (s, e) in s_spans:
        tset = set()
        for c in range(s, min(e, len(cmap))):
            tc = cmap[c]
            if tc < 0:
                continue
            for ti, (t, f) in enumerate(t_spans):
                if t <= tc < f:
                    tset.add(ti)
                    break
        result.append(sorted(tset))
    return result


def aggregate_aligned_logits(teacher_logits, alignment):
    """Mean-pool teacher logits onto student positions via ``alignment``.

    ``teacher_logits`` is ``[Tt, Vt]`` (a single sequence). ``alignment``
    is ``list[list[int]]`` of length ``Ts``. Returns ``[Ts, Vt]`` — the
    mean of the aligned teacher logits per student position; positions with
    no aligned teacher token are zeros. Out-of-range teacher indices are
    ignored (defence-in-depth against a stale alignment).
    """

    ts = len(alignment)
    vt = teacher_logits.shape[-1]
    tt = teacher_logits.shape[0]
    out = teacher_logits.new_zeros((ts, vt))
    for i, idxs in enumerate(alignment):
        valid = [j for j in idxs if 0 <= j < tt]
        if valid:
            out[i] = teacher_logits[valid].mean(dim=0)
    return out


def uld_aligned_loss(
    student_logits,
    teacher_logits,
    student_tokens,
    teacher_tokens,
    *,
    config: ULDConfig,
    attention_mask=None,
):
    """Wasserstein-1 ULD loss with token-sequence alignment (v0.71.18 #258).

    Handles fully-disjoint tokenizers: for each batch element the student
    and teacher token sequences are aligned over their decoded character
    spans, the teacher logits are mean-pooled onto the student positions,
    and the existing sorted-Wasserstein-1 surrogate is applied per student
    position. Differentiable w.r.t. the student logits.

    The alignment is a heuristic (offset-overlap when both sides decode to the
    same text, difflib Ratcliff-Obershelp char matching otherwise), so a
    student token may be mean-pooled against several teacher tokens. That is an
    accuracy surrogate, not an exact 1:1 map.

    Args:
        student_logits: ``[B, Ts, Vs]``.
        teacher_logits: ``[B, Tt, Vt]`` (may differ in length AND vocab).
        student_tokens: per-batch list of student per-token decoded strings.
            Must have at least ``B`` entries.
        teacher_tokens: per-batch list of teacher per-token decoded strings.
            Must have at least ``B`` entries.
        config: a :class:`ULDConfig` (strategy ``wasserstein_aligned``).
        attention_mask: optional ``[B, Ts]`` student mask (1 = real token).
    """
    import torch

    if not isinstance(config, ULDConfig):
        raise TypeError(
            f"config must be ULDConfig, got {type(config).__name__}"
        )
    batch = student_logits.shape[0]
    if len(student_tokens) < batch or len(teacher_tokens) < batch:
        raise ValueError(
            "student_tokens / teacher_tokens must each have at least "
            f"{batch} entries (logit batch dim), got "
            f"{len(student_tokens)} / {len(teacher_tokens)}"
        )
    per_seq = []
    for b in range(batch):
        align = align_token_sequences(student_tokens[b], teacher_tokens[b])
        agg_t = aggregate_aligned_logits(teacher_logits[b], align)  # [Ts, Vt]
        ts = min(agg_t.shape[0], student_logits.shape[1])
        if ts == 0:
            per_seq.append(student_logits.new_zeros(()))
            continue
        s = student_logits[b, :ts]
        te = agg_t[:ts]
        p_s = torch.softmax(s, dim=-1)
        p_t = torch.softmax(te, dim=-1)
        s_sorted, _ = torch.sort(p_s, dim=-1, descending=True)
        t_sorted, _ = torch.sort(p_t, dim=-1, descending=True)
        per_pos = _sorted_w1(s_sorted, t_sorted)  # [Ts]
        if attention_mask is not None:
            mask = attention_mask[b, :ts].to(per_pos.dtype)
            denom = mask.sum().clamp(min=1.0)
            per_seq.append((per_pos * mask).sum() / denom)
        else:
            per_seq.append(per_pos.mean())
    return torch.stack(per_seq).mean()
