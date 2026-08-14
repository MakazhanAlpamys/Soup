"""LoRA adapter task-vector arithmetic (v0.71.34).

Task arithmetic (arXiv:2212.04089) applied to LoRA adapters: add / scale /
NEGATE task vectors via an expression such as ``"coder + 0.5*math - toxic"``.

Two paths:

* Same-rank (common case): a **signed, un-normalized, element-wise** combine over
  the intersection of ``lora_A`` / ``lora_B`` tensor names (mirrors PEFT
  ``combination_type="linear"``): ``out[k] = Σ cᵢ·tensorᵢ[k]`` with a ``√|c|``
  factor split so the reconstructed ΔW scales linearly.
* Mixed-rank (#305): an exact **concatenation** combine (PEFT
  ``combination_type="cat"``) — stack the factors so ``B_out @ A_out =
  Σ cᵢ·(Bᵢ@Aᵢ)`` exactly for any per-adapter rank, with an optional truncated-SVD
  refactor (``rank=``) to cap the concatenated output rank. See
  :func:`merge_task_arithmetic_concat`.

No top-level torch/transformers/peft — the parser + numpy merge stay light.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MAX_EXPR_LEN = 4096
_MAX_TERMS = 64
_MAX_COEFF = 1e6
_MAX_ADAPTER_CONFIG_BYTES = 256 * 1024
# DoS cap on the concatenated rank of any one module (Σ rᵢ) in the concat path.
# Real LoRA ranks are 8-256; even 64 adapters at rank 64 = 4096. A single
# module with an abnormally high rank would otherwise force every OTHER module
# to zero-pad up to it (uniform-rank output), amplifying a <1 GB input adapter
# into tens of GB. Above this, refuse and tell the operator to pass --rank
# (which SVD-truncates to a bounded k). Mirrors neat_packing._MAX_MASK_ELEMENTS.
_MAX_CONCAT_RANK = 4096
# Aggregate ceiling on total emitted tensor elements. The per-module rank cap
# bounds a single module, but uniform-rank padding across MANY modules still
# amplifies (one high-rank module forces every module to pad up), so bound the
# whole output too. 2**31 elements ≈ 8.6 GB (float32) — well above any legit
# full-model LoRA merge (rank-256 full model ≈ 0.5 B elements) but a hard wall
# against the padding-amplification DoS. Mirrors neat_packing._MAX_MASK_ELEMENTS.
_MAX_OUTPUT_ELEMENTS = 2**31

_NAME_RE = re.compile(r"[A-Za-z0-9_.\-]+")
_FLOAT_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?")


@dataclass(frozen=True)
class TaskTerm:
    """A single ``coeff * adapter`` term of a task-arithmetic expression."""

    name: str
    coeff: float


@dataclass(frozen=True)
class ArithmeticReport:
    """Result of a ``soup adapters arithmetic`` run."""

    expression: str
    terms: tuple[TaskTerm, ...]
    output_dir: str
    merged_layers: int
    skipped_layers: tuple[str, ...]
    base_model: str | None


def parse_expression(expr: str, known_names: set[str]) -> list[TaskTerm]:
    """Parse a task-arithmetic expression into signed ``TaskTerm``s.

    Grammar (NO ``eval``): ``expr := term (('+'|'-') term)*`` where
    ``term := [sign] [coeff '*'] name`` (also ``name ['*' coeff]``). Signs fold
    into the coefficient; an omitted coefficient is ``1.0``. Duplicate adapter
    names sum their coefficients; a term summing to ``0.0`` is dropped.

    Raises:
        ValueError: empty / over-length expression, unknown token, adapter name
            not in ``known_names``, non-finite coefficient, all terms cancel,
            or more than ``_MAX_TERMS`` distinct adapters.
    """
    if not isinstance(expr, str):
        raise TypeError("expression must be a string")
    s = expr.strip()
    if not s:
        raise ValueError("empty expression")
    if len(s) > _MAX_EXPR_LEN:
        raise ValueError(f"expression too long (> {_MAX_EXPR_LEN} chars)")

    n = len(s)
    i = 0
    coeffs: dict[str, float] = {}
    order: list[str] = []
    seen_term = False

    def _skip_ws() -> None:
        nonlocal i
        while i < n and s[i] in " \t":
            i += 1

    _skip_ws()
    while i < n:
        _skip_ws()
        if i >= n:
            break
        # Leading sign(s) — only valid before the first term or between terms.
        sign = 1.0
        if s[i] in "+-":
            while i < n and s[i] in "+- \t":
                if s[i] == "-":
                    sign = -sign
                i += 1
        elif seen_term:
            # Two terms with no operator between them → malformed.
            raise ValueError(
                f"expected '+' or '-' between terms at pos {i}: {s[i:i + 8]!r}"
            )
        _skip_ws()
        if i >= n:
            raise ValueError("expression ends with a dangling operator")

        coeff = 1.0
        name: str | None = None
        m = _FLOAT_RE.match(s, i)
        if m:
            coeff = float(m.group())
            i = m.end()
            _skip_ws()
            if i < n and s[i] == "*":
                i += 1
                _skip_ws()
            nm = _NAME_RE.match(s, i)
            if not nm:
                raise ValueError(
                    f"expected adapter name after coefficient at pos {i}"
                )
            name = nm.group()
            i = nm.end()
        else:
            nm = _NAME_RE.match(s, i)
            if not nm:
                raise ValueError(
                    f"unexpected token at pos {i}: {s[i:i + 8]!r}"
                )
            name = nm.group()
            i = nm.end()
            _skip_ws()
            if i < n and s[i] == "*":
                i += 1
                _skip_ws()
                cm = _FLOAT_RE.match(s, i)
                if not cm:
                    raise ValueError(
                        f"expected coefficient after '*' at pos {i}"
                    )
                coeff = float(cm.group())
                i = cm.end()

        if not math.isfinite(coeff):
            raise ValueError("coefficient must be finite")
        if abs(coeff) > _MAX_COEFF:
            raise ValueError(
                f"coefficient magnitude {coeff} exceeds cap {_MAX_COEFF} "
                "(would overflow the float32 output)"
            )
        if name not in known_names:
            raise ValueError(
                f"unknown adapter name {name!r} "
                f"(declare it with --adapter {name}=<path>)"
            )
        signed = sign * coeff
        if name not in coeffs:
            coeffs[name] = 0.0
            order.append(name)
            if len(order) > _MAX_TERMS:
                raise ValueError(f"too many distinct terms (> {_MAX_TERMS})")
        coeffs[name] += signed
        seen_term = True
        _skip_ws()

    if not seen_term:
        raise ValueError("expression has no terms")
    terms = [TaskTerm(nm, coeffs[nm]) for nm in order if coeffs[nm] != 0.0]
    if not terms:
        raise ValueError("all terms cancelled to zero — nothing to merge")
    return terms


def _factor_coeff(name: str, c: float) -> float:
    """Per-factor coefficient so the reconstructed delta scales *linearly*.

    A LoRA contributes ``ΔW = B @ A``. Applying the raw coefficient ``c`` to
    both ``lora_A`` and ``lora_B`` would scale ``ΔW`` by ``c²`` (so negation is
    a no-op and 0.5·adapter halves twice). Instead split the magnitude as
    ``√|c|`` across both factors and carry the sign on ``lora_B`` only, so the
    self (diagonal) term of the reconstruction is exactly ``c·B@A``
    (``√|c| · sign(c)√|c| = c``). Mirrors PEFT's ``combination_type='linear'``
    task-arithmetic. Non-LoRA tensors (biases / direct deltas) scale linearly
    by ``c``.
    """
    lname = name.lower()
    if "lora_a" in lname or "lora_embedding_a" in lname:
        return math.sqrt(abs(c))
    if "lora_b" in lname or "lora_embedding_b" in lname:
        return math.copysign(math.sqrt(abs(c)), c)
    return float(c)


def merge_task_arithmetic(
    weights_list: Sequence[Mapping[str, Any]],
    coeffs: Sequence[float],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Signed task-vector combine over the intersection of tensor names.

    Per adapter ``i`` and tensor ``k`` the effective factor coefficient is
    :func:`_factor_coeff` of ``coeffs[i]`` so that the reconstructed LoRA delta
    ``B_out @ A_out`` scales linearly with each coefficient (negation flips the
    delta, ``0.5·`` halves it — not the ``c²`` a naive element-wise sum gives).
    Names present in only some adapters are reported in ``skipped``. A shape
    mismatch on a *shared* name is a rank mismatch and raises (same-rank
    contract).
    """
    import numpy as np

    if len(weights_list) != len(coeffs):
        raise ValueError(
            f"weights_list ({len(weights_list)}) and coeffs "
            f"({len(coeffs)}) length mismatch"
        )
    if not weights_list:
        raise ValueError("need at least one adapter")

    shared = set(weights_list[0].keys())
    all_keys = set(weights_list[0].keys())
    for w in weights_list[1:]:
        shared &= set(w.keys())
        all_keys |= set(w.keys())

    merged: dict[str, Any] = {}
    for name in sorted(shared):
        tensors = [np.asarray(w[name], dtype=np.float64) for w in weights_list]
        if len({t.shape for t in tensors}) > 1:
            raise ValueError(
                f"rank/shape mismatch on {name!r} across adapters — task "
                f"arithmetic requires same-rank adapters (harmonize the LoRA "
                f"rank first, or merge with `soup adapters merge --strategy svd`)"
            )
        acc = np.zeros_like(tensors[0])
        for c, t in zip(coeffs, tensors):
            acc += _factor_coeff(name, float(c)) * t
        merged[name] = acc.astype(np.float32)

    skipped = tuple(sorted(all_keys - shared))
    return merged, skipped


def _read_adapter_config_dict(adapter_dir: str) -> dict | None:
    """Read + parse an adapter's ``adapter_config.json`` into a dict.

    Returns ``None`` when the config file is absent or its top-level JSON value
    is not an object. The read is symlink-rejecting (O_NOFOLLOW + explicit
    ``S_ISLNK`` for Windows), size-capped at 256 KiB, and containment-checked
    under cwd — mirrors ``adapter_merge.write_merged_adapter``'s config-read
    guards. Shared by :func:`read_adapter_base` and
    :func:`read_adapter_lora_scaling`.

    Raises:
        ValueError: dir outside cwd / symlinked / unreadable / oversize config /
            malformed JSON.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    # Defence-in-depth: re-validate the adapter dir stays under cwd (callers
    # already check, but these readers are public) and is not a symlink.
    enforce_under_cwd_and_no_symlink(adapter_dir, "adapter")
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not os.path.lexists(str(cfg_path)):
        return None
    # O_NOFOLLOW is a no-op on Windows, so explicitly reject a symlinked config
    # file (an attacker-shipped adapter could point it at an arbitrary file).
    if stat.S_ISLNK(os.lstat(str(cfg_path)).st_mode):
        raise ValueError("adapter_config.json must not be a symlink")
    try:
        fd = os.open(str(cfg_path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(
            f"adapter_config.json unreadable: {type(exc).__name__}"
        ) from exc
    fh = None
    try:
        if os.fstat(fd).st_size > _MAX_ADAPTER_CONFIG_BYTES:
            raise ValueError(
                f"adapter_config.json exceeds {_MAX_ADAPTER_CONFIG_BYTES} byte cap"
            )
        fh = os.fdopen(fd, "r", encoding="utf-8")
        raw = fh.read()
    finally:
        if fh is not None:
            fh.close()
        else:
            os.close(fd)
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"adapter_config.json is not valid JSON: {exc}") from exc
    return data if isinstance(data, dict) else None


def read_adapter_base(adapter_dir: str) -> str | None:
    """Read ``base_model_name_or_path`` from an adapter's ``adapter_config.json``.

    Returns ``None`` when the config is absent or the field is missing. The read
    is symlink-rejecting (O_NOFOLLOW) and size-capped at 256 KiB (mirrors
    ``adapter_merge.write_merged_adapter``'s config-read guards).
    """
    data = _read_adapter_config_dict(adapter_dir)
    if data is None:
        return None
    base = data.get("base_model_name_or_path")
    return base if isinstance(base, str) else None


def read_adapter_lora_scaling(adapter_dir: str) -> float | None:
    """Return a LoRA adapter's decode-time scaling ``lora_alpha / r``.

    Returns ``None`` when the config is absent, ``r`` / ``lora_alpha`` are
    missing or non-numeric, or ``r <= 0`` (a degenerate config the caller
    should treat as "unknown scaling", falling back to ``1.0``). Used by the
    concat path so a mixed-rank merge can bake each adapter's own scaling into
    its block (mirrors PEFT ``add_weighted_adapter`` cat, which folds
    ``weight * scaling`` into ``lora_A``).
    """
    data = _read_adapter_config_dict(adapter_dir)
    if data is None:
        return None
    r = data.get("r")
    alpha = data.get("lora_alpha")
    if isinstance(r, bool) or isinstance(alpha, bool):
        return None
    if not isinstance(r, (int, float)) or not isinstance(alpha, (int, float)):
        return None
    if r <= 0:
        return None
    scaling = float(alpha) / float(r)
    return scaling if math.isfinite(scaling) else None


# ---------------------------------------------------------------------------
# Mixed-rank concat+SVD path (#305) — PEFT ``combination_type='cat'`` style
# ---------------------------------------------------------------------------
_LORA_ROLE_SUFFIXES: tuple[tuple[str, str], ...] = (
    (".lora_A.weight", "A"),
    (".lora_B.weight", "B"),
    (".lora_embedding_A", "A"),
    (".lora_embedding_B", "B"),
)


def _lora_role(name: str) -> tuple[str, str, str] | None:
    """Classify a tensor name as a LoRA factor.

    Returns ``(stem, role, suffix)`` where ``role`` is ``"A"`` / ``"B"`` and
    ``suffix`` is the matched suffix (so the emitted tensor keeps the adapter's
    own naming convention), or ``None`` when the name is not a LoRA factor.
    """
    for suffix, role in _LORA_ROLE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)], role, suffix
    return None


def _detect_lora_rank(weights: Mapping[str, Any]) -> int | None:
    """Infer the LoRA rank of an adapter from its first ``lora_A`` tensor.

    ``lora_A`` has shape ``(r, in)`` so ``shape[0]`` is the rank. Returns
    ``None`` when the adapter has no LoRA factors (non-LoRA / dense delta).
    """
    import numpy as np

    for name, tensor in weights.items():
        role = _lora_role(name)
        if role is not None and role[1] == "A":
            arr = np.asarray(tensor)
            if arr.ndim >= 1:
                return int(arr.shape[0])
    return None


def _validate_optional_rank(rank: int | None) -> int | None:
    if rank is None:
        return None
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise ValueError("rank must be an int")
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return rank


def merge_task_arithmetic_concat(
    weights_list: Sequence[Mapping[str, Any]],
    coeffs: Sequence[float],
    scalings: Sequence[float] | None = None,
    rank: int | None = None,
) -> tuple[dict[str, Any], tuple[str, ...], int]:
    """Exact concatenation-based task arithmetic over MIXED-rank LoRA adapters.

    For each LoRA module shared (as a complete A/B pair) across every adapter,
    stack the factors so the reconstructed delta is exact regardless of per-
    adapter rank (PEFT ``combination_type='cat'``)::

        A_out = [ s₁·A₁ ; s₂·A₂ ; ... ]   (rows = Σ rᵢ, cols = in)
        B_out = [ c₁·B₁ | c₂·B₂ | ... ]   (rows = out, cols = Σ rᵢ)
        B_out @ A_out = Σ cᵢ·sᵢ·(Bᵢ @ Aᵢ)   exactly

    ``coeffs[i]`` is folded into ``Bᵢ`` and the per-adapter scaling ``scalings[i]``
    (``lora_alpha/r``, default ``1.0``) into ``Aᵢ`` — so a single-scaling output
    adapter (``lora_alpha = r_out``) reproduces ``Σ cᵢ·(effective ΔWᵢ)``. Negation
    is exact (flips the ``Bᵢ`` block). When ``rank`` is given AND smaller than the
    concatenated rank ``Σ rᵢ``, each module's delta is refactored to the best
    rank-``rank`` approximation via truncated SVD.

    A LoRA ``adapter_config.json`` carries a single scalar ``r``, but different
    modules can end up at different ranks (heterogeneous input ``rank_pattern``,
    or ``rank`` truncating only the modules whose concat rank exceeds it). To
    stay loadable, every module is zero-padded to the global max rank — the zero
    block contributes nothing to ``B @ A``, so the reconstruction stays exact —
    and ``new_rank`` is that uniform max. Non-LoRA tensors (bias /
    ``modules_to_save``) present in every adapter with a matching shape are
    combined linearly ``Σ cᵢ·tᵢ`` (mirrors the element-wise path).

    Returns ``(merged, skipped, new_rank)`` where ``new_rank`` is the uniform rank
    of the emitted adapter so the caller can patch ``adapter_config.json``'s
    ``r`` / ``lora_alpha``. Tensor names present in only some adapters (LoRA
    factors whose pair is not shared everywhere, or shape-mismatched non-LoRA
    names) are reported in ``skipped``.

    Raises:
        ValueError: length mismatch, no adapters, invalid ``rank``, or an
            in-/out-dim mismatch on a shared LoRA module (a genuine shape
            conflict, not a rank difference — those the concat path handles).
    """
    import numpy as np

    rank = _validate_optional_rank(rank)
    if len(weights_list) != len(coeffs):
        raise ValueError(
            f"weights_list ({len(weights_list)}) and coeffs "
            f"({len(coeffs)}) length mismatch"
        )
    if not weights_list:
        raise ValueError("need at least one adapter")
    if scalings is None:
        scalings = [1.0] * len(weights_list)
    elif len(scalings) != len(weights_list):
        raise ValueError(
            f"scalings ({len(scalings)}) and weights_list "
            f"({len(weights_list)}) length mismatch"
        )

    # Per-adapter: stem -> {"A": name, "B": name}; and a flat set of all names.
    per_adapter_pairs: list[dict[str, dict[str, tuple[str, str]]]] = []
    for weights in weights_list:
        pairs: dict[str, dict[str, tuple[str, str]]] = {}
        for name in weights:
            role = _lora_role(name)
            if role is None:
                continue
            stem, which, suffix = role
            pairs.setdefault(stem, {})[which] = (name, suffix)
        per_adapter_pairs.append(pairs)

    # Stems that are a COMPLETE (A and B) pair in EVERY adapter.
    shared_stems: set[str] | None = None
    for pairs in per_adapter_pairs:
        complete = {s for s, roles in pairs.items() if "A" in roles and "B" in roles}
        shared_stems = complete if shared_stems is None else (shared_stems & complete)
    shared_stems = shared_stems or set()

    all_keys: set[str] = set()
    merged_keys: set[str] = set()
    for weights in weights_list:
        all_keys |= set(weights.keys())

    # First pass: build each stem's concatenated (a_out, b_out); stems may end
    # up at DIFFERENT ranks (heterogeneous input rank_pattern, or --rank
    # truncating only the stems whose concat rank exceeds it). A LoRA adapter
    # config carries a single scalar ``r``, so a self-consistent, loadable output
    # must be uniform-rank — we zero-pad every stem to the global max in the
    # second pass (zeros contribute nothing to ``B @ A``, so the delta is exact).
    stem_results: list[tuple[str, str, Any, Any]] = []  # (a_key, b_key, a_out, b_out)
    max_stem_rank = 0
    for stem in sorted(shared_stems):
        a_blocks = []
        b_blocks = []
        a_suffix = per_adapter_pairs[0][stem]["A"][1]
        b_suffix = per_adapter_pairs[0][stem]["B"][1]
        in_dim: int | None = None
        out_dim: int | None = None
        stem_rank_sum = 0
        for idx, weights in enumerate(weights_list):
            a_name = per_adapter_pairs[idx][stem]["A"][0]
            b_name = per_adapter_pairs[idx][stem]["B"][0]
            a_mat = np.asarray(weights[a_name], dtype=np.float64)
            b_mat = np.asarray(weights[b_name], dtype=np.float64)
            if a_mat.ndim != 2 or b_mat.ndim != 2:
                raise ValueError(
                    f"LoRA factor for {stem!r} is not 2-D "
                    f"(A{a_mat.shape}, B{b_mat.shape})"
                )
            r_a, in_a = a_mat.shape
            out_b, r_b = b_mat.shape
            if r_a != r_b:
                raise ValueError(
                    f"A/B rank mismatch within adapter {idx} for {stem!r} "
                    f"(A rows {r_a} != B cols {r_b})"
                )
            if in_dim is None:
                in_dim, out_dim = in_a, out_b
            elif in_a != in_dim or out_b != out_dim:
                raise ValueError(
                    f"in/out dim mismatch on {stem!r} across adapters "
                    f"(expected in={in_dim}, out={out_dim}; got in={in_a}, "
                    f"out={out_b}) — task vectors are not comparable"
                )
            stem_rank_sum += r_a
            # Cap the concatenated rank BEFORE the np.concatenate allocation so a
            # single abnormally-high-rank module cannot balloon (and, via uniform
            # padding, force every other module to balloon) the output. Bounded
            # further downstream by --rank's SVD truncation, but the concat itself
            # must be bounded regardless of --rank.
            if stem_rank_sum > _MAX_CONCAT_RANK:
                raise ValueError(
                    f"concatenated rank for {stem!r} exceeds cap "
                    f"{_MAX_CONCAT_RANK} (Σ rᵢ > {_MAX_CONCAT_RANK}); pass "
                    "--rank N to SVD-truncate the merged adapter to a bounded rank"
                )
            a_blocks.append(float(scalings[idx]) * a_mat)
            b_blocks.append(float(coeffs[idx]) * b_mat)
        a_out = np.concatenate(a_blocks, axis=0)  # (Σr, in)
        b_out = np.concatenate(b_blocks, axis=1)  # (out, Σr)
        concat_rank = a_out.shape[0]
        if rank is not None and rank < concat_rank:
            delta = b_out @ a_out  # (out, in)
            u, s, vt = np.linalg.svd(delta, full_matrices=False)
            k = min(rank, s.shape[0])
            b_out = u[:, :k] * s[:k]  # (out, k)
            a_out = vt[:k, :]  # (k, in)
        max_stem_rank = max(max_stem_rank, a_out.shape[0])
        stem_results.append((stem + a_suffix, stem + b_suffix, a_out, b_out))

    new_rank = max_stem_rank
    # Bound the total emitted element count BEFORE the padding allocations, so
    # the padding-amplification DoS (one high-rank module forcing every module up
    # to new_rank) cannot exhaust memory even within the per-module cap.
    projected = sum(
        (a_out.shape[1] + b_out.shape[0]) * new_rank
        for _a, _b, a_out, b_out in stem_results
    )
    if projected > _MAX_OUTPUT_ELEMENTS:
        raise ValueError(
            f"merged adapter would emit ~{projected} tensor elements, exceeding "
            f"the cap {_MAX_OUTPUT_ELEMENTS}; pass --rank N to SVD-truncate to a "
            "bounded rank"
        )

    merged: dict[str, Any] = {}
    for a_key, b_key, a_out, b_out in stem_results:
        stem_rank = a_out.shape[0]
        if stem_rank < new_rank:
            # Pad to the uniform output rank with zero rows (A) / cols (B). The
            # zero block multiplies to nothing, so B_out @ A_out is unchanged.
            pad = new_rank - stem_rank
            a_out = np.pad(a_out, ((0, pad), (0, 0)))  # (new_rank, in)
            b_out = np.pad(b_out, ((0, 0), (0, pad)))  # (out, new_rank)
        merged[a_key] = a_out.astype(np.float32)
        merged[b_key] = b_out.astype(np.float32)
        merged_keys.add(a_key)
        merged_keys.add(b_key)

    # Non-LoRA tensors (bias / modules_to_save) present in EVERY adapter with a
    # matching shape are combined linearly ``Σ cᵢ·tᵢ`` — mirrors the element-wise
    # path's treatment of non-LoRA names, so third-party adapters with
    # ``bias != 'none'`` / ``modules_to_save`` are not silently dropped. Rank
    # factors are never padded here (these are full deltas, not LoRA A/B). Names
    # missing from some adapter, or shape-mismatched, fall through to ``skipped``.
    role_names: set[str] = set()
    for weights in weights_list:
        role_names |= {n for n in weights if _lora_role(n) is not None}
    non_lora_shared = set(weights_list[0].keys())
    for weights in weights_list[1:]:
        non_lora_shared &= set(weights.keys())
    for name in sorted(non_lora_shared - role_names):
        tensors = [np.asarray(w[name], dtype=np.float64) for w in weights_list]
        if len({t.shape for t in tensors}) > 1:
            continue  # shape mismatch -> report in skipped
        acc = np.zeros_like(tensors[0])
        for c, t in zip(coeffs, tensors):
            acc += float(c) * t
        merged[name] = acc.astype(np.float32)
        merged_keys.add(name)

    skipped = tuple(sorted(all_keys - merged_keys))
    return merged, skipped, new_rank
