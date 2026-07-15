"""LoRA adapter task-vector arithmetic (v0.71.34).

Task arithmetic (arXiv:2212.04089) applied to LoRA adapters: add / scale /
NEGATE task vectors via an expression such as ``"coder + 0.5*math - toxic"``.

The engine is **signed, un-normalized, element-wise** over the intersection of
``lora_A`` / ``lora_B`` tensor names (mirrors PEFT ``combination_type="linear"``):
``out[k] = Σ cᵢ·tensorᵢ[k]``. Same-rank inputs only — a shape mismatch on a shared
tensor is a rank mismatch and is rejected loudly (harmonize rank first). Exact
concatenation+SVD arithmetic for mixed-rank adapters is a future enhancement.

No top-level torch/transformers/peft — the parser + numpy merge stay light.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MAX_EXPR_LEN = 4096
_MAX_TERMS = 64
_MAX_ADAPTER_CONFIG_BYTES = 256 * 1024

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


def read_adapter_base(adapter_dir: str) -> str | None:
    """Read ``base_model_name_or_path`` from an adapter's ``adapter_config.json``.

    Returns ``None`` when the config is absent or the field is missing. The read
    is symlink-rejecting (O_NOFOLLOW) and size-capped at 256 KiB (mirrors
    ``adapter_merge.write_merged_adapter``'s config-read guards).
    """
    cfg_path = Path(adapter_dir) / "adapter_config.json"
    if not os.path.lexists(str(cfg_path)):
        return None
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
    if not isinstance(data, dict):
        return None
    base = data.get("base_model_name_or_path")
    return base if isinstance(base, str) else None
