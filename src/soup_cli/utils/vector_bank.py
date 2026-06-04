"""VeRA / VB-LoRA vector-bank storage format (v0.67.0 schema / v0.71.12 #221 live).

A vector bank is the data structure behind multi-tenant LoRA serving at
MB-per-user instead of hundreds-of-MB per LoRA:

    Bank = {
        shared random projection matrix P : (d_model × d_model),
        per-user scaling vector v_u : (vector_dim,),
    }

The per-user delta at inference time is ``v_u ⊙ Px`` (VeRA) or a
codebook lookup (VB-LoRA). Storage size is dominated by ``vector_dim``
per user — a 128-D vector at fp32 is 512 bytes vs ~30 MB for a rank-16
LoRA on a 7B model.

v0.67.0 shipped the schema + atomic disk I/O + validators with a deferred
``apply_bank_to_serve`` stub; v0.71.12 #221 lifts the stub to a live
``LoadedVectorBank`` (reconstructed ``P`` + per-user vectors) with a
decode-time forward hook (``install_serve_hook``) wired into
``soup serve --bank`` (per-request user via the ``X-User-Id`` header).

Public surface:

- ``VectorBank`` / ``BankEntry`` frozen dataclasses
- ``validate_bank_name`` / ``validate_user_id`` / ``validate_scaling_vector``
- ``estimate_bank_size(num_users, vector_dim)`` -> bytes
- ``write_bank(bank, path)`` / ``load_bank(path)`` atomic JSON I/O
- ``reconstruct_projection(seed, dim)`` -> torch tensor
- ``LoadedVectorBank`` runtime bank + ``apply_bank_to_serve(bank)``
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, Iterable, Tuple

from soup_cli.utils.paths import (
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bounds (closed, locked at module load)
# ---------------------------------------------------------------------------

MAX_NAME_LEN = 128
MAX_USER_ID_LEN = 256
MAX_BASE_MODEL_LEN = 512
MAX_VECTOR_DIM = 16_384
MIN_VECTOR_DIM = 1
MAX_ENTRIES_PER_BANK = 1_000_000
_MAX_FILE_BYTES = 16 * 1024 * 1024  # 16 MiB cap on bank file size
# Sanity cap on the serve-time steering magnitude (matches the spirit of the
# v0.62.0 steering ±10 cap, but generous — banks may want stronger personas).
_MAX_BANK_STRENGTH = 100.0

# kebab-case + `._-`, leading alnum, no path separators / shell metacharacters
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]{0,127}$")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_bank_name(name: object) -> str:
    """Canonical kebab-case bank name (case-insensitive)."""
    if isinstance(name, bool):
        raise TypeError("name must not be bool")
    if not isinstance(name, str):
        raise TypeError(
            f"name must be str, got {type(name).__name__}"
        )
    if not name:
        raise ValueError("name must be non-empty")
    if "\x00" in name:
        raise ValueError("name must not contain null bytes")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(
            f"name length {len(name)} > {MAX_NAME_LEN}"
        )
    canonical = name.lower()
    if not _NAME_RE.match(canonical):
        raise ValueError(
            f"name must be kebab-case alphanumeric + `._-`, got {name!r}"
        )
    return canonical


def validate_user_id(user_id: object) -> str:
    if isinstance(user_id, bool):
        raise TypeError("user_id must not be bool")
    if not isinstance(user_id, str):
        raise TypeError("user_id must be str")
    if not user_id:
        raise ValueError("user_id must be non-empty")
    if "\x00" in user_id:
        raise ValueError("user_id must not contain null bytes")
    if len(user_id) > MAX_USER_ID_LEN:
        raise ValueError(
            f"user_id length {len(user_id)} > {MAX_USER_ID_LEN}"
        )
    return user_id


def validate_scaling_vector(values: object) -> Tuple[float, ...]:
    """Per-user scaling vector — non-empty, finite, ≤MAX_VECTOR_DIM."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError("scaling vector must be a non-string iterable")
    out: list[float] = []
    for v in values:
        if isinstance(v, bool):
            raise TypeError("scaling vector entries must not be bool")
        if not isinstance(v, (int, float)):
            raise TypeError("scaling vector entries must be numeric")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError("scaling vector entries must be finite")
        out.append(f)
    if not out:
        raise ValueError("scaling vector must be non-empty")
    if len(out) > MAX_VECTOR_DIM:
        raise ValueError(
            f"scaling vector length {len(out)} > {MAX_VECTOR_DIM}"
        )
    return tuple(out)


def _validate_base_model(base_model: object) -> str:
    if isinstance(base_model, bool):
        raise TypeError("base_model must not be bool")
    if not isinstance(base_model, str):
        raise TypeError("base_model must be str")
    if not base_model:
        raise ValueError("base_model must be non-empty")
    if "\x00" in base_model:
        raise ValueError("base_model must not contain null bytes")
    if len(base_model) > MAX_BASE_MODEL_LEN:
        raise ValueError(
            f"base_model length {len(base_model)} > {MAX_BASE_MODEL_LEN}"
        )
    return base_model


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("projection_seed must be int")
    if seed < 0:
        raise ValueError("projection_seed must be non-negative")
    if seed > 2**63 - 1:
        raise ValueError("projection_seed too large")
    return seed


def _validate_vector_dim(dim: object) -> int:
    if isinstance(dim, bool) or not isinstance(dim, int):
        raise TypeError("vector_dim must be int")
    if dim < MIN_VECTOR_DIM:
        raise ValueError(
            f"vector_dim {dim} below floor {MIN_VECTOR_DIM}"
        )
    if dim > MAX_VECTOR_DIM:
        raise ValueError(
            f"vector_dim {dim} above cap {MAX_VECTOR_DIM}"
        )
    return dim


# ---------------------------------------------------------------------------
# Frozen dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BankEntry:
    """One per-user scaling vector in the bank."""

    user_id: str
    scaling: Tuple[float, ...]

    def __post_init__(self) -> None:
        # Re-validate so callers constructing BankEntry directly cannot
        # smuggle in invalid state (matches v0.61 EditPlan policy).
        validate_user_id(self.user_id)
        if not isinstance(self.scaling, tuple):
            raise TypeError("scaling must be tuple")
        validate_scaling_vector(self.scaling)


@dataclass(frozen=True)
class VectorBank:
    """A multi-tenant LoRA storage bank.

    ``projection_seed`` is the deterministic seed for the shared random
    projection P. Each ``BankEntry`` carries a per-user scaling vector
    of length ``vector_dim``.

    Live serving wiring is deferred to v0.67.1. The v0.67.0 surface is
    the schema + atomic disk I/O.
    """

    name: str
    base_model: str
    projection_seed: int
    vector_dim: int
    entries: Tuple[BankEntry, ...]

    def __post_init__(self) -> None:
        validate_bank_name(self.name)
        _validate_base_model(self.base_model)
        _validate_seed(self.projection_seed)
        _validate_vector_dim(self.vector_dim)
        if not isinstance(self.entries, tuple):
            raise TypeError("entries must be tuple")
        if len(self.entries) > MAX_ENTRIES_PER_BANK:
            raise ValueError(
                f"entries length {len(self.entries)} > "
                f"{MAX_ENTRIES_PER_BANK}"
            )
        for entry in self.entries:
            if not isinstance(entry, BankEntry):
                raise TypeError("entries must be BankEntry instances")
            if len(entry.scaling) != self.vector_dim:
                raise ValueError(
                    f"entry user_id={entry.user_id!r} scaling length "
                    f"{len(entry.scaling)} != vector_dim "
                    f"{self.vector_dim}"
                )


# ---------------------------------------------------------------------------
# Size estimate
# ---------------------------------------------------------------------------


def estimate_bank_size(*, num_users: int, vector_dim: int) -> int:
    """Estimate disk + memory cost of a bank in bytes.

    The shared projection matrix dominates for small ``num_users``;
    per-user scaling dominates as the bank fills up. Both are fp32
    (4 bytes/element).
    """
    if isinstance(num_users, bool) or not isinstance(num_users, int):
        raise TypeError("num_users must be int")
    if num_users < 0:
        raise ValueError("num_users must be non-negative")
    if isinstance(vector_dim, bool) or not isinstance(vector_dim, int):
        raise TypeError("vector_dim must be int")
    if vector_dim < 1:
        raise ValueError("vector_dim must be positive")
    projection_bytes = vector_dim * vector_dim * 4  # P : (d×d) fp32
    user_bytes = num_users * vector_dim * 4         # v_u : (d,) fp32
    return projection_bytes + user_bytes


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------


def _bank_to_dict(bank: VectorBank) -> dict:
    return {
        "name": bank.name,
        "base_model": bank.base_model,
        "projection_seed": bank.projection_seed,
        "vector_dim": bank.vector_dim,
        "entries": [
            {"user_id": e.user_id, "scaling": list(e.scaling)}
            for e in bank.entries
        ],
    }


def _bank_from_dict(data: object) -> VectorBank:
    if not isinstance(data, dict):
        raise ValueError("bank file root must be JSON object")
    name = data.get("name")
    base_model = data.get("base_model")
    seed = data.get("projection_seed")
    dim = data.get("vector_dim")
    raw_entries = data.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("entries field must be a list")
    entries: list[BankEntry] = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("each entry must be a JSON object")
        user_id = raw.get("user_id")
        scaling = raw.get("scaling")
        if not isinstance(scaling, list):
            raise ValueError("scaling must be a list")
        entries.append(
            BankEntry(
                user_id=user_id if isinstance(user_id, str) else "",
                scaling=tuple(validate_scaling_vector(scaling)),
            )
        )
    return VectorBank(
        name=name if isinstance(name, str) else "",
        base_model=base_model if isinstance(base_model, str) else "",
        projection_seed=seed if isinstance(seed, int) else 0,
        vector_dim=dim if isinstance(dim, int) else 0,
        entries=tuple(entries),
    )


def write_bank(bank: VectorBank, path: str) -> str:
    """Atomically write a bank to JSON under cwd containment."""
    if not isinstance(bank, VectorBank):
        raise TypeError("bank must be VectorBank")
    text = json.dumps(_bank_to_dict(bank), indent=2, sort_keys=True)
    return atomic_write_text(text, path, field="bank path")


def load_bank(path: str) -> VectorBank:
    """Read + validate a bank JSON. cwd-contained, symlink-rejected."""
    if not isinstance(path, str):
        raise TypeError("path must be str")
    if not path:
        raise ValueError("path must be non-empty")
    if "\x00" in path:
        raise ValueError("path must not contain null bytes")
    enforce_under_cwd_and_no_symlink(path, field="bank path")
    # Size cap defence (matches v0.55+ policy)
    real = os.path.realpath(path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"bank file not found: {path!r}")
    st = os.lstat(real)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("bank path must not be a symlink (TOCTOU defence)")
    if st.st_size > _MAX_FILE_BYTES:
        raise ValueError(
            f"bank file size {st.st_size} > {_MAX_FILE_BYTES}"
        )
    with open(real, encoding="utf-8") as fh:
        data = json.load(fh)
    return _bank_from_dict(data)


# ---------------------------------------------------------------------------
# Live serving wiring (v0.71.12 #221)
# ---------------------------------------------------------------------------


def reconstruct_projection(seed: int, dim: int):
    """Deterministically rebuild the shared projection matrix ``P`` (``[dim, dim]``).

    Seeded with ``projection_seed`` so every server reconstructs the same ``P``
    without storing the (dim × dim) matrix on disk. Returns a float32 torch
    tensor. Lazy torch import.
    """
    import torch

    seed = _validate_seed(seed)
    dim = _validate_vector_dim(dim)
    gen = torch.Generator().manual_seed(int(seed))
    # 1/sqrt(dim) scale keeps Px in a sane magnitude regardless of dim.
    return torch.randn(dim, dim, generator=gen) * (dim ** -0.5)


class LoadedVectorBank:
    """Runtime VeRA / VB-LoRA bank — reconstructed ``P`` + per-user vectors.

    The per-token delta for the active user ``u`` is ``v_u ⊙ (x @ Pᵀ)`` where
    ``x`` is a token hidden state of dim ``vector_dim``. ``set_active_user``
    selects which user the decode hook applies; an unknown / unset user is a
    no-op (zero delta).
    """

    def __init__(self, bank: VectorBank):
        import torch

        if not isinstance(bank, VectorBank):
            raise TypeError("bank must be VectorBank")
        self.name = bank.name
        self.base_model = bank.base_model
        self.vector_dim = bank.vector_dim
        self.projection = reconstruct_projection(
            bank.projection_seed, bank.vector_dim
        )
        self._user_vectors = {
            e.user_id: torch.tensor(e.scaling, dtype=torch.float32)
            for e in bank.entries
        }
        self._active_user: str | None = None

    def has_user(self, user_id: object) -> bool:
        return isinstance(user_id, str) and user_id in self._user_vectors

    def set_active_user(self, user_id: object) -> bool:
        """Select the active user for the decode hook. Returns ``True`` if known."""
        if isinstance(user_id, str) and user_id in self._user_vectors:
            self._active_user = user_id
            return True
        self._active_user = None
        return False

    def delta_for_user(self, user_id: str, hidden):
        """Per-token delta ``v_u ⊙ (hidden @ Pᵀ)`` for a known user.

        ``hidden`` is ``[..., vector_dim]``. Raises ``KeyError`` for an unknown
        user (callers that prefer a no-op use :meth:`set_active_user` + the hook).
        """
        if user_id not in self._user_vectors:
            raise KeyError(f"unknown user_id {user_id!r}")
        v = self._user_vectors[user_id]
        proj = self.projection.to(dtype=hidden.dtype, device=hidden.device)
        v = v.to(dtype=hidden.dtype, device=hidden.device)
        projected = hidden @ proj.transpose(0, 1)
        return v * projected

    def install_serve_hook(self, model: Any, *, layer: int = -1, strength: float = 1.0):
        """Install a decode-time forward hook applying the active user's delta.

        Adds ``strength * delta_for_user(active, x)`` to the chosen decoder
        layer's residual-stream output. When no active user is set (or the
        hidden dim does not match ``vector_dim``) the hook is a no-op.

        Returns the hook handle so the caller can ``.remove()`` it on shutdown.
        """
        if isinstance(strength, bool) or not isinstance(strength, (int, float)):
            raise TypeError("strength must be numeric")
        strength_f = float(strength)
        if not math.isfinite(strength_f):
            raise ValueError("strength must be finite")
        if abs(strength_f) > _MAX_BANK_STRENGTH:
            raise ValueError(
                f"bank strength magnitude {strength_f} exceeds "
                f"{_MAX_BANK_STRENGTH} (sanity cap)"
            )

        from soup_cli.utils.edit_kernels import _locate_decoder_layers

        layers = _locate_decoder_layers(model)
        n_layers = len(layers)
        idx = layer if layer >= 0 else n_layers + layer
        if idx < 0 or idx >= n_layers:
            raise ValueError(
                f"bank serve layer {layer} out of range for {n_layers}-layer model"
            )
        block = layers[idx]
        bank_ref = self
        warned = {"dim": False}

        def _hook(_mod, _args, output):
            active = bank_ref._active_user
            if active is None:
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not hasattr(hidden, "shape") or hidden.shape[-1] != bank_ref.vector_dim:
                # Hidden-dim mismatch is a misconfiguration, not a legit no-op:
                # warn once so the operator does not silently see zero effect.
                if not warned["dim"]:
                    warned["dim"] = True
                    logger.warning(
                        "vector bank %r vector_dim=%d does not match the layer "
                        "hidden size; the serve hook is a no-op. Check --bank "
                        "matches the served model.",
                        bank_ref.name,
                        bank_ref.vector_dim,
                    )
                return output
            delta = strength_f * bank_ref.delta_for_user(active, hidden)
            new_hidden = hidden + delta
            if isinstance(output, (tuple, list)):
                return (new_hidden,) + tuple(output[1:])
            return new_hidden

        return block.register_forward_hook(_hook)


def apply_bank_to_serve(bank: VectorBank, *, server: Any = None) -> "LoadedVectorBank":
    """Apply a vector bank to a running ``soup serve`` instance (v0.71.12 #221).

    Reconstructs the shared projection ``P`` from ``projection_seed`` and builds
    the per-user scaling map, returning a :class:`LoadedVectorBank`. The caller
    (``soup serve --bank``) installs the decode hook and selects the active user
    per request via the ``X-User-Id`` header.

    Memory cost ≈ ``vector_dim²`` (the shared P) + ``N × vector_dim`` (per-user
    vectors) fp32 — far below ``N × rank × hidden`` for native per-user LoRA.
    """
    if not isinstance(bank, VectorBank):
        raise TypeError("bank must be VectorBank")
    return LoadedVectorBank(bank)
