"""v0.62.0 Part C — Activation steering (CAA / ITI / RepE).

Three control-vector backends for inference-time intervention:

* ``caa`` — Contrastive Activation Addition (Panickssery et al., 2023).
  Add a contrastive vector to the residual stream.
* ``iti`` — Inference-Time Intervention (Li et al., 2023). Shift specific
  attention heads.
* ``repe`` — Representation Engineering (Zou et al., 2023). PCA-based
  direction in the residual stream.

Schema-only release: validators + frozen dataclasses + CLI surface ship
in v0.62.0. The live forward-hook + decode-time intervention land in
v0.62.1, mirroring the v0.50.0 / v0.52.0 / v0.61.0 stub-then-live cadence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

SUPPORTED_STEERING_METHODS: frozenset[str] = frozenset({"caa", "iti", "repe"})

_MAX_METHOD_LEN: int = 32
_MAX_NAME_LEN: int = 128
_MAX_STRENGTH_ABS: float = 10.0  # |strength| <= 10 sanity cap.

# Kebab-case + underscore + dots only. Path-separators / whitespace /
# shell-metacharacters all rejected so the name can be safely embedded in
# CLI args, filenames, and Rich markup. Mirrors v0.57.0 adapter-branch
# policy (alphanumeric + `._-`).
_NAME_RE: re.Pattern[str] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")


@dataclass(frozen=True)
class SteeringMethodSpec:
    """Metadata for a single steering backend. Frozen post-construction."""

    name: str
    description: str
    needs_contrastive_pairs: bool
    needs_attention_heads: bool
    live_wired: bool


_STEERING_METHOD_METADATA: Mapping[str, SteeringMethodSpec] = MappingProxyType({
    "caa": SteeringMethodSpec(
        name="caa",
        description=(
            "Contrastive Activation Addition — add a contrastive vector "
            "to the residual stream during decoding. Trains on "
            "(positive, negative) prompt pairs (Panickssery et al., 2023)."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=False,
        live_wired=False,
    ),
    "iti": SteeringMethodSpec(
        name="iti",
        description=(
            "Inference-Time Intervention — shift specific attention heads "
            "along a learned direction (Li et al., 2023). Needs per-head "
            "calibration."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=True,
        live_wired=False,
    ),
    "repe": SteeringMethodSpec(
        name="repe",
        description=(
            "Representation Engineering — PCA over hidden states to "
            "extract a behavioural direction (Zou et al., 2023)."
        ),
        needs_contrastive_pairs=True,
        needs_attention_heads=False,
        live_wired=False,
    ),
})


def validate_steering_method(value: object) -> str:
    """Normalise + validate a steering-method name.

    Mirrors v0.41.0 / v0.51.0 / v0.61.0 validator policy: bool-rejected,
    null-byte-rejected, oversize-rejected, case-insensitive normalisation,
    unknown rejected with friendly actionable message.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering method must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"steering method must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("steering method must be non-empty")
    if "\x00" in value:
        raise ValueError("steering method must not contain null bytes")
    if len(value) > _MAX_METHOD_LEN:
        raise ValueError(
            f"steering method must be <= {_MAX_METHOD_LEN} chars"
        )
    canonical = value.lower()
    if canonical not in SUPPORTED_STEERING_METHODS:
        supported = ", ".join(sorted(SUPPORTED_STEERING_METHODS))
        raise ValueError(
            f"unknown steering method {value!r}; supported: {supported}"
        )
    return canonical


def validate_steering_name(value: object) -> str:
    """Validate an operator-supplied steering-vector name.

    Returns the value unchanged on success. Closed regex allowlist
    (alphanumeric + ``._-``, leading-alnum, ≤128 chars) so the name can
    be safely used as a Registry artifact id, CLI flag, and filename
    fragment on every platform.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering name must not be bool, got {value!r}"
        )
    if not isinstance(value, str):
        raise TypeError(
            f"steering name must be str, got {type(value).__name__}"
        )
    if not value:
        raise ValueError("steering name must be non-empty")
    if "\x00" in value:
        raise ValueError("steering name must not contain null bytes")
    if len(value) > _MAX_NAME_LEN:
        raise ValueError(
            f"steering name must be <= {_MAX_NAME_LEN} chars"
        )
    if not _NAME_RE.match(value):
        raise ValueError(
            f"steering name {value!r} must match {_NAME_RE.pattern!r} "
            "(alphanumeric + `._-`, leading alnum)."
        )
    return value


def validate_steering_strength(value: object) -> float:
    """Validate a steering strength multiplier.

    Bool-rejected (bool is a subclass of int), NaN/Inf-rejected via
    ``math.isfinite``, bounded ``|strength| <= 10.0`` as a sanity cap.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"steering strength must not be bool, got {value!r}"
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"steering strength must be a number, got {type(value).__name__}"
        )
    fval = float(value)
    if not math.isfinite(fval):
        raise ValueError("steering strength must be finite (no NaN / Inf)")
    if abs(fval) > _MAX_STRENGTH_ABS:
        raise ValueError(
            f"steering strength must satisfy |s| <= {_MAX_STRENGTH_ABS}; "
            f"got {fval}"
        )
    return fval


def get_steering_method_spec(name: str) -> SteeringMethodSpec:
    """Return the frozen :class:`SteeringMethodSpec` for ``name`` or raise."""
    canonical = validate_steering_method(name)
    return _STEERING_METHOD_METADATA[canonical]


# --- v0.71.10 #201 — live CAA / ITI / RepE fitting + decode hook ----------

_MAX_PAIRS = 2000  # cap on contrastive-pair count (DoS guard on the JSONL).
_DEFAULT_TOP_K_HEADS = 8  # ITI: top-K attention heads to intervene on.
_MAX_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB cap on the pairs JSONL.
_MAX_PAIR_FIELD_LEN = 65_536  # per-field cap (parity with raft._check_str).
_MIN_PAIRS_FOR_PCA = 2  # repe / iti need >= 2 pairs (SVD degenerate at N=1).
_CONFIG_NAME = "steering_config.json"
_VECTOR_NAME = "steering_vector.safetensors"


@dataclass(frozen=True)
class SteeringArtifact:
    """On-disk result of ``build_steering_vector`` (frozen)."""

    method: str
    name: str
    layer: int
    hidden_dim: int
    intervention_point: str  # "residual" | "attn_o_proj_input"
    output_dir: str
    base: str
    num_pairs: int


@dataclass(frozen=True)
class LoadedSteering:
    """A steering vector loaded from disk, ready for the decode hook."""

    method: str
    name: str
    layer: int
    intervention_point: str
    vector: Any  # numpy float32 [D]
    default_strength: float


def compute_caa_vector(positive: Any, negative: Any) -> Any:
    """CAA control vector = ``mean(positive) - mean(negative)`` (Panickssery 2023).

    ``positive`` / ``negative`` are ``[N, D]`` activation matrices (one mean-
    pooled residual-stream vector per prompt). The raw mean-difference is
    returned (NOT unit-normalised) so its magnitude reflects one "contrast
    unit" — the decode-time ``strength`` multiplier scales it. Returns a
    float32 ``[D]`` numpy vector.
    """
    import numpy as np

    pos = _steer_as_2d(positive, "positive")
    neg = _steer_as_2d(negative, "negative")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(
            f"hidden-dim mismatch: positive[{pos.shape[1]}] vs negative[{neg.shape[1]}]"
        )
    vec = (pos.mean(axis=0) - neg.mean(axis=0)).astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError("CAA vector is not finite")
    return vec


def compute_repe_direction(diffs: Any) -> Any:
    """RepE behavioural direction (Zou 2023) — top PCA component of the diffs.

    ``diffs`` is the ``[N, D]`` per-example ``positive - negative`` matrix. The
    top principal component is extracted via SVD, sign-aligned so the mean diff
    projects positively, and scaled by that mean projection so the magnitude is
    comparable to the CAA vector. Returns float32 ``[D]``.
    """
    import numpy as np

    d = _steer_as_2d(diffs, "diffs")
    centered = d - d.mean(axis=0, keepdims=True)
    # full_matrices=False keeps the SVD cheap; Vt rows are the right-singular
    # vectors (principal directions in feature space).
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc = vt[0]
    mean_diff = d.mean(axis=0)
    scale = float(mean_diff @ pc)
    if scale < 0:
        pc = -pc
        scale = -scale
    vec = (pc * scale).astype(np.float32)
    if not np.all(np.isfinite(vec)):
        raise ValueError("RepE direction is not finite")
    return vec


def compute_iti_directions(
    pos_heads: Any, neg_heads: Any, *, top_k: int = _DEFAULT_TOP_K_HEADS
) -> Tuple[Any, Tuple[int, ...]]:
    """ITI per-head intervention directions (Li 2023).

    ``pos_heads`` / ``neg_heads`` are ``[N, H, Dh]`` per-example, per-head
    activations (the input to ``o_proj`` reshaped into heads). The per-head
    mean difference is computed, heads are ranked by the L2 norm of their diff,
    and the top ``top_k`` heads keep their direction (others zeroed). Returns
    ``(directions[H, Dh] float32, selected_heads tuple)``.
    """
    import numpy as np

    pos = _steer_as_3d(pos_heads, "pos_heads")
    neg = _steer_as_3d(neg_heads, "neg_heads")
    if pos.shape[1:] != neg.shape[1:]:
        raise ValueError(
            f"head-shape mismatch: pos{pos.shape[1:]} vs neg{neg.shape[1:]}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive int")
    num_heads = pos.shape[1]
    diff = pos.mean(axis=0) - neg.mean(axis=0)  # [H, Dh]
    head_scores = np.linalg.norm(diff, axis=1)  # [H]
    k = min(top_k, num_heads)
    selected = tuple(sorted(int(i) for i in np.argsort(head_scores)[-k:]))
    directions = np.zeros_like(diff, dtype=np.float32)
    for head in selected:
        directions[head] = diff[head]
    if not np.all(np.isfinite(directions)):
        raise ValueError("ITI directions are not finite")
    return directions, selected


def _steer_as_2d(value: Any, field: str) -> Any:
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{field} must be a non-empty 2D array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} must be finite")
    return arr


def _steer_as_3d(value: Any, field: str) -> Any:
    import numpy as np

    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(f"{field} must be a non-empty 3D [N, H, Dh] array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} must be finite")
    return arr


def load_contrastive_pairs(path: str) -> list:
    """Load ``[(positive, negative), ...]`` from a JSONL of ``{positive, negative}``.

    cwd-contained + symlink-rejected (via the shared
    ``enforce_under_cwd_and_no_symlink`` helper) + ``O_NOFOLLOW`` open (closes
    the TOCTOU window the inline ``lexists``-gated lstat left) + size-capped.
    Skips malformed / incomplete rows; per-field cap ``_MAX_PAIR_FIELD_LEN``.
    Capped at ``_MAX_PAIRS``.
    """
    import json
    import os
    import stat

    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "pairs_path")
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"pairs file not found: {path!r}") from exc
    except OSError as exc:
        raise ValueError(
            f"pairs file cannot be opened (symlink?): {type(exc).__name__}"
        ) from exc
    pairs: list = []
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            raise ValueError("pairs_path must be a regular file")
        if st.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"pairs file exceeds {_MAX_FILE_BYTES} bytes")
        with os.fdopen(fd, "r", encoding="utf-8-sig") as fh:
            for line in fh:
                if len(pairs) >= _MAX_PAIRS:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                pos = row.get("positive")
                neg = row.get("negative")
                if (
                    isinstance(pos, str) and isinstance(neg, str)
                    and pos and neg
                    and len(pos) <= _MAX_PAIR_FIELD_LEN
                    and len(neg) <= _MAX_PAIR_FIELD_LEN
                ):
                    pairs.append((pos, neg))
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    if not pairs:
        raise ValueError(f"pairs file {path!r} yielded no usable (positive, negative) rows")
    return pairs


def _validated_steer_output_dir(output_dir: str) -> str:
    """Reject an output dir outside cwd or pointing at a symlink (TOCTOU).

    Delegates to the shared ``enforce_under_cwd_and_no_symlink`` helper for the
    containment + leaf-symlink rejection (SEC LOW-1 — was an inline
    ``lexists``-gated lstat).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(output_dir, "output dir")
    return output_dir


def build_steering_vector(
    *,
    method: str,
    name: str,
    pairs_path: Optional[str] = None,
    base: Optional[str] = None,
    layer: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    top_k: int = _DEFAULT_TOP_K_HEADS,
    trust_remote_code: bool = False,
) -> SteeringArtifact:
    """Fit a steering vector from contrastive pairs and persist it.

    Validates method + name FIRST (so a typo surfaces before the model load),
    then requires ``base`` + ``pairs_path``. Loads the model, captures residual
    -stream (CAA / RepE) or per-head (ITI) activations on the contrastive
    prompts, computes the control vector, and writes
    ``<output_dir>/steering_vector.safetensors`` + ``steering_config.json``.

    Returns a frozen :class:`SteeringArtifact`. Heavy imports are local.
    """
    import json
    import os

    import numpy as np

    canonical = validate_steering_method(method)
    canonical_name = validate_steering_name(name)
    if not base or not isinstance(base, str):
        raise ValueError("build_steering_vector requires a non-empty base model")
    if not pairs_path or not isinstance(pairs_path, str):
        raise ValueError("build_steering_vector requires a pairs_path (JSONL)")
    if layer is not None:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be int or None")
        if layer < 0 or layer > 2048:
            raise ValueError(f"layer must satisfy 0 <= layer <= 2048, got {layer}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive int")

    pairs = load_contrastive_pairs(pairs_path)
    # repe (PCA) and iti (per-head ranking) need >= 2 pairs — a single pair
    # makes the centered SVD degenerate (code-review L5). CAA (mean-diff) is
    # well-defined at N=1.
    if canonical in ("repe", "iti") and len(pairs) < _MIN_PAIRS_FOR_PCA:
        raise ValueError(
            f"method {canonical!r} requires >= {_MIN_PAIRS_FOR_PCA} contrastive "
            f"pairs (got {len(pairs)}); use 'caa' for a single pair"
        )
    pos_prompts = [p for p, _ in pairs]
    neg_prompts = [n for _, n in pairs]

    from soup_cli.utils.edit_kernels import _locate_decoder_layers
    from soup_cli.utils.live_eval import (
        extract_layer_activations,
        load_model_and_tokenizer,
    )

    model, tokenizer, dev = load_model_and_tokenizer(
        base, device=device, trust_remote_code=trust_remote_code
    )
    layers = _locate_decoder_layers(model)
    n_layers = len(layers)
    resolved_layer = n_layers // 2 if layer is None else layer
    if resolved_layer < 0 or resolved_layer >= n_layers:
        raise ValueError(
            f"layer {resolved_layer} out of range for a {n_layers}-layer model"
        )

    extra: dict = {}
    if canonical in ("caa", "repe"):
        layer_path = f"model.layers.{resolved_layer}"
        pos = extract_layer_activations(
            model, tokenizer, pos_prompts, layer=layer_path, device=dev, pool="mean"
        )
        neg = extract_layer_activations(
            model, tokenizer, neg_prompts, layer=layer_path, device=dev, pool="mean"
        )
        if canonical == "caa":
            vector = compute_caa_vector(pos, neg)
        else:
            vector = compute_repe_direction(np.asarray(pos) - np.asarray(neg))
        intervention = "residual"
    else:  # iti
        num_heads = int(getattr(model.config, "num_attention_heads", 0)) or 1
        pos_h = _capture_attn_heads(
            model, tokenizer, pos_prompts, resolved_layer, num_heads, dev
        )
        neg_h = _capture_attn_heads(
            model, tokenizer, neg_prompts, resolved_layer, num_heads, dev
        )
        dirs, selected = compute_iti_directions(pos_h, neg_h, top_k=top_k)
        vector = dirs.reshape(-1).astype(np.float32)
        intervention = "attn_o_proj_input"
        extra = {
            "num_heads": num_heads,
            "head_dim": int(dirs.shape[1]),
            "selected_heads": list(selected),
        }

    hidden_dim = int(vector.shape[0])
    out_dir = output_dir or os.path.join("steering", canonical_name)
    out_dir = _validated_steer_output_dir(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from safetensors.numpy import save_file

    save_file(
        {"vector": np.ascontiguousarray(vector, dtype=np.float32)},
        os.path.join(out_dir, _VECTOR_NAME),
    )
    config = {
        "method": canonical,
        "name": canonical_name,
        "layer": resolved_layer,
        "hidden_dim": hidden_dim,
        "intervention_point": intervention,
        "base": base,
        "num_pairs": len(pairs),
        "default_strength": 1.0,
        **extra,
    }
    with open(os.path.join(out_dir, _CONFIG_NAME), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, allow_nan=False)

    return SteeringArtifact(
        method=canonical,
        name=canonical_name,
        layer=resolved_layer,
        hidden_dim=hidden_dim,
        intervention_point=intervention,
        output_dir=out_dir,
        base=base,
        num_pairs=len(pairs),
    )


def _capture_attn_heads(model, tokenizer, prompts, layer_idx, num_heads, device):
    """Capture per-head ``o_proj``-input activations: ``[N, H, Dh]`` (ITI)."""
    import numpy as np
    import torch

    from soup_cli.utils.edit_kernels import _locate_decoder_layers
    from soup_cli.utils.raft import render_raft_prompt

    layers = _locate_decoder_layers(model)
    attn = getattr(layers[layer_idx], "self_attn", None)
    o_proj = getattr(attn, "o_proj", None) if attn is not None else None
    if o_proj is None or not hasattr(o_proj, "weight"):
        raise ValueError(
            f"layer {layer_idx} has no self_attn.o_proj (unsupported arch for ITI)"
        )
    captured: list = []

    def _pre_hook(_mod, args):
        # args[0]: [batch, seq, in_features]
        captured.append(args[0][0].detach().to(torch.float32).mean(dim=0).cpu())

    handle = o_proj.register_forward_pre_hook(_pre_hook)
    rows: list = []
    try:
        model.eval()
        with torch.no_grad():
            for prompt in prompts:
                text = render_raft_prompt(tokenizer, prompt)
                inputs = tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=1024
                ).to(device)
                captured.clear()
                model(**inputs)
                if captured:
                    rows.append(captured[-1].numpy())
    finally:
        handle.remove()
    if not rows:
        raise ValueError("no attention activations captured for ITI")
    flat = np.stack(rows).astype(np.float32)  # [N, in_features]
    in_features = flat.shape[1]
    head_dim = in_features // num_heads
    if head_dim * num_heads != in_features:
        raise ValueError(
            f"o_proj in_features {in_features} not divisible by num_heads {num_heads}"
        )
    return flat.reshape(flat.shape[0], num_heads, head_dim)


def resolve_steering_dir(name: str) -> str:
    """Resolve a steering-vector NAME to its on-disk directory.

    Resolution order: (1) the default ``./steering/<name>`` directory if it
    holds a ``steering_config.json``; (2) the most recent Registry entry named
    ``<name>`` carrying a ``steering_vector`` artifact. Raises ``ValueError``
    when neither resolves.
    """
    import os

    from soup_cli.utils.paths import is_under_cwd

    canonical = validate_steering_name(name)
    local = os.path.join("steering", canonical)
    if os.path.isfile(os.path.join(local, _CONFIG_NAME)):
        return local
    try:
        from soup_cli.registry.store import RegistryStore

        with RegistryStore() as store:
            for entry in store.list():
                if str(entry.get("name", "")) != canonical:
                    continue
                for art in store.get_artifacts(entry["id"]):
                    if art.get("kind") == "steering_vector":
                        art_path = str(art.get("path", ""))
                        # Only trust a Registry-supplied path that is still
                        # under cwd (a shared/copied Registry DB could hold an
                        # absolute out-of-tree path) — SEC LOW-2.
                        if (
                            art_path
                            and is_under_cwd(art_path)
                            and os.path.isfile(os.path.join(art_path, _CONFIG_NAME))
                        ):
                            return art_path
    except (ImportError, OSError, ValueError):
        pass
    raise ValueError(
        f"no steering vector named {canonical!r} (looked under ./steering/ "
        "and the Registry). Train one with `soup steer train`."
    )


def load_steering_artifact(dir_path: str) -> LoadedSteering:
    """Load a steering vector + config from ``dir_path`` into a :class:`LoadedSteering`."""
    import json
    import os

    import numpy as np
    from safetensors import safe_open

    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    # The dir (and its two leaf files) may come from a Registry row — contain
    # them under cwd + reject symlinks before reading (SEC LOW-2).
    enforce_under_cwd_and_no_symlink(dir_path, "steering dir")
    config_path = os.path.join(dir_path, _CONFIG_NAME)
    vector_path = os.path.join(dir_path, _VECTOR_NAME)
    if not os.path.isfile(config_path) or not os.path.isfile(vector_path):
        raise ValueError(f"steering dir {dir_path!r} missing config / vector file")
    enforce_under_cwd_and_no_symlink(config_path, "steering config")
    enforce_under_cwd_and_no_symlink(vector_path, "steering vector")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    method = validate_steering_method(config.get("method", "caa"))
    name = validate_steering_name(config.get("name", "steer"))
    layer = int(config.get("layer", 0))
    intervention = config.get("intervention_point", "residual")
    if intervention not in ("residual", "attn_o_proj_input"):
        raise ValueError(f"unknown intervention_point {intervention!r}")
    with safe_open(vector_path, framework="numpy") as handle:
        if "vector" not in handle.keys():
            raise ValueError("steering safetensors missing 'vector'")
        vector = np.asarray(handle.get_tensor("vector"), dtype=np.float32)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError("steering vector must be a finite 1D array")
    default_strength = float(config.get("default_strength", 1.0))
    return LoadedSteering(
        method=method,
        name=name,
        layer=layer,
        intervention_point=intervention,
        vector=vector,
        default_strength=default_strength,
    )


def install_steering_hook(model: Any, loaded: LoadedSteering, *, strength: float):
    """Install a decode-time forward hook that adds ``strength * vector``.

    * ``residual`` (CAA / RepE) — a forward hook on the decoder layer adds the
      vector to the residual-stream output.
    * ``attn_o_proj_input`` (ITI) — a forward pre-hook on ``self_attn.o_proj``
      shifts the per-head activations before the output projection.

    Returns the hook handle so the caller can ``.remove()`` it on shutdown.
    """
    import torch

    if not isinstance(loaded, LoadedSteering):
        raise TypeError("loaded must be a LoadedSteering")
    strength_f = validate_steering_strength(strength)

    from soup_cli.utils.edit_kernels import _locate_decoder_layers

    layers = _locate_decoder_layers(model)
    if loaded.layer < 0 or loaded.layer >= len(layers):
        raise ValueError(
            f"steering layer {loaded.layer} out of range for "
            f"{len(layers)}-layer model"
        )
    block = layers[loaded.layer]
    param = next(model.parameters())
    vec = torch.tensor(loaded.vector, dtype=param.dtype, device=param.device)

    if loaded.intervention_point == "residual":
        def _hook(_mod, _args, output):
            if isinstance(output, (tuple, list)):
                hidden = output[0] + strength_f * vec
                return (hidden,) + tuple(output[1:])
            return output + strength_f * vec

        return block.register_forward_hook(_hook)

    o_proj = getattr(getattr(block, "self_attn", None), "o_proj", None)
    if o_proj is None:
        raise ValueError("ITI steering requires self_attn.o_proj on the layer")

    def _pre_hook(_mod, args):
        shifted = args[0] + strength_f * vec
        return (shifted,) + tuple(args[1:])

    return o_proj.register_forward_pre_hook(_pre_hook)
