"""v0.53.0 Part F + v0.53.1 #142 — Advanced save / merge formats.

Two new save-format surfaces ship this release (schema-only):

* ``soup merge --save-format <fmt>`` where fmt ∈ {fp16, 4bit, 4bit_forced}.
  ``4bit`` writes a single BNB-4bit-quantized merged checkpoint without
  the dequant → merge → requant cycle (unsloth ``merged_4bit`` recipe).
  ``4bit_forced`` is the unsloth ``4bit_forced`` shortcut.
* ``soup export --format torchao --quant-config <yaml>`` — invokes
  ``torchao.quantize_`` then ``save_pretrained`` for the
  ``Int4WeightOnly`` / ``Int8DynActInt4`` / ``Float8DynActFloat8`` /
  ``NVFP4`` PTQ schemes (unsloth + axolotl parity).

Live wiring deferred to v0.53.1.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

# --- Merge save formats ------------------------------------------------------
MERGE_SAVE_FORMATS: frozenset[str] = frozenset({
    "fp16", "4bit", "4bit_forced",
})

# --- TorchAO PTQ schemes (a closed allowlist, mirrors the v0.38.0 Quant Menu
# string convention so YAML round-trips cleanly).
TORCHAO_PTQ_SCHEMES: frozenset[str] = frozenset({
    "Int4WeightOnly",
    "Int8DynActInt4",
    "Float8DynActFloat8",
    "NVFP4",
})

_MAX_SAVE_FORMAT_LEN: int = 32
_MAX_TORCHAO_SCHEME_LEN: int = 48


@dataclass(frozen=True)
class MergeSaveSpec:
    """Frozen metadata for a merge-save format."""

    name: str
    bits: int
    description: str
    live_wired: bool


_MERGE_METADATA: Mapping[str, MergeSaveSpec] = MappingProxyType({
    "fp16": MergeSaveSpec(
        name="fp16", bits=16,
        description="Standard FP16 merged checkpoint (default — pre-v0.53.0)",
        live_wired=True,
    ),
    "4bit": MergeSaveSpec(
        name="4bit", bits=4,
        description="Single BNB-4bit-quantized merged checkpoint",
        live_wired=False,
    ),
    "4bit_forced": MergeSaveSpec(
        name="4bit_forced", bits=4,
        description="Forced BNB-4bit merge (unsloth 4bit_forced)",
        live_wired=False,
    ),
})


@dataclass(frozen=True)
class TorchAOPTQSpec:
    """Frozen metadata for a TorchAO PTQ scheme."""

    name: str
    bits: int
    description: str
    live_wired: bool


_TORCHAO_METADATA: Mapping[str, TorchAOPTQSpec] = MappingProxyType({
    "Int4WeightOnly": TorchAOPTQSpec(
        name="Int4WeightOnly", bits=4,
        description="TorchAO Int4WeightOnly (weight-only int4)",
        live_wired=False,
    ),
    "Int8DynActInt4": TorchAOPTQSpec(
        name="Int8DynActInt4", bits=4,
        description="TorchAO Int8DynActInt4 (dynamic int8 act + int4 weight)",
        live_wired=False,
    ),
    "Float8DynActFloat8": TorchAOPTQSpec(
        name="Float8DynActFloat8", bits=8,
        description="TorchAO FP8 dynamic activations + FP8 weight",
        live_wired=False,
    ),
    "NVFP4": TorchAOPTQSpec(
        name="NVFP4", bits=4,
        description="TorchAO NVFP4 (Blackwell FP4 PTQ)",
        live_wired=False,
    ),
})


def _validate_string_field(value: object, field: str, max_len: int) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > max_len:
        raise ValueError(f"{field} too int (max {max_len} chars)")
    return value


def validate_merge_save_format(value: object) -> str:
    """Validate ``--save-format`` arg. Case-insensitive normalisation.

    Mirrors v0.41.0 ``optimizer`` policy. Returns the lowercase canonical
    form so ``_MERGE_METADATA`` (all-lowercase keys) lookups are O(1).
    """
    validated = _validate_string_field(value, "save_format", _MAX_SAVE_FORMAT_LEN)
    canonical = validated.lower()
    if canonical not in MERGE_SAVE_FORMATS:
        supported = ", ".join(sorted(MERGE_SAVE_FORMATS))
        raise ValueError(
            f"save_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_torchao_scheme(value: object) -> str:
    """Validate a TorchAO PTQ scheme name.

    INTENTIONALLY CASE-SENSITIVE — these are PyTorch class names
    (``Int4WeightOnly`` / ``NVFP4`` etc.) that ``torchao.quantize_`` looks up
    by exact name. Diverges from ``validate_merge_save_format`` (which
    lowercase-normalises) and ``validate_kv_cache_type`` (also lowercase)
    on purpose: TorchAO uses CapWords, the others are operator-facing flags.
    """
    validated = _validate_string_field(
        value, "torchao_scheme", _MAX_TORCHAO_SCHEME_LEN,
    )
    if validated not in TORCHAO_PTQ_SCHEMES:
        supported = ", ".join(sorted(TORCHAO_PTQ_SCHEMES))
        raise ValueError(
            f"torchao_scheme {value!r} not supported. Supported: {supported}"
        )
    return validated


def get_merge_save_spec(name: str) -> MergeSaveSpec:
    """Return the frozen :class:`MergeSaveSpec` for ``name`` (case-insensitive)."""
    canonical = validate_merge_save_format(name)
    return _MERGE_METADATA[canonical]


def get_torchao_spec(name: str) -> TorchAOPTQSpec:
    """Return the frozen :class:`TorchAOPTQSpec` for ``name`` (case-sensitive)."""
    canonical = validate_torchao_scheme(name)
    return _TORCHAO_METADATA[canonical]


def validate_quant_config_path(path: object) -> str:
    """Validate ``--quant-config <yaml>`` argument shape.

    Boundary contract — what's enforced HERE vs at CLI dispatch (v0.53.1):

    THIS helper enforces non-empty ``str``, no null bytes, length <= 4096.

    CLI dispatch in v0.53.1 MUST additionally apply (mirrors v0.43.0 /
    v0.46.0 / v0.47.0 TOCTOU policy):
    * ``os.path.realpath`` + ``os.path.commonpath`` cwd containment
    * ``os.lstat`` + ``stat.S_ISLNK`` rejection before any ``open()``
    * existence + extension check (``.yaml`` / ``.yml``)
    * ``yaml.safe_load`` only.
    """
    if isinstance(path, bool):
        raise TypeError(f"quant_config must not be bool, got {path!r}")
    if not isinstance(path, str):
        raise TypeError(
            f"quant_config must be str, got {type(path).__name__}"
        )
    if not path:
        raise ValueError("quant_config must be non-empty")
    if "\x00" in path:
        raise ValueError("quant_config must not contain null bytes")
    if len(path) > 4096:
        raise ValueError("quant_config path too int (max 4096 chars)")
    return path


# --- v0.53.1 #142 — Live path-containment + symlink TOCTOU helpers ---------

_MAX_QUANT_CONFIG_BYTES: int = 256 * 1024


def _enforce_under_cwd_and_no_symlink(path: str, field: str) -> str:
    """Re-export the shared helper from :mod:`soup_cli.utils.paths`.

    Kept as a module-level alias so external callers (and the v0.53.1 CLI
    dispatch path) can keep their existing imports.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    return enforce_under_cwd_and_no_symlink(path, field)


def load_quant_config(path: object) -> Mapping[str, Any]:
    """Load + validate a TorchAO ``--quant-config`` YAML file.

    Enforces (per v0.53.0 docstring contract):
    * Shape validation via :func:`validate_quant_config_path`
    * ``os.path.realpath`` + ``os.path.commonpath`` cwd containment
    * ``os.lstat + stat.S_ISLNK`` rejection (TOCTOU)
    * Extension allowlist (``.yaml`` / ``.yml``)
    * ``yaml.safe_load`` only
    * 256 KB size cap
    """
    import yaml

    validated_shape = validate_quant_config_path(path)
    _enforce_under_cwd_and_no_symlink(validated_shape, "quant_config")
    lower = validated_shape.lower()
    if not (lower.endswith(".yaml") or lower.endswith(".yml")):
        raise ValueError(
            f"quant_config extension must be .yaml or .yml: "
            f"{os.path.basename(validated_shape)!r}"
        )
    if not os.path.isfile(validated_shape):
        raise FileNotFoundError(
            f"quant_config not found: {os.path.basename(validated_shape)!r}"
        )
    size = os.path.getsize(validated_shape)
    if size > _MAX_QUANT_CONFIG_BYTES:
        raise ValueError(
            f"quant_config too large ({size} bytes; "
            f"cap {_MAX_QUANT_CONFIG_BYTES})"
        )
    with open(validated_shape, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"quant_config must be a YAML mapping, got {type(data).__name__}"
        )
    return data


def merge_4bit(
    *,
    merged_dir: str,
    output_dir: str,
    forced: bool = False,
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> None:
    """Write a single BNB-4bit-quantized merged checkpoint.

    Unsloth ``merged_4bit`` / ``4bit_forced`` recipe — no
    dequant → merge → requant cycle. ``forced=True`` quantizes ALL linear
    layers including embeddings; default ``False`` follows BNB's default
    skip-modules behaviour.
    """
    if not isinstance(forced, bool):
        raise TypeError(f"forced must be bool, got {type(forced).__name__}")
    if not isinstance(dtype, str):
        raise TypeError(f"dtype must be str, got {type(dtype).__name__}")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(
            f"dtype {dtype!r} invalid; expected float16 / bfloat16 / float32"
        )

    _enforce_under_cwd_and_no_symlink(merged_dir, "merged_dir")
    _enforce_under_cwd_and_no_symlink(output_dir, "output_dir")

    if not os.path.isdir(merged_dir):
        raise FileNotFoundError(
            f"merged_dir not a directory: {os.path.basename(merged_dir)!r}"
        )

    import torch
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    bnb_kwargs: dict[str, Any] = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": dtype_map[dtype],
        "bnb_4bit_use_double_quant": True,
    }
    if forced:
        # ``forced`` => no skip-modules, so every Linear (incl. lm_head) is
        # 4-bit quantized. BNB 4-bit uses ``bnb_4bit_skip_modules`` (the
        # legacy 8-bit name was ``llm_int8_skip_modules`` — only emit it
        # when the installed BNB exposes the 8-bit kwarg as a fallback).
        bnb_kwargs["bnb_4bit_skip_modules"] = []
    bnb_config = BitsAndBytesConfig(**bnb_kwargs)

    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        merged_dir,
        quantization_config=bnb_config,
        trust_remote_code=trust_remote_code,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        merged_dir, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)


def export_torchao(
    *,
    model_dir: str,
    output_dir: str,
    scheme: str,
    quant_config_data: Optional[Mapping[str, Any]] = None,
    trust_remote_code: bool = False,
) -> None:
    """Run ``torchao.quantize_(scheme)`` + ``save_pretrained``.

    Schemes (CASE-SENSITIVE) from :data:`TORCHAO_PTQ_SCHEMES`:
    ``Int4WeightOnly`` / ``Int8DynActInt4`` / ``Float8DynActFloat8`` / ``NVFP4``.
    """
    canonical_scheme = validate_torchao_scheme(scheme)
    _enforce_under_cwd_and_no_symlink(model_dir, "model_dir")
    _enforce_under_cwd_and_no_symlink(output_dir, "output_dir")
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model_dir not a directory: {os.path.basename(model_dir)!r}"
        )
    if quant_config_data is not None and not isinstance(quant_config_data, Mapping):
        raise TypeError(
            f"quant_config_data must be Mapping, got {type(quant_config_data).__name__}"
        )

    from torchao import quantization as ao_q  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    scheme_factory_map = {
        "Int4WeightOnly": "Int4WeightOnlyConfig",
        "Int8DynActInt4": "Int8DynActInt4Config",
        "Float8DynActFloat8": "Float8DynActFloat8Config",
        "NVFP4": "NVFP4Config",
    }
    factory_name = scheme_factory_map[canonical_scheme]
    if not hasattr(ao_q, factory_name):
        raise RuntimeError(
            f"torchao does not expose {factory_name}; "
            f"upgrade torchao or pick a different scheme."
        )

    # Build the config from quant_config_data if provided; else defaults.
    # Apply a per-scheme closed key allowlist to defeat kwarg injection
    # (security review H1).
    scheme_kwarg_allowlist: dict[str, frozenset[str]] = {
        "Int4WeightOnly": frozenset({"group_size", "inner_k_tiles"}),
        "Int8DynActInt4": frozenset({"group_size"}),
        "Float8DynActFloat8": frozenset(),
        "NVFP4": frozenset(),
    }
    allowed = scheme_kwarg_allowlist.get(canonical_scheme, frozenset())
    raw_kwargs = dict(quant_config_data or {})
    raw_kwargs.pop("scheme", None)
    bad_keys = [
        k for k in raw_kwargs
        if (not isinstance(k, str)) or k.startswith("__") or k not in allowed
    ]
    if bad_keys:
        allowed_str = (
            ", ".join(sorted(allowed))
            if allowed
            else "(none — scheme takes no extra args)"
        )
        raise ValueError(
            f"quant_config keys not allowed for scheme {canonical_scheme}: "
            f"{bad_keys}. Allowed: {allowed_str}"
        )
    config_obj = getattr(ao_q, factory_name)(**raw_kwargs)

    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code,
    )
    # torchao quantize in-place. Some torchao versions ship `quantize_` at
    # top level; others under quantization. Try both.
    try:
        import torchao  # type: ignore[import-not-found]
        quantize_fn = getattr(torchao, "quantize_", None) or getattr(ao_q, "quantize_")
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            f"torchao.quantize_ entry point not found: {type(exc).__name__}"
        ) from exc

    quantize_fn(model, config_obj)
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=trust_remote_code
    )
    tokenizer.save_pretrained(output_dir)
