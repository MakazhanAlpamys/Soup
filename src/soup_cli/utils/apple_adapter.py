"""``soup apple-adapter`` — HF / PEFT ↔ MLX ↔ Apple FoundationModels (v0.68.0 Part D).

v0.71.21 (#228) lifts ``convert_apple_adapter`` to live:

- ``hf-to-mlx`` reads a PEFT LoRA adapter (``adapter_model.safetensors``)
  and writes an mlx-lm-shaped ``adapters.safetensors`` +
  ``adapter_config.json`` (with ``num_layers`` derived from the converted
  keys — mlx-lm's ``load_adapters`` reads both unconditionally). PEFT
  stores ``...lora_A.weight`` as ``[r, in]`` / ``...lora_B.weight`` as
  ``[out, r]``; mlx-lm's ``LoRALinear`` computes ``(x @ lora_a) @ lora_b``
  with ``lora_a [in, r]`` / ``lora_b [r, out]``, so both matrices
  transpose on the way through. Pure numpy — no mlx import needed, so the
  conversion runs on any OS; loading the artifact into mlx-lm itself
  requires Apple hardware (documented BETA gate). bf16 adapters are
  upcast to float32 via the torch loader (numpy has no bf16).
- ``mlx-to-hf`` reverses the conversion (``adapters.safetensors`` or
  legacy ``adapters.npz`` → ``adapter_model.safetensors`` + a PEFT-style
  ``adapter_config.json``).
- ``hf-to-apple`` / ``mlx-to-apple`` raise a friendly upstream-gate
  RuntimeError: Apple has not published a stable FoundationModels adapter
  spec — refusing to export wrong-shaped weights (per #228 fix path).
- ``sign=True`` reuses v0.60 Part B Merkle-root signing to emit a
  ``.soup-signature.json`` next to the converted adapter.

Public surface:

- ``SUPPORTED_ADAPTER_DIRECTIONS`` — closed frozenset
- ``validate_direction(name)`` — bool-first / null-byte / case-insensitive
- ``validate_source_adapter(path)`` — cwd containment + directory check + symlink reject
- ``AppleAdapterPlan`` frozen dataclass + ``build_apple_adapter_plan(...)``
- ``hf_key_to_mlx`` / ``mlx_key_to_hf`` — LoRA key mapping
- ``convert_hf_to_mlx_arrays`` / ``convert_mlx_to_hf_arrays`` — pure kernels
- ``convert_apple_adapter(plan)`` — LIVE; returns ``ConversionReport``
"""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from soup_cli.utils.paths import (
    atomic_write_bytes,
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    is_under_cwd,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# PEFT LoRA weight keys: ``base_model.model.<path>.lora_A.weight`` (the
# leading prefix is optional — some exporters strip it).
_HF_LORA_KEY_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<path>.+)\.(?P<matrix>lora_[AB])\.weight$"
)
# mlx-lm LoRA keys: ``<path>.lora_a`` / ``<path>.lora_b``.
_MLX_LORA_KEY_RE = re.compile(r"^(?P<path>.+)\.(?P<matrix>lora_[ab])$")
# Decoder-layer index inside an mlx key (``model.layers.N.…`` /
# GPT-2-style ``transformer.h.N.…``) — used to derive the ``num_layers``
# field mlx-lm's load_adapters reads.
_LAYER_INDEX_RE = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")

# Adapters are small; a multi-GiB "adapter" is a red flag, not a use case.
# The cap also bounds DECOMPRESSED npz arrays (zip bomb defence).
_MAX_ADAPTER_FILE_BYTES = 4 * 1024**3  # 4 GiB

SUPPORTED_ADAPTER_DIRECTIONS: frozenset[str] = frozenset(
    {"hf-to-mlx", "mlx-to-hf", "hf-to-apple", "mlx-to-apple"}
)

_MAX_DIRECTION_LEN = 32


def validate_direction(name: object) -> str:
    """Canonicalise a conversion direction against the closed allowlist."""
    if isinstance(name, bool):
        raise TypeError("direction must not be bool")
    if not isinstance(name, str):
        raise TypeError("direction must be str")
    if not name:
        raise ValueError("direction must be non-empty")
    if "\x00" in name:
        raise ValueError("direction must not contain null bytes")
    if len(name) > _MAX_DIRECTION_LEN:
        raise ValueError(
            f"direction length {len(name)} > {_MAX_DIRECTION_LEN}"
        )
    canonical = name.lower()
    if canonical not in SUPPORTED_ADAPTER_DIRECTIONS:
        raise ValueError(
            f"unknown direction {name!r}; supported: "
            + ", ".join(sorted(SUPPORTED_ADAPTER_DIRECTIONS))
        )
    return canonical


def validate_source_adapter(path: object) -> str:
    """Validate a source-adapter directory (cwd-contained, no symlink)."""
    if isinstance(path, bool):
        raise TypeError("source_dir must not be bool")
    if not isinstance(path, str):
        raise TypeError("source_dir must be str")
    if not path:
        raise ValueError("source_dir must be non-empty")
    if "\x00" in path:
        raise ValueError("source_dir must not contain null bytes")
    if not is_under_cwd(path):
        raise ValueError(
            f"source_dir {os.path.basename(path)!r} must stay under cwd"
        )
    if os.path.lexists(path):
        try:
            link_stat = os.lstat(path)
        except OSError as exc:
            raise ValueError(
                f"source_dir unreadable: {type(exc).__name__}"
            ) from exc
        if stat.S_ISLNK(link_stat.st_mode):
            raise ValueError(
                "source_dir must not be a symlink (TOCTOU defence)"
            )
        if not stat.S_ISDIR(link_stat.st_mode):
            raise ValueError("source_dir must be a directory")
    return os.path.realpath(path)


def _validate_output_dir(path: object) -> str:
    if isinstance(path, bool):
        raise TypeError("output_dir must not be bool")
    if not isinstance(path, str):
        raise TypeError("output_dir must be str")
    if not path:
        raise ValueError("output_dir must be non-empty")
    if "\x00" in path:
        raise ValueError("output_dir must not contain null bytes")
    return path


@dataclass(frozen=True)
class AppleAdapterPlan:
    """Resolved conversion plan (validated in ``__post_init__``)."""

    source_dir: str
    output_dir: str
    direction: str
    sign: bool

    def __post_init__(self) -> None:
        validate_source_adapter(self.source_dir)
        _validate_output_dir(self.output_dir)
        object.__setattr__(
            self, "direction", validate_direction(self.direction)
        )
        if not isinstance(self.sign, bool):
            raise TypeError("sign must be bool")


def build_apple_adapter_plan(
    *,
    source_dir: str,
    output_dir: str,
    direction: str,
    sign: bool = False,
) -> AppleAdapterPlan:
    """Build a validated :class:`AppleAdapterPlan` from raw CLI inputs."""
    return AppleAdapterPlan(
        source_dir=source_dir,
        output_dir=output_dir,
        direction=validate_direction(direction),
        sign=sign,
    )


@dataclass(frozen=True)
class ConversionReport:
    """Outcome of a live adapter conversion (v0.71.21 #228)."""

    direction: str
    output_dir: str
    converted_keys: int
    skipped_keys: tuple[str, ...]
    signed: bool

    def __post_init__(self) -> None:
        validate_direction(self.direction)
        if isinstance(self.converted_keys, bool) or not isinstance(
            self.converted_keys, int
        ):
            raise TypeError("converted_keys must be an int")
        if self.converted_keys < 0:
            raise ValueError("converted_keys must be >= 0")
        if not isinstance(self.skipped_keys, tuple):
            raise TypeError("skipped_keys must be a tuple")
        if not isinstance(self.signed, bool):
            raise TypeError("signed must be bool")


def hf_key_to_mlx(key: object) -> Optional[str]:
    """Map a PEFT LoRA key to its mlx-lm name (None for non-LoRA keys)."""
    if not isinstance(key, str):
        return None
    match = _HF_LORA_KEY_RE.match(key)
    if match is None:
        return None
    matrix = "lora_a" if match.group("matrix") == "lora_A" else "lora_b"
    return f"{match.group('path')}.{matrix}"


def mlx_key_to_hf(key: object) -> Optional[str]:
    """Map an mlx-lm LoRA key back to its PEFT name (None for non-LoRA)."""
    if not isinstance(key, str):
        return None
    match = _MLX_LORA_KEY_RE.match(key)
    if match is None:
        return None
    matrix = "lora_A" if match.group("matrix") == "lora_a" else "lora_B"
    return f"base_model.model.{match.group('path')}.{matrix}.weight"


def convert_hf_to_mlx_arrays(
    arrays: "Mapping[str, Any]",
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Rename + transpose PEFT LoRA arrays into mlx-lm shape.

    Returns ``(converted, skipped_keys)``. Non-LoRA keys (embeddings,
    ``modules_to_save`` etc.) are skipped — mlx-lm adapters carry only the
    LoRA matrices. Raises ``ValueError`` when no LoRA keys exist at all.
    """
    converted: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in arrays.items():
        mlx_key = hf_key_to_mlx(key)
        if mlx_key is None:
            skipped.append(key)
            continue
        converted[mlx_key] = value.T  # [r, in] -> [in, r] / [out, r] -> [r, out]
    if not converted:
        raise ValueError(
            "no LoRA keys found in the source adapter (expected "
            "'...lora_A.weight' / '...lora_B.weight' PEFT keys)"
        )
    return converted, tuple(skipped)


def convert_mlx_to_hf_arrays(
    arrays: "Mapping[str, Any]",
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Rename + transpose mlx-lm LoRA arrays back into PEFT shape."""
    converted: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in arrays.items():
        hf_key = mlx_key_to_hf(key)
        if hf_key is None:
            skipped.append(key)
            continue
        converted[hf_key] = value.T
    if not converted:
        raise ValueError(
            "no LoRA keys found in the source adapter (expected "
            "'...lora_a' / '...lora_b' mlx-lm keys)"
        )
    return converted, tuple(skipped)


def _read_adapter_file(path: str, field: str) -> str:
    """Per-file symlink rejection + size cap before any open (TOCTOU)."""
    try:
        file_stat = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(
            f"{field} not found: {os.path.basename(path)}"
        ) from exc
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"{field} must not be a symlink (TOCTOU defence)")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"{field} must be a regular file")
    if file_stat.st_size > _MAX_ADAPTER_FILE_BYTES:
        raise ValueError(
            f"{field} exceeds the {_MAX_ADAPTER_FILE_BYTES // 1024**3} GiB "
            "adapter cap"
        )
    return path


def _load_safetensors_arrays(path: str, field: str) -> dict[str, Any]:
    """Load a safetensors file as numpy arrays (bf16 upcast via torch).

    ``safetensors.numpy.load_file`` raises ``SafetensorError`` (a direct
    ``Exception`` subclass) on bf16 tensors — the overwhelmingly common
    PEFT adapter dtype — so the fallback decision catches broadly, then
    the torch loader either succeeds (bf16 upcast) or proves the file is
    genuinely corrupt (friendly ``ValueError``).
    """
    _read_adapter_file(path, field)
    try:
        from safetensors.numpy import load_file as np_load_file
    except ImportError as exc:  # pragma: no cover — safetensors in [train]
        raise ImportError(
            "apple-adapter conversion requires safetensors "
            "(pip install safetensors)"
        ) from exc
    try:
        return dict(np_load_file(path))
    except Exception as np_exc:  # noqa: BLE001 — SafetensorError is a bare Exception
        # bf16 tensors are not representable in numpy — fall back to the
        # torch loader and upcast to float32 (documented precision note).
        try:
            import torch  # noqa: F401
            from safetensors.torch import load_file as torch_load_file
        except ImportError as exc:
            raise ImportError(
                "this adapter holds non-numpy dtypes (likely bf16); "
                "converting it requires torch "
                "(pip install 'soup-cli[train]')"
            ) from exc
        try:
            tensors = torch_load_file(path)
        except Exception as exc:  # noqa: BLE001 — corrupt file, not a dtype issue
            raise ValueError(
                f"{field} is not a valid safetensors file: "
                f"{type(np_exc).__name__}"
            ) from exc
        logger.warning(
            "adapter %s holds non-numpy dtypes (likely bf16); upcasting "
            "to float32 for conversion",
            os.path.basename(path),
        )
        return {
            key: tensor.float().numpy()
            for key, tensor in tensors.items()
        }


def _load_npz_arrays(path: str, field: str) -> dict[str, Any]:
    """Load a legacy mlx-lm ``adapters.npz`` as a plain dict of arrays.

    The 4 GiB cap is re-applied to the DECOMPRESSED arrays — the on-disk
    cap in ``_read_adapter_file`` bounds only the compressed container
    (zip bomb defence).
    """
    _read_adapter_file(path, field)
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as bundle:
            arrays = {key: bundle[key] for key in bundle.files}
    except Exception as exc:  # noqa: BLE001 — BadZipFile etc. are bare Exception
        raise ValueError(
            f"{field} is not a valid npz file: {type(exc).__name__}"
        ) from exc
    total_bytes = sum(int(getattr(arr, "nbytes", 0)) for arr in arrays.values())
    if total_bytes > _MAX_ADAPTER_FILE_BYTES:
        raise ValueError(
            f"{field} decompresses past the "
            f"{_MAX_ADAPTER_FILE_BYTES // 1024**3} GiB adapter cap"
        )
    return arrays


def _read_source_config(source_dir: str) -> dict[str, Any]:
    """Best-effort read of the source adapter_config.json (never raises)."""
    config_path = os.path.join(source_dir, "adapter_config.json")
    try:
        _read_adapter_file(config_path, "adapter_config.json")
        with open(config_path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _save_safetensors(arrays: dict[str, Any], output_path: str) -> None:
    """Serialise arrays to safetensors bytes and write atomically.

    The converted matrices are ``.T`` views (reversed strides) —
    safetensors serialises the raw base buffer, silently mangling
    non-contiguous input, so every array is made C-contiguous first
    (bug caught by the v0.71.21 review-wave round-trip assertions).
    """
    import numpy as np
    from safetensors.numpy import save as st_save

    contiguous = {
        key: np.ascontiguousarray(value) for key, value in arrays.items()
    }
    atomic_write_bytes(st_save(contiguous), output_path, field="output")


def _infer_rank(hf_arrays: dict[str, Any]) -> Optional[int]:
    """Infer the LoRA rank from any ``lora_A.weight`` matrix ([r, in])."""
    for key, value in hf_arrays.items():
        if key.endswith(".lora_A.weight") and getattr(value, "ndim", 0) == 2:
            return int(value.shape[0])
    return None


def _infer_num_layers(mlx_keys: "Mapping[str, Any]") -> Optional[int]:
    """Derive ``num_layers`` (max decoder index + 1) from converted keys.

    mlx-lm's ``load_adapters`` reads ``config.num_layers`` unconditionally
    before wiring LoRA layers, so the emitted adapter_config.json must
    carry it whenever it is derivable.
    """
    max_index = -1
    for key in mlx_keys:
        match = _LAYER_INDEX_RE.search(key)
        if match is not None:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1 if max_index >= 0 else None


def _source_dropout(source_config: dict[str, Any]) -> float:
    """Carry the source PEFT ``lora_dropout`` through (default 0.0)."""
    dropout = source_config.get("lora_dropout")
    if (
        isinstance(dropout, (int, float))
        and not isinstance(dropout, bool)
        and 0.0 <= float(dropout) < 1.0
    ):
        return float(dropout)
    return 0.0


def _convert_hf_to_mlx(plan: AppleAdapterPlan) -> tuple[int, tuple[str, ...]]:
    source_file = os.path.join(plan.source_dir, "adapter_model.safetensors")
    if not os.path.lexists(source_file):
        if os.path.lexists(os.path.join(plan.source_dir, "adapter_model.bin")):
            raise ValueError(
                "adapter_model.bin (pickle) is not supported — re-save the "
                "adapter as safetensors first (v0.57.0 policy)"
            )
        raise FileNotFoundError(
            "adapter_model.safetensors not found in the source adapter"
        )
    arrays = _load_safetensors_arrays(source_file, "adapter_model.safetensors")
    converted, skipped = convert_hf_to_mlx_arrays(arrays)
    _save_safetensors(
        converted, os.path.join(plan.output_dir, "adapters.safetensors")
    )

    source_config = _read_source_config(plan.source_dir)
    rank = source_config.get("r")
    if not isinstance(rank, int) or isinstance(rank, bool):
        rank = _infer_rank(arrays)
    alpha = source_config.get("lora_alpha")
    mlx_config: dict[str, Any] = {
        "fine_tune_type": "lora",
        "soup_converted_from": "peft",
    }
    num_layers = _infer_num_layers(converted)
    if num_layers is not None:
        mlx_config["num_layers"] = num_layers
    lora_parameters: dict[str, Any] = {
        "dropout": _source_dropout(source_config)
    }
    if isinstance(rank, int) and rank > 0:
        lora_parameters["rank"] = rank
        if isinstance(alpha, (int, float)) and not isinstance(alpha, bool):
            lora_parameters["scale"] = float(alpha) / float(rank)
    mlx_config["lora_parameters"] = lora_parameters
    atomic_write_text(
        json.dumps(mlx_config, indent=2),
        os.path.join(plan.output_dir, "adapter_config.json"),
        field="output",
    )
    return len(converted), skipped


def _convert_mlx_to_hf(plan: AppleAdapterPlan) -> tuple[int, tuple[str, ...]]:
    st_file = os.path.join(plan.source_dir, "adapters.safetensors")
    npz_file = os.path.join(plan.source_dir, "adapters.npz")
    if os.path.lexists(st_file):
        arrays = _load_safetensors_arrays(st_file, "adapters.safetensors")
    elif os.path.lexists(npz_file):
        arrays = _load_npz_arrays(npz_file, "adapters.npz")
    else:
        raise FileNotFoundError(
            "no adapters.npz or adapters.safetensors found in the source "
            "adapter (expected an mlx-lm adapter directory)"
        )
    converted, skipped = convert_mlx_to_hf_arrays(arrays)
    _save_safetensors(
        converted, os.path.join(plan.output_dir, "adapter_model.safetensors")
    )

    source_config = _read_source_config(plan.source_dir)
    rank = _infer_rank(converted)
    target_modules = sorted({
        key[: -len(".lora_A.weight")].rsplit(".", 1)[-1]
        for key in converted
        if key.endswith(".lora_A.weight")
    })
    hf_config: dict[str, Any] = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "soup_converted_from": "mlx",
        "target_modules": target_modules,
    }
    if isinstance(rank, int) and rank > 0:
        hf_config["r"] = rank
        lora_params = source_config.get("lora_parameters")
        scale = (
            lora_params.get("scale") if isinstance(lora_params, dict) else None
        )
        if isinstance(scale, (int, float)) and not isinstance(scale, bool):
            hf_config["lora_alpha"] = float(scale) * rank
    atomic_write_text(
        json.dumps(hf_config, indent=2),
        os.path.join(plan.output_dir, "adapter_config.json"),
        field="output",
    )
    return len(converted), skipped


def convert_apple_adapter(plan: AppleAdapterPlan) -> ConversionReport:
    """Run the live adapter conversion described by ``plan``.

    Live since v0.71.21 (#228) for the ``hf-to-mlx`` / ``mlx-to-hf``
    directions; the two ``*-to-apple`` directions raise a friendly
    upstream-gate RuntimeError until Apple publishes a stable
    FoundationModels adapter spec.
    """
    if not isinstance(plan, AppleAdapterPlan):
        raise TypeError("plan must be AppleAdapterPlan")
    if plan.direction in ("hf-to-apple", "mlx-to-apple"):
        raise RuntimeError(
            "the Apple FoundationModels adapter format has no stable public "
            "spec yet — refusing to export wrong-shaped weights. Track "
            "https://developer.apple.com/documentation/foundationmodels "
            "for the published format."
        )

    enforce_under_cwd_and_no_symlink(plan.output_dir, "output_dir")
    os.makedirs(plan.output_dir, exist_ok=True)

    if plan.direction == "hf-to-mlx":
        converted_count, skipped = _convert_hf_to_mlx(plan)
    else:  # mlx-to-hf — directions are a closed allowlist
        converted_count, skipped = _convert_mlx_to_hf(plan)

    signed = False
    if plan.sign:
        from soup_cli.utils.adapter_sign import sign_adapter

        sign_adapter(plan.output_dir)
        signed = True

    return ConversionReport(
        direction=plan.direction,
        output_dir=plan.output_dir,
        converted_keys=converted_count,
        skipped_keys=skipped,
        signed=signed,
    )


__all__ = [
    "SUPPORTED_ADAPTER_DIRECTIONS",
    "validate_direction",
    "validate_source_adapter",
    "AppleAdapterPlan",
    "ConversionReport",
    "build_apple_adapter_plan",
    "hf_key_to_mlx",
    "mlx_key_to_hf",
    "convert_hf_to_mlx_arrays",
    "convert_mlx_to_hf_arrays",
    "convert_apple_adapter",
]
