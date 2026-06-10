"""v0.52.0 Part D — BitNet 1.58-bit fine-tuning + export schema helpers.

Schema-only support for ``quantization='bitnet_1.58'`` and the new
``soup export --format bitnet`` / ``--format tq1_0`` GGUF flavours.

Live ``onebitllms`` wrapping + llama.cpp ``TQ1_0`` export wiring are
deferred to v0.52.1 (mirrors v0.50.0 stub-then-live pattern).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

# Closed allowlist of BitNet-flavoured quant strings exposed to YAML.
BITNET_QUANT_FORMATS: frozenset[str] = frozenset({"bitnet_1.58"})

# Closed allowlist of BitNet export targets (the actual export formats).
BITNET_EXPORT_FORMATS: frozenset[str] = frozenset({"bitnet", "tq1_0"})

_BITNET_FAMILY_RE_PREFIXES: tuple[str, ...] = (
    "bitnet", "falcon-e", "falcone", "1bitllm", "onebit",
)


@dataclass(frozen=True)
class BitNetSpec:
    """Metadata for the BitNet quant path. Frozen — immutable."""

    name: str
    description: str
    bits: float
    live_wired: bool


_BITNET_METADATA: Mapping[str, BitNetSpec] = MappingProxyType({
    "bitnet_1.58": BitNetSpec(
        name="bitnet_1.58",
        description="BitNet 1.58-bit ternary weights (axolotl + onebitllms)",
        bits=1.58,
        live_wired=False,
    ),
})


def is_bitnet_quant(value: object) -> bool:
    """Return True iff ``value`` is a BitNet quant string."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return False
    return value in BITNET_QUANT_FORMATS


def is_bitnet_export_format(value: object) -> bool:
    """Return True iff ``value`` is a BitNet export-format string."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, str):
        return False
    return value in BITNET_EXPORT_FORMATS


def get_bitnet_spec(name: str) -> BitNetSpec:
    """Return the frozen :class:`BitNetSpec` for ``name`` or raise."""
    if not is_bitnet_quant(name):
        supported = ", ".join(sorted(BITNET_QUANT_FORMATS))
        raise ValueError(
            f"BitNet quant {name!r} not supported. Supported: {supported}"
        )
    return _BITNET_METADATA[name]


def is_bitnet_model(model_name: object) -> bool:
    """Best-effort detect whether ``model_name`` references a BitNet family.

    Checks every slash-delimited component (lowercased) against the
    ``_BITNET_FAMILY_RE_PREFIXES`` prefix list. This is intentionally more
    permissive than v0.39.0 ``is_gemma4_model`` because BitNet families
    typically live under namespaced orgs (``1bitllm/...``, ``OneBitLLM/...``)
    rather than being identifiable by repo name alone.
    """
    if isinstance(model_name, bool):
        return False
    if not isinstance(model_name, str):
        return False
    if not model_name or "\x00" in model_name:
        return False
    # Check each path component so an org name like "1bitllm/foo" matches
    # while still rejecting unrelated substrings (e.g. an SFT model that
    # happens to embed "bitnet" inside a description path).
    for part in model_name.lower().split("/"):
        if any(part.startswith(prefix) for prefix in _BITNET_FAMILY_RE_PREFIXES):
            return True
    return False


def validate_bitnet_compat(*, task: str, backend: str, modality: str) -> None:
    """Schema-time gate for ``quantization='bitnet_1.58'``.

    Rejects:
    - non-string / bool args (defence-in-depth).
    - ``backend == 'mlx'`` — onebitllms is CUDA-only in v0.52.0.
    - ``modality != 'text'`` — vision/audio BitNet not modelled.
    - ``task`` outside {sft, pretrain, dpo} — BitNet wiring is text-LM
      training only this release.
    """
    for name, value in (("task", task), ("backend", backend), ("modality", modality)):
        if isinstance(value, bool):
            raise TypeError(f"{name} must not be bool, got {value!r}")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if backend == "mlx":
        raise ValueError(
            "quantization='bitnet_1.58' is not supported on backend=mlx "
            "(onebitllms is CUDA-only). Use backend='transformers'."
        )
    if modality != "text":
        raise ValueError(
            f"quantization='bitnet_1.58' is wired for modality='text' only; "
            f"got modality={modality!r}"
        )
    if task not in ("sft", "pretrain", "dpo"):
        raise ValueError(
            f"quantization='bitnet_1.58' is only wired for "
            f"task in (sft, pretrain, dpo); got task={task!r}"
        )


def validate_bitnet_export(format_name: object) -> str:
    """Validate a BitNet export-format string. Returns canonical form."""
    if isinstance(format_name, bool):
        raise TypeError(
            f"bitnet export format must not be bool, got {format_name!r}"
        )
    if not isinstance(format_name, str):
        raise TypeError(
            f"bitnet export format must be str, "
            f"got {type(format_name).__name__}"
        )
    if not format_name:
        raise ValueError("bitnet export format must be non-empty")
    if "\x00" in format_name:
        raise ValueError(
            "bitnet export format must not contain null bytes"
        )
    canonical = format_name.lower()
    if canonical not in BITNET_EXPORT_FORMATS:
        supported = ", ".join(sorted(BITNET_EXPORT_FORMATS))
        raise ValueError(
            f"bitnet export format {format_name!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


# BitNet export-format → llama.cpp ternary quantize CLI arg. Both the
# friendly ``bitnet`` alias and the explicit ``tq1_0`` map to TQ1_0 (the
# llama.cpp 1.58-bit ternary GGUF type).
_BITNET_GGUF_QUANT_ARG: Mapping[str, str] = MappingProxyType({
    "bitnet": "TQ1_0",
    "tq1_0": "TQ1_0",
})


def build_bitnet_trainer(config: object, **kwargs: object):
    """Live BitNet 1.58-bit trainer factory (v0.71.20 #134).

    Returns a :class:`~soup_cli.trainer.bitnet.BitNetTrainerWrapper`. BitNet
    1.58 fine-tuning trains an SFT-style next-token CE objective on a model
    whose ``BitLinear`` layers carry ternary weights. The faithful training
    path needs the upstream ``onebitllms`` package (CUDA / Linux only); the
    wrapper surfaces a friendly ``RuntimeError`` naming it when absent.

    Lazy import keeps ``soup_cli.utils.bitnet`` torch-free.
    """
    from soup_cli.trainer.bitnet import BitNetTrainerWrapper

    return BitNetTrainerWrapper(config, **kwargs)


def export_bitnet_gguf(
    *,
    model_dir: str,
    output_path: str,
    export_format: str,
    llama_cpp_dir: str,
) -> None:
    """Export a BitNet model as a TQ1_0 (1.58-bit ternary) GGUF (v0.71.20 #134).

    Two-stage llama.cpp pipeline (reuses the v0.53.1 gguf machinery):
    1. ``convert_hf_to_gguf.py`` → ``f16.gguf``
    2. ``llama-quantize`` with the ``TQ1_0`` flavour → ``output_path``

    No importance matrix is needed — ternary weights export directly. All
    subprocess invocations use argv-list form (no shell). cwd containment +
    symlink rejection mirror ``export_advanced_gguf``.

    Requires a built llama.cpp toolchain; the convert/quantize binaries raise a
    friendly ``FileNotFoundError`` naming the missing piece when absent.
    """
    import os
    import tempfile
    from pathlib import Path

    from soup_cli.utils.gguf_quant import (
        _enforce_under_cwd_and_no_symlink,
        _run_convert_to_f16,
        _run_quantize_binary,
    )

    flavour = validate_bitnet_export(export_format)
    quant_arg = _BITNET_GGUF_QUANT_ARG[flavour]

    _enforce_under_cwd_and_no_symlink(model_dir, "model_dir")
    _enforce_under_cwd_and_no_symlink(output_path, "output_path")
    _enforce_under_cwd_and_no_symlink(llama_cpp_dir, "llama_cpp_dir")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model_dir not a directory: {os.path.basename(model_dir)!r}"
        )
    if not os.path.isdir(llama_cpp_dir):
        raise FileNotFoundError(
            f"llama_cpp_dir not a directory: {os.path.basename(llama_cpp_dir)!r}"
        )

    with tempfile.TemporaryDirectory(
        prefix=".soup_bitnet_gguf_", dir=str(Path.cwd()),
    ) as staged:
        f16_path = Path(staged) / "model.f16.gguf"
        _run_convert_to_f16(llama_cpp_dir, model_dir, str(f16_path))
        _run_quantize_binary(
            llama_cpp_dir=llama_cpp_dir,
            f16_path=str(f16_path),
            output_path=output_path,
            flavour=quant_arg,
            imatrix_path=None,
        )

    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"llama-quantize did not produce {os.path.basename(output_path)!r}"
        )
