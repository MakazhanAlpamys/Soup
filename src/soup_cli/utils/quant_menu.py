"""Train-time quantization menu (v0.38.0).

Wires up GPTQ / AWQ / HQQ / AQLM / EETQ / MXFP4 + FP8-dequant load paths so
LoRA can train on top of pre-quantized base checkpoints. Each format follows
the same shape:

* ``build_<fmt>_config(...)``  — returns a dict-shaped quantization_config that
  the trainer hands to ``AutoModel.from_pretrained``. Keeping the helper
  output dict-shaped lets us mock heavy library imports in tests.
* ``validate_<fmt>_checkpoint(ref)`` — runtime probe (fail-fast at train start)
  that the referenced model is genuinely the right format. Local paths are
  inspected; HF repo IDs (no path separator) fall through and let HF surface
  any error itself.

All public helpers reject null bytes in caller-supplied strings — matches the
project-wide hardening policy (see CLAUDE.md security section).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console

    from soup_cli.config.schema import TrainingConfig


# Module-level set so schema.py validators can reference the same source of truth.
PREQUANTIZED_FORMATS: frozenset[str] = frozenset(
    {"gptq", "awq", "aqlm", "eetq", "mxfp4", "fp8"}
)


def is_quant_menu_format(quantization: str) -> bool:
    """True for v0.38.0 quant-menu values — seven formats total: HQQ at any
    bit-rate plus ``gptq`` / ``awq`` / ``aqlm`` / ``eetq`` / ``mxfp4`` /
    ``fp8``. False for legacy ``4bit`` / ``8bit`` / ``none``.
    """
    return (
        quantization in PREQUANTIZED_FORMATS
        or quantization.startswith("hqq:")
    )

_NULL_BYTE = "\x00"


def _reject_null_bytes(value: str, label: str) -> None:
    if _NULL_BYTE in value:
        raise ValueError(f"{label} must not contain null bytes")


def _looks_like_local_path(ref: str) -> bool:
    """A reference points to a local checkpoint iff it actually exists on
    disk. HF repo IDs (e.g. ``TheBloke/Llama-2-7B-GPTQ``) fall through.

    Edge case: if the user happens to have a local directory whose path
    matches an HF repo id (e.g. they ran ``git clone`` into ``./TheBloke/...``)
    they will hit the local-path branch — at which point the ``quantize_config.json``
    check will run on real bytes, which is the safer outcome.
    """
    return os.path.exists(ref)


# ---------------------------------------------------------------------------
# Part A — GPTQ
# ---------------------------------------------------------------------------


def build_gptq_config(
    *,
    bits: int = 4,
    disable_exllama: bool = True,
) -> dict[str, Any]:
    """Build a GPTQ quantization-config dict.

    The returned dict can be passed via ``quantization_config=...`` after
    converting to ``transformers.GPTQConfig``. We deliberately return a dict
    rather than constructing the GPTQConfig at module import time so tests
    don't need transformers + optimum installed.

    PEFT requires the triton backend — exllama silently breaks adapter
    gradient flow. Default is ``disable_exllama=True``.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"GPTQ bits must be one of (2, 3, 4, 8); got {bits}")
    return {
        "bits": bits,
        "use_exllama": not disable_exllama,
    }


def validate_gptq_checkpoint(ref: str) -> None:
    """Best-effort check that ``ref`` resolves to a GPTQ-quantized model.

    Local paths must contain ``quantize_config.json``; HF repo IDs are
    accepted on faith (the actual download will fail loudly if wrong).
    Raises ``FileNotFoundError`` for missing local config; ``ValueError``
    for null bytes.
    """
    _check_local_marker(ref, label="GPTQ", marker="quantize_config.json")


def _check_local_marker(ref: str, *, label: str, marker: str) -> None:
    if not isinstance(ref, str):
        raise TypeError(f"{label} ref must be str, got {type(ref).__name__}")
    _reject_null_bytes(ref, f"{label} ref")
    if _looks_like_local_path(ref):
        marker_path = os.path.join(ref, marker)
        if not os.path.isfile(marker_path):
            # Reduce ref to basename in error to avoid leaking absolute paths
            # (matches v0.34.0 crash.py policy).
            displayed = os.path.basename(os.path.normpath(ref)) or ref
            raise FileNotFoundError(
                f"{label} checkpoint {displayed!r} has no {marker}. "
                f"Either point at a pre-quantized {label} model, or convert first."
            )


# ---------------------------------------------------------------------------
# Part B — AWQ
# ---------------------------------------------------------------------------


def build_awq_config(
    *,
    bits: int = 4,
    version: str = "gemm",
) -> dict[str, Any]:
    """Build an AWQ quantization-config dict.

    AWQ is always 4-bit (autoawq's only supported width). ``version`` is
    "gemm" for general use (default) or "gemv" for low-batch inference.
    """
    if bits != 4:
        raise ValueError(f"AWQ supports only 4-bit; got {bits}")
    if version not in ("gemm", "gemv"):
        raise ValueError(f"AWQ version must be 'gemm' or 'gemv'; got {version!r}")
    return {"bits": bits, "version": version}


def validate_awq_checkpoint(ref: str) -> None:
    """Local paths must contain ``quant_config.json``; HF repo IDs accepted."""
    _check_local_marker(ref, label="AWQ", marker="quant_config.json")


# ---------------------------------------------------------------------------
# Part C — HQQ 1-8 bit
# ---------------------------------------------------------------------------


_HQQ_BITS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8)


def parse_hqq_bits(quantization: str) -> int:
    """Extract bit-rate from a ``hqq:Nbit`` string."""
    if not quantization.startswith("hqq:"):
        raise ValueError(
            f"build_hqq_config expects a 'hqq:Nbit' string, got {quantization!r}"
        )
    suffix = quantization.split(":", 1)[1]
    # DoS cap — closed Literal in schema makes this defence-in-depth, but
    # public callers (Autopilot, recipe builders) might pass arbitrary strings.
    if len(suffix) > 16:
        raise ValueError("HQQ suffix exceeds 16 chars")
    if not suffix.endswith("bit"):
        raise ValueError(f"HQQ suffix must end with 'bit', got {suffix!r}")
    try:
        bits = int(suffix[:-3])
    except ValueError as exc:
        raise ValueError(f"HQQ bits must be int, got {suffix!r}") from exc
    if bits not in _HQQ_BITS:
        raise ValueError(
            f"HQQ bits must be one of {_HQQ_BITS}; got {bits}"
        )
    return bits


def build_hqq_config(
    *,
    quantization: str,
    group_size: int = 64,
) -> dict[str, Any]:
    """Build an HQQ quantization-config dict.

    Returned shape mirrors ``hqq.core.quantize.BaseQuantizeConfig`` kwargs
    (``nbits`` + ``group_size``). We expose ``bits`` for parity with the
    other builders, callers translate to ``nbits`` at HfApi seam.
    """
    bits = parse_hqq_bits(quantization)
    if (
        isinstance(group_size, bool)
        or not isinstance(group_size, int)
        or group_size < 32
    ):
        raise ValueError(
            f"HQQ group_size must be int >= 32, got {group_size!r}"
        )
    return {"bits": bits, "group_size": group_size}


# ---------------------------------------------------------------------------
# Part D — AQLM 2-bit
# ---------------------------------------------------------------------------


def build_aqlm_config() -> dict[str, Any]:
    """AQLM is locked to 2-bit (LlamaFactory enforces this — quantization.py:125)."""
    return {"bits": 2}


# ---------------------------------------------------------------------------
# Part E — EETQ 8-bit
# ---------------------------------------------------------------------------


def build_eetq_config() -> dict[str, Any]:
    """EETQ is locked to 8-bit."""
    return {"bits": 8}


# ---------------------------------------------------------------------------
# Part F — MXFP4 / FP8 dequantize-on-load
# ---------------------------------------------------------------------------


def build_mxfp4_config() -> dict[str, Any]:
    """BNB 4-bit MXFP4 quant_type — newer-than-NF4 4-bit format with better
    activation distribution. Requires bitsandbytes >= 0.45 + CUDA.
    """
    return {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "mxfp4",
    }


def build_fp8_dequant_config() -> dict[str, Any]:
    """Dequantize-on-load for FP8 inference checkpoints. Trains in fp16/bf16
    after upcast — useful for fine-tuning FP8-released base models without
    requiring a Hopper+ GPU at training time.
    """
    return {"dequantize": True}


# ---------------------------------------------------------------------------
# Part H — Compatibility matrix (quant × multi-GPU)
# ---------------------------------------------------------------------------


# Source-of-truth matrix. Each entry is a tuple (problems, warnings) keyed by
# (quant_family, distributed_strategy). Family names: "bnb4", "bnb8", "gptq",
# "awq", "hqq", "aqlm", "eetq", "mxfp4", "fp8", "none". Strategies: "ddp",
# "fsdp", "zero1", "zero2", "zero3".
#
# Derived from LlamaFactory `quantization.py` (see plan file:line refs in
# v0.38.0 plan), Axolotl docs/multipack-vs-quant.md, and bitsandbytes README.
_INCOMPATIBLE: dict[tuple[str, str], str] = {
    ("hqq", "zero3"): (
        "HQQ is incompatible with DeepSpeed ZeRO-3 "
        "(LlamaFactory quantization.py:199)."
    ),
    ("hqq", "fsdp"): (
        "HQQ is incompatible with FSDP (state-dict assumes dense weights)."
    ),
    ("eetq", "zero3"): (
        "EETQ is incompatible with DeepSpeed ZeRO-3 "
        "(LlamaFactory quantization.py:211)."
    ),
    ("eetq", "fsdp"): (
        "EETQ is incompatible with FSDP (8-bit kernels assume single-device)."
    ),
    ("aqlm", "zero3"): (
        "AQLM is incompatible with DeepSpeed ZeRO-3 "
        "(sharded codes break dequant)."
    ),
    ("aqlm", "fsdp"): (
        "AQLM is incompatible with FSDP (sharded codes break dequant)."
    ),
}


def _quant_family(quantization: str) -> str:
    if quantization == "4bit":
        return "bnb4"
    if quantization == "8bit":
        return "bnb8"
    if quantization == "none":
        return "none"
    if quantization.startswith("hqq:"):
        return "hqq"
    if quantization in {"gptq", "awq", "aqlm", "eetq", "mxfp4", "fp8"}:
        return quantization
    raise ValueError(f"unknown quantization {quantization!r}")


_DS_ZERO3_TOKENS: frozenset[str] = frozenset(
    {"zero3", "zero++", "zeropp", "zero_pp", "stage3"}
)
_DS_ZERO2_TOKENS: frozenset[str] = frozenset({"zero2", "stage2"})
_DS_ZERO1_TOKENS: frozenset[str] = frozenset({"zero1", "stage1"})


def _distributed_strategy(deepspeed: object, fsdp: bool) -> str:
    """Map ``--deepspeed <preset>`` + ``--fsdp <preset>`` to a strategy key.

    Returns one of: ``"fsdp"``, ``"zero1"``, ``"zero2"``, ``"zero3"``, or
    ``"ddp"`` (fallback). FSDP wins over DeepSpeed when both are set —
    matches the train.py ordering and the LlamaFactory matrix.
    """
    if fsdp:
        return "fsdp"
    if deepspeed is None or deepspeed == "":
        return "ddp"
    token = str(deepspeed).lower().replace("-", "").replace("_", "")
    if token in _DS_ZERO3_TOKENS:
        return "zero3"
    if token in _DS_ZERO2_TOKENS:
        return "zero2"
    if token in _DS_ZERO1_TOKENS:
        return "zero1"
    # Fall through: be conservative — anything unrecognized is treated as
    # DDP rather than guessing a stage from ambiguous substrings.
    return "ddp"


def build_quantization_config_for_loader(
    *,
    tcfg: TrainingConfig,
    base: str,
    console: Console | None = None,
) -> Any:
    """Single entry point for trainers — returns a transformers
    quantization_config object (or ``None`` for ``quantization='none'``).

    Lazy-imports transformers / bitsandbytes / torch so the CLI startup stays
    lean. Validates pre-quantized checkpoints at the seam.
    """
    quantization = tcfg.quantization
    if quantization == "none":
        return None
    if quantization == "4bit":
        from transformers import BitsAndBytesConfig

        from soup_cli.utils.gpu import get_compute_dtype

        kwargs: dict[str, Any] = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": get_compute_dtype(),
            "bnb_4bit_use_double_quant": True,
        }
        storage = getattr(tcfg, "bnb_4bit_quant_storage", None)
        if storage:
            kwargs["bnb_4bit_quant_storage"] = _resolve_torch_dtype(storage)
        return BitsAndBytesConfig(**kwargs)
    if quantization == "8bit":
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "gptq":
        from transformers import GPTQConfig

        validate_gptq_checkpoint(base)
        gptq_kwargs = build_gptq_config(
            disable_exllama=tcfg.gptq_disable_exllama
        )
        if console is not None:
            console.print(
                f"[green]GPTQ:[/] pre-quantized checkpoint "
                f"(use_exllama={gptq_kwargs['use_exllama']})"
            )
        return GPTQConfig(**gptq_kwargs)
    if quantization == "awq":
        from transformers import AwqConfig

        validate_awq_checkpoint(base)
        awq_kwargs = build_awq_config()
        if console is not None:
            console.print(
                f"[green]AWQ:[/] pre-quantized checkpoint "
                f"(version={awq_kwargs['version']})"
            )
        return AwqConfig(**awq_kwargs)
    if quantization.startswith("hqq:"):
        from transformers import HqqConfig

        hqq_kwargs = build_hqq_config(quantization=quantization)
        if console is not None:
            console.print(
                f"[green]HQQ:[/] {hqq_kwargs['bits']}-bit "
                f"(group_size={hqq_kwargs['group_size']})"
            )
        # transformers HqqConfig uses nbits, not bits.
        return HqqConfig(
            nbits=hqq_kwargs["bits"], group_size=hqq_kwargs["group_size"]
        )
    if quantization == "aqlm":
        from transformers import AqlmConfig

        if console is not None:
            console.print(
                "[green]AQLM:[/] 2-bit pre-quantized checkpoint"
            )
        return AqlmConfig()
    if quantization == "eetq":
        from transformers import EetqConfig

        if console is not None:
            console.print(
                "[green]EETQ:[/] 8-bit (transformers EetqConfig)"
            )
        return EetqConfig()
    if quantization == "mxfp4":
        from transformers import BitsAndBytesConfig

        kwargs = build_mxfp4_config()
        storage = getattr(tcfg, "bnb_4bit_quant_storage", None)
        if storage:
            kwargs["bnb_4bit_quant_storage"] = _resolve_torch_dtype(storage)
        if console is not None:
            console.print("[green]MXFP4:[/] BNB 4-bit quant_type='mxfp4'")
        return BitsAndBytesConfig(**kwargs)
    if quantization == "fp8":
        try:
            from transformers import FineGrainedFP8Config as _FP8Cfg
        except ImportError:
            try:
                from transformers import FP8Config as _FP8Cfg
            except ImportError as exc:
                raise RuntimeError(
                    "transformers does not expose an FP8 config — "
                    "upgrade transformers >= 4.45 to use quantization='fp8'."
                ) from exc
        if console is not None:
            console.print("[green]FP8:[/] dequantize-on-load")
        return _FP8Cfg(**build_fp8_dequant_config())
    raise ValueError(f"unknown quantization {quantization!r}")


def _resolve_torch_dtype(name: str) -> Any:
    """Translate 'bfloat16' / 'float16' / etc. to torch dtypes (lazy import)."""
    import torch

    mapping = {
        "uint8": torch.uint8,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"unsupported quant_storage dtype: {name!r}")
    return mapping[name]


def check_quant_distributed_compat(
    *,
    quantization: str,
    deepspeed: object = None,
    fsdp: bool = False,
    bnb_4bit_quant_storage: object = None,
) -> list[str]:
    """Return a list of human-readable problem strings (empty = OK).

    Hard incompatibilities are emitted as plain strings; advisories use a
    leading ``"warning:"`` prefix so callers can split severity. Used by
    ``soup train`` at startup.
    """
    family = _quant_family(quantization)
    strategy = _distributed_strategy(deepspeed, fsdp)
    problems: list[str] = []
    msg = _INCOMPATIBLE.get((family, strategy))
    if msg:
        problems.append(msg)
    # BNB 4-bit + FSDP without quant_storage = silent perf foot-gun.
    if family == "bnb4" and strategy == "fsdp" and not bnb_4bit_quant_storage:
        problems.append(
            "warning: BNB 4-bit + FSDP without bnb_4bit_quant_storage causes "
            "all-gather to upcast to fp32. Set bnb_4bit_quant_storage to your "
            "compute dtype (bfloat16 or float16) for a 2-3x speed-up. "
            "(LlamaFactory quantization.py:178 'crucial for fsdp+qlora')"
        )
    return problems
