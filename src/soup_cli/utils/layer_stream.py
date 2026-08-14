"""training.stream_layers — layer streaming planner (v0.72.0 BETA).

The pure half: tier choice, pinned-vs-pageable decision, the architecture
allowlist, and the VRAM / throughput arithmetic. **No top-level torch** — this
module sits on the light CLI's import path so `soup profile` and friends can
forecast a streaming run without pulling in the training stack.

The runtime half (buffer pool, weight source, prefetch scheduler, layer
wrapper) lives in ``layer_stream_runtime.py``; the checkpoint sharder lives in
``layer_shard.py``.

Model of the mechanism: the frozen base lives in CPU RAM and is streamed into a
small pool of pre-allocated VRAM buffers one decoder layer at a time, so peak
VRAM is bounded by ONE layer rather than the whole model. Only the LoRA
adapters, their gradients and optimizer state stay resident.
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

from rich.panel import Panel

# Quantisation names are re-exported, NOT redeclared: layer_shard owns the shard
# format and these same strings key its cache-invalidation check. Two copies
# could drift and leave this module's pre-flight RAM estimate disagreeing with
# what was actually written to disk. (layer_shard has no top-level torch either,
# so the light CLI import path is unaffected.)
from soup_cli.utils.layer_shard import QUANT_NONE, SUPPORTED_STREAM_QUANTS

# --- tiers (plan 5.1) -----------------------------------------------------
TIER_RAM = "ram"
TIER_DISK = "disk"
STREAM_SOURCES = ("auto", "ram", "disk")

#: model_bytes must be under this fraction of free RAM to claim the RAM tier.
RAM_TIER_HEADROOM = 0.7

# --- buffers --------------------------------------------------------------
MIN_STREAM_BUFFERS = 2
MAX_STREAM_BUFFERS = 8
DEFAULT_STREAM_BUFFERS = 2

# --- tasks ----------------------------------------------------------------
#: Tasks whose trainers can run against a streamed base (v0.72.4).
#:
#: DPO and KTO take their reference model from the SAME streamed base with the
#: adapters disabled (TRL's ``null_ref_context``), so the reference costs no
#: extra weights at all — measured 0.914x the SFT peak, where forcing a real
#: second instance cost 9.92x. ORPO and SimPO are reference-free.
SUPPORTED_STREAM_TASKS = ("sft", "dpo", "orpo", "simpo", "kto")

#: Tasks PERMANENTLY excluded, not merely unimplemented. Generation rollouts
#: re-read every layer once per generated token, which destroys the whole
#: premise: streaming amortises one weight read over a training step, not over a
#: single decoded token (plan §3.2).
ROLLOUT_STREAM_TASKS = ("grpo", "ppo")

#: FLOPs per parameter per token. 6 == WITH gradient checkpointing
#: (2 forward + 2 recompute + 2 dL/dx; base weight-grads are skipped because
#: the base is frozen). Streaming always checkpoints, so this is never 4.
FLOPS_PER_PARAM_PER_TOKEN = 6

#: Families whose decoder layout has been proven bit-exact against a resident
#: run of the same numerics (v0.72.0 llama/qwen2/qwen3; v0.72.3 GATE 1 added the
#: rest, under bf16 AND NF4). An allowlist rather than a heuristic because a
#: half-supported architecture streams weights into the wrong module and
#: mis-trains silently instead of crashing.
#:
#: ``gemma3_text`` is present but ``gemma3`` is NOT: a real ``google/gemma-3-*``
#: reports ``model_type='gemma3'`` for the vision-capable wrapper, and streaming
#: a multimodal wrapper as though it were a causal LM is exactly the failure the
#: allowlist exists to prevent.
SUPPORTED_STREAM_ARCHS = (
    "llama",
    "qwen2",
    "qwen3",
    "mistral",
    "gemma",
    "gemma2",
    "gemma3_text",
    "phi",
    "phi3",
)

#: The loss path's own arithmetic, in VRAM bytes per logit element. **Measured
#: stage by stage (issue #327), not derived.** ``ForCausalLMLoss`` upcasts to
#: fp32, hands the view to ``cross_entropy`` and returns; the residency at each
#: stage, per element, excluding the source logits tensor:
#:
#:   after the loss returns   4  — only log-softmax's saved fp32 output survives;
#:                                 the fp32 upcast is freed with the local
#:   backward peak           12  — that saved output plus the two transient fp32
#:                                 gradient buffers (nll -> log-softmax)
#:
#: So the peak holds THREE fp32 logits-shaped buffers, not "upcast + softmax +
#: grad" as v0.72.3 recorded — the totals agree, the decomposition did not.
#: Measured 12.000000 exactly, spread 0.00e+00 over 3 repeats, at every
#: (vocab, tokens) probed, and **byte-identical under torch 2.5.1 and 2.13.0
#: and under trl 0.19.1 and 0.26.2** — this term is not stack-dependent.
LOGITS_LOSS_BYTES_PER_ELEMENT = 12

#: One further bf16 logits-shaped tensor, charged unconditionally.
#:
#: This is the whole of the inter-stack disagreement in issue #327. The v0.72.3
#: grid (RTX 3050 / Windows) needs 13.869-13.901 bytes per element across its
#: ten rows; the H100 grid fits 12.311, and a real ``trl`` training step
#: measured on that box marginals at 12.0955 (12.0821 once the probe fixture's
#: own per-token cost is removed). The excess the RTX grid requires over that
#: is **1.8031 bytes per element**, fitted across its 3.1x vocab contrast with a
#: flat residual of 21.0 bytes per token — i.e. the excess is vocab-scaled, the
#: size of one bf16 copy of the logits, and not a mis-modelled activation term.
#: A single-variable control reproduces it directly: holding the model
#: output object across ``backward()`` instead of letting it die with the local
#: costs **+2.0000 bytes per element**, at 5 of 6 (vocab, tokens) cells and
#: +2.0480 at the sixth (allocator rounding on a 32k-vocab, 256-token cell).
#:
#: What retains it on that stack is NOT identified. torch, trl and transformers
#: were each swapped to the RTX grid's versions on the H100 box and the real
#: training step measured byte-identical, so none of them is the cause. Until
#: something can OBSERVE the retention at pre-flight — a synthetic probe cannot,
#: it sees only its own reference — this stays charged: dropping it under-
#: predicts 10 of the 10 v0.72.3 rows by up to 10.49%, and on Windows an
#: under-prediction does not raise, it silently spills to host memory.
LOGITS_RETENTION_BYTES_PER_ELEMENT = 2

#: VRAM bytes per logit element at peak: the loss arithmetic plus the retained
#: copy. v0.72.0 charged 6 from first principles, which under-predicts this term
#: by 2.33x — a ~5 GB error on a 152k-vocab model at batch 8, where the logits
#: tensor is 146x the entire buffer pool.
LOGITS_BYTES_PER_ELEMENT = LOGITS_LOSS_BYTES_PER_ELEMENT + LOGITS_RETENTION_BYTES_PER_ELEMENT
#: Forward only, no loss: just the bf16 logits.
LOGITS_BYTES_PER_ELEMENT_NO_LOSS = 2

#: Bytes per trainable adapter parameter at peak: fp32 weight + fp32 grad +
#: Adam ``m`` + ``v``. Charged even under an 8-bit optimizer (which needs 10) —
#: over-charging ~6 bytes across a few million adapter parameters is noise next
#: to the logits term, and it errs in the safe direction.
OPTIMIZER_BYTES_PER_PARAM = 16

#: Constant offset measured across both GATE 2 models (RoPE caches, allocator
#: rounding, small per-run buffers). It came out at ~13.3 MB for a 135M model
#: and a 0.5B model alike, so it is charged as a constant rather than scaled.
STREAM_FIXED_SLACK_BYTES = 13_500_000

#: Fraction of the *measured same-session GEMM ceiling* that real streamed
#: training actually reached on the dev box: 68% (Llama-3.1-8B NF4, 5.26 of
#: 7.74 TFLOPS) up to ~100% (small models, fully pinned store). A forecast
#: quoted as a single number would over-promise by up to 1.5x, so the bracket
#: travels with it.
MEASURED_CEILING_FRACTION = (0.68, 1.0)

#: ``uint8`` is a STORAGE dtype only — NF4 packs two nibbles per byte and (under
#: double quant) stores absmax as uint8 too. It is never a base/compute dtype,
#: which is what ``SUPPORTED_STREAM_DTYPES`` lists.
_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "uint8": 1}
SUPPORTED_STREAM_DTYPES = ("bfloat16", "float16", "float32")

# --- NF4 (v0.72.2) --------------------------------------------------------
#: Streamed bytes per parameter under NF4 + double quant: packed ``N/2`` +
#: absmax ``N/64`` + nested absmax ``N/64/256*4`` + a 4-byte offset per weight.
NF4_BYTES_PER_PARAM = 0.5 + 1 / 64 + 4 / (64 * 256)
#: Without double quant the per-block absmax stays float32: ``N/2 + 4*N/64``.
NF4_BYTES_PER_PARAM_SINGLE = 0.5 + 4 / 64

#: CUDA context + allocator fragmentation headroom (plan 4.1).
DEFAULT_WORKSPACE_BYTES = 1_000_000_000

_NVME = "nvme"


def dtype_bytes(name: str) -> int:
    """Bytes per element. THE single dtype-size table for layer streaming —
    the sharder, the planner and the trainer must not each keep their own."""
    try:
        return _DTYPE_BYTES[name]
    except KeyError:
        raise ValueError(
            f"unsupported dtype {name!r} for layer streaming; "
            f"supported: {', '.join(SUPPORTED_STREAM_DTYPES)}"
        ) from None


def resolve_stream_dtype(device: str = "cuda") -> str:
    """The store/compute dtype for a streamed run, chosen from the CARD.

    Until v0.73.x this was the literal ``"bfloat16" if on_cuda else "float32"``
    in ``trainer/stream_setup.py``, which is wrong on every pre-Ampere GPU:
    Colab's free tier is a T4 (sm_75), Kaggle is a T4 or a P100, and V100 /
    GTX 16xx / RTX 20xx all report ``is_bf16_supported() == False``. That is
    most of the hardware this feature exists for, and it could not fail on the
    Ampere card every published measurement was taken on (#385).

    The capability question is delegated to ``utils.gpu.get_compute_dtype``
    rather than answered again here — one question, one answer, or the two
    drift. CPU stays float32: CPU streaming is a test convenience, and
    half-precision CPU kernels are not uniformly available.

    Correctness is unaffected by the choice. Streamed-vs-resident logits were
    measured bit-exact (``0.000000e+00``) in float16 as well as bfloat16, in
    both quantisations, against resident references of matching numerics.
    """
    if not str(device).lower().startswith("cuda"):
        return "float32"

    from soup_cli.utils.gpu import get_compute_dtype

    name = str(get_compute_dtype()).removeprefix("torch.")
    # get_compute_dtype already returns float32 when no CUDA runtime is present,
    # so a "cuda" string on a CPU-only box does not claim a GPU dtype.
    return name if name in SUPPORTED_STREAM_DTYPES else "float32"


# ==========================================================================
# architecture gate
# ==========================================================================
def stream_arch_of(config: Any) -> str:
    """Return the streaming architecture family, or raise naming the allowlist.

    Mirrors ``utils/shrink.arch_family_of_config`` — an allowlist, not a
    heuristic, because a half-supported architecture streams weights into the
    wrong module and produces silently wrong numbers rather than a crash.
    """
    model_type = getattr(config, "model_type", None)
    if not model_type or not isinstance(model_type, str):
        raise ValueError(
            "layer streaming needs config.model_type to pick an architecture; "
            "none was found on the model config"
        )
    family = model_type.strip().lower()
    if family not in SUPPORTED_STREAM_ARCHS:
        raise ValueError(
            f"layer streaming does not support model_type={family!r}. "
            f"Supported: {', '.join(SUPPORTED_STREAM_ARCHS)}. "
            f"More architectures land in v0.72.3."
        )
    return family


# ==========================================================================
# storage tier (plan 5.1)
# ==========================================================================
DISK_KINDS = (_NVME, "ssd", "hdd", "unknown")

#: A media type, or a thunk that determines one. ``choose_tier`` accepts the
#: thunk form so the ~9 s Windows probe is paid only when the tier decision
#: actually depends on the answer.
DiskKind = Union[str, Callable[[], str]]

_DISK_KIND_CACHE: Dict[str, str] = {}


def detect_disk_kind(path: str = ".") -> str:
    """Media type of the volume holding ``path``: nvme / ssd / hdd / unknown.

    **Costs about 9 s on Windows** (measured), because the only reliable source
    is a PowerShell ``Get-PhysicalDisk`` CIM query. That is why ``choose_tier``
    takes a *callable* and only invokes it when the base does not fit in RAM —
    the answer is irrelevant on the RAM tier, which is the common case. The
    result is cached per process.

    ``unknown`` is returned rather than guessed whenever the platform cannot be
    probed, and ``choose_tier`` refuses it. Refusing is the safe direction: the
    cost of wrongly believing a spinning disk is NVMe is a run that thrashes for
    hours (plan P11 — 80 shards x 2 reads = 160 seeks per step).
    """
    import os

    resolved = os.path.realpath(os.path.expanduser(path))
    # Key on the volume where there is one (Windows) and on the resolved path
    # otherwise, so two spellings of the same location share a cache entry.
    key = os.path.splitdrive(resolved)[0] or resolved
    if key in _DISK_KIND_CACHE:
        return _DISK_KIND_CACHE[key]
    kind = _probe_disk_kind(path)
    _DISK_KIND_CACHE[key] = kind
    return kind


def _resolve_tool(name: str, *fallbacks: str) -> Optional[str]:
    """Absolute path to a system tool, or None.

    Bare names are deliberately never handed to ``subprocess``: on Windows
    ``CreateProcess`` searches the CURRENT DIRECTORY before ``PATH``, so
    ``["powershell", ...]`` run from a freshly-cloned project would execute an
    attacker-planted ``powershell.exe`` sitting in that checkout (CWE-427). No
    shell metacharacters required. POSIX ``execvp`` searches ``$PATH`` only, so
    this matters most on Windows — but resolving everywhere costs nothing.
    """
    import os
    import shutil

    found = shutil.which(name)
    if found:
        return found
    for candidate in fallbacks:
        if os.path.isfile(candidate):
            return candidate
    return None


_POWERSHELL_FALLBACK = r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"


def _probe_disk_kind(path: str) -> str:
    import os
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Linux":
            # /sys/block/<dev>/queue/rotational: 1 spinning, 0 solid state.
            # Instant and authoritative; the device name distinguishes NVMe.
            names = sorted(os.listdir("/sys/block"))
            for dev in names:
                if not dev.startswith("nvme"):
                    continue
                return _NVME
            for dev in names:
                rot = os.path.join("/sys/block", dev, "queue", "rotational")
                if os.path.exists(rot):
                    with open(rot, encoding="utf-8") as handle:
                        return "hdd" if handle.read().strip() == "1" else "ssd"
            return "unknown"
        if system == "Windows":
            shell = _resolve_tool("powershell", _POWERSHELL_FALLBACK)
            if shell is None:
                return "unknown"
            out = subprocess.run(
                [
                    shell, "-NoProfile", "-NonInteractive", "-Command",
                    "Get-PhysicalDisk | Select-Object MediaType,BusType "
                    "| ConvertTo-Json -Compress",
                ],
                capture_output=True, text=True, timeout=60, check=False,
            )
            if out.returncode != 0 or not out.stdout.strip():
                return "unknown"
            import json

            payload = json.loads(out.stdout)
            disks = payload if isinstance(payload, list) else [payload]
            kinds = {_windows_kind(d) for d in disks if isinstance(d, dict)}
            # Several physical disks and no way to attribute the volume to one:
            # report the WORST, so a machine with a spinning disk is not told it
            # has NVMe.
            for candidate in ("hdd", "unknown", "ssd", _NVME):
                if candidate in kinds:
                    return candidate
            return "unknown"
        if system == "Darwin":
            tool = _resolve_tool("diskutil", "/usr/sbin/diskutil")
            if tool is None:
                return "unknown"
            out = subprocess.run(
                [tool, "info", "-plist", "/"],
                capture_output=True, text=True, timeout=60, check=False,
            )
            text = out.stdout.lower()
            if "nvme" in text:
                return _NVME
            if "solid state" in text or ("<true/>" in text and "solidstate" in text):
                return "ssd"
            return "unknown"
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unknown"
    return "unknown"


def _windows_kind(disk: dict) -> str:
    bus = str(disk.get("BusType", "")).strip().lower()
    media = str(disk.get("MediaType", "")).strip().lower()
    if bus == "nvme":
        return _NVME
    # MSFT_PhysicalDisk.MediaType ValueMap is {0 Unspecified, 3 HDD, 4 SSD,
    # 5 SCM} — verified on this box, whose NVMe reports raw value 4. PowerShell
    # usually renders the friendly string, but ConvertTo-Json emits the integer
    # on some systems, so both forms are handled and neither is guessed.
    if media in ("hdd", "3"):
        return "hdd"
    if media in ("ssd", "4"):
        return "ssd"
    return "unknown"


def choose_tier(
    model_bytes: int,
    free_ram_bytes: int,
    disk_kind: DiskKind,
    *,
    headroom: float = RAM_TIER_HEADROOM,
) -> str:
    """RAM is Tier 1; disk is overflow only; spinning rust is refused.

    plan 2.3: streaming from NVMe measured 2-11 TFLOPS against multiples of
    that from CPU RAM, so RAM is the primary source and disk is a fallback.
    plan P11: on an HDD, 80 shards x 2 reads = 160 seeks per step.

    ``disk_kind`` may be a **callable**, and normally should be: detecting the
    media type costs ~9 s on Windows and is irrelevant whenever the base fits in
    RAM. Passing ``detect_disk_kind`` here means that cost is paid only by runs
    that are actually about to use the disk tier.
    """
    if model_bytes < free_ram_bytes * headroom:
        return TIER_RAM
    kind = disk_kind() if callable(disk_kind) else disk_kind
    if kind == _NVME:
        return TIER_DISK
    raise ValueError(
        f"layer streaming needs NVMe or more RAM: the base needs "
        f"{model_bytes / 1e9:.1f} GB, only {free_ram_bytes / 1e9:.1f} GB of RAM "
        f"is free, and the detected disk is {kind!r} (not NVMe). "
        f"Free RAM, pick a smaller base, or move the model to an NVMe drive."
    )


@dataclass(frozen=True)
class PinDecision:
    """Whether the RAM store can be page-locked, and why not when it cannot."""

    pinned: bool
    reason: str


def decide_pinning(store_bytes: int, pinned_limit_bytes: Optional[int]) -> PinDecision:
    """Pin the RAM store when the box can actually page-lock it.

    Measured on the dev box (RTX 3050 4 GB / 16.9 GB RAM): the maximum
    page-locked host allocation was 7.12 GB even with 9.1 GB "available", so a
    5.55 GB base plus a CUDA context and a model skeleton did not fit. Falling
    back to a pageable store is correct, but it makes
    ``copy_(non_blocking=True)`` synchronous and therefore costs overlap:
    measured GPU utilisation dropped from 96.8% (pinned) to 79.3% (pageable).
    That cost is stated out loud rather than absorbed silently.
    """
    if pinned_limit_bytes is None:
        return PinDecision(
            pinned=True,
            reason="pinned-host ceiling unknown; attempting a page-locked store",
        )
    if store_bytes <= pinned_limit_bytes:
        return PinDecision(pinned=True, reason="store fits under the page-locked ceiling")
    return PinDecision(
        pinned=False,
        reason=(
            f"base store is {store_bytes / 1e9:.2f} GB but this box can only "
            f"page-lock {pinned_limit_bytes / 1e9:.2f} GB — falling back to a "
            f"pageable store. Host-to-device copies become synchronous, which "
            f"costs overlap: measured GPU utilisation drops from ~97% to ~79%. "
            f"Free RAM or use a smaller base to keep the pinned store."
        ),
    )


# ==========================================================================
# buffers
# ==========================================================================
def validate_stream_buffers(value: Any) -> int:
    """plan 8: 1 buffer is a scheduler bug, not a config."""
    if isinstance(value, bool):
        raise ValueError("training.stream_buffers must be an int, not bool")
    if not isinstance(value, int):
        raise ValueError(f"training.stream_buffers must be an int; got {type(value).__name__}")
    if value < MIN_STREAM_BUFFERS or value > MAX_STREAM_BUFFERS:
        raise ValueError(
            f"training.stream_buffers must be between {MIN_STREAM_BUFFERS} and "
            f"{MAX_STREAM_BUFFERS}; got {value}. A single buffer cannot overlap "
            f"load with compute — that is a scheduler bug, not a configuration."
        )
    return value


# ==========================================================================
# arithmetic (plan 4.1 / 4.3)
# ==========================================================================
def should_enable_hf_gradient_checkpointing(
    gradient_checkpointing: Any, *, stream_layers: bool
) -> bool:
    """Whether HF's own gradient checkpointing should be switched on.

    ``StreamedDecoderLayer`` already wraps every layer in
    ``checkpoint(use_reentrant=False)`` — that is what bounds activation memory
    AND what triggers the second weight read per step. Letting the HF Trainer
    also enable checkpointing double-recomputes every layer: the run still
    converges, it is just silently ~1.5x slower. So streaming always wins.
    """
    return bool(gradient_checkpointing) and not stream_layers


def free_ram_bytes() -> Optional[int]:
    """Available host RAM, or None when psutil is not installed."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except (AttributeError, OSError, ValueError):
        return None


def estimate_stream_store_bytes(
    source_bytes: int,
    *,
    dtype: str,
    quant: str = QUANT_NONE,
    double_quant: bool = True,
) -> int:
    """Scale an on-disk checkpoint size to the RAM store streaming will hold.

    The pre-flight probe compares the base against free RAM *before* sharding,
    to avoid spending minutes rewriting a checkpoint that will then be refused.
    Under NF4 the store is ~0.26x the bf16 on-disk size, so measuring the raw
    file size would refuse an 8B run on a 16.9 GB box — precisely the
    configuration this release exists to enable.

    Deliberately coarse: it assumes the source is stored at ``dtype`` and
    charges the NF4 rate to every parameter, while layernorms and (in the shard
    set) the embeddings actually stay unquantised. That errs slightly LOW, which
    is the safe direction — the authoritative check runs after sharding, on the
    real shard sizes.
    """
    if quant not in SUPPORTED_STREAM_QUANTS:
        raise ValueError(
            f"unsupported quant {quant!r} for layer streaming; supported: "
            f"{', '.join(SUPPORTED_STREAM_QUANTS)}"
        )
    if source_bytes < 0:
        raise ValueError(f"source_bytes must be non-negative; got {source_bytes}")
    if quant == QUANT_NONE:
        return int(source_bytes)
    per_param = NF4_BYTES_PER_PARAM if double_quant else NF4_BYTES_PER_PARAM_SINGLE
    return int(source_bytes * per_param / dtype_bytes(dtype))


def estimate_stream_tokens_per_sec(params: int, effective_tflops: float) -> float:
    """tok/s ceiling when compute-bound: TFLOPS_eff / (C * P), C = 6."""
    if params <= 0:
        raise ValueError(f"params must be positive; got {params}")
    if effective_tflops <= 0 or not math.isfinite(effective_tflops):
        raise ValueError(f"effective_tflops must be positive and finite; got {effective_tflops}")
    return (effective_tflops * 1e12) / (FLOPS_PER_PARAM_PER_TOKEN * params)


def estimate_epoch_seconds(tokens: int, tokens_per_sec: float) -> float:
    """Wall-clock for a token budget — the pre-flight forecast (plan 10)."""
    if tokens_per_sec <= 0 or not math.isfinite(tokens_per_sec):
        raise ValueError(f"tokens_per_sec must be positive and finite; got {tokens_per_sec}")
    if tokens < 0:
        raise ValueError(f"tokens must be non-negative; got {tokens}")
    return tokens / tokens_per_sec


def estimate_logits_bytes(
    *,
    vocab_size: int,
    seq_len: int,
    batch_size: int = 1,
    upcast_fp32: bool = True,
    bytes_per_element: Optional[float] = None,
) -> int:
    """plan P5: logits, not weights, OOM you first on a small card.

    Measured (GATE 2), not derived: the loss path holds ``LOGITS_BYTES_PER_ELEMENT``
    bytes per element live at peak, not the 2 + 4 a first-principles reading
    suggests. At Qwen2.5-0.5B (vocab 151936), batch 8, S=512 this single term is
    8.71 GB — 146x the whole buffer pool — so a pre-flight that budgets only
    weights and buffers green-lights a config that cannot run.

    ``bytes_per_element`` accepts a stack measurement from
    :func:`calibrated_logits_bytes_per_element`. It is **floored at the shipped
    constant and can therefore only raise the budget**, which is the only
    direction the evidence supports: the loss arithmetic is measurable at
    pre-flight, the retention (issue #327) is not, so a reading below the
    constant is a probe that could not see the retention rather than a stack
    that does not pay it.
    """
    if vocab_size <= 0 or seq_len <= 0 or batch_size <= 0:
        raise ValueError("vocab_size, seq_len and batch_size must all be positive")
    elements = vocab_size * seq_len * batch_size
    per: float = LOGITS_BYTES_PER_ELEMENT if upcast_fp32 else LOGITS_BYTES_PER_ELEMENT_NO_LOSS
    if bytes_per_element is not None:
        if not math.isfinite(bytes_per_element):
            raise ValueError(f"bytes_per_element must be finite; got {bytes_per_element}")
        per = max(per, bytes_per_element)
    return math.ceil(elements * per)


def measure_logits_loss_bytes_per_element(
    *,
    vocab_size: int = 8192,
    tokens: Tuple[int, int] = (1024, 2048),
    device: Optional[str] = None,
) -> Optional[float]:
    """Measure THIS stack's loss-path cost per logit element, or ``None``.

    Reports the **marginal** slope between two token counts, so the fixed
    overhead of the probe itself cancels rather than being modelled. The source
    logits tensor is allocated *before* the peak is reset, so what comes back is
    the loss arithmetic alone — :data:`LOGITS_LOSS_BYTES_PER_ELEMENT`, not the
    retained copy on top of it.

    Costs one transient allocation of ``14 * vocab_size * max(tokens)`` bytes —
    96 MiB at the defaults. Returns ``None`` rather than raising when there is no
    CUDA device or the probe cannot run: a pre-flight that dies because its own
    instrument failed is worse than one that falls back to the shipped constant.

    Measured 12.000000 with spread 0.00e+00 over 3 repeats at each of
    (8192, 1024->2048), (4096, 1024->2048) and (16384, 512->1024).
    """
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive; got {vocab_size}")
    small, large = tokens
    if small <= 0 or large <= small:
        raise ValueError(f"tokens must be an increasing positive pair; got {tokens!r}")
    try:  # pragma: no cover - exercised only where torch + CUDA exist
        import torch
        from transformers.loss.loss_utils import ForCausalLMLoss
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    where = device or "cuda"

    def _peak(count: int) -> int:
        torch.cuda.empty_cache()
        labels = torch.randint(0, vocab_size, (1, count), device=where)
        logits = torch.randn(
            1, count, vocab_size, dtype=torch.bfloat16, device=where, requires_grad=True
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(where)
        base = torch.cuda.memory_allocated(where)
        loss = ForCausalLMLoss(logits, labels, vocab_size)
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated(where) - base
        del loss, logits, labels
        torch.cuda.empty_cache()
        return peak

    try:  # pragma: no cover - exercised only where torch + CUDA exist
        low, high = _peak(small), _peak(large)
    except Exception:
        return None
    return (high - low) / (vocab_size * (large - small))


def calibrated_logits_bytes_per_element(**kwargs: Any) -> float:
    """The shipped constant, raised if this stack's loss path costs more.

    Never lowered. The measurable half is the loss arithmetic; the retention
    term is charged on top of whatever is measured, because no synthetic probe
    can observe whether the training loop that will actually run still holds the
    logits when the loss backward peaks (issue #327). A stack whose loss path
    grew a fourth fp32 buffer would otherwise be under-budgeted by 12.5% with no
    guard at all, and under-prediction is the failure that does not raise.
    """
    measured = measure_logits_loss_bytes_per_element(**kwargs)
    if measured is None:
        return float(LOGITS_BYTES_PER_ELEMENT)
    return max(float(LOGITS_BYTES_PER_ELEMENT), measured + LOGITS_RETENTION_BYTES_PER_ELEMENT)


def estimate_activation_bytes(
    *,
    hidden_size: int,
    intermediate_size: int,
    n_layers: int,
    seq_len: int,
    batch_size: int = 1,
    dtype: str = "bfloat16",
) -> int:
    """Activation memory for one streamed step (GATE 2).

    Two terms, and keeping them separate is the whole memory argument for
    streaming:

    * ``2 * n_layers * hidden`` per token — ``checkpoint(use_reentrant=False)``
      saves each layer's *input*, so this one does scale with depth. It is small.
    * ``4 * (hidden + intermediate)`` per token — the tensors live inside the ONE
      layer currently being recomputed. Independent of depth, which is precisely
      why a 32-layer model costs no more here than a 4-layer one.
    """
    for name, value in (
        ("hidden_size", hidden_size),
        ("intermediate_size", intermediate_size),
        ("n_layers", n_layers),
        ("seq_len", seq_len),
        ("batch_size", batch_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive; got {value}")
    element = dtype_bytes(dtype)
    tokens = seq_len * batch_size
    boundary = element * n_layers * hidden_size
    transient = 2 * element * (hidden_size + intermediate_size)
    return tokens * (boundary + transient)


def estimate_optimizer_bytes(
    adapter_params: int, *, bytes_per_param: int = OPTIMIZER_BYTES_PER_PARAM
) -> int:
    """Adapter weights + gradients + optimizer moments."""
    if adapter_params < 0:
        raise ValueError(f"adapter_params must be non-negative; got {adapter_params}")
    return adapter_params * bytes_per_param


def estimate_stream_peak_vram(
    *,
    layer_bytes: int,
    buffers: int,
    extras_bytes: int,
    adapter_params: int,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    n_layers: int,
    seq_len: int,
    batch_size: int = 1,
    dtype: str = "bfloat16",
    logits_bytes_per_element: Optional[float] = None,
) -> int:
    """Predicted ``torch.cuda.max_memory_allocated()`` for a streaming step.

    Assembles the GATE 2 terms. Validated against 10 real runs across two models
    (a 3.1x vocab contrast), batch 1..8 and two sequence lengths: **worst
    absolute error 0.85%, and it never under-predicts** — the only safe direction
    for a number that is allowed to refuse a run. An independent check against
    the published v0.72.2 Llama-3.1-8B NF4 row (untied embeddings, different
    quantisation, different session, nothing fitted to it) brackets it at +7.5%.

    ``extras_bytes`` is what makes an 8B run predictable: its embeddings and
    ``lm_head`` are UNTIED, so two 1.05 GB matrices sit resident and account for
    2.10 GB of that run's 3.32 GB peak.

    Returns allocator-visible bytes only. The CUDA context and driver reservation
    sit outside the caching allocator (0.85 GB on the dev box, which also drives
    a display), so the fit decision compares this against *measured free VRAM*
    rather than against the card's nameplate size.

    ``logits_bytes_per_element`` forwards a stack measurement from
    :func:`calibrated_logits_bytes_per_element`; it is floored at the shipped
    constant, so passing it can only raise the prediction.
    """
    return (
        layer_bytes * buffers
        + extras_bytes
        + estimate_optimizer_bytes(adapter_params)
        + STREAM_FIXED_SLACK_BYTES
        + estimate_activation_bytes(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            n_layers=n_layers,
            seq_len=seq_len,
            batch_size=batch_size,
            dtype=dtype,
        )
        + estimate_logits_bytes(
            vocab_size=vocab_size,
            seq_len=seq_len,
            batch_size=batch_size,
            bytes_per_element=logits_bytes_per_element,
        )
    )


@dataclass(frozen=True)
class VramFit:
    """Whether a streaming step is predicted to fit in the VRAM actually free.

    ``measured_bytes`` is ``None`` for a decision taken from the prediction alone
    and carries the real peak once :func:`decide_measured_fit` has run, so a
    caller can print both numbers and let a divergence be seen rather than
    silently replacing one with the other.
    """

    fits: bool
    predicted_bytes: int
    available_bytes: int
    reason: str
    measured_bytes: Optional[int] = None


def decide_stream_fit(*, predicted_bytes: int, available_bytes: int) -> VramFit:
    """Compare predicted demand against measured free VRAM.

    Refusing rather than warning is deliberate. On Linux an over-budget step is a
    hard OOM. On Windows it is worse: WDDM silently spills to host memory and the
    run merely becomes an order of magnitude slower — measured here as a 9.27 GB
    peak on a 4.29 GB card with no exception raised at all. A user reading that
    as "streaming is slow" would draw exactly the wrong conclusion.
    """
    if predicted_bytes <= available_bytes:
        return VramFit(
            fits=True,
            predicted_bytes=predicted_bytes,
            available_bytes=available_bytes,
            reason=(
                f"predicted peak {predicted_bytes / 1e9:.2f} GB fits in "
                f"{available_bytes / 1e9:.2f} GB of free VRAM"
            ),
        )
    return VramFit(
        fits=False,
        predicted_bytes=predicted_bytes,
        available_bytes=available_bytes,
        reason=(
            f"a streaming step is predicted to need "
            f"{predicted_bytes / 1e9:.2f} GB of VRAM but only "
            f"{available_bytes / 1e9:.2f} GB is free. Streaming bounds the "
            f"WEIGHTS, not the activations or the logits — lower "
            f"training.batch_size or data.max_length, both of which scale this "
            f"linearly."
        ),
    )


def decide_measured_fit(
    *, measured_bytes: int, predicted_bytes: int, available_bytes: int
) -> VramFit:
    """Decide the fit on the REAL peak of one step, with the prediction beside it.

    :func:`decide_stream_fit` compares a formula against free VRAM, and the
    formula's documented contract is that it never under-predicts. Measured on an
    RTX 3050 Laptop against SmolLM2-135M streamed in bf16, that holds to seq 3072
    (prediction 1.6%-2.9% high) and then fails: at seq 4096 the prediction is
    0.992x the real peak and at seq 5120 it is 0.830x, i.e. it under-predicts by
    17% on a shape a user can reach by editing one line of YAML. Three repeats at
    4096 returned a bit-identical peak, so it is deterministic. The mechanism is
    NOT established — an attention ``seq**2`` term is the obvious candidate and
    does not fit the numbers — which is exactly why this takes a measurement
    rather than another coefficient: a formula cannot model a term nobody has
    identified.

    ``measured_bytes`` is ``torch.cuda.max_memory_allocated``, deliberately not
    ``max_memory_reserved``. Reserved runs 1.08x-1.41x allocated here and
    overshoots what has to fit, because the caching allocator keeps freed blocks
    and hands them back under pressure: the flagship Llama-3.1-8B NF4
    configuration reserved 3.70 GB against 3.45 GB free and runs, with
    ``num_alloc_retries`` at 0 on every shape measured. Gating on reserved would
    refuse the headline config of the feature it protects.
    """
    if measured_bytes <= available_bytes:
        return VramFit(
            fits=True,
            predicted_bytes=predicted_bytes,
            available_bytes=available_bytes,
            measured_bytes=measured_bytes,
            reason=(
                f"measured peak {measured_bytes / 1e9:.2f} GB fits in "
                f"{available_bytes / 1e9:.2f} GB of free VRAM "
                f"(predicted {predicted_bytes / 1e9:.2f} GB)"
            ),
        )
    return VramFit(
        fits=False,
        predicted_bytes=predicted_bytes,
        available_bytes=available_bytes,
        measured_bytes=measured_bytes,
        reason=(
            f"a streaming step MEASURED {measured_bytes / 1e9:.2f} GB of VRAM at "
            f"the configured shape but only {available_bytes / 1e9:.2f} GB is "
            f"free (the formula predicted {predicted_bytes / 1e9:.2f} GB). "
            f"Streaming bounds the WEIGHTS, not the activations or the logits — "
            f"lower training.batch_size or data.max_length, both of which scale "
            f"this. This run was measured, not estimated: it does not fit."
        ),
    )


def resolve_available_vram_bytes(
    *, measured_bytes: int, override_bytes: Optional[int]
) -> int:
    """The free-VRAM figure the pre-flight fit check measures against.

    ``mem_get_info()`` is a device-level driver query, so it cannot see a
    per-process cap (``set_per_process_memory_fraction``, a MIG slice, a card
    another process is also using). ``training.stream_vram_override``, when
    set, REPLACES the driver reading rather than padding it, in either
    direction: raised to let a documented over-prediction through, or lowered
    to enforce a cap the driver itself cannot report.
    """
    return measured_bytes if override_bytes is None else override_bytes


#: Measured (GATE 3) speed-up from reaching an effective batch by raising
#: ``batch_size`` rather than ``gradient_accumulation_steps``: Qwen2.5-0.5B bf16,
#: S=256, pinned store — 1393.4 tok/s at batch 4 / accum 1 against 553.2 at
#: batch 1 / accum 4. One weight read amortised over 4x the tokens.
ACCUM_VS_BATCH_SPEEDUP = 2.5


def accumulation_advice(*, batch_size: int, accum: int) -> Optional[str]:
    """Tell an accumulating user what the measurement says, or stay quiet.

    The intuition most people bring — "accumulation costs streaming I/O linearly"
    — is right per optimizer step and wrong per token, which is the unit that
    decides wall-clock: ``accum=N`` reads the base N times *and* processes N
    times the tokens, so layer reads per 1k tokens do not move at all (measured
    constant at 175.78 across accum 1/2/4).

    What is true is the trade the user is actually making. Raising ``batch_size``
    reaches the same effective batch in ONE read instead of N, and measured
    2.52x faster — but it costs VRAM, while accumulation holds peak flat. So the
    advice is only worth printing when there might be VRAM headroom to spend.
    """
    if batch_size <= 0 or accum <= 0:
        raise ValueError(
            f"batch_size and accum must be positive; got {batch_size}, {accum}"
        )
    if accum == 1:
        return None
    return (
        f"accumulating {accum}x at batch {batch_size}: the base is re-read once "
        f"per micro-batch. Per token that is free, but reaching effective batch "
        f"{batch_size * accum} by raising training.batch_size instead measured "
        f"~{ACCUM_VS_BATCH_SPEEDUP:.1f}x faster. Accumulation holds peak VRAM "
        f"flat, so raise batch_size while the budget above allows, then "
        f"accumulate for the rest."
    )


@dataclass(frozen=True)
class ThroughputForecast:
    """A compute-bound CEILING, plus the observed fraction of it. Never a point
    estimate: real streamed runs landed at 68%-100% of the measured ceiling."""

    effective_tflops: float
    tokens_per_sec_ceiling: float
    tokens_per_sec_low: float
    epoch_seconds_floor: float
    epoch_seconds_high: float
    sm_clock_mhz: Optional[int] = None


def forecast_stream_throughput(
    *,
    params: int,
    effective_tflops: float,
    tokens_per_epoch: int,
    sm_clock_mhz: Optional[int] = None,
) -> ThroughputForecast:
    """Bracket a streaming run's throughput from a *measured* GEMM ceiling.

    The ceiling itself is arithmetic (``TFLOPS / (6 * P)``); the honest part is
    that ``effective_tflops`` must come from a matmul benchmarked on the user's
    own card in the same session — a per-card constant baked into the source
    would be a fabrication, and this box's boost clock alone moved 442..952 MHz
    inside a single measurement run.
    """
    ceiling = estimate_stream_tokens_per_sec(params, effective_tflops)
    low_fraction, high_fraction = MEASURED_CEILING_FRACTION
    low = ceiling * low_fraction
    return ThroughputForecast(
        effective_tflops=effective_tflops,
        tokens_per_sec_ceiling=ceiling * high_fraction,
        tokens_per_sec_low=low,
        epoch_seconds_floor=(
            estimate_epoch_seconds(tokens_per_epoch, ceiling * high_fraction)
            if tokens_per_epoch
            else 0.0
        ),
        epoch_seconds_high=(
            estimate_epoch_seconds(tokens_per_epoch, low) if tokens_per_epoch else 0.0
        ),
        sm_clock_mhz=sm_clock_mhz,
    )


def estimate_stream_vram(
    *,
    layer_bytes: int,
    buffers: int,
    embed_bytes: int,
    adapter_bytes: int = 0,
    activation_bytes: int = 0,
    logits_bytes: int = 0,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> int:
    """Peak VRAM for a streaming step (plan 4.1)."""
    return (
        layer_bytes * buffers
        + embed_bytes
        + adapter_bytes
        + activation_bytes
        + logits_bytes
        + workspace_bytes
    )


# ==========================================================================
# specs and plans
# ==========================================================================
@dataclass(frozen=True)
class LayerSpec:
    """Shape + dtype of ONE tensor inside a decoder layer."""

    name: str
    shape: Tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        dtype_bytes(self.dtype)  # fail fast on an unsupported dtype

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * dtype_bytes(self.dtype)


@dataclass(frozen=True)
class StreamPlan:
    """What a streaming run will do, before it does it."""

    arch: str
    tier: str
    n_layers: int
    layer_bytes: int
    store_bytes: int
    embed_bytes: int
    buffers: int
    buffer_bytes: int
    pinned: bool
    notes: Tuple[str, ...]


def build_stream_plan(
    *,
    arch: str,
    n_layers: int,
    layer_bytes: int,
    embed_bytes: int,
    available_ram_bytes: int,
    pinned_limit_bytes: Optional[int],
    buffers: int = DEFAULT_STREAM_BUFFERS,
    disk_kind: DiskKind = _NVME,
) -> StreamPlan:
    """Decide tier + pinning and record every caveat as a visible note."""
    buffers = validate_stream_buffers(buffers)
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive; got {n_layers}")
    store_bytes = n_layers * layer_bytes
    tier = choose_tier(store_bytes + embed_bytes, available_ram_bytes, disk_kind)
    notes = []
    if tier == TIER_DISK:
        # Falling back is the point of stream_source='auto', but a silent
        # fallback to a slower path is the failure mode this project keeps
        # calling out elsewhere. Say what happened and be explicit that the
        # slowdown is NOT quantified: safetensors memory-maps the shards, so the
        # OS page cache blurs the RAM-vs-disk boundary, and the dev box could not
        # produce a trustworthy gap measurement.
        notes.append(
            "base does not fit in RAM — streaming from the NVMe disk tier "
            "instead. Nothing is held resident, and the slowdown versus the RAM "
            "tier is unmeasured on this hardware. Set stream_source='ram' to "
            "refuse rather than fall back."
        )
    decision = decide_pinning(store_bytes, pinned_limit_bytes)
    if not decision.pinned:
        notes.append(decision.reason)
    return StreamPlan(
        arch=arch,
        tier=tier,
        n_layers=n_layers,
        layer_bytes=layer_bytes,
        store_bytes=store_bytes,
        embed_bytes=embed_bytes,
        buffers=buffers,
        buffer_bytes=layer_bytes * buffers,
        pinned=decision.pinned,
        notes=tuple(notes),
    )


def render_stream_panel(plan: StreamPlan, extra_lines: Sequence[str] = ()) -> Panel:
    """Pre-flight summary. plan 10: tell the user the cost BEFORE the run."""
    if plan.tier == TIER_DISK:
        # "store ... (pinned)" is meaningless here: the disk tier deliberately
        # holds nothing resident, so reporting a pinned store of 0.00 GB reads
        # as a bug rather than as the design.
        store_line = (
            f"  base         streamed from disk across {plan.n_layers} layers, "
            f"nothing held resident"
        )
    else:
        store_line = (
            f"  base store   {plan.store_bytes / 1e9:.2f} GB across "
            f"{plan.n_layers} layers "
            f"({'pinned' if plan.pinned else 'pageable'})"
        )
    lines = [
        f"[bold]Layer streaming[/] [yellow]BETA[/] — arch [cyan]{plan.arch}[/], "
        f"tier [cyan]{plan.tier}[/]",
        store_line,
        f"  VRAM buffers {plan.buffers} x {plan.layer_bytes / 1e6:.0f} MB "
        f"= {plan.buffer_bytes / 1e6:.0f} MB",
        f"  resident     {plan.embed_bytes / 1e6:.0f} MB embeddings + adapters",
    ]
    lines.extend(extra_lines)
    for note in plan.notes:
        lines.append(f"  [yellow]![/] {note}")
    return Panel("\n".join(lines), title="training.stream_layers", border_style="cyan")
