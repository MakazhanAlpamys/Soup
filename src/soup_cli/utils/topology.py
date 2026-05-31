"""Multi-GPU topology detection + distributed strategy suggestions (v0.27.0)."""

from __future__ import annotations

from typing import TypedDict

MAX_GPU_COUNT = 128
# NVLink links per GPU — NVML_NVLINK_MAX_LINKS = 18 on H100/H200.
# A100 caps out at 12, V100 at 6; probing 18 is safe on older GPUs (missing
# links raise, and we continue on Exception).
_MAX_NVLINK_LINKS_PER_GPU = 18


class TopologyInfo(TypedDict):
    gpu_count: int
    nvlink_pairs: int
    interconnect: str  # "none" | "single" | "nvlink" | "pcie"


class StrategyRecommendation(TypedDict):
    strategy: str  # "none" | "single" | "ddp" | "zero2" | "zero3" | ...
    reason: str


def _detected_gpu_count() -> int:
    """Return number of visible CUDA devices (0 if torch/cuda unavailable)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except ImportError:
        return 0


def _count_nvlink_pairs() -> int:
    """Count NVLink peer-to-peer connected GPU pairs via NVML.

    Returns 0 when NVML is unavailable or no NVLink links are active.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
    except (ImportError, Exception):  # noqa: BLE001 - NVMLError is subclass of Exception
        return 0

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        pairs = 0
        for src in range(device_count):
            h_src = pynvml.nvmlDeviceGetHandleByIndex(src)
            for link in range(_MAX_NVLINK_LINKS_PER_GPU):
                try:
                    state = pynvml.nvmlDeviceGetNvLinkState(h_src, link)
                except Exception:  # noqa: BLE001
                    continue
                if state == 1:  # NVML_FEATURE_ENABLED
                    pairs += 1
        return pairs
    except Exception:  # noqa: BLE001 - defensive: NVML may be partially broken
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def resolve_num_gpus(spec: int | str | None) -> int | None:
    """Resolve --gpus flag into concrete int.

    Accepts:
        - None: returns None (user did not pass --gpus)
        - int: returns the int
        - str "auto": returns detected GPU count
        - str digit: returns int
        - anything else: raises ValueError
    """
    if spec is None:
        return None

    if isinstance(spec, bool):  # bool is a subclass of int — reject explicitly
        raise ValueError(f"Invalid --gpus value: {spec!r}")
    if isinstance(spec, int):
        value = spec
    elif isinstance(spec, str):
        stripped = spec.strip()
        if stripped.lower() == "auto":
            return _detected_gpu_count()
        # Accept optional leading sign; reject anything else (the ``value < 1``
        # guard below catches negatives and zero).
        if not stripped.lstrip("+-").isdigit():
            raise ValueError(
                f"Invalid --gpus value: {spec!r}. Use 'auto' or a positive integer."
            )
        value = int(stripped)
    else:
        raise ValueError(f"Invalid --gpus value: {spec!r}")

    if value < 1:
        raise ValueError(f"--gpus must be >= 1 (got {value}).")
    if value > MAX_GPU_COUNT:
        raise ValueError(
            f"--gpus value {value} exceeds cap of {MAX_GPU_COUNT}."
        )
    return value


def detect_topology() -> TopologyInfo:
    """Detect GPU count + interconnect type (NVLink / PCIe / single / none).

    Returns:
        {"gpu_count": int, "nvlink_pairs": int, "interconnect": str}
        interconnect ∈ {"none", "single", "nvlink", "pcie"}.
    """
    gpu_count = _detected_gpu_count()
    if gpu_count == 0:
        return {"gpu_count": 0, "nvlink_pairs": 0, "interconnect": "none"}
    if gpu_count == 1:
        return {"gpu_count": 1, "nvlink_pairs": 0, "interconnect": "single"}

    pairs = _count_nvlink_pairs()
    interconnect = "nvlink" if pairs > 0 else "pcie"
    return {
        "gpu_count": gpu_count,
        "nvlink_pairs": pairs,
        "interconnect": interconnect,
    }


def suggest_nccl_env(gpu_count: int, interconnect: str) -> dict[str, str]:
    """Suggest NCCL environment variables tuned for the detected topology.

    Returns an empty dict for single-GPU / CPU. Local multi-GPU disables
    InfiniBand probing to avoid startup hangs on machines without IB.
    """
    if gpu_count <= 1:
        return {}

    env = {
        "NCCL_P2P_DISABLE": "0",  # allow peer-to-peer when available
        "NCCL_IB_DISABLE": "1",   # local only: skip IB probing
    }
    if interconnect == "nvlink":
        env["NCCL_NVLS_ENABLE"] = "1"
    return env


def suggest_strategy(gpu_count: int, model_size_b: float) -> StrategyRecommendation:
    """Suggest a distributed strategy based on GPU count + model size (billions).

    Strategies:
        - "none"            CPU / no GPUs
        - "single"          1 GPU
        - "ddp"             small model + multi-GPU (DDP / ZeRO-1-like)
        - "zero2"           mid model + multi-GPU
        - "zero3"           large model + multi-GPU (sharded params)
        - "zero2_offload"   large model + few GPUs (CPU offload)
        - "fsdp_full_shard" modern alternative to zero3
    """
    if gpu_count <= 0:
        return {"strategy": "none", "reason": "No GPUs detected."}
    if gpu_count == 1:
        return {"strategy": "single", "reason": "Single GPU - no sharding needed."}

    # Multi-GPU — heuristics by model size (billions).
    if model_size_b <= 3.0:
        return {
            "strategy": "ddp",
            "reason": f"Small model ({model_size_b}B) fits per GPU; use DDP.",
        }
    if model_size_b <= 13.0:
        return {
            "strategy": "zero2",
            "reason": f"Mid model ({model_size_b}B); ZeRO-2 shards optimizer.",
        }
    if model_size_b <= 34.0:
        if gpu_count >= 4:
            return {
                "strategy": "fsdp_full_shard",
                "reason": f"Large model ({model_size_b}B); FSDP2 full shard.",
            }
        return {
            "strategy": "zero2_offload",
            "reason": f"Large model ({model_size_b}B) + few GPUs; ZeRO-2 offload.",
        }
    # Huge (>34B)
    if gpu_count >= 4:
        return {
            "strategy": "zero3",
            "reason": f"Huge model ({model_size_b}B); ZeRO-3 shards params.",
        }
    return {
        "strategy": "zero2_offload",
        "reason": f"Huge model ({model_size_b}B) + few GPUs; CPU offload.",
    }
