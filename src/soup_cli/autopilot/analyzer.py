"""Autopilot analyzers — dataset, model, and hardware profiling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetProfile:
    samples: int
    format: str
    avg_tokens: int
    p95_tokens: int
    quality: float


@dataclass
class ModelProfile:
    name: str
    params_b: float
    context: int
    architecture: str
    modality: str


@dataclass
class HardwareProfile:
    device: str
    gpu_name: str
    vram_gb: float
    compute_capability: float
    system_ram_gb: float


def analyze_dataset(path: str) -> DatasetProfile:
    """Inspect a dataset file and produce a profile.

    Delegates to Soup's existing loader/validator/format-detector so we don't
    duplicate heuristics.
    """
    from soup_cli.data.formats import detect_format
    from soup_cli.data.loader import load_raw_data
    from soup_cli.data.validator import extended_stats, validate_and_stats

    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"Dataset not found: {file_path}")

    data = load_raw_data(file_path)
    if not data:
        raise ValueError("Dataset is empty")

    try:
        fmt = detect_format(data)
    except ValueError:
        fmt = "unknown"

    ext = extended_stats(data)
    stats = validate_and_stats(data, expected_format=fmt if fmt != "unknown" else None)

    # Rough quality score — 1.0 minus penalty for issues
    total_issues = sum(
        1 for key in ("empty_fields", "duplicates") if stats.get(key, 0) > 0
    )
    quality = max(0.0, 1.0 - 0.15 * total_issues - 0.05 * len(stats.get("issues", [])))
    quality = round(quality, 2)

    # token_counts is already approximate tokens; compute p95 directly.
    token_counts = sorted(ext.get("token_counts") or [])
    if token_counts:
        p95_idx = min(len(token_counts) - 1, int(len(token_counts) * 0.95))
        p95_tokens = int(token_counts[p95_idx])
    else:
        p95_tokens = int(ext["avg_tokens"])
    if p95_tokens <= 0:
        p95_tokens = max(1, int(ext["avg_tokens"]))

    return DatasetProfile(
        samples=stats["total"],
        format=fmt,
        avg_tokens=int(ext["avg_tokens"]),
        p95_tokens=p95_tokens,
        quality=quality,
    )


_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb]")


def _guess_params_from_name(name: str) -> float:
    """Extract the parameter count (in billions) from a model name.

    v0.40.1 Part C / C3 — fall back to **1B** (not 7B) when the name has no
    embedded size hint. The previous 7.0 default made tiny models like
    ``tiny-gpt2`` (5 MB) fail VRAM-budget checks on machines that could
    trivially train them. We also recognise the ``-Mm`` (millions) format
    used by SmolLM2 / Phi-3 family. Probes for a local safetensors index
    (cached HF snapshot) before falling back.
    """
    match = _MODEL_SIZE_RE.search(name)
    if match:
        return float(match.group(1))
    # Recognise <N>m / <N>M for sub-billion models (e.g. SmolLM2-135M,
    # tiny-gpt2). Convert to billions.
    m_match = re.search(r"(\d+(?:\.\d+)?)\s*[Mm](?![a-zA-Z])", name)
    if m_match:
        return float(m_match.group(1)) / 1000.0
    # Probe local HF cache for safetensors index size if we can.
    cache_size = _probe_cache_param_count(name)
    if cache_size is not None:
        return cache_size
    # Conservative default — small assumption (yellow advisory should fire).
    return 1.0


def _probe_cache_param_count(name: str) -> Optional[float]:
    """Best-effort: read parameter count from cached safetensors index.

    Looks at ``~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/*/
    model.safetensors.index.json`` and returns ``total_size / 4 / 1e9`` (fp32
    bytes per param). Returns ``None`` if not found.

    v0.40.1 review fix — reject empty / null-byte names (project policy
    mirroring v0.26.0 registry / v0.39.0 ReLoRAPolicy) before constructing
    the cache path.
    """
    if not isinstance(name, str) or not name or "\x00" in name:
        return None
    try:
        from pathlib import Path as _Path

        owner_repo = name.replace("/", "--")
        cache = _Path.home() / ".cache" / "huggingface" / "hub" / f"models--{owner_repo}"
        if not cache.is_dir():
            return None
        for idx in cache.rglob("model.safetensors.index.json"):
            try:
                import json as _json
                data = _json.loads(idx.read_text(encoding="utf-8"))
                total_bytes = data.get("metadata", {}).get("total_size")
                if isinstance(total_bytes, (int, float)) and total_bytes > 0:
                    # Assume fp32 storage (4 bytes/param) — generous upper
                    # bound; bf16/fp16 cuts it in half.
                    return float(total_bytes) / 4.0 / 1e9
            except (OSError, ValueError):
                continue
    except Exception:  # noqa: BLE001
        return None
    return None


def analyze_model(name: str, params_b: Optional[float] = None) -> ModelProfile:
    """Build a model profile from a HF model name.

    Uses ``utils.gpu.model_size_from_name`` when available, otherwise estimates
    from the model name. Context length defaults to 8192 unless the name
    signals otherwise (e.g. ``-128k``, ``-longctx``).
    """
    if params_b is None:
        try:
            from soup_cli.utils.gpu import model_size_from_name

            params_b = float(model_size_from_name(name)) or _guess_params_from_name(name)
        except Exception:  # noqa: BLE001
            params_b = _guess_params_from_name(name)

    lowered = name.lower()
    context = 8192
    if "128k" in lowered or "long" in lowered:
        context = 131072
    elif "32k" in lowered:
        context = 32768
    elif "qwen" in lowered or "llama-3" in lowered or "llama3" in lowered:
        context = 8192

    architecture = "dense"
    if "moe" in lowered or "a3b" in lowered or "deepseek-v3" in lowered:
        architecture = "moe"

    modality = "text"
    if "vision" in lowered or "vl-" in lowered:
        modality = "vision"
    elif "audio" in lowered:
        modality = "audio"

    return ModelProfile(
        name=name,
        params_b=float(params_b),
        context=context,
        architecture=architecture,
        modality=modality,
    )


def analyze_hardware() -> HardwareProfile:
    """Profile the current training hardware (GPU or CPU fallback)."""
    try:
        from soup_cli.utils.gpu import detect_device, get_gpu_info

        gpu_info = get_gpu_info()
        device = detect_device()
        vram_bytes = gpu_info.get("memory_total_bytes", 0) or 0
        vram_gb = vram_bytes / 1024**3
        gpu_name = gpu_info.get("name", "unknown") or "unknown"
    except Exception:  # noqa: BLE001
        device = "cpu"
        gpu_name = "none"
        vram_gb = 0.0

    # Compute capability: best-effort, falls back to 0
    compute_capability = 0.0
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            major, minor = torch.cuda.get_device_capability(0)
            compute_capability = float(f"{major}.{minor}")
    except Exception:  # noqa: BLE001
        compute_capability = 0.0

    try:
        import psutil

        system_ram_gb = psutil.virtual_memory().total / 1024**3
    except Exception:  # noqa: BLE001
        system_ram_gb = 0.0

    return HardwareProfile(
        device=str(device),
        gpu_name=str(gpu_name),
        vram_gb=float(vram_gb),
        compute_capability=compute_capability,
        system_ram_gb=float(system_ram_gb),
    )
