"""v0.44.0 Part D — `soup delinearize-llama4` weight reshape.

Llama 4 ships its fused-MoE expert weights as parameters on
``Llama4TextExperts`` (``...feed_forward.experts.gate_up_proj`` /
``...experts.down_proj``). Checkpoints exported through tooling that
flattens parameters carry those tensors in *linearised* 2-D form
``[num_experts * dim_in, dim_out]``; HF transformers (and most downstream
backends) expect the 3-D ``[num_experts, dim_in, dim_out]`` layout.

v0.71.21 (#97) lifts the v0.44.0 planner to a live runtime:
:func:`run_delinearize` loads each ``.safetensors`` shard (torch loader —
handles bf16), reshapes the fused expert tensors row-major to 3-D, passes
every other tensor through unchanged, and writes the shards to the target
directory (atomic write via the shared ``paths.atomic_write_bytes``
helper, mirroring the v0.71.14 FSDP-consolidate policy). JSON sidecars
(``config.json``, tokenizer files, the safetensors index) are copied so
the target stays a loadable checkpoint. Per-expert numbered keys
(``...experts.0.gate_proj.weight`` — Mixtral-style, already unfused) are
intentionally NOT matched.
"""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from soup_cli.utils.paths import (
    atomic_write_bytes,
    atomic_write_text,
    enforce_under_cwd_and_no_symlink,
    is_under_cwd,
)

# Restrict to canonical Llama 4 model id shape; reject crafted names.
_LLAMA4_RE = re.compile(r"(?i)(?:^|[^a-z0-9])llama-?4(?:[^a-z0-9]|$)")

# Fused-expert parameter keys on Llama4TextExperts. Numbered per-expert
# keys (``.experts.0.gate_proj``) are already unfused — never matched.
_EXPERT_FUSED_KEY_RE = re.compile(
    r"\.experts\.(?:gate_up_proj|down_proj)(?:\.weight)?$"
)

# Mirrors the v0.71.14 fsdp_consolidate per-shard cap.
_MAX_WEIGHT_FILE_BYTES = 16 * 1024**3  # 16 GiB
_MAX_SIDECAR_BYTES = 256 * 1024**2  # 256 MiB (tokenizer.json scale)
_MAX_NUM_EXPERTS = 4096


@dataclass(frozen=True)
class DelinearizePlan:
    """Planned weights to reshape, source path, target path.

    `weight_files` is a `tuple` for genuine immutability (matches the
    project frozen-collection policy). ``__post_init__`` re-runs the
    containment checks so a directly constructed plan cannot bypass
    ``plan_delinearize`` and route writes outside cwd.
    """

    source_dir: str
    target_dir: str
    weight_files: Tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_dir", "target_dir"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty str")
            if "\x00" in value:
                raise ValueError(f"{field_name} contains NUL byte")
            if not is_under_cwd(value):
                raise ValueError(
                    f"{field_name} is outside cwd: {os.path.basename(value)}"
                )
        if not isinstance(self.weight_files, tuple):
            raise TypeError("weight_files must be a tuple")


def is_llama4_model(name: str) -> bool:
    """Return True iff `name` looks like a Llama 4 family model."""
    if not isinstance(name, str) or not name or "\x00" in name:
        return False
    return bool(_LLAMA4_RE.search(name))


def discover_weight_files(source_dir: str) -> List[str]:
    """List `.safetensors` weight files in `source_dir`."""
    if not isinstance(source_dir, str):
        raise TypeError("source_dir must be str")
    if not is_under_cwd(source_dir):
        raise ValueError(
            f"source_dir is outside cwd: {os.path.basename(source_dir)}"
        )
    real = os.path.realpath(source_dir)
    if not os.path.isdir(real):
        raise FileNotFoundError(
            f"source_dir not found: {os.path.basename(real)}"
        )
    files = sorted(
        entry for entry in os.listdir(real) if entry.endswith(".safetensors")
    )
    if not files:
        raise FileNotFoundError(
            "no .safetensors files found in source_dir"
        )
    return files


def plan_delinearize(source_dir: str, target_dir: str) -> DelinearizePlan:
    """Build a `DelinearizePlan`. Raises on bad inputs."""
    if not isinstance(target_dir, str) or not target_dir:
        raise ValueError("target_dir must be non-empty str")
    if "\x00" in target_dir:
        raise ValueError("target_dir contains NUL byte")
    if not is_under_cwd(target_dir):
        raise ValueError(
            f"target_dir is outside cwd: {os.path.basename(target_dir)}"
        )
    files = discover_weight_files(source_dir)
    return DelinearizePlan(
        source_dir=os.path.realpath(source_dir),
        target_dir=os.path.realpath(target_dir),
        weight_files=tuple(files),
    )


# --- v0.71.21 #97 — live runtime -------------------------------------------


@dataclass(frozen=True)
class DelinearizeResult:
    """Outcome of a live delinearize run (v0.71.21 #97)."""

    source_dir: str
    target_dir: str
    files_written: Tuple[str, ...]
    reshaped_keys: int
    passthrough_keys: int
    already_3d_keys: int
    sidecars_copied: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.files_written, tuple):
            raise TypeError("files_written must be a tuple")
        for field_name in (
            "reshaped_keys",
            "passthrough_keys",
            "already_3d_keys",
            "sidecars_copied",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an int")
            if value < 0:
                raise ValueError(f"{field_name} must be >= 0")


def is_expert_weight_key(key: object) -> bool:
    """True for fused Llama-4 expert keys (``...experts.gate_up_proj``).

    Defensive surface — returns False (never raises) on non-string /
    empty / null-byte input. Numbered per-expert keys are NOT matched
    (those are already unfused, Mixtral-style).
    """
    if not isinstance(key, str) or not key or "\x00" in key:
        return False
    return bool(_EXPERT_FUSED_KEY_RE.search(key))


def _validate_num_experts(value: object) -> int:
    """Bounds-check ``num_experts`` (bool-rejected, [1, 4096])."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"num_experts must be an int, got {type(value).__name__}"
        )
    if not 1 <= value <= _MAX_NUM_EXPERTS:
        raise ValueError(
            f"num_experts must be in [1, {_MAX_NUM_EXPERTS}], got {value}"
        )
    return value


def read_num_experts(source_dir: str) -> Optional[int]:
    """Probe ``config.json`` for the expert count (None when absent).

    Checks ``text_config.num_local_experts`` (HF Llama4TextConfig),
    then top-level ``num_local_experts`` / ``num_experts``.
    """
    config_path = os.path.join(source_dir, "config.json")
    try:
        file_stat = os.lstat(config_path)
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            return None
        if file_stat.st_size > _MAX_SIDECAR_BYTES:
            return None
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(config, dict):
        return None
    candidates: List[Any] = []
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        candidates.append(text_config.get("num_local_experts"))
        candidates.append(text_config.get("num_experts"))
    candidates.append(config.get("num_local_experts"))
    candidates.append(config.get("num_experts"))
    for candidate in candidates:
        if (
            isinstance(candidate, int)
            and not isinstance(candidate, bool)
            and 1 <= candidate <= _MAX_NUM_EXPERTS
        ):
            return candidate
    return None


def delinearize_tensor(tensor: Any, *, num_experts: int) -> Tuple[Any, str]:
    """Reshape a linearised 2-D expert tensor to 3-D.

    ``[num_experts * dim_in, dim_out]`` reshapes row-major to
    ``[num_experts, dim_in, dim_out]``; an already-3-D tensor passes
    through unchanged (``status='already_3d'``).

    Returns ``(tensor, status)`` with status ``'reshaped'`` or
    ``'already_3d'``.
    """
    _validate_num_experts(num_experts)
    ndim = getattr(tensor, "ndim", None)
    if ndim == 3:
        return tensor, "already_3d"
    if ndim != 2:
        raise ValueError(
            f"expert tensor must be 2-D (linearised) or 3-D, got {ndim}-D"
        )
    rows = int(tensor.shape[0])
    if rows % num_experts != 0:
        raise ValueError(
            f"expert tensor dim 0 ({rows}) is not divisible by "
            f"num_experts={num_experts}"
        )
    return tensor.reshape(num_experts, rows // num_experts, tensor.shape[1]), "reshaped"


# JSON sidecars copied so the target stays a loadable checkpoint.
_SIDECAR_SUFFIXES = (".json",)


def _copy_json_sidecars(source_dir: str, target_dir: str) -> int:
    """Best-effort copy of top-level ``.json`` sidecars (config/tokenizer)."""
    copied = 0
    for entry in sorted(os.listdir(source_dir)):
        if not entry.endswith(_SIDECAR_SUFFIXES):
            continue
        source_path = os.path.join(source_dir, entry)
        try:
            file_stat = os.lstat(source_path)
            if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
                continue
            if file_stat.st_size > _MAX_SIDECAR_BYTES:
                continue
            with open(source_path, encoding="utf-8") as handle:
                text = handle.read()
            atomic_write_text(
                text, os.path.join(target_dir, entry), field="target_dir",
            )
            copied += 1
        except (OSError, ValueError, UnicodeDecodeError):
            continue
    return copied


def run_delinearize(
    plan: DelinearizePlan,
    *,
    num_experts: Optional[int] = None,
) -> DelinearizeResult:
    """Run the live Llama-4 expert delinearization described by ``plan``.

    Live since v0.71.21 (#97). ``num_experts`` defaults to the value read
    from the source ``config.json``; an explicit argument wins. Raises a
    friendly ``ValueError`` naming ``--num-experts`` when neither is
    available, and names the offending key when a fused expert tensor's
    leading dim is not divisible by the expert count.
    """
    if not isinstance(plan, DelinearizePlan):
        raise TypeError("plan must be a DelinearizePlan")
    if num_experts is None:
        num_experts = read_num_experts(plan.source_dir)
        if num_experts is None:
            raise ValueError(
                "could not determine the expert count from config.json — "
                "pass --num-experts explicitly"
            )
    _validate_num_experts(num_experts)

    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file
        from safetensors.torch import save as st_save
    except ImportError as exc:
        raise ImportError(
            "delinearize-llama4 requires torch + safetensors "
            "(pip install 'soup-cli[train]')"
        ) from exc

    # Containment BEFORE makedirs — a directly constructed plan must not
    # be able to create directories outside cwd (defence-in-depth on top
    # of DelinearizePlan.__post_init__).
    if not is_under_cwd(plan.target_dir):
        raise ValueError(
            f"target_dir is outside cwd: {os.path.basename(plan.target_dir)}"
        )
    os.makedirs(plan.target_dir, exist_ok=True)

    reshaped = 0
    passthrough = 0
    already_3d = 0
    written: List[str] = []
    for name in plan.weight_files:
        source_path = os.path.join(plan.source_dir, name)
        enforce_under_cwd_and_no_symlink(source_path, "source file")
        size = os.path.getsize(source_path)
        if size > _MAX_WEIGHT_FILE_BYTES:
            raise ValueError(
                f"{name} exceeds the "
                f"{_MAX_WEIGHT_FILE_BYTES // 1024**3} GiB per-file cap"
            )
        try:
            tensors = load_file(source_path)
        except Exception as exc:  # noqa: BLE001 — SafetensorError is bare Exception
            raise ValueError(
                f"{name} is not a valid safetensors file: "
                f"{type(exc).__name__}"
            ) from exc
        out_tensors = {}
        for key, tensor in tensors.items():
            if is_expert_weight_key(key):
                try:
                    out_tensor, status = delinearize_tensor(
                        tensor, num_experts=num_experts
                    )
                except ValueError as exc:
                    raise ValueError(f"{key}: {exc}") from exc
                out_tensors[key] = out_tensor
                if status == "reshaped":
                    reshaped += 1
                else:
                    already_3d += 1
            else:
                out_tensors[key] = tensor
                passthrough += 1
        target_path = os.path.join(plan.target_dir, name)
        atomic_write_bytes(st_save(out_tensors), target_path, field="target_dir")
        written.append(name)

    sidecars_copied = _copy_json_sidecars(plan.source_dir, plan.target_dir)
    return DelinearizeResult(
        source_dir=plan.source_dir,
        target_dir=plan.target_dir,
        files_written=tuple(written),
        reshaped_keys=reshaped,
        passthrough_keys=passthrough,
        already_3d_keys=already_3d,
        sidecars_copied=sidecars_copied,
    )
