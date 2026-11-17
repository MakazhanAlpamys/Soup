"""LoRA adapter merge strategies (v0.57.0 Part B).

Four strategies wrap mergekit-style math in pure numpy:

- ``linear``: weighted average per-layer (baseline)
- ``ties``: trim-elect-disjoint average (Yadav et al. 2023)
- ``dare``: drop and rescale by 1/(1-density) (Yu et al. 2024)
- ``svd``: low-rank SVD reconstruction of the merged delta

All strategies operate on the intersection of layer names. ``merge_adapters``
refuses inputs that cannot be merged before anything is written (#2907): a
shared tensor whose shape differs between inputs (the LoRA rank changed), or
inputs that share no tensor name at all. The per-strategy functions still skip
a shape-mismatched tensor when called directly. Live canary
verdict via v0.55 eval gate is deferred to v0.57.1; stub returns ``None``.

The merged adapter keeps the first input's ``adapter_config.json``, so every
input's ``lora_B`` is first rescaled to decode at that adapter's LoRA scaling
(:func:`fold_lora_scaling`).
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, FrozenSet, Literal, Mapping, Optional, Sequence, Tuple

from soup_cli.utils.adapter_diff import (
    WRITE_ADAPTER_DEPS,
    WRITE_ADAPTER_DEPS_MSG,
    lora_module_scaling,
    peft_module_name,
    read_adapter_config_or_none,
    require_adapter_dependencies,
)
from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink, ensure_distinct_paths

MergeStrategy = Literal["linear", "ties", "dare", "svd", "cmaes"]
# A canary scorer: ``scorer(role, tasks) -> per-prompt scores`` where ``role``
# is "baseline" / "candidate" and ``tasks`` is the parsed canary task list.
CanaryScorer = Callable[[str, Sequence[Mapping[str, Any]]], Sequence[float]]
# frozenset for O(1) membership + immutability (mirrors v0.41.0 / v0.51.0 policy);
# tuple alias preserved for caller code that iterates in canonical order.
# v0.67.0 Part A: added "cmaes" — evolutionary search dispatched separately
# in commands/adapters.py (requires --eval suite + --budget).
SUPPORTED_STRATEGIES: FrozenSet[str] = frozenset(
    {"linear", "ties", "dare", "svd", "cmaes"}
)
STRATEGY_ORDER: Tuple[MergeStrategy, ...] = (
    "linear", "ties", "dare", "svd", "cmaes"
)

_MAX_ADAPTERS = 16
_MIN_ADAPTERS = 2


@dataclass(frozen=True)
class MergeReport:
    strategy: str
    adapters: Tuple[str, ...]
    weights: Tuple[float, ...]
    merged_layers: int
    skipped_layers: Tuple[str, ...]
    output_dir: str
    verdict: str  # "OK" / "MINOR" / "MAJOR" / "UNKNOWN" (v0.57.1 live)


def _validate_weights(weights: Sequence[float], n: int) -> Tuple[float, ...]:
    if not isinstance(weights, Sequence) or isinstance(weights, str):
        raise TypeError("weights must be a sequence of floats")
    if len(weights) != n:
        raise ValueError(f"weights length {len(weights)} != number of adapters {n}")
    out: list[float] = []
    for w in weights:
        if isinstance(w, bool):
            raise TypeError("weights must not contain bool")
        if not isinstance(w, (int, float)):
            raise TypeError("weights must be numeric")
        wf = float(w)
        if not math.isfinite(wf):
            raise ValueError("weights must be finite")
        if wf < 0:
            raise ValueError("weights must be non-negative")
        out.append(wf)
    total = sum(out)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return tuple(out)


def _validate_density(density: float, field: str) -> float:
    if isinstance(density, bool) or not isinstance(density, (int, float)):
        raise TypeError(f"{field} must be float")
    val = float(density)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    if not 0 < val <= 1:
        raise ValueError(f"{field} must be in (0, 1]")
    return val


def _lora_rank_of(name: str, shape: Tuple[int, ...]) -> Optional[int]:
    """LoRA rank a factor tensor carries, or ``None`` for any other tensor.

    ``lora_A`` / ``lora_embedding_A`` hold the rank on axis 0, ``lora_B`` /
    ``lora_embedding_B`` on axis 1 (linear, embedding and conv layers alike).
    """
    if "lora_A" in name or "lora_embedding_A" in name:
        return int(shape[0]) if len(shape) >= 1 else None
    if "lora_B" in name or "lora_embedding_B" in name:
        return int(shape[1]) if len(shape) >= 2 else None
    return None


def _module_of(name: str) -> str:
    """``...q_proj.lora_A.weight`` -> ``...q_proj``; other names unchanged."""
    for marker in (".lora_embedding_A", ".lora_embedding_B", ".lora_A", ".lora_B"):
        head, sep, _tail = name.partition(marker)
        if sep:
            return head
    return name


def check_mergeable(weights_list: Sequence[Mapping[str, Any]]) -> None:
    """Refuse inputs that ``soup adapters merge`` cannot merge (#2907).

    Every strategy merges only the tensors all inputs share and skips a shared
    tensor whose shapes differ. Inputs with different LoRA ranks differ in
    shape on every factor, so the merge used to be empty and still exit 0.
    Raises ``ValueError`` when:

    - the inputs share no tensor name at all (disjoint ``target_modules``), or
    - a shared LoRA tensor differs in shape between inputs (on every module,
      or on some only, as with ``rank_pattern``), or
    - no shared tensor has one shape, so the merge would be empty.

    Not a refusal, as before: a tensor present in only some inputs stays out
    of the merge, and a shared tensor that is not a LoRA one (a
    ``modules_to_save`` head, say) and differs in shape is skipped by the
    strategy and listed as skipped.
    """
    import numpy as np

    if len(weights_list) < _MIN_ADAPTERS:
        return
    shared = set(weights_list[0].keys())
    for adapter in weights_list[1:]:
        shared &= set(adapter.keys())
    if not shared:
        raise ValueError(
            "cannot merge: the adapters share no tensor name, so the merged "
            "adapter would hold no weights (the inputs target different "
            "modules). Nothing was written."
        )
    mergeable = 0
    for name in sorted(shared):
        shapes = [tuple(np.shape(adapter[name])) for adapter in weights_list]
        if len(set(shapes)) == 1:
            mergeable += 1
            continue
        if "lora_" not in name:
            continue
        ranks = [_lora_rank_of(name, shape) for shape in shapes]
        if None not in ranks and len(set(ranks)) > 1:
            per_input = ", ".join(
                f"rank {rank} in adapter {pos}" for pos, rank in enumerate(ranks, 1)
            )
            raise ValueError(
                f"cannot merge: module {_module_of(name)!r} has LoRA {per_input}. "
                "`soup adapters merge` needs the same rank on every module; for "
                "mixed ranks use `soup adapters arithmetic ... --rank N`. "
                "Nothing was written."
            )
        per_input = ", ".join(
            f"{shape} in adapter {pos}" for pos, shape in enumerate(shapes, 1)
        )
        raise ValueError(
            f"cannot merge: tensor {name!r} has shape {per_input}. "
            "`soup adapters merge` needs every shared tensor to have one shape "
            "(were the adapters trained on the same base model?). "
            "Nothing was written."
        )
    if not mergeable:
        raise ValueError(
            "cannot merge: no tensor the adapters share has the same shape in "
            "every input, so the merged adapter would hold no weights. "
            "Nothing was written."
        )


def merge_linear(
    weights_list: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
) -> Tuple[dict[str, Any], Tuple[str, ...]]:
    """Weighted merge per layer over the intersection of names.

    A LoRA's delta is ``B @ A``, so a weight ``c`` on both factors would scale
    that input's own delta by ``c²`` (#2908). Each ``lora_A`` / ``lora_B`` (and
    ``lora_embedding_A`` / ``lora_embedding_B``) tensor is scaled by ``√c``
    instead, so an input's own term is ``c`` times its delta — the rule
    ``soup adapters arithmetic`` uses on its same-rank path. Any other tensor
    keeps ``c``. Inputs that use the same rank slots also contribute cross
    terms ``√(cᵢcⱼ)·Bᵢ@Aⱼ``, as they do there.
    """
    import numpy as np

    from soup_cli.utils.adapter_arithmetic import _factor_coeff

    if len(weights_list) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    if len(weights_list) > _MAX_ADAPTERS:
        raise ValueError(f"at most {_MAX_ADAPTERS} adapters")
    w = _validate_weights(weights, len(weights_list))
    norm = sum(w)
    coeffs = [wi / norm for wi in w]

    shared = set(weights_list[0].keys())
    for adapters in weights_list[1:]:
        shared &= set(adapters.keys())

    merged: dict[str, Any] = {}
    skipped: list[str] = []
    for name in sorted(shared):
        tensors = [np.asarray(a[name], dtype=np.float64) for a in weights_list]
        shapes = {t.shape for t in tensors}
        if len(shapes) > 1:
            skipped.append(name)
            continue
        acc = np.zeros_like(tensors[0])
        for c, t in zip(coeffs, tensors):
            acc += _factor_coeff(name, c) * t
        merged[name] = acc.astype(np.float32)
    return merged, tuple(skipped)


def merge_ties(
    weights_list: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    density: float = 0.2,
) -> Tuple[dict[str, Any], Tuple[str, ...]]:
    """TIES merge: trim low magnitudes, elect majority sign, disjoint avg."""
    import numpy as np

    density_val = _validate_density(density, "density")
    if len(weights_list) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    w = _validate_weights(weights, len(weights_list))

    shared = set(weights_list[0].keys())
    for adapters in weights_list[1:]:
        shared &= set(adapters.keys())

    merged: dict[str, Any] = {}
    skipped: list[str] = []
    for name in sorted(shared):
        tensors = [np.asarray(a[name], dtype=np.float64) for a in weights_list]
        shapes = {t.shape for t in tensors}
        if len(shapes) > 1:
            skipped.append(name)
            continue
        # Per-adapter trim: keep top density% by magnitude, zero out rest
        trimmed = []
        for t in tensors:
            flat = np.abs(t).flatten()
            if flat.size == 0:
                trimmed.append(t.copy())
                continue
            k = max(1, int(flat.size * density_val))
            threshold = np.partition(flat, -k)[-k] if k < flat.size else 0.0
            mask = np.abs(t) >= threshold
            trimmed.append(np.where(mask, t, 0.0))
        # Elect majority sign per element. On a tie (sign_sum == 0) default
        # to positive (TIES paper §3) so all-tied entries are not silently
        # zeroed — `np.sign(0) == 0` would set `elected_sign` to 0 and the
        # subsequent agree_mask would drop every parameter (review fix HIGH).
        stacked = np.stack(trimmed)  # (n_adapters, ...)
        sign_sum = np.sign(stacked).sum(axis=0)
        elected_sign = np.where(sign_sum >= 0, 1.0, -1.0)
        # Disjoint average: only average elements that agree with elected sign
        agree_mask = np.sign(stacked) == elected_sign[None, ...]
        weighted = np.zeros_like(trimmed[0])
        weight_sum = np.zeros_like(trimmed[0])
        for wi, t, mask in zip(w, trimmed, agree_mask):
            weighted += wi * t * mask
            weight_sum += wi * mask
        out = np.where(weight_sum > 0, weighted / np.maximum(weight_sum, 1e-12), 0.0)
        merged[name] = out.astype(np.float32)
    return merged, tuple(skipped)


def merge_dare(
    weights_list: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    density: float = 0.5,
    seed: int = 0,
) -> Tuple[dict[str, Any], Tuple[str, ...]]:
    """DARE merge: random drop with prob (1-density), rescale 1/density, average."""
    import numpy as np

    density_val = _validate_density(density, "density")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be int")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if len(weights_list) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    w = _validate_weights(weights, len(weights_list))
    norm = sum(w)
    coeffs = [wi / norm for wi in w]

    shared = set(weights_list[0].keys())
    for adapters in weights_list[1:]:
        shared &= set(adapters.keys())

    from soup_cli.utils.adapter_arithmetic import _factor_coeff

    rng = np.random.default_rng(seed)
    merged: dict[str, Any] = {}
    skipped: list[str] = []
    for name in sorted(shared):
        tensors = [np.asarray(a[name], dtype=np.float64) for a in weights_list]
        shapes = {t.shape for t in tensors}
        if len(shapes) > 1:
            skipped.append(name)
            continue
        acc = np.zeros_like(tensors[0])
        for c, t in zip(coeffs, tensors):
            mask = rng.random(t.shape) < density_val
            rescaled = np.where(mask, t / density_val, 0.0)
            acc += _factor_coeff(name, c) * rescaled
        merged[name] = acc.astype(np.float32)
    return merged, tuple(skipped)


def merge_svd(
    weights_list: Sequence[Mapping[str, Any]],
    weights: Sequence[float],
    *,
    rank: int | None = None,
) -> Tuple[dict[str, Any], Tuple[str, ...]]:
    """SVD merge: linear-average, then low-rank reconstruct each 2D tensor.

    Non-2D tensors are merged linearly without truncation.
    """
    import numpy as np

    if rank is not None:
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("rank must be int or None")
        if rank < 1:
            raise ValueError("rank must be ≥ 1")

    linear_merged, skipped = merge_linear(weights_list, weights)
    if rank is None:
        return linear_merged, skipped

    out: dict[str, Any] = {}
    for name, tensor in linear_merged.items():
        arr = np.asarray(tensor, dtype=np.float64)
        if arr.ndim != 2:
            out[name] = arr.astype(np.float32)
            continue
        try:
            u_mat, sv, vt_mat = np.linalg.svd(arr, full_matrices=False)
        except np.linalg.LinAlgError:
            out[name] = arr.astype(np.float32)
            continue
        r = min(rank, sv.shape[0])
        reconstructed = u_mat[:, :r] @ np.diag(sv[:r]) @ vt_mat[:r, :]
        out[name] = reconstructed.astype(np.float32)
    return out, skipped


#: The bias PEFT adds to ``lora_B``'s output for ``lora_bias=True``; scaled with it.
_LORA_B_BIAS = ".lora_B.bias"


def fold_lora_scaling(
    weights_list: Sequence[Mapping[str, Any]],
    adapter_dirs: Sequence[str],
) -> list[Mapping[str, Any]]:
    """Rescale every adapter's ``lora_B`` to decode at the first adapter's scaling.

    The merge strategies combine raw ``lora_A`` / ``lora_B`` tensors, and the merged
    adapter is written with the FIRST adapter's ``adapter_config.json``, so PEFT
    decodes every merged module at that adapter's scaling. An input trained with a
    different ``lora_alpha``, ``use_rslora`` or per-module ``alpha_pattern`` /
    ``rank_pattern`` would otherwise contribute at the wrong strength (a
    ``lora_alpha=32`` adapter merged under a ``lora_alpha=16`` template counts for
    half). Each ``lora_B`` / ``lora_embedding_B``, and the ``lora_B`` bias of an adapter
    saved with ``lora_bias=True``, is multiplied by its own module
    scaling over the template's, so every ``B @ A`` keeps its effective delta.

    Inputs whose scaling already matches the template are passed through
    unchanged, as is any module whose scaling cannot be read from either config.
    An input whose own scaling is 0 gets a zero ``lora_B``: it decoded to no delta and
    still adds none. A template whose scaling is 0 (``lora_alpha: 0``) decodes every
    module at 0 whatever the tensors hold, so the ratio is undefined and every input is
    passed through as it is. DoRA's ``lora_magnitude_vector`` is not multiplied by the
    scaling in PEFT's forward, so it is never rescaled; every ``lora_A`` is left alone.
    """
    import numpy as np

    from soup_cli.utils.adapter_arithmetic import _lora_role

    if len(weights_list) != len(adapter_dirs):
        raise ValueError(
            f"weights_list ({len(weights_list)}) and adapter_dirs "
            f"({len(adapter_dirs)}) length mismatch"
        )
    if not weights_list:
        return []
    template = read_adapter_config_or_none(adapter_dirs[0])
    if template is None:
        return list(weights_list)

    folded: list[Mapping[str, Any]] = [weights_list[0]]
    for weights, adapter_dir in zip(weights_list[1:], adapter_dirs[1:]):
        config = read_adapter_config_or_none(adapter_dir)
        rescaled: Optional[dict[str, Any]] = None
        if config is not None:
            for name, tensor in weights.items():
                role = _lora_role(name)
                if role is None and name.endswith(_LORA_B_BIAS):
                    # PEFT scales lora_B's output, its bias (lora_bias=True) included.
                    role = (name[: -len(_LORA_B_BIAS)], "B", _LORA_B_BIAS)
                if role is None or role[1] != "B":
                    continue
                module = peft_module_name(role[0])
                target = lora_module_scaling(template, module)
                own = lora_module_scaling(config, module)
                if target is None or own is None or target == 0.0 or own == target:
                    continue
                if rescaled is None:
                    rescaled = dict(weights)
                rescaled[name] = np.asarray(tensor, dtype=np.float64) * (own / target)
        folded.append(weights if rescaled is None else rescaled)
    return folded


def merge_adapters(
    adapter_paths: Sequence[str],
    output_dir: str,
    *,
    strategy: MergeStrategy,
    weights: Sequence[float] | None = None,
    density: float = 0.2,
    seed: int = 0,
    rank: int | None = None,
) -> MergeReport:
    """End-to-end: load adapters, merge, write output as safetensors."""
    from soup_cli.utils.adapter_diff import load_adapter_weights

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {SUPPORTED_STRATEGIES}, got {strategy!r}"
        )
    if strategy == "cmaes":
        # cmaes is an evolutionary *search* over linear weights, not a
        # one-shot tensor merge — routing it through this function would
        # silently fall into the svd branch. Fail loudly with the real path.
        raise ValueError(
            "cmaes is an evolutionary strategy; use run_cmaes_merge / "
            "`soup adapters merge --strategy cmaes --eval ...`, "
            "not merge_adapters(strategy='cmaes')"
        )
    if not isinstance(adapter_paths, Sequence) or isinstance(adapter_paths, str):
        raise TypeError("adapter_paths must be a sequence of strings")
    if len(adapter_paths) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    if len(adapter_paths) > _MAX_ADAPTERS:
        raise ValueError(f"at most {_MAX_ADAPTERS} adapters")

    require_adapter_dependencies(WRITE_ADAPTER_DEPS)
    enforce_under_cwd_and_no_symlink(output_dir, "output_dir")
    ensure_distinct_paths(
        [("adapter", path) for path in adapter_paths], [("output_dir", output_dir)]
    )

    weights_list = [load_adapter_weights(p) for p in adapter_paths]
    check_mergeable(weights_list)
    # Inputs that pass are rescaled to the template's LoRA scaling (#2861); the
    # fold changes lora_B values only, never a shape, so the check above holds.
    weights_list = fold_lora_scaling(weights_list, adapter_paths)

    if weights is None:
        eq = 1.0 / len(adapter_paths)
        weights = [eq] * len(adapter_paths)
    coeffs = _validate_weights(weights, len(adapter_paths))

    if strategy == "linear":
        merged, skipped = merge_linear(weights_list, coeffs)
    elif strategy == "ties":
        merged, skipped = merge_ties(weights_list, coeffs, density=density)
    elif strategy == "dare":
        merged, skipped = merge_dare(weights_list, coeffs, density=density, seed=seed)
    else:  # svd
        merged, skipped = merge_svd(weights_list, coeffs, rank=rank)

    write_merged_adapter(output_dir, adapter_paths[0], merged)

    return MergeReport(
        strategy=strategy,
        adapters=tuple(adapter_paths),
        weights=coeffs,
        merged_layers=len(merged),
        skipped_layers=skipped,
        output_dir=output_dir,
        verdict="UNKNOWN",  # v0.57.1: live canary eval via v0.55 gate
    )


_MAX_ADAPTER_CONFIG_BYTES = 256 * 1024


def _reject_if_symlink(path: Path, field: str) -> None:
    """TOCTOU defence: lstat the raw path before opening."""
    if os.path.lexists(str(path)):
        st = os.lstat(str(path))
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(f"{field} must not be a symlink: {path.name}")


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Atomic write via mkstemp + os.replace in target's parent dir."""
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_if_symlink(target, "output file")
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, str(target))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_merged_adapter(
    output_dir: str,
    template_source: str,
    weights: Mapping[str, Any],
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> None:
    """Write adapter_model.safetensors + copy adapter_config.json from source.

    Both writes are atomic (tempfile + os.replace) and reject pre-placed
    symlinks at the target path (TOCTOU defence; mirrors v0.53.1 policy).
    The source config is lstat-checked before reading and size-capped at
    256 KB per the v0.53.0 ``load_quant_config`` precedent.

    ``config_overrides`` (e.g. ``{"r": 12, "lora_alpha": 12}``) is shallow-merged
    into the copied config before writing — used by the mixed-rank concat path
    (#305) so the emitted ``adapter_config.json``'s ``r`` matches the
    concatenated tensors and the adapter stays loadable.
    """
    require_adapter_dependencies(WRITE_ADAPTER_DEPS)
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise RuntimeError(WRITE_ADAPTER_DEPS_MSG) from exc

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    safetensors_target = out_path / "adapter_model.safetensors"
    _reject_if_symlink(safetensors_target, "output safetensors")
    # safetensors.save_file writes atomically internally via mmap-rename on
    # POSIX, but on Windows it does not — write to sibling tmp + replace
    # ourselves so behaviour is consistent across platforms.
    fd, tmp_safe = tempfile.mkstemp(
        dir=str(out_path), prefix=".tmp_", suffix=".safetensors"
    )
    os.close(fd)
    try:
        save_file(weights, tmp_safe)
        os.replace(tmp_safe, str(safetensors_target))
    except Exception:
        try:
            os.unlink(tmp_safe)
        except OSError:
            pass
        raise

    # Copy adapter_config.json from first source so the merged adapter is loadable
    source_cfg = Path(template_source) / "adapter_config.json"
    if os.path.lexists(str(source_cfg)):
        st = os.lstat(str(source_cfg))
        if stat.S_ISLNK(st.st_mode):
            raise ValueError(
                "source adapter_config.json must not be a symlink"
            )
        if st.st_size > _MAX_ADAPTER_CONFIG_BYTES:
            raise ValueError(
                f"source adapter_config.json > {_MAX_ADAPTER_CONFIG_BYTES} byte cap"
            )
        cfg = json.loads(source_cfg.read_text(encoding="utf-8"))
        if config_overrides:
            if not isinstance(cfg, dict):
                raise ValueError(
                    "source adapter_config.json is not a JSON object; cannot "
                    "apply config overrides"
                )
            cfg.update(dict(config_overrides))
        target_cfg = out_path / "adapter_config.json"
        _atomic_write_bytes(
            target_cfg,
            json.dumps(cfg, indent=2).encode("utf-8"),
        )


# Back-compat alias: ``_write_merged_adapter`` was private through v0.71.4;
# promoted to the public ``write_merged_adapter`` so cmaes_merge imports a
# public name (review MEDIUM-3). Keep the old name for any external caller.
_write_merged_adapter = write_merged_adapter


_MAX_CANARY_BYTES = 16 * 1024 * 1024  # 16 MiB cap on the canary-suite JSON
_VERDICT_MINOR = 0.02
_VERDICT_MAJOR = 0.05


def _classify_drop(delta: float) -> str:
    """OK / MINOR / MAJOR per the v0.26.0 Quant-Lobotomy taxonomy.

    ``delta`` is ``candidate_mean - baseline_mean``: positive (improvement)
    or a small drop is OK, a 2-5 % drop is MINOR, a >5 % drop is MAJOR.
    """
    if delta >= 0:
        return "OK"
    drop = -delta
    if drop < _VERDICT_MINOR:
        return "OK"
    if drop < _VERDICT_MAJOR:
        return "MINOR"
    return "MAJOR"


def _require_score_list(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a JSON list of numbers")
    out: list[float] = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{field} entries must be numbers")
        vf = float(v)
        if not math.isfinite(vf):
            raise ValueError(f"{field} entries must be finite")
        out.append(vf)
    if not out:
        raise ValueError(f"{field} must be non-empty")
    return out


def _load_canary_scores(
    canary_suite: str,
    scorer: Optional[CanaryScorer],
) -> Tuple[list[float], list[float]]:
    """Resolve baseline + candidate per-prompt scores from a canary suite.

    Two supported shapes:

    - ``{"baseline_scores": [...], "candidate_scores": [...]}`` — pre-scored
      (no model load; the no-GPU workflow). Operators run ``soup eval custom``
      against the baseline and merged adapters, then assemble the two arrays.
    - ``{"tasks": [{"prompt", "expected"}, ...]}`` — requires an injectable
      ``scorer(role, tasks) -> list[float]`` (the live path).
    """
    # Defence-in-depth cwd containment + symlink rejection at the read site so
    # the public ``predict_merged_verdict`` entry point is safe even when a
    # non-CLI caller skips the CLI-boundary check (mirrors every other read
    # surface in this changeset).
    enforce_under_cwd_and_no_symlink(canary_suite, "canary_suite")
    path = Path(canary_suite)
    if not path.is_file():
        raise ValueError(f"canary suite not found: {path.name}")
    # Open ONCE with O_NOFOLLOW and enforce the size cap on the same fd
    # (os.fstat), so the symlink-rejection + 16 MiB cap cannot be defeated by a
    # local file swap between the lstat/stat and the read (TOCTOU) — matches the
    # O_NOFOLLOW+fstat read pattern used elsewhere in the codebase.
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"canary suite unreadable: {type(exc).__name__}") from exc
    fh = None
    try:
        if os.fstat(fd).st_size > _MAX_CANARY_BYTES:
            raise ValueError("canary suite exceeds 16 MiB cap")
        fh = os.fdopen(fd, "r", encoding="utf-8")
        raw = fh.read()
    finally:
        if fh is not None:
            fh.close()
        else:
            os.close(fd)
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"canary suite is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("canary suite root must be a JSON object")

    if "baseline_scores" in data and "candidate_scores" in data:
        baseline = _require_score_list(data["baseline_scores"], "baseline_scores")
        candidate = _require_score_list(data["candidate_scores"], "candidate_scores")
    else:
        tasks = data.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            raise ValueError(
                "canary suite must contain 'baseline_scores'+'candidate_scores' "
                "or a non-empty 'tasks' list"
            )
        if scorer is None or not callable(scorer):
            raise ValueError(
                "canary suite has 'tasks' but no scorer; supply pre-scored "
                "'baseline_scores'/'candidate_scores' arrays or pass scorer="
            )
        baseline = _require_score_list(scorer("baseline", tasks), "baseline scorer output")
        candidate = _require_score_list(scorer("candidate", tasks), "candidate scorer output")

    if len(baseline) != len(candidate):
        raise ValueError(
            f"baseline ({len(baseline)}) and candidate ({len(candidate)}) "
            "score counts must match"
        )
    return baseline, candidate


def predict_merged_verdict(
    report: MergeReport,
    canary_suite: str | None = None,
    *,
    scorer: Optional[CanaryScorer] = None,
) -> str:
    """Live canary verdict for a merged adapter (v0.71.4 #172).

    Returns the merged adapter's OK / MINOR / MAJOR verdict by comparing
    per-prompt canary scores against the first input adapter (the "baseline"
    by convention) using the v0.26.0 Quant-Lobotomy thresholds. When
    ``canary_suite`` is ``None`` the existing ``report.verdict`` (``UNKNOWN``
    for a fresh merge) is returned unchanged — back-compat with v0.57.0.
    """
    if not isinstance(report, MergeReport):
        raise TypeError("report must be MergeReport")
    if canary_suite is not None and not isinstance(canary_suite, str):
        raise TypeError("canary_suite must be str or None")
    if canary_suite is None:
        return report.verdict
    baseline, candidate = _load_canary_scores(canary_suite, scorer)
    delta = (sum(candidate) / len(candidate)) - (sum(baseline) / len(baseline))
    return _classify_drop(delta)
