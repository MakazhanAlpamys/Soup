"""How much of the base a LoRA adapter's merged update changes (#2762).

``soup ship --adapter`` prints this number beside the verdict. For every module
the adapter targets, the update is added to the base weight in the base's own
dtype, with the same arithmetic ``soup merge --low-memory`` writes
(:func:`soup_cli.utils.lowmem_merge.merged_tensor`). The number is the share of
those elements whose value differs from the base weight afterwards.

Values, not bits, are compared: adding a zero contribution turns ``-0.0`` into
``+0.0``, which is not a change, and a NaN that stays NaN is not one either.

Only the base tensors the adapter targets are read, one at a time, from the
base's safetensors files (memory-mapped); the model itself is not loaded.

The number is a display line, not a rule. The calibration in
``docs/evaluation.md`` found no fraction that separates adapters trained with a
near-zero learning rate from early checkpoints of real runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union

from soup_cli.utils.lowmem_merge import (
    _FLOAT_DTYPES,
    LowMemoryMergeError,
    _checked_source,
    _flag,
    _source_roots,
    _weight_files,
    load_adapter,
    lora_scaling,
    merged_tensor,
)
from soup_cli.utils.ship_verdict import AdapterChange


def value_changes(merged: Any, weight: Any) -> int:
    """Elements whose value differs (``-0.0 == +0.0``; NaN vs NaN is no change)."""
    import torch

    differ = merged != weight
    differ &= ~(torch.isnan(merged) & torch.isnan(weight))
    return int(differ.sum())


def measure_adapter_change(
    adapter_dir: Union[str, Path], base_dir: Union[str, Path]
) -> AdapterChange:
    """Measure the changed fraction, or say why this adapter is not measured.

    Adapters the low-memory reader refuses (DoRA, ``bias != none``,
    ``modules_to_save``, embeddings, ``lm_head``, non-LoRA types, ...) and bases
    it cannot read come back as ``AdapterChange.not_measured(reason)``.
    """
    try:
        return _measure(Path(adapter_dir), Path(base_dir))
    except LowMemoryMergeError as exc:
        return AdapterChange.not_measured(str(exc))


def _measure(adapter_dir: Path, base_dir: Path) -> AdapterChange:
    from safetensors import safe_open

    config, pairs = load_adapter(adapter_dir)
    if not pairs:
        raise LowMemoryMergeError("the adapter has no LoRA A/B weights")
    scaling = lora_scaling(config)
    fan_in_fan_out = _flag(config, "fan_in_fan_out")

    roots = _source_roots(base_dir)
    files, index = _weight_files(base_dir, roots)
    by_file: Dict[str, List[str]] = {}
    if index is not None:
        weight_map = index["weight_map"]
        missing = sorted(key for key in pairs if key not in weight_map)
        for key in pairs:
            if key in weight_map:
                by_file.setdefault(weight_map[key], []).append(key)
    else:
        by_file[files[0]] = list(pairs)
        missing = []

    elements = changed = 0
    dtypes = set()
    seen = set()
    for name in sorted(by_file):
        path = _checked_source(base_dir, name, roots)
        with safe_open(str(path), framework="pt") as source:
            present = set(source.keys())
            for key in sorted(by_file[name]):
                if key not in present:
                    missing.append(key)
                    continue
                stored = source.get_slice(key).get_dtype()
                if stored not in _FLOAT_DTYPES:
                    raise LowMemoryMergeError(
                        f"base tensor {key!r} is stored as {stored}, not a float type"
                    )
                weight = source.get_tensor(key)
                lora_a, lora_b = pairs[key]
                merged = merged_tensor(
                    weight,
                    lora_a,
                    lora_b,
                    scaling=scaling,
                    fan_in_fan_out=fan_in_fan_out,
                    dtype=weight.dtype,
                )
                changed += value_changes(merged, weight)
                elements += weight.numel()
                dtypes.add(str(weight.dtype).replace("torch.", ""))
                seen.add(key)
    if missing:
        raise LowMemoryMergeError(
            f"{len(missing)} adapter weights match no base tensor, e.g. {sorted(missing)[0]!r}"
        )
    return AdapterChange(
        changed=changed,
        elements=elements,
        modules=len(seen),
        base_dtypes=tuple(sorted(dtypes)),
    )
