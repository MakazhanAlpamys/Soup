"""v0.71.8 #218 — live measurement for ``soup probe interference --measure``.

The v0.66.0 :mod:`soup_cli.utils.interference` matrix builder consumes a dict of
pre-measured per-pair losses. This module produces that dict by actually loading
the base model + each LoRA adapter (via PEFT multi-adapter), measuring loss on a
shared eval suite for each adapter alone (diagonal) and for each co-loaded pair
(off-diagonal, ``add_weighted_adapter(combination_type="cat")`` — the honest
"both adapters live" semantics), and returning the losses keyed by
``(target, co_loaded)``.

Heavy imports (``torch`` / ``transformers`` / ``peft``) live inside the function
and in :mod:`soup_cli.utils.live_eval`; importing this module is cheap.

On a 4 GB box keep the adapters tiny (SmolLM2-135M LoRA) and the eval suite
small — the eval pairs are capped at ``_MAX_EVAL_PAIRS``.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Dict, Optional, Tuple

from soup_cli.utils.interference import _MAX_ADAPTERS, _MIN_ADAPTERS

_MAX_EVAL_PAIRS = 64


def _row_input(row: object) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("prompt", "instruction", "input", "question", "query"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def _row_output(row: object) -> str:
    if not isinstance(row, Mapping):
        return ""
    for key in ("response", "completion", "output", "answer", "chosen", "text"):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return ""


def measure_interference_losses(
    base: str,
    adapters: Mapping[str, str],
    eval_rows: Sequence[Mapping[str, object]],
    *,
    device: Optional[str] = None,
    max_length: int = 256,
    trust_remote_code: bool = False,
) -> Dict[Tuple[str, str], float]:
    """Measure per-pair losses for a fleet of LoRA adapters (#218).

    ``adapters`` maps adapter name → local adapter directory. Returns a dict of
    ``{(target, co_loaded): loss}`` — diagonal ``(a, a)`` is adapter A alone,
    off-diagonal ``(a, b)`` is the loss with A AND B both loaded. The result is
    directly consumable by
    :func:`soup_cli.utils.interference.build_interference_matrix`.
    """
    if not isinstance(base, str) or not base.strip():
        raise ValueError("base must be a non-empty string")
    if not isinstance(adapters, Mapping):
        raise TypeError("adapters must be a mapping of name -> path")
    names = list(adapters)
    if len(names) < _MIN_ADAPTERS:
        raise ValueError(f"need at least {_MIN_ADAPTERS} adapters")
    if len(names) > _MAX_ADAPTERS:
        raise ValueError(f"too many adapters (>{_MAX_ADAPTERS}); split your fleet")
    # (dict keys are inherently unique — the CLI `_parse_adapter_specs` rejects
    #  duplicate `--adapter` names before this point.)
    for name, path in adapters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("adapter names must be non-empty strings")
        if not isinstance(path, str) or not path:
            raise ValueError(f"adapter path for {name!r} must be a non-empty string")

    from peft import PeftModel

    from soup_cli.utils import live_eval

    model, tokenizer, dev = live_eval.load_model_and_tokenizer(
        base, device=device, trust_remote_code=trust_remote_code
    )

    first = names[0]
    peft_model = PeftModel.from_pretrained(
        model, adapters[first], adapter_name=first
    )
    try:
        for name in names[1:]:
            peft_model.load_adapter(adapters[name], adapter_name=name)

        pairs = live_eval._build_pairs(
            eval_rows, input_extractor=_row_input, output_extractor=_row_output
        )[:_MAX_EVAL_PAIRS]
        if not pairs:
            raise ValueError("eval suite has no usable (prompt, target) pairs")

        losses: Dict[Tuple[str, str], float] = {}

        # ``active_model`` is passed explicitly (not closed over) so the
        # ``finally`` block below can ``del peft_model`` — Python forbids
        # deleting a name referenced by a nested function.
        def _measure(active_model: object, key: Tuple[str, str]) -> float:
            loss = live_eval.compute_eval_loss(
                active_model, tokenizer, pairs, device=dev, max_length=max_length
            )
            # Fail fast with a targeted message instead of letting a NaN
            # surface as a confusing "loss must be float" deep in the matrix
            # builder after the whole O(N^2) measurement already ran.
            if not math.isfinite(loss):
                raise ValueError(
                    f"eval loss for {key} was non-finite — the eval suite rows "
                    "likely have empty target spans"
                )
            return loss

        # Diagonal — each adapter alone.
        for name in names:
            peft_model.set_adapter(name)
            losses[(name, name)] = _measure(peft_model, (name, name))

        # Off-diagonal — A and B both loaded (concatenated LoRA).
        for target in names:
            for co_loaded in names:
                if target == co_loaded:
                    continue
                combo = f"__combo__{target}__{co_loaded}"
                peft_model.add_weighted_adapter(
                    [target, co_loaded], weights=[1.0, 1.0],
                    adapter_name=combo, combination_type="cat",
                )
                try:
                    peft_model.set_adapter(combo)
                    losses[(target, co_loaded)] = _measure(
                        peft_model, (target, co_loaded)
                    )
                finally:
                    # Free the combo even if the eval raised, so the live
                    # adapter set never blows up to N^2 (review fix).
                    peft_model.delete_adapter(combo)

        return losses
    finally:
        # Best-effort free of the base + adapter weights so a 4 GB box can run
        # a subsequent --measure without a wedged GPU (review fix).
        try:
            import torch

            del peft_model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — cleanup must never mask the result
            pass


__all__ = ["measure_interference_losses"]
