"""k-bit model preparation that keeps the non-quantised weights in bf16 (#2609).

PEFT's ``prepare_model_for_kbit_training`` freezes the base, turns on input gradients
and checkpointing, and casts every fp16/bf16 parameter that is not a ``Params4bit`` to
fp32 (embeddings, norms, the output head). The cast feeds an fp32 residual stream through
a model whose weights are bf16: on a resident 4-bit Qwen2.5-7B SFT step (RTX 4090, NF4,
LoRA r16, 8 x 512 tokens) 837 of 916 ms of elementwise GPU time ran on fp32 inputs.
Casting embeddings, head and norms back to bf16 after the call made the step 24.9% faster
(2.256 s to 1.695 s), lowered the allocator peak by 3,905 MiB (17,425 to 13,520 MiB) and
kept the loss inside the tolerance written down before the run
(``benchmarks/gate-2609-kbit-fp32-ab.md``).

``prepare_kbit_model`` is the one place Soup calls PEFT's function. It calls it and casts
back exactly what PEFT upcast, so PEFT keeps owning the freezing, the input-grad hook, the
``use_reentrant`` rule and the quantisation-method gates (GPTQ, AQLM, EETQ, HQQ and torchao
models are not cast by PEFT and so are not touched here either). It restores only when the
run computes in bf16:

- a parameter that loaded as bf16 and came back fp32 goes back to bf16;
- a parameter that loaded as fp16 or fp32 keeps PEFT's result: the dtype it was loaded in
  is not the compute dtype, and that case has no measurement;
- ``lora_*`` parameters keep what PEFT gave them (the adapter-dtype rules of #1503 and
  #1573 live in ``utils/mixed_precision.py``);
- a card without bf16 hardware (``compute_dtype`` fp16) and the CPU keep PEFT's fp32, which
  fp16 training needs.

Lazy imports: torch and peft are imported inside the functions.
"""

from __future__ import annotations

from typing import Any, Optional


def kbit_compute_dtype(device: str, *, prefer_fp16: bool = False):
    """The torch dtype a 4-bit/8-bit run computes in on ``device``.

    The same answer the trainers give ``TrainingArguments`` (``bf16_fp16_flags``) and
    bitsandbytes (``get_compute_dtype``): bf16 where the card has bf16 hardware, fp16
    where it does not, fp32 with no mixed precision (the CPU). ``prefer_fp16`` is for a
    run that trains in fp16 on a bf16 card (``grpo_fp16``, an ``auto_mixed_precision``
    pick of fp16).
    """
    import torch

    from soup_cli.utils.gpu import bf16_fp16_flags

    bf16, fp16 = bf16_fp16_flags(device)
    if bf16 and not prefer_fp16:
        return torch.bfloat16
    if bf16 or fp16:
        return torch.float16
    return torch.float32


def prepare_kbit_model(
    model: Any,
    *,
    use_gradient_checkpointing: bool,
    gradient_checkpointing_kwargs: Optional[dict] = None,
    compute_dtype: Any,
) -> Any:
    """PEFT's k-bit preparation, then bf16 back where the run computes in bf16 (#2609).

    ``compute_dtype`` is the dtype the run computes in (``kbit_compute_dtype``). Only
    ``torch.bfloat16`` restores anything; every other dtype is PEFT's behaviour unchanged.
    Returns the prepared model (PEFT prepares it in place).
    """
    import torch
    from peft import prepare_model_for_kbit_training

    if not isinstance(compute_dtype, torch.dtype):
        raise TypeError(
            f"compute_dtype must be a torch.dtype, got {type(compute_dtype).__name__}"
        )

    kwargs: dict[str, Any] = {"use_gradient_checkpointing": use_gradient_checkpointing}
    if gradient_checkpointing_kwargs is not None:
        kwargs["gradient_checkpointing_kwargs"] = gradient_checkpointing_kwargs

    if compute_dtype is not torch.bfloat16:
        return prepare_model_for_kbit_training(model, **kwargs)

    loaded = {name: param.dtype for name, param in model.named_parameters()}
    prepared = prepare_model_for_kbit_training(model, **kwargs)

    restored = 0
    for name, param in prepared.named_parameters():
        if (
            loaded.get(name) is torch.bfloat16
            and param.dtype is torch.float32
            and "lora_" not in name
        ):
            param.data = param.data.to(torch.bfloat16)
            restored += 1
    if restored and torch.cuda.is_available():
        # PEFT cleared the cache while the fp32 copies were still live; the freed
        # blocks are only returned to the driver once they are dropped here.
        torch.cuda.empty_cache()
    return prepared
