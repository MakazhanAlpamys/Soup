"""v0.71.7 — shared live model-loading helpers for the eval-runner family.

Every heavy import (``torch`` / ``transformers`` / ``peft``) is performed
*inside* a function so importing this module stays cheap (project lazy-import
policy — keeps ``soup --help`` fast).

Consumers:

* ``utils/advise.py`` — ``synth_probe_baselines`` / ``synth_probe_lora_delta``
  (#161) + ``measure_base_model_proximity`` (#162).
* ``utils/tunability.py`` — ``live_lora_probe`` (#208).
* ``utils/behavior_battery.py`` — base + adapter generation (#212).
* ``utils/diagnose/live.py`` — generator / multi-generator closures (#165).

The shared primitives are ``make_generator`` / ``make_multi_generator``
(text generation) and ``lora_probe`` / ``compute_eval_loss`` /
``measure_logit_agreement`` (loss + agreement measurement). All accept a
``device=None`` that resolves to CUDA when available else CPU.

Tests mock at this boundary (monkeypatch ``make_generator`` etc.) so the
orchestration logic in every consumer is exercised without a GPU; the real
model load is covered by the release-step-6 smoke on SmolLM2-135M.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from typing import Callable, Dict, List, Optional, Tuple

# Public closure types (mirror the diagnose protocols).
GeneratorFn = Callable[[str], str]
MultiGen = Callable[[str, int], "list[str]"]

# Bounds — keep the live paths from running away on a 4 GB box.
_MAX_PROMPT_TOKENS = 1024
_MAX_TRAIN_ROWS = 512
_MIN_TRAIN_ROWS = 1
_DEFAULT_LORA_R = 8
_DEFAULT_LR = 2e-4
_MAX_AGREEMENT_PAIRS = 128
# v0.71.8 — activation-capture bounds (residual-stream probe surface).
_MAX_CAPTURE_PROMPTS = 256
_MAX_CAPTURE_TOKENS = 8192
_LAYER_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_]+$")


def token_f1(predicted: str, target: str) -> float:
    """Token-overlap F1 (canonical SQuAD multiset overlap) of two strings.

    Shared by the advise baseline probe (#161) and the diagnose forgetting
    probe (#165). Returns ``0.0`` when either side has no alphanumeric tokens.
    """
    pred = re.findall(r"[A-Za-z0-9]+", predicted.lower())
    gold = re.findall(r"[A-Za-z0-9]+", target.lower())
    if not pred or not gold:
        return 0.0
    counts: Dict[str, int] = {}
    for tok in pred:
        counts[tok] = counts.get(tok, 0) + 1
    overlap = 0
    for tok in gold:
        if counts.get(tok, 0) > 0:
            counts[tok] -= 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def _check_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def resolve_device(device: Optional[str] = None) -> str:
    """Return ``device`` if given, else ``"cuda"`` when available, else ``"cpu"``.

    Never raises — falls back to ``"cpu"`` if torch is unavailable.
    """
    if device is not None:
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a non-empty string or None")
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001 — torch missing → cpu
        return "cpu"


def _apply_prompt_template(tokenizer: object, prompt: str) -> str:
    """Render a single user turn through the tokenizer's chat template.

    Falls back to the raw prompt when the tokenizer has no chat template.
    """
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        try:
            return tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001 — malformed template → raw prompt
            return prompt
    return prompt


def load_model_and_tokenizer(
    model_id: str,
    *,
    adapter: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    dtype: Optional[str] = None,
):
    """Load an ``AutoModelForCausalLM`` + tokenizer, optionally with a LoRA adapter.

    Returns ``(model, tokenizer, device)``. ``model`` is ``.eval()``-ed and
    moved to the resolved device. Heavy imports are local. ``dtype`` (e.g.
    ``"auto"``) is forwarded as ``torch_dtype`` so a caller can preserve the
    checkpoint's native precision instead of upcasting to fp32 (``soup shrink``
    needs this so the shipped smaller model is not silently re-widened).
    """
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    import torch  # noqa: F401 — ensures torch present
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {"trust_remote_code": trust_remote_code}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if adapter is not None:
        if not isinstance(adapter, str) or not adapter.strip():
            raise ValueError("adapter must be a non-empty string or None")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(dev)
    model.eval()
    return model, tokenizer, dev


def make_generator(
    model_id: str,
    *,
    adapter: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    trust_remote_code: bool = False,
    loaded: Optional[tuple] = None,
) -> GeneratorFn:
    """Build a deterministic ``GeneratorFn`` closure (greedy decode).

    ``loaded`` lets a caller share an already-loaded ``(model, tokenizer,
    device)`` triple across several closures (base + multi off one load).
    """
    _check_positive_int(max_new_tokens, "max_new_tokens")
    if loaded is not None and (not isinstance(loaded, tuple) or len(loaded) != 3):
        raise ValueError("loaded must be a (model, tokenizer, device) tuple")
    import torch

    model, tokenizer, dev = loaded or load_model_and_tokenizer(
        model_id, adapter=adapter, device=device, trust_remote_code=trust_remote_code
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def _gen(prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        text = _apply_prompt_template(tokenizer, prompt)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=_MAX_PROMPT_TOKENS
        ).to(dev)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_id,
            )
        gen_ids = out[0][prompt_len:]
        return tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _gen


def make_multi_generator(
    model_id: str,
    *,
    adapter: Optional[str] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    trust_remote_code: bool = False,
    loaded: Optional[tuple] = None,
) -> MultiGen:
    """Build a sampling ``MultiGen`` closure: ``multi(prompt, k) -> [str, ...]``."""
    _check_positive_int(max_new_tokens, "max_new_tokens")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not (temperature > 0)
    ):
        raise ValueError("temperature must be a positive number")
    if loaded is not None and (not isinstance(loaded, tuple) or len(loaded) != 3):
        raise ValueError("loaded must be a (model, tokenizer, device) tuple")
    import torch

    model, tokenizer, dev = loaded or load_model_and_tokenizer(
        model_id, adapter=adapter, device=device, trust_remote_code=trust_remote_code
    )
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    def _multi(prompt: str, k: int) -> List[str]:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive int")
        text = _apply_prompt_template(tokenizer, prompt)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=_MAX_PROMPT_TOKENS
        ).to(dev)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                num_return_sequences=k,
                pad_token_id=pad_id,
            )
        return [tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in out]

    return _multi


def _build_pairs(
    rows: Sequence[Mapping[str, object]],
    *,
    input_extractor: Callable[[Mapping[str, object]], str],
    output_extractor: Callable[[Mapping[str, object]], str],
) -> List[Tuple[str, str]]:
    """Build (prompt, target) text pairs, skipping rows missing either side."""
    pairs: List[Tuple[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        prompt = input_extractor(row)
        target = output_extractor(row)
        if prompt and target:
            pairs.append((prompt, target))
    return pairs


def _tokenize_pair(tokenizer: object, prompt: str, target: str, *, max_length: int):
    """Tokenise prompt+target, masking the prompt span in ``labels`` with -100."""
    import torch

    rendered = _apply_prompt_template(tokenizer, prompt)
    prompt_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token_id
    if eos is not None:
        target_ids = target_ids + [eos]
    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
    return (
        torch.tensor([input_ids], dtype=torch.long),
        torch.tensor([labels], dtype=torch.long),
    )


def compute_eval_loss(
    model: object,
    tokenizer: object,
    pairs: Sequence[Tuple[str, str]],
    *,
    device: str,
    max_length: int = 256,
) -> float:
    """Mean masked cross-entropy of ``model`` over (prompt, target) pairs.

    Returns ``float('nan')`` when no usable pair has a non-empty target span.
    """
    _check_positive_int(max_length, "max_length")
    import torch

    losses: List[float] = []
    model.eval()
    with torch.no_grad():
        for prompt, target in pairs:
            input_ids, labels = _tokenize_pair(
                tokenizer, prompt, target, max_length=max_length
            )
            if (labels != -100).sum().item() == 0:
                continue
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            out = model(input_ids=input_ids, labels=labels)
            loss = float(out.loss.item())
            if loss == loss:  # not NaN
                losses.append(loss)
    if not losses:
        return float("nan")
    return sum(losses) / len(losses)


def lora_probe(
    base: str,
    rows: Sequence[Mapping[str, object]],
    *,
    input_extractor: Callable[[Mapping[str, object]], str],
    output_extractor: Callable[[Mapping[str, object]], str],
    n_steps: int = 100,
    holdout_size: int = 64,
    device: Optional[str] = None,
    lr: float = _DEFAULT_LR,
    max_length: int = 256,
    trust_remote_code: bool = False,
) -> Tuple[float, float, float]:
    """Measure held-out loss before/after a short LoRA train. Returns
    ``(base_loss, probe_loss, wall_clock_seconds)``.

    The held-out slice is the last ``holdout_size`` pairs; the train slice is
    the remainder (capped at ``_MAX_TRAIN_ROWS``). LoRA-trains for ``n_steps``
    optimiser steps with batch size 1, then re-measures the SAME held-out
    slice with the adapter attached.
    """
    _check_positive_int(n_steps, "n_steps")
    _check_positive_int(holdout_size, "holdout_size")
    _check_positive_int(max_length, "max_length")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not (lr > 0):
        raise ValueError("lr must be a positive number")
    import torch
    from peft import LoraConfig, get_peft_model

    started = time.monotonic()
    model, tokenizer, dev = load_model_and_tokenizer(
        base, device=device, trust_remote_code=trust_remote_code
    )
    pairs = _build_pairs(
        rows, input_extractor=input_extractor, output_extractor=output_extractor
    )
    if len(pairs) < _MIN_TRAIN_ROWS + 1:
        raise ValueError("dataset has too few usable (prompt, target) pairs for a probe")

    holdout = pairs[-holdout_size:]
    train = pairs[:-holdout_size][:_MAX_TRAIN_ROWS]
    if not train:
        # Not enough rows to both train and hold out — train on the holdout.
        train = pairs[:_MAX_TRAIN_ROWS]

    base_loss = compute_eval_loss(
        model, tokenizer, holdout, device=dev, max_length=max_length
    )

    lora_cfg = LoraConfig(
        r=_DEFAULT_LORA_R,
        lora_alpha=_DEFAULT_LORA_R * 2,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.train()
    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=lr
    )

    step = 0
    while step < n_steps:
        progressed = False
        for prompt, target in train:
            if step >= n_steps:
                break
            input_ids, labels = _tokenize_pair(
                tokenizer, prompt, target, max_length=max_length
            )
            if (labels != -100).sum().item() == 0:
                continue
            input_ids = input_ids.to(dev)
            labels = labels.to(dev)
            optimizer.zero_grad(set_to_none=True)
            out = peft_model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
            step += 1
            progressed = True
        if not progressed:
            # Every train row had an empty target span — avoid an infinite loop.
            break

    probe_loss = compute_eval_loss(
        peft_model, tokenizer, holdout, device=dev, max_length=max_length
    )
    wall = time.monotonic() - started
    return base_loss, probe_loss, wall


def measure_logit_agreement(
    base: str,
    rows: Sequence[Mapping[str, object]],
    *,
    input_extractor: Callable[[Mapping[str, object]], str],
    output_extractor: Callable[[Mapping[str, object]], str],
    device: Optional[str] = None,
    max_pairs: int = _MAX_AGREEMENT_PAIRS,
    max_length: int = 256,
    trust_remote_code: bool = False,
) -> float:
    """Fraction of held-out target tokens the base model already predicts top-1.

    This is the #162 ``base_model_proximity`` signal: how close the base
    model's next-token distribution already is to the dataset's targets,
    normalised to ``[0, 1]`` (1.0 = the model already produces the targets).
    Returns ``float('nan')`` when no target token can be scored.
    """
    _check_positive_int(max_pairs, "max_pairs")
    _check_positive_int(max_length, "max_length")
    import torch

    model, tokenizer, dev = load_model_and_tokenizer(
        base, device=device, trust_remote_code=trust_remote_code
    )
    pairs = _build_pairs(
        rows, input_extractor=input_extractor, output_extractor=output_extractor
    )[:max_pairs]
    matched = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for prompt, target in pairs:
            input_ids, labels = _tokenize_pair(
                tokenizer, prompt, target, max_length=max_length
            )
            target_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if target_positions.numel() == 0:
                continue
            input_ids = input_ids.to(dev)
            logits = model(input_ids=input_ids).logits[0]
            preds = logits.argmax(dim=-1)
            for pos in target_positions.tolist():
                if pos == 0:
                    continue
                pred_tok = int(preds[pos - 1].item())
                true_tok = int(labels[0][pos].item())
                total += 1
                if pred_tok == true_tok:
                    matched += 1
    if total == 0:
        return float("nan")
    return matched / total


# ---------------------------------------------------------------------------
# v0.71.8 — residual-stream activation capture (shared by #215 / #217 / #219)
# ---------------------------------------------------------------------------


def resolve_layer_module(model: object, layer_path: str) -> object:
    """Resolve a dotted ``layer_path`` (e.g. ``model.layers.5``) to a submodule.

    Each segment is either an attribute name (``getattr``) or an integer index
    into a ``ModuleList`` (``module[int(seg)]``). Rejects empty / ``..`` /
    non-``[A-Za-z0-9_]`` segments before walking so a crafted path cannot reach
    a dunder. Raises ``ValueError`` when the path does not resolve.
    """
    if not isinstance(layer_path, str) or not layer_path.strip():
        raise ValueError("layer_path must be a non-empty string")
    segments = layer_path.split(".")
    # Validate every segment ONCE up front (security check) before any walk so
    # the PEFT fallback below cannot bypass the dunder / regex guards.
    for seg in segments:
        if not seg or not _LAYER_SEGMENT_RE.match(seg):
            raise ValueError(
                f"invalid layer path segment {seg!r} in {layer_path!r}"
            )
        # Reject dunder attributes (``__class__`` etc.) — they pass the
        # ``[A-Za-z0-9_]`` regex but must never be reachable via getattr.
        if seg.startswith("__") and seg.endswith("__"):
            raise ValueError(
                f"invalid layer path segment {seg!r} (dunder) in {layer_path!r}"
            )

    def _walk(root: object) -> object:
        current = root
        for seg in segments:
            if seg.isdigit():
                current = current[int(seg)]  # type: ignore[index]
            else:
                current = getattr(current, seg)
        return current

    try:
        return _walk(model)
    except (AttributeError, IndexError, KeyError, TypeError) as exc:
        # PEFT wraps the base model (``PeftModel.base_model.model.<arch>...``),
        # so a natural path like ``model.layers.5`` won't resolve against the
        # wrapper. Retry against the unwrapped base so the user supplies the
        # same path whether or not a LoRA adapter is loaded (the hook still
        # sees the LoRA delta — it is applied inside the layer's submodules).
        get_base = getattr(model, "get_base_model", None)
        if callable(get_base):
            try:
                return _walk(get_base())
            except (AttributeError, IndexError, KeyError, TypeError):
                pass
        raise ValueError(
            f"could not resolve layer path {layer_path!r}: {type(exc).__name__}"
        ) from exc


def extract_layer_activations(
    model: object,
    tokenizer: object,
    prompts: Sequence[str],
    *,
    layer: str,
    device: str,
    pool: str = "mean",
    max_tokens: int = _MAX_CAPTURE_TOKENS,
):
    """Capture residual-stream activations at ``layer`` for ``prompts``.

    Registers a forward hook on the resolved layer module, runs each prompt
    through the model under ``torch.no_grad()``, and returns a 2D float32
    numpy array.

    ``pool='mean'`` returns one mean-pooled vector per prompt (``[N, D]`` — used
    for contrast-probe computation, one example per prompt). ``pool='none'``
    returns per-token rows (``[total_tokens, D]`` capped at ``max_tokens`` — used
    for SAE feature diff). The hook is always removed in a ``finally``.
    """
    if pool not in ("mean", "none"):
        raise ValueError("pool must be 'mean' or 'none'")
    _check_positive_int(max_tokens, "max_tokens")
    if not isinstance(prompts, Sequence) or isinstance(prompts, (str, bytes)):
        raise TypeError("prompts must be a sequence of strings")
    prompt_list = [p for p in prompts if isinstance(p, str) and p.strip()]
    if not prompt_list:
        raise ValueError("prompts must contain at least one non-empty string")
    if len(prompt_list) > _MAX_CAPTURE_PROMPTS:
        prompt_list = prompt_list[:_MAX_CAPTURE_PROMPTS]
    import numpy as np
    import torch

    module = resolve_layer_module(model, layer)
    captured: List[object] = []

    def _hook(_mod, _inp, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden.detach())

    handle = module.register_forward_hook(_hook)
    rows: List[object] = []
    total_tokens = 0
    try:
        model.eval()  # type: ignore[attr-defined]
        with torch.no_grad():
            for prompt in prompt_list:
                text = _apply_prompt_template(tokenizer, prompt)
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=_MAX_PROMPT_TOKENS,
                ).to(device)
                captured.clear()
                model(**inputs)
                if not captured:
                    continue
                # Single forward per prompt under no_grad → the layer fires once;
                # ``captured[-1]`` is that activation. Validated for plain decoder
                # -layer paths (model.layers.N); an exotic module re-invoked within
                # one forward would only contribute its last call.
                hidden = captured[-1]
                # hidden: [batch, seq, D]; batch is 1 here.
                seq = hidden[0].to(torch.float32).cpu().numpy()
                if pool == "mean":
                    rows.append(seq.mean(axis=0))
                    total_tokens += 1
                else:
                    remaining = max_tokens - total_tokens
                    if remaining <= 0:
                        break
                    take = seq[:remaining]
                    rows.append(take)
                    total_tokens += int(take.shape[0])
    finally:
        handle.remove()

    if not rows:
        raise ValueError("no activations captured (empty model output?)")
    if pool == "mean":
        return np.stack(rows).astype(np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)
