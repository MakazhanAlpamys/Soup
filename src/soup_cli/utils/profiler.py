"""Training profile estimator — memory, speed, and recommendations."""

import dataclasses
import math
import re
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Union

if TYPE_CHECKING:
    from soup_cli.utils.hardware_fit import FitUnavailable

# GPU memory lookup table (name → GB VRAM). Keys are normalized by
# ``normalize_gpu_key``. Every laptop part has its own key, including those with
# the same memory as the desktop card: that is what lets the device name torch
# reports ("... Laptop GPU") resolve at all. A part that ships in two sizes gets a
# ``_<N>gb`` key for the size its name carries ("RTX 3050 6GB Laptop GPU"); where
# one name covers both sizes, the plain key is the larger one.
GPU_MEMORY: dict[str, int] = {
    "rtx3050": 8,
    "rtx3050_6gb": 6,
    "rtx3060": 12,
    "rtx3070": 8,
    "rtx3070ti": 8,
    "rtx3080": 10,
    "rtx3080ti": 12,
    "rtx3090": 24,
    "rtx4060": 8,
    "rtx4060ti": 16,
    "rtx4070": 12,
    "rtx4070ti": 12,
    "rtx4080": 16,
    "rtx4090": 24,
    "rtx5060": 8,
    "rtx5060ti": 16,
    "rtx5060ti_8gb": 8,
    "rtx5070": 12,
    "rtx5070ti": 16,
    "rtx5080": 16,
    "rtx5090": 32,
    # NVIDIA's 30-series laptop specification: "RTX 3050: 4 GB or 6 GB" and
    # "RTX 3080: 8 GB or 16 GB". The driver names the 3050 by size (4GB, 6GB)
    # or not at all (the original 4 GB part); both 3080 sizes share one name.
    "rtx3050laptop": 4,
    "rtx3050laptop_4gb": 4,
    "rtx3050laptop_6gb": 6,
    "rtx3050tilaptop": 4,
    "rtx3060laptop": 6,
    "rtx3070laptop": 8,
    "rtx3070tilaptop": 8,
    "rtx3080laptop": 16,
    "rtx3080laptop_8gb": 8,
    "rtx3080tilaptop": 16,
    "rtx4050laptop": 6,
    "rtx4060laptop": 8,
    "rtx4070laptop": 8,
    "rtx4080laptop": 12,
    "rtx4090laptop": 16,
    "rtx5060laptop": 8,
    "rtx5070laptop": 8,
    "rtx5070tilaptop": 12,
    "rtx5080laptop": 16,
    "rtx5090laptop": 24,
    "a10": 24,
    "a30": 24,
    "a40": 48,
    "a100": 80,
    "a100_40gb": 40,
    "h100": 80,
    "h200": 141,
    # DGX B200: 1,440 GB across eight GPUs.
    "b200": 180,
    "l4": 24,
    "l40": 48,
    "l40s": 48,
    "t4": 16,
    "v100": 32,
}

# What ``soup profile`` and the MCP ``profile`` tool fall back to when there is
# no --gpu and no device is detected. Always reported as assumed, not measured.
ASSUMED_GPU_MEMORY_GB = 24.0

#: ``gpu_memory_source`` when the memory is Metal's recommended working set (#2049).
MLX_WORKING_SET_SOURCE = "mlx working set"


def detected_gpu_memory() -> tuple[float, str]:
    """The card's memory in GiB and its source, with no ``--gpu``.

    On Apple Silicon that is Metal's recommended working set (``"mlx working
    set"``), never the total unified memory; elsewhere the CUDA device
    (``"detected"``). Neither: ``ASSUMED_GPU_MEMORY_GB``, ``"assumed"``.
    """
    from soup_cli.utils import mlx

    if mlx.is_apple_silicon():
        working_set = mlx.get_mlx_working_set_bytes()
        if working_set:
            return working_set / 1024**3, MLX_WORKING_SET_SOURCE
        return ASSUMED_GPU_MEMORY_GB, "assumed"
    try:
        from soup_cli.utils.gpu import get_gpu_info

        mem_bytes = get_gpu_info().get("memory_total_bytes", 0)
        if mem_bytes > 0:
            return mem_bytes / (1024**3), "detected"
    except (ImportError, RuntimeError, OSError):
        pass
    return ASSUMED_GPU_MEMORY_GB, "assumed"


_GPU_NAME_NOISE = re.compile(r"\b(nvidia|geforce|tesla|gpu)\b")
# A memory size written as its own word, which the table spells as a suffix.
_GPU_NAME_SIZE = re.compile(r"[\s-](\d+)\s*gb\b")


def normalize_gpu_key(name: str) -> str:
    """Map a ``--gpu`` value or a device name torch reports to a ``GPU_MEMORY`` key.

    ``"NVIDIA GeForce RTX 5070 Laptop GPU"`` and ``"rtx 5070 laptop"`` both
    become ``"rtx5070laptop"``; ``"RTX-4090"`` becomes ``"rtx4090"``. A size
    word moves to the end as a suffix: ``"NVIDIA GeForce RTX 3050 6GB Laptop
    GPU"`` becomes ``"rtx3050laptop_6gb"``. A size whose suffixed key does not
    exist but which the plain key holds resolves to the plain key: ``"rtx 3080
    laptop 16gb"`` becomes ``"rtx3080laptop"``.
    """
    key = _GPU_NAME_NOISE.sub("", name.lower())
    size = _GPU_NAME_SIZE.search(key)
    if size is None:
        return re.sub(r"[\s-]+", "", key)
    key = re.sub(r"[\s-]+", "", key[: size.start()] + " " + key[size.end() :])
    suffixed = f"{key}_{size.group(1)}gb"
    if suffixed not in GPU_MEMORY and str(GPU_MEMORY.get(key)) == size.group(1):
        return key
    return suffixed


def get_model_arch(model_name: str, model_params_b: float) -> dict:
    """Get model architecture estimates (hidden_size, num_layers).

    A guess from the parameter count alone (``model_shape.guessed_arch``). The VRAM
    estimate reads the real dims from ``config.json`` whenever it can (#2049).
    """
    from soup_cli.utils.model_shape import guessed_arch

    hidden, layers, intermediate = guessed_arch(model_params_b)
    return {
        "hidden_size": hidden,
        "num_layers": layers,
        "intermediate_size": intermediate,
    }


def checkpointing_tier(value: Any) -> str:
    """``off``, ``full``, ``medium`` or ``selective``: the checkpointing a run is priced with.

    ``value`` is what the estimate was given: ``False`` or a tier name, or a config's
    ``True`` / ``"auto"``, which price as ``full``.
    """
    if not value:
        return "off"
    return value if value in ("medium", "selective") else "full"


def profile_result(
    breakdown: Any, *, model_name: str, gradient_checkpointing: Any = None
) -> dict:
    """``soup profile``'s result fields from a ``hardware_fit.VRAMBreakdown``.

    The old field names keep their meaning (``lora_memory_gb`` is the adapters);
    ``logits_memory_gb``, ``dequant_memory_gb``, the exact byte counts, the shape's
    source and the notes are new with #2049. GB is 1e9 bytes.
    ``gradient_checkpointing`` is reported as the config wrote it (``True``, ``False``,
    ``"auto"`` or a tier), as it always was; ``gradient_checkpointing_tier`` is the
    tier the estimate priced, which ``auto`` and the task resolve.
    """
    est = breakdown.estimate
    shape, run = est.shape, est.run
    return {
        "model_name": model_name,
        "model_params_b": round(shape.params / 1e9, 3),
        "trainable_params": int(
            shape.params if run.peft == "full" else est.terms["adapters"] // 4
        ),
        "quantization": run.quant,
        "peft": run.peft,
        "lora_targets": list(est.lora_targets),
        "model_memory_gb": round(breakdown.weights_gb, 2),
        "lora_memory_gb": round(breakdown.adapters_gb, 2),
        "gradients_memory_gb": round(breakdown.gradients_gb, 2),
        "optimizer_memory_gb": round(breakdown.optimizer_gb, 2),
        "activation_memory_gb": round(breakdown.activations_gb, 2),
        "logits_memory_gb": round(breakdown.logits_gb, 2),
        "dequant_memory_gb": round(breakdown.dequant_gb, 2),
        "optimizer_step_memory_gb": round(breakdown.optimizer_step_gb, 2),
        # #2909: task distill's frozen teacher, its divergence over the vocabulary, and
        # with distill_mode: sequence how far the teacher's generation exceeds training.
        "teacher_memory_gb": round(breakdown.teacher_gb, 2),
        "distill_memory_gb": round(breakdown.distill_gb, 2),
        "teacher_generation_memory_gb": round(breakdown.teacher_generation_gb, 2),
        "overhead_gb": round(breakdown.overhead_gb, 2),
        "total_memory_gb": round(breakdown.total_gb, 2),
        "total_memory_bytes": est.total_bytes,
        "memory_terms_bytes": dict(est.terms),
        # A full fine-tune has no adapters, so no rank (Spectrum and LISA too, whose
        # lora.r keeps its default).
        "lora_r": None if run.peft == "full" else run.lora_r,
        "batch_size": run.batch_size,
        "seq_len": run.seq_len,
        "gradient_checkpointing": (
            run.gradient_checkpointing
            if gradient_checkpointing is None
            else gradient_checkpointing
        ),
        "gradient_checkpointing_tier": checkpointing_tier(run.gradient_checkpointing),
        "hidden_size": shape.hidden_size,
        "num_layers": shape.num_layers,
        "vocab_size": shape.vocab_size,
        "shape_source": shape.shape_source,
        "params_source": shape.params_source,
        "notes": list(est.notes),
    }


#: What ``profile_from_config(shape=)`` must carry, all together: without the vocabulary the
#: logits term, most of a small model's estimate, is not priced.
SHAPE_REQUIRED_FIELDS = ("hidden_size", "num_layers", "intermediate_size", "vocab_size", "params")
#: Optional, for a model whose attention is not hidden_size wide; the defaults are the ones a
#: shape guessed from the name uses.
SHAPE_OPTIONAL_FIELDS = ("num_attention_heads", "num_key_value_heads", "head_dim")
_SHAPE_FIELDS = (*SHAPE_REQUIRED_FIELDS, *SHAPE_OPTIONAL_FIELDS, "tie_word_embeddings")


def _checked_seq_len(value: Any) -> Optional[int]:
    """``None`` or an int of at least 1; anything else is refused by name, before any read."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"seq_len must be an int or None, got {type(value).__name__}")
    if value < 1:
        # An empty dataset has no longest row; do not turn it into a 64-token estimate.
        raise ValueError(f"seq_len must be at least 1, got {value}")
    return value


def _not_padded_to_the_longest_row(cfg: Any) -> Optional[str]:
    """Why ``cfg``'s micro-batches are not padded to their longest row; ``None`` when they are.

    Only text SFT on the transformers backend, with no packing, no multipack, no streamed data
    and no layer streaming, builds a micro-batch of ``batch_size`` rows padded to the longest of
    them (measured: the same peak at ``max_length`` 1024 and 256 on rows of 79 tokens). A caller's
    ``seq_len`` would under-price every other run, so it is not honoured for them:

    - packing and multipack fill every sequence to ``data.max_length``;
    - grpo, ppo and online_dpo generate tokens on top of the prompt; dpo, kto, orpo, simpo, ipo,
      bco, preference and reward_model run a chosen and a rejected sequence per row; distill and
      unlearn run a second model or a second set; pretrain's rows are long documents cut to
      ``max_length``; tts, asr, vision and audio carry media tokens; the classifier family,
      embedding, prm and moe_lora_routing have their own collators. None of them is measured, so
      sft is the only task listed here;
    - the longest row of a streamed dataset is not known before the run, and layer streaming has
      its own planner (the pre-flight skips it).
    """
    training, data = getattr(cfg, "training", None), getattr(cfg, "data", None)
    filled = "every sequence of a micro-batch is filled to data.max_length"
    if getattr(training, "packing", False):
        return f"training.packing is on: {filled}"
    if getattr(training, "multipack", False):
        return f"training.multipack is on: {filled}"
    task = getattr(cfg, "task", "sft")
    if task != "sft":
        return (
            f"task {task!r} is not priced at a shorter length: only task sft pads each "
            "micro-batch to its longest row (the others pack, pair rows, generate on top of the "
            "prompt, or are not measured)"
        )
    modality = getattr(cfg, "modality", "text")
    if modality != "text":
        return f"modality {modality!r} carries image or audio tokens on top of the text"
    backend = getattr(cfg, "backend", "transformers")
    if backend != "transformers":
        return f"backend {backend!r} is not measured (only transformers is)"
    if getattr(data, "streaming", False):
        return (
            "data.streaming is on: the longest row of a streamed dataset is not known before "
            "the run"
        )
    if getattr(training, "stream_layers", False):
        return (
            "training.stream_layers is on: layer streaming has its own planner and the "
            "resident estimate does not apply"
        )
    return None


def _shape_from_caller(shape: Any) -> Any:
    """A ``ModelShape`` from ``profile_from_config(shape=)``; every refusal names its field."""
    from soup_cli.utils.hardware_fit import _MAX_PARAMS_B
    from soup_cli.utils.model_shape import _MAX_DIM, _MAX_LAYERS, ModelShape

    if not isinstance(shape, Mapping):
        raise TypeError(f"shape must be a mapping, got {type(shape).__name__}")
    for key in shape:
        if key not in _SHAPE_FIELDS:
            raise ValueError(f"unknown shape field {key!r}; known: {', '.join(_SHAPE_FIELDS)}")
    missing = [name for name in SHAPE_REQUIRED_FIELDS if name not in shape]
    if missing:
        raise ValueError(
            f"shape is missing {', '.join(missing)}. A shape needs "
            f"{', '.join(SHAPE_REQUIRED_FIELDS[:-1])} and params together: a parameter count "
            "alone leaves the logits term, most of a small model's estimate, unpriced."
        )
    limits = {"params": int(_MAX_PARAMS_B * 1e9), "num_layers": _MAX_LAYERS}
    values: dict[str, int] = {}
    for name in (*SHAPE_REQUIRED_FIELDS, *SHAPE_OPTIONAL_FIELDS):
        if name not in shape:
            continue
        value = shape[name]
        if isinstance(value, bool):
            raise TypeError(f"shape[{name!r}] must be an int, not bool")
        if not isinstance(value, int):
            raise TypeError(f"shape[{name!r}] must be an int, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"shape[{name!r}] must be positive, got {value}")
        if value > limits.get(name, _MAX_DIM):
            raise ValueError(f"shape[{name!r}] is larger than {limits.get(name, _MAX_DIM):,}")
        values[name] = value
    floor = values["vocab_size"] * values["hidden_size"]
    if values["params"] < floor:
        raise ValueError(
            f"shape['params'] ({values['params']:,}) is below vocab_size x hidden_size "
            f"({floor:,}), the embedding table alone: params is a raw parameter count"
        )
    tied = shape.get("tie_word_embeddings", False)
    if not isinstance(tied, bool):
        raise TypeError(f"shape['tie_word_embeddings'] must be a bool, got {type(tied).__name__}")
    hidden = values["hidden_size"]
    heads = values.get("num_attention_heads") or max(1, hidden // 128)
    return ModelShape(
        params=values["params"],
        hidden_size=hidden,
        intermediate_size=values["intermediate_size"],
        num_layers=values["num_layers"],
        num_attention_heads=heads,
        num_kv_heads=values.get("num_key_value_heads") or heads,
        head_dim=values.get("head_dim") or max(1, hidden // heads),
        vocab_size=values["vocab_size"],
        tie_word_embeddings=tied,
        model_type="",
        params_source="caller",
        shape_source="caller",
    )


def profile_from_config(
    cfg: Any,
    *,
    allow_network: bool = True,
    gpu_memory_gb: Optional[float] = None,
    on_hub_request: Optional[Callable[[], None]] = None,
    seq_len: Optional[int] = None,
    shape: Optional[Mapping[str, Any]] = None,
) -> Union[dict, "FitUnavailable"]:
    """The memory profile of a ``SoupConfig``, from the model's own shape.

    The entry point for a program that wants the estimate without running ``soup train``.
    It builds the same input as the train pre-flight
    (``hardware_fit.hardware_fit_input_from_config``, #2049), with one exception: the
    pre-flight passes ``kbit_fp32`` on a CUDA card without bf16 hardware
    (``commands/train.py``), which this function cannot take. When the run cannot be priced,
    the :class:`~soup_cli.utils.hardware_fit.FitUnavailable` naming the field that stops
    it (#2306). ``gpu_memory_gb`` (GiB) is the card ``gradient_checkpointing: auto``
    resolves on. ``on_hub_request`` is called just before the first request to the
    Hub, if one is made.

    The estimate prices rows that FILL the length it is given: ``data.max_length`` tokens
    in every row of a micro-batch, as its calibration runs did. For data whose rows are
    shorter, ``seq_len`` is the length to price instead: an int of at least 1 (a bool or a
    non-int raises ``TypeError``, a value below 1 ``ValueError``), clamped to
    ``[64, data.max_length]``. It is honoured only for a run that pads each micro-batch to its
    longest row (text SFT without packing; ``_not_padded_to_the_longest_row`` lists what is
    not), where the longest row of the data is the figure to pass. Any other run is priced at
    ``data.max_length``. The result then has ``seq_len_requested``, a note, and
    ``seq_len_source``: ``"caller"`` when the caller's length was priced, ``"max_length"``
    when ``data.max_length`` was (the run does not pad to its longest row, or the caller's
    value was above it); ``seq_len`` is the length priced. With ``seq_len=None`` the result is
    what it always was.

    ``shape`` gives the model's shape when it cannot be read (not cached, no network, a
    gated repo): a mapping with ``hidden_size``, ``num_layers``, ``intermediate_size``,
    ``vocab_size`` and ``params``, all required together (a parameter count alone is
    refused: it leaves the logits unpriced), and optionally ``num_attention_heads``,
    ``num_key_value_heads``, ``head_dim`` and ``tie_word_embeddings``. ``params`` is a raw
    parameter count of at least ``vocab_size * hidden_size``. Pass the sum of the tensors the
    checkpoint stores (its safetensors header sum, which is what this function reads for a
    model it can see); whether a tied head is stored once or twice varies by checkpoint. With
    only a loaded count of a tied model, add ``vocab_size * hidden_size``: never too small,
    at most one head matrix too large. A wrong ``tie_word_embeddings: true`` under-prices
    k-bit loading, so omit it when unsure.
    The shape is used only when the model's own ``config.json`` cannot be read, which
    otherwise wins (a note says so); the result then has ``shape_source: "caller"``. Weights
    on disk without a ``config.json`` keep their measured parameter count over the caller's.
    """
    from soup_cli.utils.hardware_fit import (
        _MIN_SEQ_LEN,
        FitUnavailable,
        estimate_peak_vram_gb,
        hardware_fit_input_or_reason,
    )

    caller_seq = _checked_seq_len(seq_len)
    caller_shape = None if shape is None else _shape_from_caller(shape)
    inp = hardware_fit_input_or_reason(
        cfg,
        allow_network=allow_network,
        profile_defaults=True,
        gpu_memory_gb=gpu_memory_gb,
        on_hub_request=on_hub_request,
    )
    if isinstance(inp, FitUnavailable):
        return inp
    notes: list[str] = []
    if caller_shape is not None:
        if not inp.shape.guessed:
            notes.append(
                "The supplied shape was not used: the model's own config.json was readable."
            )
        else:
            note = (
                "The model's shape was supplied by the caller (shape=): "
                f"hidden_size {caller_shape.hidden_size}, {caller_shape.num_layers} layers, "
                f"intermediate_size {caller_shape.intermediate_size}, vocab_size "
                f"{caller_shape.vocab_size}, {caller_shape.params:,} parameters."
            )
            if inp.shape.params_source == "safetensors":
                note += (
                    f" The weights are on disk without a config.json: their "
                    f"{inp.shape.params:,} parameters, read from the safetensors headers, are "
                    "used instead of the supplied count."
                )
                caller_shape = dataclasses.replace(
                    caller_shape, params=inp.shape.params, params_source="safetensors"
                )
            else:
                note += " The estimate takes it as given."
            notes.append(note)
            inp = dataclasses.replace(inp, shape=caller_shape)
    seq_source = None
    if caller_seq is not None:
        max_length = inp.seq_len
        not_padded = _not_padded_to_the_longest_row(cfg)
        lead = f"seq_len {caller_seq} was supplied by the caller (data.max_length is {max_length})"
        if not_padded is not None:
            priced, seq_source = max_length, "max_length"
            notes.append(f"{lead} but {not_padded}; priced at data.max_length ({max_length}).")
        else:
            priced = max(_MIN_SEQ_LEN, min(caller_seq, max_length))
            seq_source = "max_length" if caller_seq > max_length else "caller"
            note = (
                f"{lead}; the estimate prices rows that fill {priced} tokens, so a longer row "
                "in the data is not priced."
            )
            if caller_seq > max_length:
                note += f" It was clamped to data.max_length ({max_length})."
            elif caller_seq < priced:
                note += f" It was clamped to {priced}, the smallest length the estimate prices."
            notes.append(note)
        inp = dataclasses.replace(inp, seq_len=priced)
    result = profile_result(
        estimate_peak_vram_gb(inp),
        model_name=cfg.base,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
    )
    result["quantization"] = cfg.training.quantization
    if seq_source is not None:
        result["seq_len_source"] = seq_source
        result["seq_len_requested"] = caller_seq
    result["notes"].extend(notes)
    return result


#: ``soup profile --model``'s methods: (quantization as soup.yaml names it, peft mode).
MODEL_QUERY_METHODS: dict[str, tuple[str, str]] = {
    "lora": ("none", "lora"),
    "qlora": ("4bit", "qlora"),
    "full": ("none", "full"),
}

#: ``--gradient-checkpointing`` values, as soup.yaml's ``gradient_checkpointing``.
MODEL_QUERY_CHECKPOINTING: dict[str, Any] = {
    "off": False,
    "full": True,
    "medium": "medium",
    "selective": "selective",
    "auto": "auto",
}


class ModelShapeUnavailableError(ValueError):
    """``--model``'s config.json could not be read, so nothing is estimated."""


def _unreadable_reason(model: str, *, allow_network: bool) -> str:
    import os

    from soup_cli.utils.model_shape import _REPO_ID, hub_offline

    if os.path.isdir(model):
        return "there is no readable config.json in that directory"
    if not _REPO_ID.fullmatch(model) or ".." in model:
        return "it is not a local model directory or a Hugging Face model id"
    if hub_offline():
        return (
            "its config.json is not in the Hugging Face cache, and HF_HUB_OFFLINE "
            "or TRANSFORMERS_OFFLINE is set"
        )
    if not allow_network:
        return "its config.json is not in the Hugging Face cache"
    return (
        "its config.json is not in the Hugging Face cache and could not be fetched "
        "from the Hub (no network, a gated or private repo, or no such model)"
    )


def profile_from_model(
    model: str,
    *,
    method: str = "qlora",
    seq_len: int = 2048,
    batch_size: int = 4,
    lora_r: int = 64,
    optimizer: str = "adamw_torch",
    gradient_checkpointing: str = "off",
    allow_network: bool = True,
    gpu_memory_gb: Optional[float] = None,
    on_hub_request: Optional[Callable[[], None]] = None,
) -> dict:
    """The memory profile of a model id or directory, with no soup.yaml (#2049).

    The same estimate as :func:`profile_from_config` for the same run. The shape must
    come from the model's own config.json (a local directory, the Hugging Face cache,
    then the Hub unless offline); a shape guessed from the name is not an answer to
    "will it fit", so that raises :class:`ModelShapeUnavailableError`. Out-of-range
    numbers raise ``ValueError``. ``on_hub_request`` is called just before the first
    request to the Hub, if one is made.
    """
    from soup_cli.utils.hardware_fit import (
        HardwareFitInput,
        checkpointing_for_task,
        estimate_peak_vram_gb,
        validate_batch_size,
        validate_seq_len,
    )
    from soup_cli.utils.model_shape import read_model_shape

    quant, peft = MODEL_QUERY_METHODS[method]
    shape = read_model_shape(
        model, allow_network=allow_network, on_hub_request=on_hub_request
    )
    if shape.guessed:
        raise ModelShapeUnavailableError(_unreadable_reason(model, allow_network=allow_network))
    inp = HardwareFitInput(
        params_b=shape.params / 1e9,
        seq_len=validate_seq_len(seq_len),
        batch_size=validate_batch_size(batch_size),
        optimizer=optimizer,
        quant=quant,
        peft=peft,
        gradient_checkpointing=checkpointing_for_task(
            MODEL_QUERY_CHECKPOINTING[gradient_checkpointing],
            task="sft",
            gpu_memory_gb=gpu_memory_gb,
        ),
        shape=shape,
        lora_r=lora_r,
    )
    result = profile_result(estimate_peak_vram_gb(inp), model_name=model)
    result["quantization"] = quant
    return result


def finish_profile(
    result: dict, *, seq_len: int, gpu_memory_gb: float, gpu_memory_source: str
) -> dict:
    """Add the speed, the recommendations and the fit to an estimate's ``result``.

    Shared by ``soup profile`` (``--config`` and ``--model``) and the MCP tool.
    ``gpu_memory_gb`` is the card's GiB.
    """
    from soup_cli.utils.vram_estimate import fit_verdict

    tokens_per_sec = estimate_speed(
        result["model_params_b"], result["quantization"], result["batch_size"]
    )
    result["tokens_per_sec"] = round(tokens_per_sec, 1)
    result["samples_per_sec"] = round(tokens_per_sec / max(seq_len, 1), 2)
    result["recommended_batch_size"] = recommend_batch_size(
        result["total_memory_gb"], gpu_memory_gb
    )
    result["compatible_gpus"] = recommend_gpu(result["total_memory_gb"])
    result["gpu_memory_gb"] = gpu_memory_gb
    # "flag" (--gpu), "detected" (a CUDA device), "mlx working set" (Apple Silicon)
    # or "assumed" (none of them: a default, not a measurement).
    result["gpu_memory_source"] = gpu_memory_source
    # "fits", "tight" (fits, but not with the train pre-flight's margin) or
    # "does not fit". A card's GB are binary: an "rtx3060laptop" 6 is 6144 MiB.
    result["fit"] = fit_verdict(result["total_memory_bytes"], gpu_memory_gb * 1024**3)
    if gpu_memory_source == MLX_WORKING_SET_SOURCE:
        result["notes"].append(
            f"Apple Silicon: the fit is against Metal's recommended working-set size "
            f"({gpu_memory_gb:.1f} GB), not the whole unified memory. The estimate's "
            "terms were measured with PyTorch on CUDA and are not measured on Apple "
            "hardware."
        )
    return result


def estimate_speed(
    model_params_b: float, quantization: str, batch_size: int
) -> float:
    """Estimate training tokens/sec (rough lookup-based).

    Based on typical A100 throughput for different model sizes.
    """
    # Base tokens/sec on A100 for different sizes (4bit, batch=4)
    base_speed: dict[float, float] = {
        1.0: 5000,
        3.0: 2500,
        7.0: 1200,
        8.0: 1100,
        13.0: 600,
        14.0: 550,
        32.0: 250,
        34.0: 230,
        70.0: 100,
    }

    # Find closest size
    closest_key = min(
        base_speed.keys(),
        key=lambda size: abs(size - model_params_b),
    )
    speed = base_speed[closest_key]

    # Adjust for quantization
    quant_factor = {"4bit": 1.0, "8bit": 0.8, "none": 0.5}
    speed *= quant_factor.get(quantization, 0.5)

    # Adjust for batch size (relative to base batch=4)
    speed *= min(batch_size / 4, 2.0)  # diminishing returns above 8

    return speed


def estimate_training_time(
    dataset_size: int, epochs: int, samples_per_sec: float
) -> float:
    """Estimate total training time in minutes.

    Returns float('inf') if samples_per_sec is 0.
    """
    if samples_per_sec <= 0:
        return float("inf")

    total_samples = dataset_size * epochs
    total_seconds = total_samples / samples_per_sec
    return total_seconds / 60


def recommend_batch_size(total_memory_gb: float, gpu_memory_gb: float) -> int:
    """Recommend batch size based on available GPU memory.

    Leaves headroom for memory spikes during training. ``total_memory_gb`` is the
    estimate's GB (1e9 bytes); ``gpu_memory_gb`` is a card's GB as the table and the
    driver give it (GiB), so it is converted before the two are compared.
    """
    available = gpu_memory_gb * 1024**3 / 1e9 - total_memory_gb
    if available <= 0:
        return 1

    # Each additional batch sample needs roughly (activation memory / current batch)
    # Simplified: available memory / ~1.5 GB per additional sample for 7B-class models
    extra_samples = int(available / 1.5)
    batch_size = max(1, 1 + extra_samples)

    # Clamp to power of 2
    if batch_size > 1:
        batch_size = 2 ** int(math.log2(batch_size))

    return min(batch_size, 64)


def recommend_gpu(total_memory_gb: float) -> list[str]:
    """Recommend GPUs that can fit the estimated memory.

    Returns list of GPU names that have enough VRAM.
    """
    from soup_cli.utils.vram_estimate import fit_verdict

    compatible = []
    for name, vram in sorted(GPU_MEMORY.items(), key=lambda item: item[1]):
        # In bytes, as ``soup profile``'s fit is: the table's GB are GiB, the
        # estimate's are 1e9 bytes. Only a card that fits with the 10% margin.
        if fit_verdict(total_memory_gb * 1e9, vram * 1024**3) == "fits":
            compatible.append(f"{name.upper()} ({vram} GB)")

    if not compatible:
        compatible.append(
            f"No single GPU fits {total_memory_gb:.1f} GB -- use multi-GPU (DeepSpeed/FSDP)"
        )

    return compatible
