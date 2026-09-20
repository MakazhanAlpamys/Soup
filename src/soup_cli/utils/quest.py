"""Experimental mixed-precision QuEST route for issue #674.

This is the exact *shape* of the route independently confirmed in the retained
research record: all 168 transformer linear weights use group-128 fake W4,
161 activations use group-128 fake A4, and the seven linears in block 23 keep
their rotated activations at A16.  It is not pure W4A4, packed INT4, or an
upstream-QuEST parity claim.  The quality evidence is evaluation-only; training
through this route is an engineering integration whose quality is not yet
validated.

Torch and Transformers stay lazily imported so ordinary CLI startup remains
lightweight.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
METADATA_NAME = "quest_mixed_precision.json"
RECIPE = "w4a4-group128-block23-a16"
GROUP_SIZE = 128
WEIGHT_BITS = 4
WEIGHT_SCALE = 2.513930578568423
CALIBRATION_SCALES = (2.0, WEIGHT_SCALE, 3.0, 4.0, 6.0)
CALIBRATION_EXAMPLES = 32
CALIBRATION_POSITION_LIMIT = 16
BLOCKS = tuple(range(24))
SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
EXPECTED_MODULES = tuple(
    f"model.layers.{block}.{suffix}" for block in BLOCKS for suffix in SUFFIXES
)
A16_MODULES = tuple(f"model.layers.23.{suffix}" for suffix in SUFFIXES)
QUALITY_EVIDENCE = {
    "scope": "evaluation_only",
    "model": "ahxt/LiteLlama-460M-1T",
    "examples": 704,
    "targets": 25017,
    "gap_nat": 0.0863441881,
    "ci95": [0.0802412531, 0.0931898109],
    "result_sha256": "94fee10542da29281f7753cbf221a3421ad5acf67f2b290e52d65acece359cdf",
}

_METADATA_KEYS = frozenset(
    {
        "format_version",
        "backend",
        "recipe",
        "base_model",
        "group_size",
        "weight_bits",
        "weight_scale",
        "activation_bits",
        "transform",
        "quantization_grid",
        "surrogate",
        "calibration",
        "weight_routes",
        "activation_routes",
        "a16_modules",
        "a4_modules",
        "activation_scales",
        "pure_w4a4",
        "packed_int4",
        "training_quality_validated",
        "quality_evidence",
    }
)


def validate_scale(value: Any) -> float:
    """Return a finite clipping scale, rejecting bool-as-number."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 8:
        raise ValueError("QuEST activation scale must be finite in [1, 8]")
    return float(value)


def validate_group(width: Any, group: Any = GROUP_SIZE) -> None:
    """Validate the measured full-Hadamard then group-128 geometry."""
    if (
        type(width) is not int
        or width < GROUP_SIZE
        or width & (width - 1)
        or type(group) is not int
        or group not in (GROUP_SIZE, width)
        or width % group
    ):
        raise ValueError("QuEST requires a power-of-two input width >= 128 divisible by group 128")


@lru_cache(maxsize=16)
def full_hadamard_matrix(width: int, dtype: Any, device: Any) -> Any:
    """Return the dense normalized matrix used by the measured operator.

    A fast Walsh-Hadamard kernel would change accumulation order and has not
    passed the retained quality gate, so the first integration deliberately
    keeps the measured dense reference arithmetic.
    """
    import torch

    validate_group(width)
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.shape[0] < width:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    return (matrix / math.sqrt(width)).to(device=device, dtype=dtype)


def rotate(value: Any) -> Any:
    """Apply the full-width normalized Hadamard used in the research route."""
    validate_group(int(value.shape[-1]))
    return value @ full_hadamard_matrix(value.shape[-1], value.dtype, value.device)


def quantize_rotated(value: Any, *, scale: float, group: int = GROUP_SIZE) -> Any:
    """Apply the measured 4-bit grid and trust-gradient surrogate."""
    if getattr(value, "ndim", 0) < 2:
        raise ValueError("QuEST quantization requires at least two dimensions")
    validate_group(int(value.shape[-1]), group)
    scale = validate_scale(scale)
    shape = (*value.shape[:-1], value.shape[-1] // group, group)
    grouped = value.reshape(shape)
    with _no_grad():
        rms = grouped.square().mean(-1, keepdim=True).sqrt()
        bound = rms * scale + 1e-8
        spacing = 2 * bound / 15
        quantized = (grouped.clamp(-bound, bound) / spacing + 0.5).round() * spacing - spacing / 2
        trusted = ((quantized - grouped).abs() <= rms * (scale / 15)).float()
    surrogate = grouped * trusted
    return (surrogate + (quantized - surrogate).detach()).reshape_as(value)


def _no_grad() -> Any:
    import torch

    return torch.no_grad()


def _target_linears(model: Any) -> dict[str, Any]:
    import torch

    return {
        name: module
        for name, module in model.named_modules()
        if type(module) is torch.nn.Linear and name != "lm_head"
    }


def _validate_raw_topology(model: Any) -> dict[str, Any]:
    import torch

    targets = _target_linears(model)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != 24:
        raise ValueError("QuEST first slice requires the measured 24 blocks")
    if set(targets) != set(EXPECTED_MODULES) or len(targets) != 168:
        raise ValueError(
            "QuEST calibration must cover exactly all 168 measured Llama transformer linears"
        )
    for module in targets.values():
        validate_group(int(module.in_features))
        if module.weight.dtype != torch.float32:
            raise ValueError(
                "QuEST first slice requires FP32 master weights before fake quantization"
            )
    return targets


def _validate_topology(model: Any, scales: dict[str, float]) -> dict[str, Any]:
    targets = _validate_raw_topology(model)
    if set(scales) != set(EXPECTED_MODULES):
        raise ValueError("QuEST activation calibration must cover all 168 linears")
    for name, module in targets.items():
        validate_scale(scales[name])
    return targets


def predictor_positions(labels: list[int], *, limit: int = 16) -> list[int]:
    """Choose deterministic response-predictor positions for calibration."""
    if type(limit) is not int or limit < 1:
        raise ValueError("QuEST calibration position limit must be positive")
    positions = [index - 1 for index in range(1, len(labels)) if labels[index] != -100]
    if not positions:
        raise ValueError("QuEST calibration row has no supervised response target")
    if len(positions) <= limit:
        return positions
    if limit == 1:
        return [positions[0]]
    return [positions[index * (len(positions) - 1) // (limit - 1)] for index in range(limit)]


def calibration_rows_sha256(dataset: Any, *, examples: int = CALIBRATION_EXAMPLES) -> str:
    """Bind the route to the exact tokenized TRAIN rows used for calibration."""
    if type(examples) is not int or examples != CALIBRATION_EXAMPLES:
        raise ValueError(f"QuEST first slice fixes calibration_examples at {CALIBRATION_EXAMPLES}")
    if len(dataset) < examples:
        raise ValueError(
            f"QuEST calibration requires at least {examples} training rows; got {len(dataset)}"
        )
    digest = hashlib.sha256()
    for row_index in range(examples):
        row = dataset[row_index]
        if "input_ids" not in row or "labels" not in row:
            raise ValueError(
                "QuEST calibration requires tokenized input_ids and labels; "
                f"training row {row_index + 1} is missing them"
            )
        input_ids = [int(value) for value in row["input_ids"]]
        labels = [int(value) for value in row["labels"]]
        attention = [int(value) for value in row.get("attention_mask", [1] * len(input_ids))]
        if not input_ids or len(labels) != len(input_ids) or len(attention) != len(input_ids):
            raise ValueError(
                "QuEST calibration input_ids, labels and attention_mask must "
                f"have the same non-zero length at training row {row_index + 1}"
            )
        payload = json.dumps(
            {
                "attention_mask": attention,
                "input_ids": input_ids,
                "labels": labels,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def calibrate_activation_scales(
    model: Any,
    dataset: Any,
    *,
    examples: int = CALIBRATION_EXAMPLES,
    position_limit: int = CALIBRATION_POSITION_LIMIT,
) -> dict[str, float]:
    """Select per-linear activation clips on the first training examples.

    Selection mirrors the retained research procedure: full-width Hadamard and
    full-width W4 weight quantization are used for this local-output proxy, then
    the winning fixed scales are installed into the group-128 training route.
    The rows are training data; no validation/test panel is touched.
    """
    import torch
    from torch.nn import functional

    if type(examples) is not int or examples != 32:
        raise ValueError("QuEST first slice fixes calibration_examples at 32")
    if type(position_limit) is not int or position_limit != CALIBRATION_POSITION_LIMIT:
        raise ValueError(
            "QuEST first slice fixes calibration_position_limit at "
            f"{CALIBRATION_POSITION_LIMIT}"
        )
    if len(dataset) < examples:
        raise ValueError(
            f"QuEST calibration requires at least {examples} training rows; got {len(dataset)}"
        )
    targets = _validate_raw_topology(model)
    measurements = {
        name: {str(scale): 0.0 for scale in CALIBRATION_SCALES} for name in EXPECTED_MODULES
    }
    energies = dict.fromkeys(EXPECTED_MODULES, 0.0)
    quantized_weights: dict[str, Any] = {}
    context: dict[str, list[int]] = {"positions": []}
    handles = []

    def make_hook(name: str):
        def measure(module: Any, arguments: tuple[Any, ...]) -> None:
            value = arguments[0][:, context["positions"], :]
            original = functional.linear(value, module.weight, module.bias)
            energies[name] += float(original.float().square().sum())
            rotated = rotate(value)
            if name not in quantized_weights:
                rotated_weight = rotate(module.weight)
                quantized_weights[name] = quantize_rotated(
                    rotated_weight,
                    scale=WEIGHT_SCALE,
                    group=module.in_features,
                )
            for scale in CALIBRATION_SCALES:
                operand = quantize_rotated(
                    rotated,
                    scale=scale,
                    group=module.in_features,
                )
                candidate = functional.linear(operand, quantized_weights[name], module.bias)
                measurements[name][str(scale)] += float(
                    (candidate.float() - original.float()).square().sum()
                )

        return measure

    for name in EXPECTED_MODULES:
        handles.append(targets[name].register_forward_pre_hook(make_hook(name)))

    was_training = bool(model.training)
    device = next(model.parameters()).device
    try:
        model.eval()
        for row_index in range(examples):
            row = dataset[row_index]
            if "input_ids" not in row or "labels" not in row:
                raise ValueError(
                    "QuEST calibration requires tokenized input_ids and labels; "
                    f"training row {row_index + 1} is missing them"
                )
            labels = [int(value) for value in row["labels"]]
            context["positions"] = predictor_positions(labels, limit=position_limit)
            input_ids = torch.as_tensor(row["input_ids"], dtype=torch.long, device=device)
            attention = torch.as_tensor(
                row.get("attention_mask", [1] * len(row["input_ids"])),
                dtype=torch.long,
                device=device,
            )
            arguments = {
                "input_ids": input_ids.unsqueeze(0),
                "attention_mask": attention.unsqueeze(0),
            }
            device_type = device.type
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type,
                    dtype=torch.bfloat16,
                    enabled=device_type == "cuda",
                ),
            ):
                model(**arguments)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    for name, errors in measurements.items():
        if (
            not math.isfinite(energies[name])
            or energies[name] <= 0
            or not all(math.isfinite(value) and value >= 0 for value in errors.values())
        ):
            raise ValueError("QuEST calibration produced a zero-energy or non-finite local error")
    return {
        name: min(
            CALIBRATION_SCALES,
            key=lambda scale: measurements[name][str(scale)],
        )
        for name in EXPECTED_MODULES
    }


def _make_linear(original: Any, *, activation_scale: float, activation_bits: int) -> Any:
    import torch
    from torch import nn
    from torch.nn import functional

    if activation_bits not in (4, 16):
        raise ValueError("QuEST activation precision must be A4 or A16")
    validate_group(int(original.in_features))
    activation_scale = validate_scale(activation_scale)

    class QuestMixedLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = original.weight
            self.bias = original.bias
            self.in_features = original.in_features
            self.out_features = original.out_features
            self.register_buffer(
                "quest_activation_clip_scale",
                torch.tensor(
                    activation_scale,
                    dtype=torch.float64,
                    device=self.weight.device,
                ),
                persistent=False,
            )
            for name, value in (
                ("quest_group_size", GROUP_SIZE),
                ("quest_weight_bits", WEIGHT_BITS),
                ("quest_activation_bits", activation_bits),
            ):
                self.register_buffer(
                    name,
                    torch.tensor(value, dtype=torch.int64, device=self.weight.device),
                    persistent=False,
                )

        def forward(self, value: Any) -> Any:
            group = int(self.quest_group_size)
            weight_bits = int(self.quest_weight_bits)
            activation_precision = int(self.quest_activation_bits)
            if weight_bits != WEIGHT_BITS or activation_precision not in (4, 16):
                raise ValueError("Serialized QuEST precision metadata is invalid")
            validate_group(self.in_features, group)
            operand = rotate(value)
            if activation_precision == 4:
                operand = quantize_rotated(
                    operand,
                    scale=float(self.quest_activation_clip_scale),
                    group=group,
                )
            weight = quantize_rotated(rotate(self.weight), scale=WEIGHT_SCALE, group=group)
            return functional.linear(operand, weight, self.bias)

    return QuestMixedLinear().train(original.training)


def build_metadata(
    *,
    activation_scales: dict[str, float],
    base_model: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    """Build and validate the complete serialized route description."""
    if not isinstance(activation_scales, dict) or set(activation_scales) != set(EXPECTED_MODULES):
        raise ValueError("QuEST activation calibration must cover all 168 linears")
    if not isinstance(base_model, str) or not base_model:
        raise ValueError("QuEST base_model must be a non-empty string")
    normalized = {name: validate_scale(activation_scales[name]) for name in EXPECTED_MODULES}
    metadata = {
        "format_version": FORMAT_VERSION,
        "backend": "quest-fake-quant",
        "recipe": RECIPE,
        "base_model": base_model,
        "group_size": GROUP_SIZE,
        "weight_bits": WEIGHT_BITS,
        "weight_scale": WEIGHT_SCALE,
        "activation_bits": {"quantized": 4, "bypass": 16},
        "transform": "full-width-normalized-hadamard",
        "quantization_grid": "symmetric-mid-rise-15-interval",
        "surrogate": "quest-trust-gradient",
        "calibration": {
            "split": "train",
            "examples": CALIBRATION_EXAMPLES,
            "position_limit": CALIBRATION_POSITION_LIMIT,
            "candidate_scales": list(CALIBRATION_SCALES),
            "objective": "local-output-squared-error",
            "rows_sha256": calibration_sha256,
        },
        "weight_routes": {"w4": 168},
        "activation_routes": {"a4": 161, "a16": 7},
        "a16_modules": list(A16_MODULES),
        "a4_modules": [name for name in EXPECTED_MODULES if name not in A16_MODULES],
        "activation_scales": normalized,
        "pure_w4a4": False,
        "packed_int4": False,
        "training_quality_validated": False,
        "quality_evidence": copy.deepcopy(QUALITY_EVIDENCE),
    }
    validate_metadata(metadata)
    return metadata


def install_mixed_quest(
    model: Any,
    *,
    activation_scales: dict[str, float],
    base_model: str,
    calibration_sha256: str,
) -> dict[str, Any]:
    """Install the exact 161-A4 / 7-A16 route after full validation.

    Validation completes before any module is replaced, so a malformed late
    scale or an incompatible topology cannot leave a half-converted model.
    """
    targets = _validate_topology(model, activation_scales)
    metadata = build_metadata(
        activation_scales=activation_scales,
        base_model=base_model,
        calibration_sha256=calibration_sha256,
    )
    for name in EXPECTED_MODULES:
        parent, _, child = name.rpartition(".")
        activation_bits = 16 if name in A16_MODULES else 4
        setattr(
            model.get_submodule(parent),
            child,
            _make_linear(
                targets[name],
                activation_scale=activation_scales[name],
                activation_bits=activation_bits,
            ),
        )
    assert_route(model, metadata)
    return metadata


def assert_route(model: Any, metadata: dict[str, Any]) -> None:
    """Fail when runtime modules diverge from their serialized declaration."""
    validate_metadata(metadata)
    modules = dict(model.named_modules())
    for name in EXPECTED_MODULES:
        module = modules.get(name)
        if module is None or not hasattr(module, "quest_weight_bits"):
            raise ValueError(f"QuEST route missing module {name!r}")
        expected_activation = 16 if name in A16_MODULES else 4
        if (
            int(module.quest_weight_bits) != 4
            or int(module.quest_activation_bits) != expected_activation
            or int(module.quest_group_size) != GROUP_SIZE
            or float(module.quest_activation_clip_scale) != metadata["activation_scales"][name]
        ):
            raise ValueError(f"QuEST runtime route differs at {name!r}")


def validate_metadata(metadata: Any) -> dict[str, Any]:
    """Validate a route manifest as a closed, versioned schema."""
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise ValueError("Invalid QuEST metadata fields")
    if type(metadata["format_version"]) is not int or metadata["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported QuEST metadata format_version")
    if metadata["backend"] != "quest-fake-quant":
        raise ValueError("Invalid QuEST backend")
    if metadata["recipe"] != RECIPE:
        raise ValueError("Unknown QuEST recipe")
    if not isinstance(metadata["base_model"], str) or not metadata["base_model"]:
        raise ValueError("Invalid QuEST base_model")
    if metadata["group_size"] != GROUP_SIZE or metadata["weight_bits"] != WEIGHT_BITS:
        raise ValueError("Invalid QuEST W4 group declaration")
    if metadata["weight_scale"] != WEIGHT_SCALE:
        raise ValueError("Invalid QuEST W4 clipping scale")
    if metadata["activation_bits"] != {"quantized": 4, "bypass": 16}:
        raise ValueError("Invalid QuEST activation precision declaration")
    if metadata["transform"] != "full-width-normalized-hadamard":
        raise ValueError("Invalid QuEST transform declaration")
    if metadata["quantization_grid"] != "symmetric-mid-rise-15-interval":
        raise ValueError("Invalid QuEST quantization grid declaration")
    if metadata["surrogate"] != "quest-trust-gradient":
        raise ValueError("Invalid QuEST surrogate declaration")
    calibration = metadata["calibration"]
    if calibration != {
        "split": "train",
        "examples": CALIBRATION_EXAMPLES,
        "position_limit": CALIBRATION_POSITION_LIMIT,
        "candidate_scales": list(CALIBRATION_SCALES),
        "objective": "local-output-squared-error",
        "rows_sha256": calibration.get("rows_sha256") if isinstance(calibration, dict) else None,
    }:
        raise ValueError("Invalid QuEST calibration declaration")
    calibration_sha256 = calibration["rows_sha256"]
    if (
        not isinstance(calibration_sha256, str)
        or len(calibration_sha256) != 64
        or any(character not in "0123456789abcdef" for character in calibration_sha256)
    ):
        raise ValueError("Invalid QuEST calibration row binding")
    if metadata["weight_routes"] != {"w4": 168}:
        raise ValueError("Invalid QuEST weight route accounting")
    if metadata["activation_routes"] != {"a4": 161, "a16": 7}:
        raise ValueError("Invalid QuEST activation route accounting")
    if metadata["a16_modules"] != list(A16_MODULES):
        raise ValueError("QuEST route requires exactly the seven block-23 A16 modules")
    if metadata["a4_modules"] != [name for name in EXPECTED_MODULES if name not in A16_MODULES]:
        raise ValueError("Invalid QuEST A4 route list")
    scales = metadata["activation_scales"]
    if not isinstance(scales, dict) or set(scales) != set(EXPECTED_MODULES):
        raise ValueError("QuEST scale table must cover all 168 routes")
    for value in scales.values():
        validate_scale(value)
    if metadata["pure_w4a4"] is not False:
        raise ValueError("QuEST pure_w4a4 must be false for this mixed route")
    if metadata["packed_int4"] is not False:
        raise ValueError("QuEST packed_int4 must be false for fake quantization")
    if metadata["training_quality_validated"] is not False:
        raise ValueError("QuEST training_quality_validated must remain false")
    if metadata["quality_evidence"] != QUALITY_EVIDENCE:
        raise ValueError("QuEST quality evidence binding differs")
    return metadata


def write_metadata(
    directory: str | os.PathLike[str],
    metadata: dict[str, Any],
    *,
    validate: bool = True,
) -> Path:
    """Atomically write route metadata beside a final or periodic checkpoint."""
    if validate:
        validate_metadata(metadata)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / METADATA_NAME
    temporary = root / f".{METADATA_NAME}.{os.getpid()}.tmp"
    payload = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return destination


def load_metadata(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate bounded QuEST metadata."""
    path = Path(directory) / METADATA_NAME
    if not path.is_file():
        raise ValueError(f"missing QuEST metadata: {path}")
    if path.stat().st_size > 128 * 1024:
        raise ValueError("QuEST metadata exceeds 128 KiB")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"QuEST metadata is unreadable: {exc}") from exc
    return validate_metadata(metadata)


def validate_resume_metadata(checkpoint: str | os.PathLike[str], current: dict[str, Any]) -> None:
    """Refuse resume when calibration or routing differs from the checkpoint."""
    validate_metadata(current)
    stored = load_metadata(checkpoint)
    if stored != current:
        raise ValueError("QuEST resume metadata does not match this run's calibration and route")


def restore_mixed_quest(model: Any, metadata: dict[str, Any]) -> Any:
    """Reconstruct a serialized route on an already-loaded raw model."""
    validate_metadata(metadata)
    rebuilt = install_mixed_quest(
        model,
        activation_scales=metadata["activation_scales"],
        base_model=metadata["base_model"],
        calibration_sha256=metadata["calibration"]["rows_sha256"],
    )
    if rebuilt != metadata:
        raise ValueError("Reconstructed QuEST route differs from artifact metadata")
    return model


def load_mixed_quest_artifact(
    directory: str | os.PathLike[str], **from_pretrained_kwargs: Any
) -> Any:
    """Load a Soup QuEST artifact and restore its mixed execution route.

    Generic Transformers can load the plain master weights, but it cannot infer
    fake-quant execution from tensors alone.  This explicit loader treats the
    sidecar as mandatory and refuses unknown or malformed routes.
    """
    from transformers import AutoModelForCausalLM

    metadata = load_metadata(directory)
    model = AutoModelForCausalLM.from_pretrained(directory, **from_pretrained_kwargs)
    if getattr(model.config, "soup_quest", None) != metadata:
        raise ValueError("QuEST config declaration does not match the mandatory sidecar")
    return restore_mixed_quest(model, metadata)


def validate_cuda_hardware() -> tuple[str, tuple[int, int]]:
    """Require the Ampere-or-newer CUDA surface used by the first slice."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "quantization_aware='quest' requires a CUDA GPU (Ampere or newer); "
            "no CUDA device is available"
        )
    visible = int(torch.cuda.device_count())
    if visible != 1:
        raise RuntimeError(
            "quantization_aware='quest' first slice requires exactly one visible "
            f"CUDA GPU; found {visible}. Select one card with CUDA_VISIBLE_DEVICES."
        )
    index = torch.cuda.current_device()
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    if capability < (8, 0):
        raise RuntimeError(
            "quantization_aware='quest' requires Ampere or newer "
            f"(compute capability >= 8.0); got {capability[0]}.{capability[1]}"
        )
    return str(torch.cuda.get_device_name(index)), capability


def _build_callback_class() -> type:
    from transformers import TrainerCallback

    class _QuestMetadataCallback(TrainerCallback):
        def __init__(self, output_dir: str, metadata: dict[str, Any]) -> None:
            super().__init__()
            validate_metadata(metadata)
            self.output_dir = output_dir
            self.metadata = metadata

        def on_save(self, args, state, control, **kwargs) -> None:
            if not getattr(args, "should_save", True):
                return
            step = int(getattr(state, "global_step", 0) or 0)
            if step <= 0:
                return
            output_dir = getattr(args, "output_dir", None) or self.output_dir
            checkpoint = Path(output_dir) / f"checkpoint-{step}"
            write_metadata(checkpoint, self.metadata)

    return _QuestMetadataCallback


class QuestMetadataCallback:
    """Lazy constructor for the real Transformers callback."""

    def __new__(cls, output_dir: str, metadata: dict[str, Any]) -> Any:
        callback_class = _build_callback_class()
        return callback_class(output_dir, metadata)
