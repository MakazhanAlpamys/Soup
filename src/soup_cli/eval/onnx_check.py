"""Bounded, opt-in numerical validation of a no-past causal-LM ONNX export.

All inference dependencies are lazy. This checks logits on explicit inputs; it is
neither a held-out quality evaluation nor a claim about cached generation.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MAX_PROMPT_BYTES = 1024 * 1024
MAX_PROMPTS = 32
MAX_PROMPT_TOKENS = 512
DEFAULT_ATOL = 1e-4
_COMPARE_BLOCK = 1024 * 1024


class OnnxValidationError(ValueError):
    """The requested comparison cannot be performed on these inputs or graphs."""


@dataclass(frozen=True)
class OnnxValidationResult:
    max_abs_error: float
    mean_abs_error: float
    elements: int
    prompts: int
    atol: float

    @property
    def passed(self) -> bool:
        return (math.isfinite(self.max_abs_error) and math.isfinite(self.mean_abs_error)
                and self.max_abs_error <= self.atol)


def validate_atol(atol: float) -> None:
    if isinstance(atol, bool) or not math.isfinite(atol) or atol < 0:
        raise OnnxValidationError("--atol must be a finite, non-negative number.")


def load_validation_prompts(path: Path) -> list[str]:
    """Read explicit plain-text/JSONL prompts with bounded memory and no fallback parsing."""
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_PROMPT_BYTES + 1)
        if len(data) > MAX_PROMPT_BYTES:
            raise OnnxValidationError("--prompts must be at most 1 MiB.")
        text = data.decode("utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise OnnxValidationError(f"Cannot read --prompts: {exc}") from exc
    prompts = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        if path.suffix.lower() == ".jsonl":
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise OnnxValidationError(f"Invalid JSONL at prompt line {number}.") from exc
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise OnnxValidationError(
                    f"Prompt line {number} needs a non-empty 'prompt' string."
                )
        else:
            prompt = line.strip()
        prompts.append(prompt)
        if len(prompts) > MAX_PROMPTS:
            raise OnnxValidationError(f"--prompts must contain at most {MAX_PROMPTS} prompts.")
    if not prompts:
        raise OnnxValidationError("--prompts contains no non-empty prompts.")
    return prompts


def compare_logits(pairs: Iterable[tuple], *, atol: float) -> OnnxValidationResult:
    """Compare all elements, accumulating the mean by element rather than by prompt."""
    import numpy as np

    validate_atol(atol)
    maximum = total = 0.0
    elements = prompts = 0
    for reference, exported in pairs:
        reference, exported = np.asarray(reference), np.asarray(exported)
        if reference.shape != exported.shape or reference.size == 0:
            raise OnnxValidationError(
                f"Logits shapes must agree and be non-empty: {reference.shape} vs {exported.shape}."
            )
        if reference.dtype.kind != "f" or exported.dtype.kind != "f":
            raise OnnxValidationError("Both logits arrays must have floating-point dtype.")
        left, right = reference.reshape(-1), exported.reshape(-1)
        for start in range(0, left.size, _COMPARE_BLOCK):
            stop = start + _COMPARE_BLOCK
            first, second = left[start:stop], right[start:stop]
            if not np.isfinite(first).all() or not np.isfinite(second).all():
                maximum = total = math.inf
                continue
            difference = np.abs(np.subtract(first, second, dtype=np.float64))
            maximum = max(maximum, float(difference.max()))
            total += float(difference.sum(dtype=np.float64))
        elements += int(reference.size)
        prompts += 1
    if not elements:
        raise OnnxValidationError("No logits were compared.")
    return OnnxValidationResult(maximum, total / elements, elements, prompts, atol)


def validate_onnx_export(
    source: Path, exported: Path, prompts: Sequence[str], *, atol: float,
    trust_remote_code: bool = False,
) -> OnnxValidationResult:
    """Run the same untruncated, unpadded token sequences through both CPU models.

    Only one no-past graph with standard integer token inputs and a ``logits``
    output is supported. Unknown inputs are refused, never silently omitted.
    Temporary merged adapter weights must still exist at this call's boundary.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_atol(atol)
    if not prompts or len(prompts) > MAX_PROMPTS:
        raise OnnxValidationError(f"Supply between 1 and {MAX_PROMPTS} prompts.")
    if any(not isinstance(text, str) or not text.strip() for text in prompts):
        raise OnnxValidationError("Every prompt must be a non-empty string.")
    graphs = list(exported.glob("*.onnx"))
    if len(graphs) != 1:
        raise OnnxValidationError(
            "Validation needs exactly one ONNX graph in the export directory."
        )

    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(graphs[0]), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    inputs = {item.name: item.type for item in session.get_inputs()}
    allowed = {"input_ids", "attention_mask", "position_ids", "token_type_ids"}
    if "input_ids" not in inputs or set(inputs) - allowed:
        raise OnnxValidationError(
            "Validation supports no-past token inputs only; graph inputs are "
            + ", ".join(sorted(inputs)) + "."
        )
    input_dtypes = {"tensor(int64)": np.int64, "tensor(int32)": np.int32}
    if any(dtype not in input_dtypes for dtype in inputs.values()):
        raise OnnxValidationError("Validation requires int32/int64 token inputs.")
    outputs = {item.name: item.type for item in session.get_outputs()}
    if "logits" not in outputs:
        raise OnnxValidationError("Validation needs an ONNX output named 'logits'.")
    if outputs["logits"] != "tensor(float)":
        raise OnnxValidationError("Validation requires float32 ONNX logits.")
    tokenizer = AutoTokenizer.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(source), local_files_only=True, trust_remote_code=trust_remote_code,
        dtype=torch.float32, attn_implementation="eager",
    ).cpu().eval()

    def pairs() -> Iterable[tuple]:
        with torch.inference_mode():
            for number, prompt in enumerate(prompts, 1):
                batch = tokenizer(prompt, return_tensors="pt", truncation=False, padding=False)
                ids = batch["input_ids"]
                if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < ids.shape[1] <= MAX_PROMPT_TOKENS:
                    raise OnnxValidationError(
                        f"Prompt {number} must tokenize to 1..{MAX_PROMPT_TOKENS} tokens; "
                        "validation does not truncate."
                    )
                tensors = {
                    "input_ids": ids,
                    "attention_mask": batch.get("attention_mask", torch.ones_like(ids)),
                }
                if "position_ids" in inputs:
                    tensors["position_ids"] = torch.arange(ids.shape[1]).unsqueeze(0)
                if "token_type_ids" in inputs:
                    tensors["token_type_ids"] = batch.get("token_type_ids", torch.zeros_like(ids))
                feed = {name: tensors[name].numpy().astype(input_dtypes[dtype], copy=False)
                        for name, dtype in inputs.items()}
                reference = model(**tensors, use_cache=False).logits.cpu().numpy()
                actual = session.run(["logits"], feed)[0]
                yield reference, actual

    return compare_logits(pairs(), atol=atol)
