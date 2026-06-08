"""MoLE per-token adapter routing (v0.67.0 schema / v0.71.12 #222 live).

A gating network that routes per-token activations to a weighted blend of N
task LoRAs. Following the MoLE paper (Mixture of LoRA Experts, Wu et al. 2024),
dispatch uses softmax gating over the per-token hidden state to select top-K
LoRAs and blend them by gating weights.

v0.67.0 shipped the schema + cross-validator with a deferred ``build_gating_kernel``
stub; v0.71.12 #222 lifts the stub to a live ``torch.nn.Module`` and wires the
``MoleRoutingTrainerWrapper`` (``trainer/mole_routing.py``).

Public surface:

- ``MoleGatingConfig`` frozen dataclass
- ``validate_mole_compat(task, backend, num_task_adapters)``
- ``validate_mole_task_adapters(value)``
- ``build_gating_kernel(config)`` -> live ``torch.nn.Module`` (per-token router)
- New ``task='moe_lora_routing'`` Literal on ``SoupConfig.task``

Design notes:

- ``num_task_adapters`` bounded ``[2, 64]`` — beyond that, per-token
  softmax becomes a bottleneck; operators wanting more should hierarchy
  the gating.
- ``top_k <= num_task_adapters`` so sparse top-K dispatch is sane.
- ``temperature > 0`` to keep softmax non-degenerate; finite-only.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Tuple

from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

# torch is imported lazily inside every function that needs it (CLI-startup hot
# path stays torch-free); tensor params/returns are left unannotated to keep the
# module top torch-free (matches the existing build_gating_kernel/forward style).

# ---------------------------------------------------------------------------
# Bounds (closed, locked at module load)
# ---------------------------------------------------------------------------

MIN_TASK_ADAPTERS = 2
MAX_TASK_ADAPTERS = 64
MIN_HIDDEN_DIM = 1
MAX_HIDDEN_DIM = 16_384
MIN_TEMPERATURE = 1e-6
MAX_TEMPERATURE = 100.0
_MAX_ADAPTER_PATH_LEN = 4096


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def _check_int(value: object, field: str, lo: int, hi: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    if value < lo:
        raise ValueError(f"{field} {value} below floor {lo}")
    if value > hi:
        raise ValueError(f"{field} {value} above cap {hi}")
    return value


def _check_finite_positive(
    value: object, field: str, lo: float, hi: float
) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{field} must be finite")
    if val < lo:
        raise ValueError(f"{field} {val} below floor {lo}")
    if val > hi:
        raise ValueError(f"{field} {val} above cap {hi}")
    return val


def _check_str_field(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    return value


# ---------------------------------------------------------------------------
# Frozen config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoleGatingConfig:
    """Per-token gating network over N task LoRAs.

    Frozen dataclass — post-construction mutation raises ``FrozenInstanceError``.
    """

    num_task_adapters: int
    hidden_dim: int
    temperature: float
    top_k: int

    def __post_init__(self) -> None:
        _check_int(
            self.num_task_adapters,
            "num_task_adapters",
            MIN_TASK_ADAPTERS,
            MAX_TASK_ADAPTERS,
        )
        _check_int(
            self.hidden_dim, "hidden_dim", MIN_HIDDEN_DIM, MAX_HIDDEN_DIM
        )
        _check_finite_positive(
            self.temperature,
            "temperature",
            MIN_TEMPERATURE,
            MAX_TEMPERATURE,
        )
        # top_k must be positive and not exceed num_task_adapters
        _check_int(self.top_k, "top_k", 1, MAX_TASK_ADAPTERS)
        if self.top_k > self.num_task_adapters:
            raise ValueError(
                f"top_k {self.top_k} > num_task_adapters "
                f"{self.num_task_adapters}"
            )


# ---------------------------------------------------------------------------
# Cross-validator (called from SoupConfig + standalone)
# ---------------------------------------------------------------------------


def validate_mole_task_adapters(value: object) -> list[str]:
    """Validate the per-token MoLE task-adapter path list.

    Requires a list of `[MIN_TASK_ADAPTERS, MAX_TASK_ADAPTERS]` non-empty,
    null-byte-free, `<= _MAX_ADAPTER_PATH_LEN`-char path strings. Returns a
    defensive copy. Duplicate paths are rejected (an adapter routed to twice
    is a config error). Containment / symlink checks happen at trainer load
    time (paths may be HF ids or local dirs — mirrors `cfg.base` policy).
    """
    if not isinstance(value, (list, tuple)):
        raise TypeError("mole_task_adapters must be a list of paths")
    paths = list(value)
    if len(paths) < MIN_TASK_ADAPTERS:
        raise ValueError(
            f"mole_task_adapters needs >= {MIN_TASK_ADAPTERS} adapters, "
            f"got {len(paths)}"
        )
    if len(paths) > MAX_TASK_ADAPTERS:
        raise ValueError(
            f"mole_task_adapters {len(paths)} > cap {MAX_TASK_ADAPTERS}"
        )
    seen: set[str] = set()
    for item in paths:
        if isinstance(item, bool) or not isinstance(item, str):
            raise TypeError("each mole_task_adapter must be a string path")
        if not item:
            raise ValueError("mole_task_adapter path must be non-empty")
        if "\x00" in item:
            raise ValueError("mole_task_adapter path must not contain null bytes")
        if len(item) > _MAX_ADAPTER_PATH_LEN:
            raise ValueError(
                f"mole_task_adapter path exceeds {_MAX_ADAPTER_PATH_LEN} chars"
            )
        if item in seen:
            raise ValueError(f"duplicate mole_task_adapter path {item!r}")
        seen.add(item)
    return paths


def validate_mole_compat(
    *,
    task: str,
    backend: str,
    num_task_adapters: int,
) -> None:
    """Schema-time gate.

    - Requires ``task='moe_lora_routing'`` (silent-no-op footgun rejection
      matching v0.52.0 distill / v0.62.0 citation_faithful task-gates).
    - Rejects ``backend='mlx'`` — the gating kernel needs torch dispatch
      that mlx-lm doesn't expose (deferred to a future MLX integration).
    - ``num_task_adapters`` must be in ``[MIN_TASK_ADAPTERS, MAX_TASK_ADAPTERS]``.
    """
    _check_str_field(task, "task")
    _check_str_field(backend, "backend")
    if task != "moe_lora_routing":
        raise ValueError(
            f"validate_mole_compat: task must be 'moe_lora_routing' "
            f"(got {task!r})"
        )
    if backend == "mlx":
        raise ValueError(
            "MoLE routing is not supported on the mlx backend "
            "(the gating kernel needs torch dispatch that mlx-lm does not expose)"
        )
    _check_int(
        num_task_adapters,
        "num_task_adapters",
        MIN_TASK_ADAPTERS,
        MAX_TASK_ADAPTERS,
    )


# ---------------------------------------------------------------------------
# Live gating kernel (v0.71.12 #222)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _gating_kernel_cls():
    """Materialise the gating-kernel ``nn.Module`` class (lazy torch import).

    ``lru_cache`` makes it a process-singleton (matches ``make_mole_trainer_class``
    in ``trainer/mole_routing.py``) so the module top stays torch-free for the
    CLI-startup hot path.
    """
    import torch
    from torch import nn

    class MoleGatingKernel(nn.Module):
        """Per-token router over N task LoRAs (MoLE, Wu et al. 2024).

        ``forward(hidden)`` maps a ``[..., hidden_dim]`` hidden state to
        ``[..., num_task_adapters]`` softmax routing weights via a ``nn.Linear``
        gate + temperature-scaled top-k softmax. When ``top_k < num_task_adapters``
        only the top-k adapters per token get non-zero weight (sparse dispatch),
        and the kept logits are renormalised via softmax so the weights sum to 1.
        """

        def __init__(self, config: MoleGatingConfig):
            super().__init__()
            self.num_task_adapters = config.num_task_adapters
            self.hidden_dim = config.hidden_dim
            self.temperature = float(config.temperature)
            self.top_k = config.top_k
            # bias=False — a pure linear router over the residual stream.
            self.gate = nn.Linear(
                config.hidden_dim, config.num_task_adapters, bias=False
            )

        def forward(self, hidden):  # noqa: D401 — torch forward
            logits = self.gate(hidden) / self.temperature
            if self.top_k < self.num_task_adapters:
                topv, topi = torch.topk(logits, self.top_k, dim=-1)
                masked = torch.full_like(logits, float("-inf"))
                masked.scatter_(-1, topi, topv)
                logits = masked
            return torch.softmax(logits, dim=-1)

    return MoleGatingKernel


def build_gating_kernel(config: MoleGatingConfig):
    """Build a live per-token gating kernel for MoLE dispatch (v0.71.12 #222).

    Returns a ``torch.nn.Module`` whose forward maps a ``[..., hidden_dim]``
    hidden state to ``[..., num_task_adapters]`` per-token routing weights. The
    gate is the only trainable parameter in a MoLE run — the base model and
    every task LoRA stay frozen, and the router learns which adapter(s) each
    token should be routed to.

    Lazy torch import (heavy dep stays out of the module top, per project policy).
    """
    if not isinstance(config, MoleGatingConfig):
        raise TypeError("config must be MoleGatingConfig")
    return _gating_kernel_cls()(config)


# ---------------------------------------------------------------------------
# Serve-time MoLE (v0.71.17 #259)
# ---------------------------------------------------------------------------

_MANIFEST_FILENAME = "mole_manifest.json"
_GATE_FILENAME = "mole_gate.pt"
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024  # 1 MiB cap on the manifest file


@dataclass(frozen=True)
class MoleServeManifest:
    """Self-describing MoLE deployment manifest written next to ``mole_gate.pt``.

    Captures everything ``soup serve --mole`` needs to reconstruct the
    decode-time blend: the base model, the N frozen task-LoRA paths, and the
    gate geometry (``hidden_dim`` / ``top_k`` / ``temperature``). Frozen +
    fully validated so a tampered manifest fails loud at load time.
    """

    base: str
    adapters: Tuple[str, ...]
    num_task_adapters: int
    hidden_dim: int
    top_k: int
    temperature: float

    def __post_init__(self) -> None:
        _check_str_field(self.base, "base")
        if len(self.base) > _MAX_ADAPTER_PATH_LEN:
            raise ValueError("base too long")
        if not isinstance(self.adapters, tuple):
            raise TypeError("adapters must be a tuple")
        validate_mole_task_adapters(list(self.adapters))
        _check_int(
            self.num_task_adapters,
            "num_task_adapters",
            MIN_TASK_ADAPTERS,
            MAX_TASK_ADAPTERS,
        )
        if self.num_task_adapters != len(self.adapters):
            raise ValueError(
                f"num_task_adapters {self.num_task_adapters} != "
                f"len(adapters) {len(self.adapters)}"
            )
        _check_int(self.hidden_dim, "hidden_dim", MIN_HIDDEN_DIM, MAX_HIDDEN_DIM)
        _check_int(self.top_k, "top_k", 1, MAX_TASK_ADAPTERS)
        if self.top_k > self.num_task_adapters:
            raise ValueError(
                f"top_k {self.top_k} > num_task_adapters {self.num_task_adapters}"
            )
        _check_finite_positive(
            self.temperature, "temperature", MIN_TEMPERATURE, MAX_TEMPERATURE
        )


def _manifest_to_dict(manifest: MoleServeManifest) -> dict:
    return {
        "base": manifest.base,
        "adapters": list(manifest.adapters),
        "num_task_adapters": manifest.num_task_adapters,
        "hidden_dim": manifest.hidden_dim,
        "top_k": manifest.top_k,
        "temperature": manifest.temperature,
    }


def _require_int_field(data: dict, key: str) -> int:
    val = data.get(key)
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(f"manifest {key!r} must be an int")
    return val


def _manifest_from_dict(data: object) -> MoleServeManifest:
    # The manifest is always machine-written, so a missing / wrong-typed field
    # is tampering / corruption: fail loud rather than coerce a default into a
    # silently-wrong gate geometry (code-review MEDIUM).
    if not isinstance(data, dict):
        raise ValueError("mole manifest root must be a JSON object")
    adapters = data.get("adapters")
    if not isinstance(adapters, list):
        raise ValueError("manifest 'adapters' must be a list")
    base = data.get("base")
    if not isinstance(base, str):
        raise ValueError("manifest 'base' must be a string")
    temperature = data.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("manifest 'temperature' must be a number")
    return MoleServeManifest(
        base=base,
        adapters=tuple(adapters),
        num_task_adapters=_require_int_field(data, "num_task_adapters"),
        hidden_dim=_require_int_field(data, "hidden_dim"),
        top_k=_require_int_field(data, "top_k"),
        temperature=float(temperature),
    )


def write_mole_manifest(manifest: MoleServeManifest, directory: str) -> str:
    """Atomically write ``<directory>/mole_manifest.json``.

    The directory is the trainer's output dir (alongside ``mole_gate.pt``),
    which may live anywhere ``cfg.output`` points — so this is a plain atomic
    write (``mkstemp`` + ``os.replace``) like ``torch.save`` in the same dir,
    with a symlink guard on the target. The cwd-containment gate lives on the
    LOAD side (operator-supplied ``--mole`` path at serve time).
    """
    if not isinstance(manifest, MoleServeManifest):
        raise TypeError("manifest must be MoleServeManifest")
    if not isinstance(directory, str) or not directory:
        raise ValueError("directory must be a non-empty string")
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, _MANIFEST_FILENAME)
    if os.path.lexists(target) and stat.S_ISLNK(os.lstat(target).st_mode):
        raise ValueError("mole manifest path must not be a symlink")
    text = json.dumps(_manifest_to_dict(manifest), indent=2, sort_keys=True)
    fd, tmp = tempfile.mkstemp(prefix=".mole_manifest.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return os.path.realpath(target)


def load_mole_manifest(directory: str) -> MoleServeManifest:
    """Read + validate ``<directory>/mole_manifest.json`` (serve-side).

    cwd-contained + symlink-rejected + size-capped (operator-supplied path).
    """
    if not isinstance(directory, str) or not directory:
        raise ValueError("directory must be a non-empty string")
    if "\x00" in directory:
        raise ValueError("directory must not contain null bytes")
    manifest_path = os.path.join(directory, _MANIFEST_FILENAME)
    enforce_under_cwd_and_no_symlink(manifest_path, "mole manifest")
    real = os.path.realpath(manifest_path)
    if not os.path.exists(real):
        raise FileNotFoundError(f"mole manifest not found: {manifest_path!r}")
    st = os.lstat(real)
    if stat.S_ISLNK(st.st_mode):
        raise ValueError("mole manifest must not be a symlink (TOCTOU defence)")
    if st.st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(
            f"mole manifest size {st.st_size} > {_MAX_MANIFEST_BYTES}"
        )
    with open(real, encoding="utf-8") as fh:
        data = json.load(fh)
    return _manifest_from_dict(data)


class LoadedMole:
    """Runtime serve-time MoLE: base + N frozen task LoRAs + the trained gate.

    The decode-time per-token blend mirrors the training kernel
    (:func:`soup_cli.trainer.mole_routing.make_mole_trainer_class`): the base
    model's last hidden state drives the gate, and the N adapter logits are
    blended by the per-token gate weights. At decode we only need the LAST
    token's blend per step.

    Generation recomputes the full sequence each step (no shared KV cache —
    each task adapter would need its own cache, so recompute keeps the blend
    correct). This is fine for short demo generations on a tiny model; large
    serving deployments should expect linear-per-step cost.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        gate: Any,
        adapter_names: list[str],
        *,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gate = gate
        self.adapter_names = list(adapter_names)
        if len(self.adapter_names) < MIN_TASK_ADAPTERS:
            raise ValueError("LoadedMole needs >= 2 task adapters")
        self.device = device

    def _blended_last_logits(self, input_ids, attention_mask):
        """Per-token blend, last token only — returns ``[B, V]``."""
        import torch

        model = self.model
        with torch.no_grad(), model.disable_adapter():
            base_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        router_hidden = base_out.hidden_states[-1][:, -1, :]  # [B, H]
        weights = self.gate(router_hidden.to(self.gate.gate.weight.dtype))  # [B, N]
        # top-k masking zeroes the non-routed adapters; skip those forwards so a
        # top_k < N run does NOT pay the full N-adapter cost (code-review HIGH
        # perf). amax over every dim but the adapter axis -> [N] bool.
        col_max = weights.abs().amax(dim=tuple(range(weights.dim() - 1)))  # [N]
        active = [i for i in range(len(self.adapter_names)) if float(col_max[i]) > 0.0]
        if not active:  # degenerate (softmax should always keep >=1) — run all
            active = list(range(len(self.adapter_names)))
        blended = None
        for i in active:
            model.set_adapter(self.adapter_names[i])
            with torch.no_grad():
                out_i = model(input_ids=input_ids, attention_mask=attention_mask)
            logits_i = out_i.logits[:, -1, :].detach()  # [B, V]
            w_i = weights[..., i : i + 1].to(logits_i.dtype)
            term = w_i * logits_i
            blended = term if blended is None else blended + term
        # Reset to the first adapter so an unrelated forward never silently
        # runs only the last task adapter.
        model.set_adapter(self.adapter_names[0])
        return blended

    def generate(
        self,
        input_ids,
        attention_mask,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        """Greedy / temperature-sampled blend generation. Returns the full seq.

        Single-sequence (``B == 1``): the EOS stop inspects row 0, so a batched
        caller would stop the whole batch on the first row's EOS. The serve
        handler always sends one sequence; ``B > 1`` is rejected.
        """
        import torch

        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be a positive int")
        if input_ids.shape[0] != 1:
            raise ValueError("LoadedMole.generate supports a single sequence (B == 1)")
        seq = input_ids
        attn = attention_mask
        do_sample = isinstance(temperature, (int, float)) and not isinstance(
            temperature, bool
        ) and float(temperature) > 0.0
        for _ in range(int(max_new_tokens)):
            logits = self._blended_last_logits(seq, attn)  # [B, V]
            if do_sample:
                next_id = _sample_next(logits, float(temperature), float(top_p))
            else:
                next_id = logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            seq = torch.cat([seq, next_id], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_id)], dim=1)
            if eos_token_id is not None and int(next_id[0, 0]) == int(eos_token_id):
                break
        return seq

    def generate_text(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Tuple[str, int, int]:
        """Apply the chat template, blend-generate, decode.

        Returns ``(response, prompt_tokens, completion_tokens)`` so the serve
        chat handler can use it interchangeably with ``_generate_response``.
        """
        tokenizer = self.tokenizer
        if (
            hasattr(tokenizer, "apply_chat_template")
            and getattr(tokenizer, "chat_template", None)
        ):
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            parts = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"System: {content}")
                elif role == "user":
                    parts.append(f"User: {content}")
                elif role == "assistant":
                    parts.append(f"Assistant: {content}")
            parts.append("Assistant:")
            text = "\n".join(parts)

        inputs = tokenizer(text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        if device is not None:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        out = self.generate(
            input_ids,
            attention_mask,
            max_new_tokens=int(max_tokens),
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            temperature=temperature,
            top_p=top_p,
        )
        new_tokens = out[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, int(input_ids.shape[1]), int(len(new_tokens))


def _sample_next(logits, temperature: float, top_p: float):
    """Temperature + nucleus (top-p) sample one token per row. Returns ``[B,1]``."""
    import torch

    scaled = logits / max(temperature, 1e-6)
    probs = torch.softmax(scaled, dim=-1)
    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest set whose cumulative prob >= top_p (always >= 1).
        keep = cumulative - sorted_probs < top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(filtered, num_samples=1)
        return sorted_idx.gather(-1, choice)
    return torch.multinomial(probs, num_samples=1)


def load_mole_for_serve(
    directory: str,
    *,
    base: Optional[str] = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a serve-time MoLE runtime from a trainer output dir (#259).

    Reads the ``mole_manifest.json`` (cwd-contained, symlink-rejected) + the
    ``mole_gate.pt`` gate, loads the base + N frozen task LoRAs (PEFT
    multi-adapter), reconstructs the gate, and returns a :class:`LoadedMole`.

    ``base`` overrides the manifest's base model (the operator's ``serve``
    target); when ``None`` the manifest base is used. Lazy heavy imports.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_mole_manifest(directory)
    gate_path = os.path.join(directory, _GATE_FILENAME)
    enforce_under_cwd_and_no_symlink(gate_path, "mole gate")
    real_gate = os.path.realpath(gate_path)
    if not os.path.exists(real_gate):
        raise FileNotFoundError(f"mole gate not found: {gate_path!r}")
    if stat.S_ISLNK(os.lstat(real_gate).st_mode):
        raise ValueError("mole gate must not be a symlink (TOCTOU defence)")

    resolved_base = base or manifest.base
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_base, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        resolved_base,
        trust_remote_code=trust_remote_code,
        device_map="auto" if device == "cuda" else None,
        dtype=load_dtype,
    )
    # The gate is a Linear(hidden_dim, N); a base whose hidden size differs from
    # the manifest would shape-mismatch deep in decode. Fail loud at load
    # (code-review MEDIUM — the override base can disagree with the trained gate).
    base_hidden = getattr(base_model.config, "hidden_size", None)
    if base_hidden is not None and int(base_hidden) != manifest.hidden_dim:
        raise ValueError(
            f"--mole base hidden size {base_hidden} != gate hidden_dim "
            f"{manifest.hidden_dim}; the gate was trained against a different "
            f"base. Serve with the base the gate was trained on."
        )
    adapter_names = [f"task_{i}" for i in range(len(manifest.adapters))]
    model = PeftModel.from_pretrained(
        base_model, manifest.adapters[0], adapter_name=adapter_names[0]
    )
    for name, path in zip(adapter_names[1:], manifest.adapters[1:]):
        model.load_adapter(path, adapter_name=name)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()

    gate = build_gating_kernel(
        MoleGatingConfig(
            num_task_adapters=manifest.num_task_adapters,
            hidden_dim=manifest.hidden_dim,
            temperature=manifest.temperature,
            top_k=manifest.top_k,
        )
    )
    state = torch.load(real_gate, map_location="cpu", weights_only=True)
    gate.load_state_dict(state)
    gate.eval()
    if device == "cuda":
        gate = gate.to("cuda", dtype=torch.bfloat16)
        model = model.to("cuda")
    model.mole_gate = gate
    return LoadedMole(model, tokenizer, gate, adapter_names, device=device)
