"""v0.62.0 Part E — GRACE codebook (long-running edit).

Discrete latent-space codebook for thousands of sequential knowledge edits
that survive lifelong deployments without the norm-blowup that haunts
vanilla ROME / MEMIT. Each edit stores a (key, value) pair in a learned
codebook; at inference time the model looks up the closest codebook key
to the current residual stream and applies the stored value.

Schema-only release: ``training.grace_codebook`` opt-in + codebook
size / dim validators + ``GraceCodebookConfig`` dataclass + ``grace``
added to the v0.61.0 ``SUPPORTED_EDIT_METHODS`` allowlist. Live codebook
lookup / write / EditGovernor integration lands in v0.62.1.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    from soup_cli.utils.edit_governor import EditGovernor
    from soup_cli.utils.knowledge_edit import EditPlan, EditResult

MAX_CODEBOOK_SIZE: int = 100_000
MAX_CODEBOOK_DIM: int = 16_384  # Generous upper bound matches Llama 70B hidden.

# Default epsilon-ball radius for codebook lookup (relative to key norm).
_DEFAULT_GRACE_EPSILON: float = 1.0
_GRACE_GRAD_STEPS: int = 25
_GRACE_LR: float = 0.5
_GRACE_MAX_PROMPT_TOKENS: int = 256


@dataclass(frozen=True)
class GraceCodebookConfig:
    """Resolved codebook configuration. Frozen post-construction."""

    size: int
    dim: int


def validate_grace_codebook_size(value: object) -> int:
    """Validate the codebook entry count.

    Bool-rejected (bool is a subclass of int), positive-int only, capped
    at :data:`MAX_CODEBOOK_SIZE` so a misconfigured run cannot allocate
    a multi-GB codebook by accident.
    """
    if isinstance(value, bool):
        raise TypeError(
            f"grace_codebook_size must not be bool, got {value!r}"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"grace_codebook_size must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(
            f"grace_codebook_size must be >= 1, got {value}"
        )
    if value > MAX_CODEBOOK_SIZE:
        raise ValueError(
            f"grace_codebook_size must be <= {MAX_CODEBOOK_SIZE}, got {value}"
        )
    return value


def validate_grace_codebook_dim(value: object) -> int:
    """Validate the codebook entry dimension (residual-stream width)."""
    if isinstance(value, bool):
        raise TypeError(
            f"grace_codebook_dim must not be bool, got {value!r}"
        )
    if not isinstance(value, int):
        raise TypeError(
            f"grace_codebook_dim must be int, got {type(value).__name__}"
        )
    if value < 1:
        raise ValueError(
            f"grace_codebook_dim must be >= 1, got {value}"
        )
    if value > MAX_CODEBOOK_DIM:
        raise ValueError(
            f"grace_codebook_dim must be <= {MAX_CODEBOOK_DIM}, got {value}"
        )
    return value


def build_grace_codebook_config(*, size: int, dim: int) -> GraceCodebookConfig:
    """Validate + freeze a :class:`GraceCodebookConfig`."""
    canonical_size = validate_grace_codebook_size(size)
    canonical_dim = validate_grace_codebook_dim(dim)
    return GraceCodebookConfig(size=canonical_size, dim=canonical_dim)


def apply_grace_codebook(config: GraceCodebookConfig) -> "GraceCodebook":
    """Instantiate an empty :class:`GraceCodebook` for ``config`` (v0.71.9 #203).

    Validates the config type first so callers passing a bare dict get a crisp
    ``TypeError``. Returns a usable (empty) codebook sized per ``config`` —
    callers populate it via :func:`apply_grace_edit` and apply it at decode
    time via :func:`install_grace_hook`.
    """
    if not isinstance(config, GraceCodebookConfig):
        raise TypeError(
            f"apply_grace_codebook expects GraceCodebookConfig, "
            f"got {type(config).__name__}"
        )
    return GraceCodebook(config=config)


# ---------------------------------------------------------------------------
# v0.71.9 #203 — live GRACE codebook (lookup / write + Registry persistence).
# ---------------------------------------------------------------------------


class GraceCodebook:
    """Discrete latent-space (key, value) store for sequential knowledge edits.

    Each edit appends a ``(key, value, label)`` triple. At decode time the
    residual stream is compared against every stored key; the closest key
    within ``epsilon`` (L2 distance) wins and its value replaces the residual.

    Keys / values are plain Python lists of floats (so the class stays
    numpy-free at import); :func:`save_codebook` / :func:`load_codebook`
    persist them to a JSON sidecar. Capped at ``config.size`` entries.
    """

    def __init__(
        self,
        config: GraceCodebookConfig,
        *,
        epsilon: float = _DEFAULT_GRACE_EPSILON,
        layer: int = 0,
        base_model: str = "",
    ) -> None:
        if not isinstance(config, GraceCodebookConfig):
            raise TypeError("config must be a GraceCodebookConfig")
        if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
            raise TypeError("epsilon must be a number")
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("epsilon must be a finite positive number")
        self.config = config
        self.epsilon = float(epsilon)
        self.layer = int(layer)
        self.base_model = str(base_model)
        self._keys: List[List[float]] = []
        self._values: List[List[float]] = []
        self._labels: List[str] = []

    def __len__(self) -> int:
        return len(self._keys)

    def add(self, key: List[float], value: List[float], label: str) -> None:
        """Append a (key, value, label) triple. Enforces dim + size caps."""
        if len(key) != self.config.dim or len(value) != self.config.dim:
            raise ValueError(
                f"key/value must have dim {self.config.dim}, "
                f"got key={len(key)} value={len(value)}"
            )
        if len(self._keys) >= self.config.size:
            raise ValueError(
                f"codebook is full ({self.config.size} entries); "
                "increase grace_codebook_size"
            )
        self._keys.append([float(x) for x in key])
        self._values.append([float(x) for x in value])
        self._labels.append(str(label))

    def lookup(self, query: List[float]) -> Optional[List[float]]:
        """Return the value for the nearest key within ``epsilon``, else None."""
        if not self._keys:
            return None
        if len(query) != self.config.dim:
            raise ValueError(
                f"query must have dim {self.config.dim}, got {len(query)}"
            )
        best_idx = -1
        best_dist = float("inf")
        for i, key in enumerate(self._keys):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(query, key)))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        if best_idx >= 0 and best_dist <= self.epsilon:
            return self._values[best_idx]
        return None

    def to_dict(self) -> dict:
        return {
            "size": self.config.size,
            "dim": self.config.dim,
            "epsilon": self.epsilon,
            "layer": self.layer,
            "base_model": self.base_model,
            "keys": self._keys,
            "values": self._values,
            "labels": self._labels,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GraceCodebook":
        cfg = GraceCodebookConfig(size=int(data["size"]), dim=int(data["dim"]))
        cb = cls(
            cfg,
            epsilon=float(data.get("epsilon", _DEFAULT_GRACE_EPSILON)),
            layer=int(data.get("layer", 0)),
            base_model=str(data.get("base_model", "")),
        )
        keys = data.get("keys", [])
        values = data.get("values", [])
        labels = data.get("labels", [])
        for k, v, lab in zip(keys, values, labels):
            cb.add([float(x) for x in k], [float(x) for x in v], str(lab))
        return cb


_CODEBOOK_FILENAME = "grace_codebook.json"


def save_codebook(codebook: GraceCodebook, output_dir: str) -> str:
    """Atomically write the codebook JSON under a cwd-contained ``output_dir``.

    Returns the path of the written JSON file.
    """
    from soup_cli.utils.paths import atomic_write_text, is_under_cwd

    if not isinstance(codebook, GraceCodebook):
        raise TypeError("codebook must be a GraceCodebook")
    import stat

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if "\x00" in output_dir:
        raise ValueError("output_dir must not contain null bytes")
    if not is_under_cwd(output_dir):
        raise ValueError(f"output_dir must stay under cwd: {output_dir!r}")
    if os.path.lexists(output_dir) and stat.S_ISLNK(os.lstat(output_dir).st_mode):
        raise ValueError("output_dir must not be a symlink")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, _CODEBOOK_FILENAME)
    # atomic_write_text enforces its own cwd-containment + symlink rejection
    # on the JSON file path.
    atomic_write_text(json.dumps(codebook.to_dict(), indent=2), path, field="output")
    return path


def load_codebook(output_dir: str) -> GraceCodebook:
    """Load a codebook JSON from ``output_dir`` (cwd-contained, symlink-safe)."""
    import stat

    from soup_cli.utils.paths import is_under_cwd

    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("output_dir must be a non-empty string")
    if not is_under_cwd(output_dir):
        raise ValueError(f"output_dir must stay under cwd: {output_dir!r}")
    path = os.path.join(output_dir, _CODEBOOK_FILENAME)
    # TOCTOU: reject a symlinked codebook file (raw path) before reading.
    if not os.path.lexists(path):
        raise FileNotFoundError(f"no grace codebook at {path!r}")
    if stat.S_ISLNK(os.lstat(path).st_mode):
        raise ValueError("grace codebook must not be a symlink")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no grace codebook at {path!r}")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return GraceCodebook.from_dict(data)


def install_grace_hook(model: object, codebook: GraceCodebook, *, device: str = "cpu"):
    """Install a decode-time forward hook that applies the codebook.

    For each token whose residual at the codebook's layer falls within
    ``epsilon`` of a stored key, the residual is replaced with the stored
    value. Returns the hook handle (caller removes it). Lazy torch import.
    """
    import torch

    from soup_cli.utils.edit_kernels import _locate_decoder_layers

    layers = _locate_decoder_layers(model)
    block = layers[codebook.layer]  # type: ignore[index]
    keys_t = (
        torch.tensor(codebook._keys, dtype=torch.float32, device=device)
        if codebook._keys
        else None
    )
    values_t = (
        torch.tensor(codebook._values, dtype=torch.float32, device=device)
        if codebook._values
        else None
    )

    def _hook(_mod, _args, output):
        if keys_t is None:
            return output
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        hh = hidden[0].to(torch.float32)  # [seq, dim]
        for pos in range(hh.shape[0]):
            res = hh[pos]
            dists = torch.linalg.norm(keys_t - res.unsqueeze(0), dim=1)
            best = int(torch.argmin(dists).item())
            if float(dists[best].item()) <= codebook.epsilon:
                hidden[0, pos, :] = values_t[best].to(hidden.dtype)
        if isinstance(output, (tuple, list)):
            return (hidden, *output[1:])
        return hidden

    return block.register_forward_hook(_hook)


def apply_grace_edit(
    plan: "EditPlan",
    *,
    output_dir: Optional[str] = None,
    governor: "Optional[EditGovernor]" = None,
    device: Optional[str] = None,
    trust_remote_code: bool = False,
    grad_steps: int = _GRACE_GRAD_STEPS,
    lr: float = _GRACE_LR,
) -> "EditResult":
    """Apply a GRACE knowledge edit: capture residual key + optimise value +
    append to a codebook sidecar (v0.71.9 #203).

    The base model weights are NOT modified — GRACE stores the edit in a
    discrete codebook that an inference hook applies. When ``output_dir`` is
    given the codebook is persisted (and reused/extended if one already
    exists there). Returns an :class:`EditResult`.
    """
    import torch

    from soup_cli.utils.edit_kernels import _locate_decoder_layers, measure_target_prob
    from soup_cli.utils.knowledge_edit import EditResult
    from soup_cli.utils.live_eval import load_model_and_tokenizer

    # Defensive — apply_edit already gates, but apply_grace_edit is public so
    # a direct caller must also honour the governor (review MEDIUM M5).
    if governor is not None:
        governor.check_can_edit()

    model, tokenizer, dev = load_model_and_tokenizer(
        plan.base, device=device, trust_remote_code=trust_remote_code,
    )
    layers = _locate_decoder_layers(model)
    block = layers[plan.layer]  # type: ignore[index]
    hidden_dim = int(model.config.hidden_size)

    prob_before = measure_target_prob(
        model, tokenizer, subject=plan.subject, target=plan.target, device=dev,
    )

    # Capture the residual key at the subject's last token (layer output).
    captured: List[object] = []

    def _capture(_mod, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        captured.append(hidden[0, -1, :].detach().clone())

    handle = block.register_forward_hook(_capture)
    try:
        inputs = tokenizer(
            plan.subject,
            return_tensors="pt",
            truncation=True,
            max_length=_GRACE_MAX_PROMPT_TOKENS,
        ).to(dev)
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    if not captured:
        raise ValueError("failed to capture residual key for GRACE edit")
    key_vec = captured[-1]

    # Optimise a value vector that, substituted at this position, produces the
    # target. We optimise delta added to the layer output residual.
    subj_ids = tokenizer(plan.subject, add_special_tokens=True)["input_ids"]
    tgt_ids = tokenizer(
        (" " + plan.target) if not plan.target.startswith(" ") else plan.target,
        add_special_tokens=False,
    )["input_ids"]
    if not tgt_ids:
        raise ValueError("target tokenised to an empty sequence")
    input_ids = torch.tensor([subj_ids + tgt_ids], dtype=torch.long, device=dev)
    labels = torch.tensor(
        [[-100] * len(subj_ids) + tgt_ids], dtype=torch.long, device=dev
    )
    inject_pos = len(subj_ids) - 1
    delta = torch.zeros(
        hidden_dim, device=dev, dtype=key_vec.dtype, requires_grad=True
    )

    def _inject(_mod, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.shape[1] > inject_pos:
            hidden = hidden.clone()
            hidden[0, inject_pos, :] = hidden[0, inject_pos, :] + delta
        if isinstance(output, (tuple, list)):
            return (hidden, *output[1:])
        return hidden

    optimizer = torch.optim.Adam([delta], lr=lr)
    handle = block.register_forward_hook(_inject)
    model.eval()
    try:
        for _ in range(grad_steps):
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=input_ids, labels=labels)
            out.loss.backward()
            optimizer.step()
    finally:
        handle.remove()

    value_vec = (key_vec + delta.detach()).to(torch.float32).cpu().tolist()
    key_list = key_vec.to(torch.float32).cpu().tolist()
    norm_delta = float(torch.linalg.norm(delta.detach().float()).item())

    # Build / extend the codebook.
    cfg = GraceCodebookConfig(size=MAX_CODEBOOK_SIZE, dim=hidden_dim)
    codebook: Optional[GraceCodebook] = None
    if output_dir is not None:
        from soup_cli.utils.paths import is_under_cwd

        if is_under_cwd(output_dir) and os.path.isfile(
            os.path.join(output_dir, _CODEBOOK_FILENAME)
        ):
            try:
                codebook = load_codebook(output_dir)
            except (ValueError, FileNotFoundError, json.JSONDecodeError):
                codebook = None
    if codebook is None:
        codebook = GraceCodebook(
            cfg, layer=plan.layer, base_model=plan.base,
        )
    codebook.add(key_list, value_vec, plan.subject)

    saved_dir: Optional[str] = None
    if output_dir is not None:
        save_codebook(codebook, output_dir)
        saved_dir = output_dir

    if governor is not None:
        governor.record_edit(method="grace", norm_delta=norm_delta)

    # GRACE applies at decode time; the static prob is unchanged because we
    # have not modified the weights (the codebook hook is install-time).
    prob_after = prob_before

    return EditResult(
        method="grace",
        layer=plan.layer,
        norm_delta=norm_delta,
        layers_edited=(plan.layer,),
        output_dir=saved_dir,
        target_prob_before=prob_before,
        target_prob_after=prob_after,
        governed=governor is not None,
    )
