"""PRM-as-per-step-reward for GRPO — v0.71.30 (#PRM-guided GRPO).

Use a trained Soup PRM (the v0.53.11 ``PRMTrainerWrapper``) as the reward
function inside GRPO: split each generated completion into reasoning steps,
score every step with the PRM's scalar reward head, and fold the per-step
scores into a single scalar reward (``min`` / ``prod`` / ``last``) that GRPO
optimises. This is the o1-era process-supervision training signal.

Design:
- The pure kernels (:func:`split_steps`, :func:`aggregate_step_scores`) carry
  NO torch dependency and are unit-testable on the light core.
- :class:`PRMScorer` is a stateful ``(completions, **kwargs) -> list[float]``
  callable (torch lazy-loaded inside methods) that GRPO uses as its
  ``reward_fn``. It rides the existing shaping + ``wrap_reward_funcs`` seam in
  ``trainer/grpo.py`` unchanged, so the v0.71.26 reward-hack mitigation
  controller observes the PRM reward for free.

Honesty: proof-of-mechanism only — a tiny PRM signal is noisy; this is NOT a
production reward-model claim (see #286). Known v1 caveats:
- The step split is a newline heuristic.
- ``prm_aggregate='prod'`` assumes per-step scores in ~[0,1]; the PRM head is
  trained with unconstrained MSE, so 'prod' can blow up on uncalibrated labels
  — the default 'min' (weakest-link) is the safe choice.
- The prompt context is rendered by joining message contents (matching the
  PRM's plaintext training field), NOT via the tokenizer chat template the
  policy model sees — a distributional gap, acceptable for a proof of mechanism.
- Completions are scored one forward pass each (no batching) — fine for tiny
  models; a batched path is a future optimisation.

Security:
- ``prm_reward`` local paths are containment-checked (realpath + commonpath
  under cwd) in :func:`build_prm_reward_fn`; the reward-head weights load via
  ``safetensors.safe_open`` (no pickle).
- Bounded step count / per-step chars; aggregate mode is a closed allowlist.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:  # pragma: no cover — type-only, keeps the module torch-free
    from soup_cli.config.schema import TrainingConfig

# A GRPO completion is either a plain string or a chat message list.
Completion = Union[str, list]

# Aggregation modes for folding per-step scores into a scalar reward.
AGGREGATE_MODES: tuple[str, ...] = ("min", "prod", "last")

# Bounds — a pathological completion cannot blow up the forward pass.
_MAX_STEPS = 64
_MAX_STEP_CHARS = 2_000
# Hard ceiling on the assembled (prompt + steps) token length fed to the PRM
# forward pass. The PRM is a second, independently-loaded model whose context
# window may differ from the policy model's, and the reward fires every GRPO
# step, so cap defensively regardless of the policy-side max_length.
_MAX_INPUT_TOKENS = 8_192
# Cap the rendered prompt CONTEXT chars before tokenising (a huge prompt is
# expensive to tokenise even before the token cap applies).
_MAX_PROMPT_CHARS = 8_000


def split_steps(text: Any) -> list[str]:
    """Split a completion into reasoning steps (newline heuristic, v1).

    Drops empty / whitespace-only lines, strips each line, truncates each step
    to ``_MAX_STEP_CHARS`` and the whole list to ``_MAX_STEPS``. Returns ``[]``
    for non-string input.
    """
    if not isinstance(text, str):
        return []
    steps: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        steps.append(stripped[:_MAX_STEP_CHARS])
        if len(steps) >= _MAX_STEPS:
            break
    return steps


def _finite(value: Any) -> float:
    """Coerce ``value`` to a finite float; non-finite / non-numeric → 0.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    fv = float(value)
    return fv if math.isfinite(fv) else 0.0


def aggregate_step_scores(scores: list[float], mode: Any) -> float:
    """Fold per-step scalar scores into a single reward.

    ``mode`` is one of ``AGGREGATE_MODES``. Empty ``scores`` → 0.0. Non-finite
    per-step values are coerced to 0.0 before folding (a NaN step reward must
    not poison the whole reward).
    """
    if isinstance(mode, bool) or not isinstance(mode, str) or mode not in AGGREGATE_MODES:
        raise ValueError(f"prm_aggregate must be one of {AGGREGATE_MODES}; got {mode!r}")
    if not scores:
        return 0.0
    clean = [_finite(s) for s in scores]
    if mode == "min":
        return float(min(clean))
    if mode == "last":
        return float(clean[-1])
    # prod
    return float(math.prod(clean))


class PRMScorer:
    """Stateful GRPO reward: score each completion's steps with a Soup PRM.

    Torch / transformers / safetensors are lazy-imported inside methods so the
    module stays importable on the light core. ``__name__`` is set to
    ``"prm_reward"`` so TRL's ``rewards/<func_name>`` logging key is stable and
    the mitigation buffer records under a readable name.
    """

    __name__ = "prm_reward"

    def __init__(
        self,
        prm_path: str,
        aggregate: str = "min",
        device: str = "cpu",
        trust_remote_code: bool = False,
    ) -> None:
        if isinstance(aggregate, bool) or aggregate not in AGGREGATE_MODES:
            raise ValueError(f"aggregate must be one of {AGGREGATE_MODES}; got {aggregate!r}")
        self.prm_path = prm_path
        self.aggregate = aggregate
        self.device = device
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the base CausalLM + re-attach and load the reward head."""
        if self._model is not None:
            return
        import torch
        from torch import nn
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.prm_path, trust_remote_code=self.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            self.prm_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=dtype,
        )
        head_state = load_reward_head_weights(self.prm_path)
        hidden_size = model.config.hidden_size
        reward_head = nn.Linear(hidden_size, 1, bias=True)
        # Cast to the head weight dtype so load_state_dict matches.
        reward_head.load_state_dict(
            {
                "weight": head_state["weight"].to(torch.float32),
                "bias": head_state["bias"].to(torch.float32),
            }
        )
        reward_head = reward_head.to(dtype)
        model.reward_head = reward_head
        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._model = model
        self._tokenizer = tokenizer

    def _render_prompt(self, prompt: Any) -> str:
        """Best-effort render of a GRPO prompt (str or message list) to text."""
        if isinstance(prompt, str):
            return prompt[:_MAX_PROMPT_CHARS]
        if isinstance(prompt, (list, tuple)):
            parts: list[str] = []
            for msg in prompt:
                # Truncate each message BEFORE joining so an adversarial giant
                # content field cannot force a huge transient allocation
                # (security-review LOW).
                if isinstance(msg, dict):
                    parts.append(str(msg.get("content", ""))[:_MAX_PROMPT_CHARS])
                else:
                    parts.append(str(msg)[:_MAX_PROMPT_CHARS])
            return "\n".join(p for p in parts if p)[:_MAX_PROMPT_CHARS]
        return ""

    def _completion_text(self, completion: Completion) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        if isinstance(completion, (list, tuple)):
            parts = [
                str(m.get("content", "")) if isinstance(m, dict) else str(m) for m in completion
            ]
            return "".join(parts)
        return str(completion)

    def _score_one(self, prompt_text: str, steps: list[str]) -> float:
        import torch

        if not steps:
            return 0.0
        tokenizer = self._tokenizer
        # Prompt context (trained distribution) then step boundaries.
        prefix_ids = (
            tokenizer(prompt_text, add_special_tokens=False)["input_ids"] if prompt_text else []
        )
        input_ids = list(prefix_ids)
        step_positions: list[int] = []
        for step in steps:
            step_ids = tokenizer(step, add_special_tokens=False)["input_ids"]
            if not step_ids:
                continue
            input_ids.extend(step_ids)
            step_positions.append(len(input_ids) - 1)
        if not step_positions:
            return 0.0
        # Cap the assembled length to the PRM's own context window (bounded by a
        # hard ceiling) — the PRM may have a smaller window than the policy
        # model, and this reward fires every GRPO step. Keep the early steps.
        config = getattr(self._model, "config", None)
        max_pos = getattr(config, "max_position_embeddings", _MAX_INPUT_TOKENS)
        cap = _MAX_INPUT_TOKENS
        if isinstance(max_pos, int) and not isinstance(max_pos, bool):
            cap = min(max_pos, _MAX_INPUT_TOKENS)
        if len(input_ids) > cap:
            input_ids = input_ids[:cap]
            step_positions = [p for p in step_positions if p < cap]
            if not step_positions:
                return 0.0
        ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            outputs = self._model(input_ids=ids, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][0]  # [T, H]
            pos = torch.tensor(step_positions, dtype=torch.long, device=self.device)
            step_hidden = last_hidden.index_select(0, pos)  # [S, H]
            scores = self._model.reward_head(step_hidden).squeeze(-1)  # [S]
            per_step = [float(s) for s in scores.detach().float().cpu().tolist()]
        return aggregate_step_scores(per_step, self.aggregate)

    def __call__(self, completions: Any, **kwargs: Any) -> list[float]:
        import torch

        self._ensure_loaded()
        prompts = kwargs.get("prompts")
        rewards: list[float] = []
        completion_list = list(completions) if completions is not None else []
        if not completion_list:
             return []
        config = getattr(self._model,"config",None)
        max_pos = getattr(config,"max_position_embeddings",_MAX_INPUT_TOKENS)
        cap = _MAX_INPUT_TOKENS
        if isinstance(max_pos,int)and not isinstance(max_pos,bool):
            cap = min(max_pos,_MAX_INPUT_TOKENS)
        all_input_ids: list[list[int]]=[]
        all_step_positions: list[list[int]]=[]
        skip_indices: set[int]= set()

        for i,completion in enumerate(completion_list):
            prompt = prompts[i] if isinstance(prompts,(list, tuple)) and i<len(prompts)else None
            prompt_text = self._render_prompt(prompt)
            steps = split_steps(self._completion_text(completion))
            if not steps:
                skip_indices.add(i)
                all_input_ids.append([])
                all_step_positions.append([])
                continue
            prefix_ids =(
                self._tokenizer(prompt_text,add_special_tokens=False)["input_ids"]
                if prompt_text else[])
            input_ids = list(prefix_ids)
            step_positions: list[int]=[]
            for step in steps:
                step_ids = self._tokenizer(step,add_special_tokens=False)["input_ids"]
                if not step_ids:
                    continue
                input_ids.extend(step_ids)
                step_positions.append(len(input_ids)-1)

            if len(input_ids)>cap:
                input_ids=input_ids[:cap]
                step_positions = [p for p in step_positions if p<cap]

            if not step_positions:
                skip_indices.add(i)
                all_input_ids.append([])
                all_step_positions.append([])
                continue
            all_input_ids.append(input_ids)
            all_step_positions.append(step_positions)

        valid_indices = [i for i in range(len(completion_list)) if i not in skip_indices]
        rewards = [0.0]*len(completion_list)
        if valid_indices:
            valid_ids = [all_input_ids[i] for i in valid_indices]
            max_len = max(len(ids) for ids in valid_ids)
            batch_size = len(valid_ids)
            padded = torch.zeros(batch_size, max_len, dtype=torch.long,device=self.device)
            attn_mask = torch.zeros(batch_size,max_len, dtype=torch.long,device=self.device)
            for b,ids in enumerate(valid_ids):
                length = len(ids)
                padded[b, :length] = torch.tensor(ids,dtype=torch.long,device=self.device)
                attn_mask[b, :length] = 1
            with torch.no_grad():
                import inspect
                for b, orig_idx in enumerate(valid_indices):
                    single_ids = padded[b:b+1]
                    single_mask = attn_mask[b:b+1]
                    model_kwargs = {"input_ids": single_ids, "output_hidden_states": True}
                    try:
                        sig = inspect.signature(self._model.__call__)
                        if "attention_mask" in sig.parameters:
                            model_kwargs["attention_mask"] = single_mask
                    except (ValueError, TypeError):
                        model_kwargs["attention_mask"] = single_mask
                    outputs = self._model(**model_kwargs)
                    last_hidden = outputs.hidden_states[-1]
                    if last_hidden.dim() == 2:
                        last_hidden = last_hidden.unsqueeze(0)
                    row_len = len(all_input_ids[orig_idx])
                    steps_pos = [p for p in all_step_positions[orig_idx] if p < row_len]
                    if not steps_pos:
                        continue
                    pos = torch.tensor(steps_pos, dtype=torch.long, device=self.device)
                    step_hidden = last_hidden[0].index_select(0, pos)
                    scores = self._model.reward_head(step_hidden).squeeze(-1)
                    per_step = scores.detach().float().cpu().tolist()
                    rewards[orig_idx] = aggregate_step_scores(per_step, self.aggregate)
        return rewards

def load_reward_head_weights(prm_path: str) -> dict[str, Any]:
    """Load ``reward_head.{weight,bias}`` tensors from a Soup-trained PRM dir.

    Scans every ``*.safetensors`` shard in ``prm_path`` via
    ``safetensors.safe_open`` and collects the ``reward_head.*`` tensors.
    Raises ``ValueError`` (friendly "not a Soup-trained PRM") when absent —
    ``AutoModelForCausalLM.from_pretrained`` silently drops these keys, so a
    base checkpoint without a head must be rejected loudly.
    """
    import os

    from safetensors import safe_open

    if not os.path.isdir(prm_path):
        raise ValueError(
            f"prm_reward path is not a directory: {prm_path!r}. Expected a "
            "Soup-trained PRM produced by `soup train` with task='prm'."
        )
    collected: dict = {}
    for entry in sorted(os.listdir(prm_path)):
        if not entry.endswith(".safetensors"):
            continue
        shard = os.path.join(prm_path, entry)
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 — safe_open handle API
                if key.startswith("reward_head."):
                    collected[key[len("reward_head.") :]] = handle.get_tensor(key)
    if "weight" not in collected or "bias" not in collected:
        raise ValueError(
            f"No reward_head weights found in {prm_path!r} — this is not a "
            "Soup-trained PRM. Train one with `soup train` (task='prm') first."
        )
    return collected


def build_prm_reward_fn(
    tcfg: "TrainingConfig",
    device: str,
    trust_remote_code: bool,
) -> PRMScorer:
    """Build the :class:`PRMScorer` for GRPO from a ``TrainingConfig``.

    Validates local-path containment (realpath + commonpath under cwd) and
    surfaces a ``trust_remote_code`` probe/warning, then returns the scorer.
    A non-existent local path is treated as a Hugging Face repo id (loaded via
    ``from_pretrained``); only *existing local paths* are containment-checked.
    """
    import os

    from rich.console import Console
    from rich.markup import escape

    from soup_cli.utils.paths import is_under_cwd

    console = Console()
    prm_path = tcfg.prm_reward
    if prm_path is None:
        raise ValueError("build_prm_reward_fn called with prm_reward=None")

    # Containment: only enforce for a path that exists on disk (a bare repo id
    # with no local existence is handled by from_pretrained). Reuse the shared
    # is_under_cwd helper (realpath + commonpath, Windows-safe) so future
    # hardening applies here too (security-review LOW).
    if os.path.exists(prm_path):
        if not is_under_cwd(prm_path):
            raise ValueError(
                "prm_reward path must stay under the current working "
                f"directory; got {prm_path!r}"
            )
        prm_path = os.path.realpath(prm_path)

    resolved_trust = _resolve_trust(prm_path, trust_remote_code, console)
    # Announce that the PRM reward is active AND replaces the configured
    # reward_fn — otherwise a user who also set reward_fn/verifiable_domain has
    # no signal those are being ignored (code-review MEDIUM/LOW). escape() the
    # path so a crafted config value cannot inject Rich markup (security MEDIUM).
    console.print(
        f"[dim]Using PRM reward: prm_reward={escape(repr(prm_path))}, "
        f"aggregate={escape(repr(tcfg.prm_aggregate))} "
        "(this replaces reward_fn/verifiable_domain).[/]"
    )
    return PRMScorer(
        prm_path=prm_path,
        aggregate=tcfg.prm_aggregate,
        device=device,
        trust_remote_code=resolved_trust,
    )


def _resolve_trust(base: str, requested: bool, console: Any) -> bool:
    """Trust-remote-code probe + warn, mirroring the trainer convention."""
    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    requires = model_requires_trust_remote_code(base) or False
    return resolve_trust_remote_code(
        base,
        requested=requested,
        console=console,
        requires_remote_code=requires,
    )


__all__ = [
    "AGGREGATE_MODES",
    "PRMScorer",
    "aggregate_step_scores",
    "build_prm_reward_fn",
    "load_reward_head_weights",
    "split_steps",
]
