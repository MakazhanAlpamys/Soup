"""Iterative DPO loop driver — v0.70.0 Part E.

Sample → RM-score → re-pair → retrain over N rounds. Frozen plan +
per-round artifact tracking; the actual round orchestrator (which
would invoke ``soup train --task dpo`` between rounds) is deferred to
v0.70.1 (mirrors v0.68.0 local-rl nightly-train policy).

The plan models each round explicitly so the v0.70.1 runner can:
- skip rounds whose ``adapter_path`` already exists (resume),
- re-render pairs JSONL deterministically per round,
- track per-round pairs_count for the `runs replay` integration.

Security:
- Frozen dataclasses with per-field validation (matches v0.67.0
  CmaesPlan / v0.68.0 CompilePlan policy).
- Bool / null-byte / oversize / non-int rejection on every input.
- ``rounds`` tuple required (List would not be immutable under
  ``frozen=True``; matches v0.43 Part B / v0.61 Part E policy).
- Consecutive ``round_index`` invariant: rounds must be 0..N-1 with no
  gaps (defends against caller-side bugs that would silently skip
  rounds).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

_MIN_ROUNDS = 1
_MAX_ROUNDS = 100
_MIN_PAIRS_PER_ROUND = 10
_MAX_PAIRS_PER_ROUND = 1_000_000
_MAX_PATH_LEN = 4096
_MAX_PROMPT_ROWS = 1_000_000
_MAX_ROW_BYTES = 1_000_000


def validate_rounds(value: object) -> int:
    """Validate the ``rounds`` count: int in [1, 100], bool rejected."""
    if isinstance(value, bool):
        raise ValueError("rounds must not be bool")
    if not isinstance(value, int):
        raise ValueError(f"rounds must be int, got {type(value).__name__}")
    if value < _MIN_ROUNDS:
        raise ValueError(f"rounds must be >= {_MIN_ROUNDS}, got {value}")
    if value > _MAX_ROUNDS:
        raise ValueError(
            f"rounds={value} exceeds {_MAX_ROUNDS} cap"
        )
    return value


def validate_pairs_per_round(value: object) -> int:
    """Validate the per-round pair count.

    Range: ``[10, 1_000_000]``. Below 10 pairs the DPO gradient signal
    is too noisy to be meaningful; above 1M is a clear OOM / disk hazard.
    """
    if isinstance(value, bool):
        raise ValueError("pairs_per_round must not be bool")
    if not isinstance(value, int):
        raise ValueError(
            f"pairs_per_round must be int, got {type(value).__name__}"
        )
    if value < _MIN_PAIRS_PER_ROUND:
        raise ValueError(
            f"pairs_per_round must be >= {_MIN_PAIRS_PER_ROUND}, got {value}"
        )
    if value > _MAX_PAIRS_PER_ROUND:
        raise ValueError(
            f"pairs_per_round={value} exceeds {_MAX_PAIRS_PER_ROUND} cap"
        )
    return value


def _check_path(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    # Reject newlines / CR / tab so a crafted base_model / path cannot inject
    # extra keys into the round YAML the runner renders (security review HIGH).
    if any(c in value for c in ("\n", "\r", "\t")):
        raise ValueError(f"{field} must not contain newline / tab characters")
    if len(value) > _MAX_PATH_LEN:
        raise ValueError(f"{field} exceeds {_MAX_PATH_LEN} chars")
    return value


def _check_non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must not be bool")
    if not isinstance(value, int):
        raise TypeError(f"{field} must be int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class IterativeDPORound:
    """Frozen per-round descriptor.

    - ``round_index``: 0-based, non-negative int (bool rejected).
    - ``prompts_path``: source prompts JSONL (sampled into pairs).
    - ``pairs_path``: where the round's chosen/rejected JSONL gets written.
    - ``adapter_path``: where the round's DPO adapter lands.
    - ``pairs_count``: number of pairs the round produced. Non-negative
      int (0 allowed for plan-only rendering).
    """

    round_index: int
    prompts_path: str
    pairs_path: str
    adapter_path: str
    pairs_count: int

    def __post_init__(self) -> None:
        _check_non_negative_int(self.round_index, "round_index")
        _check_path(self.prompts_path, "prompts_path")
        _check_path(self.pairs_path, "pairs_path")
        _check_path(self.adapter_path, "adapter_path")
        _check_non_negative_int(self.pairs_count, "pairs_count")


@dataclass(frozen=True)
class IterativeDPOPlan:
    """Frozen iterative-DPO plan.

    - ``base_model``: HF id / local path. Shape-validated.
    - ``reward_model``: HF id / local path of the RM used for scoring.
    - ``rounds``: tuple of :class:`IterativeDPORound`; must be
      consecutive (0..N-1 with no gaps).
    """

    base_model: str
    reward_model: str
    rounds: Tuple[IterativeDPORound, ...]

    def __post_init__(self) -> None:
        _check_path(self.base_model, "base_model")
        _check_path(self.reward_model, "reward_model")
        if not isinstance(self.rounds, tuple):
            raise TypeError(
                f"rounds must be a tuple, got {type(self.rounds).__name__}"
            )
        if len(self.rounds) < 1:
            raise ValueError("rounds must contain at least 1 round")
        for r in self.rounds:
            if not isinstance(r, IterativeDPORound):
                raise TypeError(
                    f"every rounds[] entry must be IterativeDPORound, "
                    f"got {type(r).__name__}"
                )
        for idx, r in enumerate(self.rounds):
            if r.round_index != idx:
                raise ValueError(
                    f"rounds must have consecutive round_index 0..N-1; "
                    f"rounds[{idx}].round_index={r.round_index}"
                )


def build_iterative_dpo_plan(
    *,
    base_model: str,
    reward_model: str,
    prompts_path: str,
    output_dir: str,
    rounds: int,
    pairs_per_round: int,
) -> IterativeDPOPlan:
    """Build a canonical :class:`IterativeDPOPlan` from operator inputs.

    Per-round paths follow the pattern
    ``<output_dir>/round-<NN>/{pairs.jsonl,adapter}``. The plan is
    cheap to construct; the v0.70.1 runner consumes it.
    """
    validate_rounds(rounds)
    validate_pairs_per_round(pairs_per_round)
    _check_path(base_model, "base_model")
    _check_path(reward_model, "reward_model")
    _check_path(prompts_path, "prompts_path")
    _check_path(output_dir, "output_dir")
    output_dir = output_dir.rstrip("/\\")
    per_round = []
    for i in range(rounds):
        per_round.append(
            IterativeDPORound(
                round_index=i,
                prompts_path=prompts_path,
                pairs_path=f"{output_dir}/round-{i:02d}/pairs.jsonl",
                adapter_path=f"{output_dir}/round-{i:02d}/adapter",
                pairs_count=pairs_per_round,
            )
        )
    return IterativeDPOPlan(
        base_model=base_model,
        reward_model=reward_model,
        rounds=tuple(per_round),
    )


@dataclass(frozen=True)
class IterativeDPOResult:
    """Frozen result of a completed iterative-DPO run.

    - ``rounds_completed``: number of rounds that ran end-to-end.
    - ``final_adapter``: adapter path of the last completed round.
    - ``per_round_pairs``: tuple of pair counts actually written per round.
    """

    rounds_completed: int
    final_adapter: str
    per_round_pairs: Tuple[int, ...]


def _load_prompts(path: str) -> list[str]:
    """Read prompt strings from a cwd-contained JSONL file.

    Accepts ``{"prompt": "..."}`` / ``{"prompt": [messages]}`` /
    ``{"messages": [...]}`` shapes; falls back to the raw line. Cwd
    containment + symlink rejection + DoS caps (security review MEDIUM).
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    real = enforce_under_cwd_and_no_symlink(path, "prompts_path")
    out: list[str] = []
    seen = 0
    with open(real, encoding="utf-8") as fh:
        for line in fh:
            seen += 1
            if seen > _MAX_PROMPT_ROWS:
                break
            line = line.strip()
            if not line or len(line) > _MAX_ROW_BYTES:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                out.append(line)
                continue
            if isinstance(obj, dict):
                prompt = obj.get("prompt")
                if isinstance(prompt, str):
                    out.append(prompt)
                elif isinstance(prompt, list):
                    out.append(_messages_to_text(prompt))
                elif "messages" in obj:
                    out.append(_messages_to_text(obj["messages"]))
                elif "instruction" in obj:
                    out.append(str(obj["instruction"]))
    return out


def _messages_to_text(messages: Any) -> str:
    parts: list[str] = []
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") != "assistant":
                parts.append(str(m.get("content", "")))
    return "\n".join(parts)


def build_pairs_from_scored(
    scored: list[tuple[str, float]],
) -> Optional[tuple[str, str]]:
    """Pick (chosen, rejected) from a list of (completion, score).

    Chosen = highest score, rejected = lowest. Returns ``None`` when there
    are fewer than 2 distinct-scored completions (no usable pair).
    """
    if len(scored) < 2:
        return None
    ordered = sorted(scored, key=lambda t: t[1])
    rejected, r_score = ordered[0]
    chosen, c_score = ordered[-1]
    if c_score <= r_score:
        return None
    return chosen, rejected


def _write_pairs_jsonl(pairs: list[tuple[str, str, str]], path: str) -> int:
    """Atomically write (prompt, chosen, rejected) rows; returns the count."""
    from soup_cli.utils.paths import atomic_write_text

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    for prompt, chosen, rejected in pairs:
        lines.append(
            json.dumps(
                {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            )
        )
    atomic_write_text("\n".join(lines) + ("\n" if lines else ""), path)
    return len(pairs)


def _default_sample_fn(
    *,
    base_model: str,
    adapter_path: Optional[str],
    prompts: list[str],
    num_samples: int,
    max_new_tokens: int,
    device: Optional[str],
) -> list[list[str]]:
    """Generate ``num_samples`` completions per prompt (default seam).

    Round 0 samples from ``base_model``; later rounds load the previous
    round's LoRA adapter on top of the base via PEFT (the adapter dir is
    NOT a standalone model — code-review HIGH fix).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(base_model).to(dev)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).to(dev)
    model.eval()
    out: list[list[str]] = []
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to(dev)
        completions: list[str] = []
        with torch.no_grad():
            gen = model.generate(
                **enc,
                do_sample=True,
                num_return_sequences=num_samples,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = enc["input_ids"].shape[1]
        for seq in gen:
            completions.append(
                tok.decode(seq[prompt_len:], skip_special_tokens=True)
            )
        out.append(completions)
    return out


def _default_score_fn(
    *,
    reward_model: str,
    prompt: str,
    completions: list[str],
    device: Optional[str],
) -> list[float]:
    """Score completions with a sequence-classification reward model."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(reward_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rm = AutoModelForSequenceClassification.from_pretrained(reward_model).to(dev)
    rm.eval()
    scores: list[float] = []
    with torch.no_grad():
        for completion in completions:
            enc = tok(
                prompt, completion, return_tensors="pt", truncation=True
            ).to(dev)
            logits = rm(**enc).logits
            scores.append(float(logits.reshape(-1)[0]))
    return scores


def _default_train_fn(
    *,
    base_model: str,
    pairs_path: str,
    adapter_path: str,
) -> None:
    """Run a DPO round via a ``soup train`` subprocess (no shell).

    Each round trains a fresh LoRA from ``base_model`` (always the plan's
    base, never a prior adapter dir — code-review HIGH fix) on the round's
    pairs. The YAML is rendered via ``yaml.safe_dump`` so no value can
    inject extra keys (security review HIGH).
    """
    import subprocess
    import sys
    import tempfile

    import yaml

    yaml_text = yaml.safe_dump(
        {
            "base": base_model,
            "task": "dpo",
            "data": {"train": pairs_path, "format": "dpo", "max_length": 256},
            "training": {"epochs": 1, "batch_size": 1},
            "output": {"dir": adapter_path},
        },
        default_flow_style=False,
        sort_keys=False,
    )
    fd, tmp_yaml = tempfile.mkstemp(suffix=".yaml", prefix=".soup_idpo_", dir=os.getcwd())
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(yaml_text)
        subprocess.run(  # noqa: S603 — argv list, no shell
            [sys.executable, "-m", "soup_cli.cli", "train", "--config", tmp_yaml, "--yes"],
            check=True,
        )
    finally:
        try:
            os.remove(tmp_yaml)
        except OSError:
            pass


def run_iterative_dpo(
    plan,
    *,
    sample_fn: Optional[Callable] = None,
    score_fn: Optional[Callable] = None,
    train_fn: Optional[Callable] = None,
    num_samples: int = 4,
    max_new_tokens: int = 64,
    device: Optional[str] = None,
) -> IterativeDPOResult:
    """Execute the iterative-DPO loop (v0.71.11 #239).

    For each round: sample completions from the current model, RM-score
    them, build (chosen, rejected) pairs, write them, then run a DPO
    round to produce the round's adapter. The next round samples from the
    previous adapter.

    The ``sample_fn`` / ``score_fn`` / ``train_fn`` seams default to real
    implementations (load model + generate / load RM + score / subprocess
    ``soup train``); tests inject fast fakes.
    """
    if not isinstance(plan, IterativeDPOPlan):
        raise TypeError(
            f"plan must be IterativeDPOPlan, got {type(plan).__name__}"
        )
    sample_fn = sample_fn or _default_sample_fn
    score_fn = score_fn or _default_score_fn
    train_fn = train_fn or _default_train_fn

    # ``sample_adapter`` is the LoRA the current policy is sampling from:
    # None on round 0 (sample from base), then the previous round's adapter.
    # Training always starts from ``plan.base_model`` (a fresh LoRA per
    # round) — the round's pairs carry the improvement signal. This keeps
    # the default seams from ever treating an adapter dir as a full model.
    sample_adapter: Optional[str] = None
    per_round: list[int] = []
    final_adapter = plan.base_model
    rounds_completed = 0

    for rnd in plan.rounds:
        prompts = _load_prompts(rnd.prompts_path)
        sampled = sample_fn(
            base_model=plan.base_model,
            adapter_path=sample_adapter,
            prompts=prompts,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        pairs: list[tuple[str, str, str]] = []
        for prompt, completions in zip(prompts, sampled):
            scores = score_fn(
                reward_model=plan.reward_model,
                prompt=prompt,
                completions=list(completions),
                device=device,
            )
            scored = list(zip(completions, scores))
            picked = build_pairs_from_scored(scored)
            if picked is not None:
                pairs.append((prompt, picked[0], picked[1]))
            if len(pairs) >= rnd.pairs_count:
                break
        written = _write_pairs_jsonl(pairs, rnd.pairs_path)
        per_round.append(written)
        train_fn(
            base_model=plan.base_model,
            pairs_path=rnd.pairs_path,
            adapter_path=rnd.adapter_path,
        )
        sample_adapter = rnd.adapter_path
        final_adapter = rnd.adapter_path
        rounds_completed += 1

    return IterativeDPOResult(
        rounds_completed=rounds_completed,
        final_adapter=final_adapter,
        per_round_pairs=tuple(per_round),
    )
