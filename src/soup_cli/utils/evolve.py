"""Evol-Instruct instruction evolution (WizardLM-style) — v0.71.31.

Depth deepens a seed instruction (add constraints / specificity / reasoning /
steps); breadth creates a new in-domain instruction. Pure module (NO top-level
torch); the generation backend is any ``generate(prompt) -> str`` callable
(e.g. ``utils/magpie.make_magpie_generate_fn`` for ollama / vllm).

Completes the synthetic-data suite (Magpie / Forge / Persona / evolve).
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

STRATEGIES = ("depth", "breadth")

# Depth methods rotate deterministically per (round, item) so a run is
# reproducible without RNG (WizardLM In-Depth Evolving operators).
_DEPTH_METHODS = (
    "Add one more constraint or requirement to the prompt.",
    "Replace a general concept in the prompt with a more specific one.",
    "Add one more step of reasoning the answer must show.",
    "Increase the depth and breadth of the prompt.",
)

_DEPTH_TEMPLATE = (
    "I want you to act as a Prompt Rewriter. Rewrite the #Given Prompt# into a "
    "more complex version. {method}\nKeep it a single self-contained "
    "instruction. Return ONLY the rewritten prompt.\n\n"
    "#Given Prompt#:\n{seed}\n\n#Rewritten Prompt#:"
)
_BREADTH_TEMPLATE = (
    "I want you to act as a Prompt Creator. Create a brand-new prompt in the "
    "same domain as the #Given Prompt# but rarer / different. Return ONLY the "
    "new prompt.\n\n#Given Prompt#:\n{seed}\n\n#Created Prompt#:"
)

# Markers whose presence in an evolution means the model echoed the meta-prompt
# rather than producing a real instruction (WizardLM elimination heuristic).
_META_MARKERS = ("#Given Prompt#", "Prompt Rewriter", "Prompt Creator", "#Rewritten Prompt#")


@dataclass(frozen=True)
class EvolvedRow:
    """One evolved instruction + its lineage (seed, strategy, round)."""

    instruction: str
    seed: str
    strategy: str
    round: int


def _pick_method(round_idx: int, item_idx: int) -> str:
    return _DEPTH_METHODS[(round_idx + item_idx) % len(_DEPTH_METHODS)]


def _render(seed: str, strategy: str, method: str) -> str:
    if strategy == "depth":
        return _DEPTH_TEMPLATE.format(method=method, seed=seed)
    return _BREADTH_TEMPLATE.format(seed=seed)


def _is_valid(evolved: str, seed: str) -> bool:
    """Reject empty / unchanged / meta-prompt-echo evolutions (elimination)."""
    text = (evolved or "").strip()
    if not text or text == seed.strip():
        return False
    if any(marker in text for marker in _META_MARKERS):
        return False
    return True


def evolve_instruction(
    seed: str,
    strategy: str,
    generate_fn: Callable[[str], str],
    *,
    method: Optional[str] = None,
) -> str:
    """Evolve a single seed instruction one step via ``generate_fn``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    the_method = method or _DEPTH_METHODS[0]
    return (generate_fn(_render(seed, strategy, the_method)) or "").strip()


def run_evolve(
    seeds: Iterable[str],
    strategy: str,
    rounds: int,
    generate_fn: Callable[[str], str],
) -> "list[EvolvedRow]":
    """Evolve ``seeds`` for ``rounds`` rounds; return valid ``EvolvedRow``s.

    Each round evolves every currently-live instruction; invalid evolutions
    (empty / unchanged / meta-echo) are eliminated. If a round eliminates
    everything, the previous generation is carried forward so later rounds still
    have material to work on.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not (1 <= rounds <= 5):
        raise ValueError("rounds must be an int in [1, 5]")
    rows: list = []
    current = [str(s) for s in seeds]
    for round_idx in range(1, rounds + 1):
        nxt = []
        for item_idx, seed in enumerate(current):
            evolved = evolve_instruction(
                seed, strategy, generate_fn, method=_pick_method(round_idx, item_idx)
            )
            if _is_valid(evolved, seed):
                rows.append(EvolvedRow(evolved, seed, strategy, round_idx))
                nxt.append(evolved)
        current = nxt or current
    return rows
