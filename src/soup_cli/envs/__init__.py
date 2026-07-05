"""Bundled toy rollout environments — v0.71.30.

Each module exposes a ``rollout(prompts) -> list[{"prompt", "answer"}]`` entry
point usable as ``soup_cli.envs.<name>:rollout`` in ``training.rollout_func``
(with ``training.rollout_backend='openenv'``). They let the live openenv GRPO
rollout path (#125) run out-of-the-box against a deterministic curriculum.

Honesty: these are deterministic single-shot **seeders**, NOT interactive
multi-turn model-in-the-loop episodes — the live openenv contract calls
``fn(seed_prompts)`` and does not pass the model/tokenizer. Each env produces a
fixed, seeded set of ``{prompt, answer}`` rows scored by the existing
``accuracy`` / ``math`` reward. The seed prompts are accepted (contract shape)
but their content does not change the generated curriculum.
"""

from soup_cli.envs import calculator, guess_number, retrieval_qa

__all__ = ["calculator", "guess_number", "retrieval_qa"]
